from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from types import SimpleNamespace
import torch
from torch_geometric.graphgym.config import cfg
from graphgps.logger import create_logger
from nas.algorithm.genetic_algorithm import GeneticAlgorithm
from nas.layer.darts_supernet_layer import DartsSupernetLayer
from nas.model.darts_subnet_model import DARTSSubnetModel
from nas.model.layer_type import LayerType
from nas.model.split_model import SplitModel
from nas.model.subnet_model import SubnetModel
from nas.model.supernet_model import SupernetModel
from nas.nas_register import register_eval_strategy
from nas.train.default_model_train import DefaultModelTrain
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.comp_budget import params_count

from nas.train.nas_trainer import NasTrainer
from nas.train.parallel_trainer import ParallelTrainer
from nas.utils import dim_in_and_out

@register_eval_strategy('darts_gc')
class DARTSGC():
    def __init__(self):
        super().__init__()
        self.supernet_layers_types = [cfg.nas.supernet_layer_types] * cfg.nas.layer_num
        self.supernet_max_epoch = cfg.nas.any_shot.supernet_max_epoch
        self.sub_supernet_max_epoch = cfg.nas.any_shot.sub_supernet_max_epoch
        self.retrain_search_max_epoch = cfg.nas.any_shot.retrain_search_max_epoch
        self.retrain_best_max_epoch = cfg.nas.any_shot.retrain_best_max_epoch
        self.split_num = cfg.nas.any_shot.split_num
        self.top_k = cfg.nas.any_shot.retrain_top_k
        self.loaders = create_loader()
        
    def run(self):    
        # train supernet
        cfg.train.enable_ckpt = False
        cfg.optim.max_epoch = self.supernet_max_epoch
        space = LayerType(param=[cfg.nas.supernet_layer_types] * cfg.nas.layer_num)
        
        darts_supernet = create_model().model
        split_supernet = SplitModel(darts_supernet, split_type='grad_contrib')
        logging.info(f'Supernet begin train: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        supernet_result = NasTrainer('supernet', darts_supernet, self.loaders, split_model=split_supernet).run()
        logging.info(f'Supernet end train: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        logging.info(f'[*] Supernet Model : {space.param}')
        logging.info(f'[*] Model Params: {params_count(darts_supernet)}')
        logging.info(f'[*] Performance: {supernet_result}')
        
        # train darts sub supernets
        dim_in, dim_out = dim_in_and_out()
        cfg.optim.max_epoch = cfg.nas.any_shot.sub_supernet_max_epoch
        sub_supernets_params = split_supernet.get_split_model_params(self.split_num)
        logging.info(f'Supernet end split: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        sub_supernets_layer_type = [LayerType(param=param, space=space.param) for param in sub_supernets_params]
        sub_supernets = [DARTSSubnetModel(dim_in, dim_out, layer_type).to(torch.device(cfg.accelerator))
                         for layer_type in sub_supernets_layer_type]
        # for sub_supernet in sub_supernets:
        #     sub_supernet.load_model_state(darts_supernet.state_dict(), )
        parallel_task = [
            NasTrainer(
                f'sub_supernet_{idx}',
                sub_supernet,
                self.loaders
            ) for idx, sub_supernet in enumerate(sub_supernets)
        ]
        sub_supernet_results = ParallelTrainer(parallel_task).run()
        logging.info(f'Sub supernets training finished.')
        for idx, layer_type, sub_supernet, result in zip(range(len(sub_supernets)), sub_supernets_layer_type, sub_supernets, sub_supernet_results):
            logging.info(f'[*] Sub Supernet Model {idx + 1}: {layer_type.param}')
            logging.info(f'[*] Model Params: {params_count(sub_supernet)}')
            logging.info(f'[*] Performance: {result}')
        
        # train sub supernet
        params = []
        for layer_type, darts_sub_supernet in zip(sub_supernets_layer_type, sub_supernets):
            param = []
            for i, layer in enumerate(darts_sub_supernet.layers):
                local_max_index = torch.argmax(layer.local_weights).item()
                global_max_index = torch.argmax(layer.global_weights).item()
                
                param.append([[layer_type.param[i][0][local_max_index]], [layer_type.param[i][1][global_max_index]]])
            params.append(param)
                
        cfg.optim.max_epoch = self.retrain_best_max_epoch
        sub_supernets_layer_type = [LayerType(param=param, space=space.param) for param in params]
        sub_supernets = [DARTSSubnetModel(dim_in, dim_out, layer_type).to(torch.device(cfg.accelerator))
                         for layer_type in sub_supernets_layer_type]
        parallel_task = [
            NasTrainer(
                f'sub_supernet_{idx}',
                sub_supernet,
                self.loaders
            ) for idx, sub_supernet in enumerate(sub_supernets)
        ]
        
        results = ParallelTrainer(parallel_task).run()
        combined = sorted(zip(params, sub_supernets, results), key=lambda x: x[2], reverse=cfg.metric_best=='max')
        params, sub_supernets, results = zip(*combined)
        
        # logging
        logging.info(f'darts-gc NAS results:')
        
        for param, model, result in zip(params, sub_supernets, results):
            logging.info(f'[*] Sub Supernet Model: {param}')
            logging.info(f'[*] Model Params: {params_count(model)}')
            logging.info(f'[*] Performance: {result}')
            
        logging.info(f'Finish Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')