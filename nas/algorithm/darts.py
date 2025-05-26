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

@register_eval_strategy('darts')
class DARTS():
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
        logging.info(f'Supernet begin train: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        supernet_result = NasTrainer('supernet', darts_supernet, self.loaders).run()
        logging.info(f'Supernet end train: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        logging.info(f'[*] Supernet Model : {space.param}')
        logging.info(f'[*] Model Params: {params_count(darts_supernet)}')
        logging.info(f'[*] Performance: {supernet_result}')
        
        # train sub supernet
        param = []
        for i, layer in enumerate(darts_supernet.layers):
            local_max_index = torch.argmax(layer.local_weights).item()
            global_max_index = torch.argmax(layer.global_weights).item()
            
            param.append([[space.param[i][0][local_max_index]], [space.param[i][1][global_max_index]]])
                
        dim_in, dim_out = dim_in_and_out()
        cfg.optim.max_epoch = self.retrain_best_max_epoch
        subnet_layer_type = LayerType(param=param, space=space.param)
        subnet = DARTSSubnetModel(dim_in, dim_out, subnet_layer_type).to(torch.device(cfg.accelerator))
        nas_trainer = NasTrainer('subnet', subnet, self.loaders)
        
        result = nas_trainer.run()
        
        # logging
        logging.info(f'darts NAS results:')
        logging.info(f'[*] Sub Supernet Model: {param}')
        logging.info(f'[*] Model Params: {params_count(subnet)}')
        logging.info(f'[*] Performance: {result}')
        logging.info(f'Finish Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')