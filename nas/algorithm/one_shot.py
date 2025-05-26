import logging
import time
from types import SimpleNamespace
import torch
from torch_geometric.graphgym.config import cfg
from graphgps.logger import create_logger
from nas.algorithm.genetic_algorithm import GeneticAlgorithm
from nas.model.layer_type import LayerType
from nas.model.split_model import SplitModel
from nas.model.subnet_model import SubnetModel
from nas.model.supernet_model import SupernetModel
from nas.nas_register import register_eval_strategy
from nas.train.default_model_train import DefaultModelTrain
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.comp_budget import params_count

from nas.train.nas_trainer import NasTrainer
from nas.train.parallel_trainer import ParallelTrainer
from nas.utils import dim_in_and_out

@register_eval_strategy('one_shot')
class OneShot():
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
        dim_in, dim_out = dim_in_and_out()
        
        # train supernet
        cfg.train.enable_ckpt = False
        cfg.optim.max_epoch = self.supernet_max_epoch
        space = LayerType(param=[cfg.nas.supernet_layer_types] * cfg.nas.layer_num)
        
        supernet = create_model().model
        print(supernet)
        supernet_result = NasTrainer('supernet', supernet, self.loaders).run()
        logging.info(f'[*] Supernet Model : {space.param}')
        logging.info(f'[*] Model Params: {params_count(supernet)}')
        logging.info(f'[*] Performance: {supernet_result}')
        
        # search for the best k
        cfg.optim.max_epoch = self.retrain_search_max_epoch
        ga = GeneticAlgorithm(supernet, space.param, self.loaders)
        ga.search()
        population, metric_values = ga.population, ga.metric_values
        
        # retrain the top k
        cfg.optim.max_epoch = self.retrain_best_max_epoch
        top_k_idx = sorted(range(len(metric_values)), key=lambda i: metric_values[i], reverse=cfg.sort_fun == 'max')[:self.top_k]
        top_k_population = [population[idx] for idx in top_k_idx]
        top_k_model = [SubnetModel(dim_in, dim_out, pop).to(torch.device(cfg.accelerator)) for pop in top_k_population]
        dim_in, dim_out = dim_in_and_out()
        parallel_task = [
            NasTrainer(
                f'top_{idx + 1}_model',
                model, 
                self.loaders
            ) for idx, model in  enumerate(top_k_model)
        ]
        results = ParallelTrainer(parallel_task).run()

        # logging
        logging.info(f'One-shot NAS results:')
        for idx, pop, model, result in zip(top_k_idx, top_k_population, top_k_model, results):
            logging.info(f'[*] Final Model {idx + 1}: {pop.param}')
            logging.info(f'[*] Model Params: {params_count(model)}')
            logging.info(f'[*] Performance: {result}')

        logging.info(f'Finish Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        