import datetime
import logging
import os

import time
from types import SimpleNamespace
import torch
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, makedirs_rm_exist
from torch_geometric.graphgym import auto_select_device
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.register import train_dict

from graphgps.logger import create_logger
from graphgps.train.custom_train import eval_epoch
from nas.model.subnet_model import SubnetModel
from nas.model.supernet_model import SupernetModel
                                             
class DefaultModelTrain():
    def __init__(self):
        super().__init__()
        self.dim_out = 1
        if cfg.dataset.name == 'MNIST' or cfg.dataset.name == 'CIFAR10':
            self.dim_out = 10
    
    def train_specific_model(self, model_name):
        self.set_output_dir(model_name)
        loaders = create_loader()
        loggers = create_logger()
        
        model = create_model()
        print(model.model)
        cfg.params = params_count(model)
        optimizer = create_optimizer(model.parameters(), self.optimizer_config())
        scheduler = create_scheduler(optimizer, self.scheduler_config())
        
        logging.info(
            f"Starting training with the following settings:\n \
            [*] Model: {model_name}\n \
            [*] layers_types: {cfg.nas.supernet_layer_types}\n \
            [*] Number of Parameters: {params_count(model)}\n \
            [*] Device: {cfg.accelerator}\n \
            [*] Seed: {cfg.seed}\n \
            [*] Start Time: {datetime.datetime.now()}"
        )
        perf = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)
        return model, perf
        
    def init_supernet(self, model_name):
        self.set_output_dir(model_name)
        loaders = create_loader()
        loggers = create_logger()
        
        model = create_model()
        # print(model)
        cfg.params = params_count(model)
        optimizer = create_optimizer(model.parameters(), self.optimizer_config())
        scheduler = create_scheduler(optimizer, self.scheduler_config())
        
        logging.info(
            f"Starting training with the following settings:\n \
            [*] Model: {model_name}\n \
            [*] layers_types: {[cfg.nas.supernet_layer_types] * cfg.nas.layer_num}\n \
            [*] Number of Parameters: {params_count(model)}\n \
            [*] Device: {cfg.accelerator}\n \
            [*] Seed: {cfg.seed}\n \
            [*] Start Time: {datetime.datetime.now()}"
        )
        
        return model, loggers, loaders, optimizer, scheduler
    
    def init_subnet(self, model_name, super_model, loaders, layers_types, enc_types=None):
        self.set_output_dir(model_name)
        loggers = create_logger()
        
        model = SubnetModel(5, 10, layers_types).to(torch.device(cfg.accelerator))
        print(model)
        if super_model is not None:
            info = super_model.state_dict()
            if 'model_state' in info:
                # info['model_state'] = {k.replace('model.', ''): v for k, v in info['model_state'].items()}
                model.load_state_dict(info['model_state'])
            model.load_state_dict(self.fix_state_dict(info), strict=True)

        cfg.params = params_count(model)
        optimizer = create_optimizer(model.parameters(), self.optimizer_config())
        scheduler = create_scheduler(optimizer, self.scheduler_config())
                
        logging.info(
            f"Starting training with the following settings:\n \
            [*] Model: {model_name}\n \
            [*] layers_types: {layers_types}\n \
            [*] enc_types: {enc_types}\n \
            [*] Number of Parameters: {params_count(model)}\n \
            [*] Device: {cfg.accelerator}\n \
            [*] Seed: {cfg.seed}\n \
            [*] Start Time: {datetime.datetime.now()}"
        )
        
        return model, loggers, loaders, optimizer, scheduler
    
    def init_components(self, model_name, loaders=None, super_model=None, layers_types=None, enc_types=None):
        if 'nas' not in cfg:
            cfg['nas'] = SimpleNamespace()
        cfg.nas.layers_params = self.layers_params

        seed_everything(self.seed)
        self.set_output_dir(model_name)
        if loaders is None:                
            loaders = create_loader()
        loggers = create_logger()
        
        if super_model is None:
            model = create_model()
        else:
            model = SubnetModel(1, self.dim_out, self.layers_params).to(torch.device(cfg.accelerator))
            info = super_model.state_dict()
            if 'model_state' in info:
                info['model_state'] = {k.replace('model.', ''): v for k, v in info['model_state'].items()}
                model.load_state_dict(info['model_state'])

        cfg.params = params_count(model)
        optimizer = create_optimizer(model.parameters(), self.optimizer_config())
        scheduler = create_scheduler(optimizer, self.scheduler_config())
        
    def train(self, model, loggers, loaders, optimizer, scheduler, split_model=None):
        # logging.info(cfg)
        # logging.info(model)
        
        if split_model != None:
            perf = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler, split_model)
        else:
            perf = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

        return model, perf
    
    def inference(self, loggers, loaders, model):
        num_splits = len(loggers)
        split_names = ['train', 'val', 'test']
        perf = [[] for _ in range(num_splits)]
        cur_epoch = 0

        for i in range(0, num_splits):
            eval_epoch(loggers[i], loaders[i], model,
                    split=split_names[i])
            perf[i].append(loggers[i].write_epoch(cur_epoch))
        
        return perf
        
    def optimizer_config(self):
        return OptimizerConfig(
            optimizer=cfg.optim.optimizer,
            base_lr=cfg.optim.base_lr,
            weight_decay=cfg.optim.weight_decay,
            momentum=cfg.optim.momentum
        )
    
    def scheduler_config(self):
        return ExtendedSchedulerConfig(
            scheduler=cfg.optim.scheduler,
            steps=cfg.optim.steps, 
            lr_decay=cfg.optim.lr_decay,
            max_epoch=cfg.optim.max_epoch, 
            reduce_factor=cfg.optim.reduce_factor,
            schedule_patience=cfg.optim.schedule_patience, 
            min_lr=cfg.optim.min_lr,
            num_warmup_epochs=cfg.optim.num_warmup_epochs,
            train_mode=cfg.train.mode, 
            eval_period=cfg.train.eval_period
        )
    

    
    def fix_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "post_mp.layer_post_mp.model" in key:
                new_state_dict[key[6:]] = value
            else:
                new_key = key.replace("model.", "") if "model." in key else key
                new_state_dict[new_key] = value
        return new_state_dict
    
    def set_output_dir(self, model_name):
        cfg.run_dir = os.path.join(cfg.out_dir, model_name)
        if cfg.train.auto_resume:
            os.makedirs(cfg.run_dir, exist_ok=True)
        else:
            makedirs_rm_exist(cfg.run_dir)