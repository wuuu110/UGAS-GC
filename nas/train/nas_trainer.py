import os
import shutil
import torch
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig
from graphgps.logger import create_logger
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from graphgps.train.custom_train import eval_epoch
from torch_geometric.graphgym.config import cfg, makedirs_rm_exist
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.checkpoint import save_ckpt
import pickle
import threading

from nas.model.layer_type import LayerType
from nas.utils import metric_best_in_perf


class NasTrainer():
    lock = threading.Lock()
    
    def __init__(
        self, 
        model_name: str,
        model: object,
        loaders: list,
        **kwargs
    ):
        self.model_name = model_name
        self.model = model
        self.loaders = loaders
        self.loggers = create_logger()
        self.optimizer = create_optimizer(model.parameters(), self._optimizer_config())
        self.scheduler = create_scheduler(self.optimizer, self._scheduler_config())
        self.kwargs = kwargs
        
        cfg.params = params_count(model)
        
        self._set_output_dir(self.model_name)
        self.layer_type = LayerType()
        self.layer_type.load_model(self.model)
        self.id = id
    
    def run(self):
        perf = self.train() if cfg.optim.max_epoch > 0 else self.inference()
        # Save checkpoint and layer type
        with NasTrainer.lock:
            self._set_output_dir(self.model_name)
            save_ckpt(self.model, self.optimizer, self.scheduler)
            self.layer_type.save(os.path.join(cfg.run_dir, 'ckpt/layer_type.pkl'))
        return perf
        
    def train(self):
        perf = train_dict[cfg.train.mode](
            self.loggers,
            self.loaders,
            self.model,
            self.optimizer,
            self.scheduler,
            self.kwargs
        )
        return metric_best_in_perf(perf)
        
    def inference(self):
        num_splits = len(self.loggers)
        split_names = ['train', 'val', 'test']
        perf = [[] for _ in range(num_splits)]
        cur_epoch = 0

        for i in range(0, num_splits):
            eval_epoch(self.loggers[i], self.loaders[i], self.model, split=split_names[i])
            perf[i].append(self.loggers[i].write_epoch(cur_epoch))
        
        return metric_best_in_perf(perf)
    
    def train_hook(self, model, epoch):
        if self.id != -1:
            print(f"Model {self.model_name} Epoch {epoch}")
            tens = model.state_dict()[f'layers.{self.id}.local_models.0.lin.weight']
            print("Sum of tensor", torch.abs(tens).sum())

    def _optimizer_config(self):
        return OptimizerConfig(
            optimizer=cfg.optim.optimizer,
            base_lr=cfg.optim.base_lr,
            weight_decay=cfg.optim.weight_decay,
            momentum=cfg.optim.momentum
        )
    
    def _scheduler_config(self):
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
    
    def _set_output_dir(self, model_name):
        cfg.run_dir = os.path.join(cfg.out_dir, model_name)
        if cfg.train.auto_resume:
            os.makedirs(cfg.run_dir, exist_ok=True)
        else:
            makedirs_rm_exist(cfg.run_dir)