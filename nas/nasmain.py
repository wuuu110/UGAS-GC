import argparse
import datetime
import os
import sys
import time
from types import SimpleNamespace
import warnings
from yacs.config import CfgNode
from torch_geometric import seed_everything
from torch_geometric.graphgym import set_printing, create_loader, auto_select_device, train, params_count
from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.register import register_network, network_dict, train_dict
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from graphgps.logger import create_logger
from nas.model.subnet_model import SubnetModel
from nas.model.supernet_model import SupernetModel
from nas.model.darts_model import DARTSModel
from nas.model.specific_model import SpecificModel
from nas.model.grad_tracker import GradTracker
from nas.model.split_model import SplitModel
from nas.algorithm.one_shot import OneShot
from nas.utils import layer_weight_datasets
from nas.algorithm.few_shot_gc import FewShotGc
from nas.algorithm.darts import DARTS
from nas.algorithm.darts_gc import DARTSGC
from nas.nas_register import eval_strategy_dict
from nas.train.default_model_train import DefaultModelTrain

import torch
import random
import yaml
import logging

import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig

warnings.filterwarnings('ignore')
torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)

def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)

def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)

def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices

def sample_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    sampled_config = {}
    for key, value in config.items():
        if isinstance(value, list):
            sampled_config[key] = random.choice(value)
        elif isinstance(value, dict):
            sampled_config[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    sampled_config[key][subkey] = random.choice(subvalue)
                else:
                    sampled_config[key][subkey] = subvalue
        else:
            sampled_config[key] = value

    return sampled_config

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def train_subnet(loaders, model, optimizer, scheduler, args):
    global cfg
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # run_id = self.output_id
        custom_set_run_dir(cfg, str(1) + "/" + str(1))
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        logging.info(f"[*] Subnet: Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Subnet: Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loggers = create_logger()
        # Print model info
        logging.info(model)
        # logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            perf = train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                              scheduler)
        return perf

def set_nas_default_cfg():
    cfg.sort_fun = ""
    if 'nas' not in cfg:
        cfg.nas = CfgNode()
    cfg.nas.test = False
    cfg.nas.experiment_name = ""
    cfg.nas.enable = False
    cfg.nas.supernet_layer_types = []
    cfg.nas.layer_num = 0
    cfg.nas.specific_layers_types = []
    cfg.nas.parallel_num = 1
    
    if 'encoder' not in cfg.nas:
        cfg.nas.encoder = CfgNode()
    cfg.nas.encoder.enable = False
    cfg.nas.encoder.fixed_types = []
    cfg.nas.encoder.variable_types = []
    
    cfg.nas.algorithm = ""
    if 'any_shot' not in cfg.nas:
        cfg.nas.any_shot = CfgNode()
    cfg.nas.any_shot.supernet_max_epoch = 0
    cfg.nas.any_shot.sub_supernet_max_epoch = 0
    cfg.nas.any_shot.retrain_search_max_epoch = 0
    cfg.nas.any_shot.retrain_best_max_epoch = 0
    cfg.nas.any_shot.split_num = 0
    cfg.nas.any_shot.retrain_top_k = 0
    
    cfg.nas.search_algorithm = ""
    if 'ga' not in cfg.nas:
        cfg.nas.ga = CfgNode()
    cfg.nas.ga.max_generations = 0
    cfg.nas.ga.population_size = 0
    cfg.nas.ga.mutation_rate = 0.0
    cfg.nas.ga.crossover_rate = 0.0
    
    cfg.accelerator = ""
    
    cfg.dataset.split_index = 0
    cfg.seed = 0
    cfg.run_id = 0

def init():

    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    set_nas_default_cfg()
    cfg.seed = 42
    seed_everything(cfg.seed)
    load_cfg(cfg, args)
    dump_cfg(cfg)
    set_printing()
    auto_select_device()
    torch.set_num_threads(cfg.num_threads)
    cfg.out_dir = os.path.join(cfg.out_dir, cfg.nas.experiment_name)
    
    # seed_everything(1)
    if cfg.metric_best == 'mae':
        cfg.sort_fun = 'min'
    elif cfg.metric_best == 'accuracy' or cfg.metric_best == 'accuracy-SBM' or cfg.metric_best == 'f1' or cfg.metric_best == 'ap' or cfg.metric_best == 'auc':
        cfg.sort_fun = 'max'
    
    return args

if __name__ == '__main__':
    args = init()
    logging.info(f'Start Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    if cfg.nas.enable == False:
        trainer = DefaultModelTrain()
        trainer.train_specific_model('specific_model')
    else:
        eval_strategy_dict[cfg.nas.algorithm]().run()
    logging.info(f'End Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    