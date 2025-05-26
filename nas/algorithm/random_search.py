import yaml
import random

from nas.model.gps_nas_model import GPSNASModel
from nas.utils import custom_set_out_dir, new_optimizer_config, new_scheduler_config, custom_set_run_dir, \
    run_loop_settings

import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs

from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
from torch_geometric.graphgym.config import cfg, load_cfg

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

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

class RandomSearch():
    def __init__(self, args, num_epochs=20):
        self.num_epochs = num_epochs
        self.args = args
    def search(self):
        global cfg
        for epoch in range(self.num_epochs):
            # Sample a model
            sampled_config = sample_config('./configs/space.yaml')
            cfg.gnn.agg = sampled_config['gnn']['agg']
            cfg.model.type = 'GPSNASModel'
            # Train the model
            for run_id, seed, split_index in zip(*run_loop_settings(self.args)):
                new_run_id = epoch
                cfg.dataset.split_index = split_index
                cfg.seed = seed
                cfg.run_id = new_run_id
                # Set configurations for each run
                custom_set_run_dir(cfg, new_run_id)
                set_printing()
                seed_everything(cfg.seed)
                auto_select_device()
                if cfg.pretrained.dir:
                    cfg = load_pretrained_model_cfg(cfg)
                logging.info(f"[*] Run ID {new_run_id}: seed={cfg.seed}, "
                             f"split_index={cfg.dataset.split_index}")
                logging.info(f"    Starting now: {datetime.datetime.now()}")
                # Set machine learning pipeline
                loaders = create_loader()
                loggers = create_logger()
                model = create_model()
                if cfg.pretrained.dir:
                    model = init_model_from_pretrained(
                        model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                        cfg.pretrained.reset_prediction_head, seed=cfg.seed
                    )
                optimizer = create_optimizer(model.parameters(),
                                             new_optimizer_config(cfg))
                scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
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
                    train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                               scheduler)
            # # Aggregate results from different seeds
            # try:
            #     agg_runs(cfg.out_dir, cfg.metric_best)
            # except Exception as e:
            #     logging.info(f"Failed when trying to aggregate multiple runs: {e}")
            # # When being launched in batch mode, mark a yaml as done
            # if args.mark_done:
            #     os.rename(args.cfg_file, f'{args.cfg_file}_done')
            # logging.info(f"[*] All done: {datetime.datetime.now()}")

            # save result
            # nas_model.evaluate()
