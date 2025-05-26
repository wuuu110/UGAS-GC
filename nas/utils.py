import argparse
import datetime
import itertools
import os
import random
import re

import torch
import logging

import yaml

import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from scipy.stats import kendalltau, spearmanr

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

def run_loop_settings(args):
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


def flatten_weights(model_state_dict, condition=None):
    shapes = {}
    flat_weights = []
    for key, weight in model_state_dict.items():
        if condition is not None and key.startswith(condition) and weight.numel() % 2 == 0:
            shapes[key] = weight.shape
            flat_weights.append(weight.cpu().numpy().flatten().tolist())
            # flat_weights.append(torch.flatten(weight))
    flat_weights = torch.tensor(list(itertools.chain(*flat_weights)))
    return flat_weights, shapes

def unflatten_weights(flat_weights, shapes):
    model_state_dict = {}
    i = 0
    for key, shape in shapes.items():
        weight_len = int(torch.prod(torch.tensor(shape)).item())
        model_state_dict[key] = flat_weights[i:i + weight_len].view(shape)
        i += weight_len
    return model_state_dict

def layer_weight_datasets(model, condition_keys=None):
    info = model.state_dict()
    if condition_keys is None:
        condition_keys = extract_layer_number(info.keys())

    datasets = []
    for condition in condition_keys:
        f_weights, shapes = flatten_weights(info, condition)
        f_weights = f_weights.reshape( 1, -1)
        datasets.append(f_weights)
    return datasets

def extract_layer_number(condition_keys):
    result = []
    for key in condition_keys:
        match = re.search(r'(layers.\d+)', key)
        if match:
            result.append(match.group(0))
    return list(set(result))

def max_depth(lst, current_depth=1):
    if not isinstance(lst, list):
        return current_depth
    if not lst:
        return current_depth
    return max(max_depth(item, current_depth + 1) for item in lst)

def types_with_none(layers_types):
    new_layers_types = []
    for layer_types in layers_types:
        new_local_gnn_types = [local_gnn_type if local_gnn_type in layer_types[0] else 'None' for local_gnn_type in cfg.nas.supernet_layer_types[0]]
        new_global_model_types = [global_model_type if global_model_type in layer_types[1] else 'None' for global_model_type in cfg.nas.supernet_layer_types[1]]
        new_layers_types.append([new_local_gnn_types, new_global_model_types])
    return new_layers_types

def types_without_none(layers_types):
    new_layers_types = []
    for layer_types in layers_types:
        new_local_gnn_types = [local_gnn_type for local_gnn_type in layer_types[0] if local_gnn_type != 'None']
        new_global_model_types = [global_model_type for global_model_type in layer_types[1] if global_model_type != 'None']
        new_layers_types.append([new_local_gnn_types, new_global_model_types])
    return new_layers_types

def fix_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "post_mp.layer_post_mp.model" in key:
            # new_state_dict[key[6:]] = value
            new_state_dict[key] = value
        else:
            new_key = key.replace("model.", "") if "model." in key else key
            new_state_dict[new_key] = value
    return new_state_dict

def metric_best_in_perf(perf):
    if cfg.sort_fun == 'min':
        metric_best = min([epoch_info[cfg.metric_best] for epoch_info in perf[2]])
    elif cfg.sort_fun == 'max':
        metric_best = max([epoch_info[cfg.metric_best] for epoch_info in perf[2]])
    return metric_best

def dim_in_and_out():
    dim_in = cfg.share.dim_in
    dim_out = cfg.share.dim_out
    # binary classification, output dim = 1
    if 'classification' == cfg.dataset.task_type and dim_out == 2:
        dim_out = 1
    return dim_in, dim_out

def get_best_metric(list):
    if cfg.sort_fun == 'min':
        return min(list)
    elif cfg.sort_fun == 'max':
        return max(list)

def get_best_metric_idx(list):
    if cfg.sort_fun == 'min':
        return list.index(min(list))
    elif cfg.sort_fun == 'max':
        return list.index(max(list))
    
def list_to_rank(list):
    sorted_data_with_index = sorted(enumerate(list), key=lambda x: x[1], reverse=cfg.sort_fun == 'max')
    index_to_rank = {original_index: rank + 1 for rank, (original_index, _) in enumerate(sorted_data_with_index)}
    rank = [index_to_rank[i] for i in range(len(list))]
    return rank
    
def cal_correlation(list1, list2):
    rank_list1, rank_list2 = list_to_rank(list1), list_to_rank(list2)
    print("rank_list1: ", rank_list1)
    print("rank_list2: ", rank_list2)
    kendall_correlation, _ = kendalltau(rank_list1, rank_list2)
    spearman_correlation, _ = spearmanr(rank_list1, rank_list2)
    return kendall_correlation, spearman_correlation

def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list