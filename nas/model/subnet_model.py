import random

import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from nas.layer.supernet_layer import SupernetLayer
from nas.model.concat_node_encoder import ConcatNodeEncoder
from nas.model.layer_type import LayerType
from nas.model.supernet_model import SupernetModel
from nas.utils import dim_in_and_out, fix_state_dict

@register_network('SubnetModel')
class SubnetModel(SupernetModel):
    """
    SubnetModel is a model that is used to train the sub supernets in the few-shot NAS algorithm.
    Different from the SupernetModel, the SubnetModel need param.
    """
    def __init__(self, dim_in=None, dim_out=None, layer_type: LayerType=None):
        super(SubnetModel, self).__init__(dim_in, dim_out)

        if layer_type is not None:
            layers = []
            param = layer_type.with_none()
            for layer_param in param:
                layers.append(SupernetLayer(
                    dim_h=cfg.gt.dim_hidden,
                    local_gnn_types=layer_param[0],
                    global_model_types=layer_param[1],
                    num_heads=cfg.gt.n_heads,
                    act=cfg.gnn.act,
                    pna_degrees=cfg.gt.pna_degrees,
                    equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                    dropout=cfg.gt.dropout,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    bigbird_cfg=cfg.gt.bigbird,
                    log_attn_weights=cfg.train.mode == 'log-attn-weights',
                ))
            self.layers = torch.nn.Sequential(*layers)

    def load_model_state(self, state_dict, strict=True, fix=True):
        self.load_state_dict(fix_state_dict(state_dict) if fix else state_dict, strict=strict)