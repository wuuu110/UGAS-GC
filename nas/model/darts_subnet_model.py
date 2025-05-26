import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from nas.layer.darts_supernet_layer import DartsSupernetLayer
from nas.layer.supernet_layer import SupernetLayer
from nas.model.darts_model import DARTSModel
from nas.model.layer_type import LayerType
from nas.model.concat_node_encoder import ConcatNodeEncoder
from nas.utils import fix_state_dict

@register_network('DARTSSubnetModel')
class DARTSSubnetModel(DARTSModel):
    """
    SupernetModel is a GNN model that can be used for NAS.
    The settings for the model are defined in the config file.
    """
    def __init__(self, dim_in=None, dim_out=None, layer_type: LayerType=None):
        super(DARTSSubnetModel, self).__init__(dim_in, dim_out)

        if layer_type is not None:
            layers = []
            param = layer_type.with_none()
            for layer_param in param:
                layers.append(DartsSupernetLayer(
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