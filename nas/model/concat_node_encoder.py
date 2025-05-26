import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.encoder import AtomEncoder
from graphgps.encoder.ast_encoder import ASTNodeEncoder
from graphgps.encoder.kernel_pos_encoder import RWSENodeEncoder, \
    HKdiagSENodeEncoder, ElstaticSENodeEncoder
from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
from graphgps.encoder.ppa_encoder import PPANodeEncoder
from graphgps.encoder.signnet_pos_encoder import SignNetNodeEncoder
from graphgps.encoder.voc_superpixels_encoder import VOCNodeEncoder
from graphgps.encoder.type_dict_encoder import TypeDictNodeEncoder
from graphgps.encoder.linear_node_encoder import LinearNodeEncoder
from graphgps.encoder.equivstable_laplace_pos_encoder import EquivStableLapPENodeEncoder
from graphgps.encoder.graphormer_encoder import GraphormerEncoder

# Dataset-specific node encoders.
ds_encs = {'Atom': AtomEncoder,
           'ASTNode': ASTNodeEncoder,
           'PPANode': PPANodeEncoder,
           'TypeDictNode': TypeDictNodeEncoder,
           'VOCNode': VOCNodeEncoder,
           'LinearNode': LinearNodeEncoder}

# Positional Encoding node encoders.
pe_encs = {'LapPE': LapPENodeEncoder,
           'RWSE': RWSENodeEncoder,
           'HKdiagSE': HKdiagSENodeEncoder,
           'ElstaticSE': ElstaticSENodeEncoder,
           'SignNet': SignNetNodeEncoder,
           'EquivStableLapPE': EquivStableLapPENodeEncoder,
           'GraphormerBias': GraphormerEncoder}    

class ConcatNodeEncoder(torch.nn.Module):
    """Encoder that concatenates any number of node encoders.
    """
    def __init__(self, dim_emb, pe_enc_names):
        super().__init__()
        self.encoder_classes = []
        self.encoder_dim_pe = []
        for name in pe_enc_names:
            if name not in ds_encs and name not in pe_encs:
                raise ValueError(f"Unknown encoder name '{name}'")
            else:
                if name in ds_encs:
                    self.encoder_classes.append(ds_encs[name])
                else:
                    self.encoder_classes.append(pe_encs[name])
            if hasattr(cfg, f"posenc_{name}") and hasattr(getattr(cfg, f"posenc_{name}"), 'dim_pe'):
                self.encoder_dim_pe.append(getattr(cfg, f"posenc_{name}").dim_pe)
            else:
                self.encoder_dim_pe.append(0)
        self.encoders = []
        dim_pe_total = 0
        if cfg.posenc_EquivStableLapPE.enable: # Special handling for Equiv_Stable LapPE where node feats and PE are not concat
            for enc_cls in self.encoder_classes:
                encoder = enc_cls(dim_emb).to(torch.device(cfg.accelerator))
                self.encoders.append(encoder)
        else:
            for i, enc_cls in enumerate(self.encoder_classes):
                if i == 0:
                    encoder = enc_cls(dim_emb - sum(self.encoder_dim_pe[i + 1:])).to(torch.device(cfg.accelerator))
                else:
                    encoder = enc_cls(dim_emb - sum(self.encoder_dim_pe[i + 1:]), expand_x=False).to(torch.device(cfg.accelerator))
                self.encoders.append(encoder)         
            
    def forward(self, batch):
        for encoder in self.encoders:
            batch = encoder(batch)
        return batch