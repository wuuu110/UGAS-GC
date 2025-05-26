import numpy as np
import torch
import torch.nn as nn
from performer_pytorch import SelfAttention
import torch_geometric.nn as pygnn
import torch_geometric.graphgym.register as register
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE

from torch_geometric.graphgym.config import cfg

class DartsSupernetLayer(nn.Module):
    def __init__(self, dim_h,
                 local_gnn_types, global_model_types, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.local_gnn_types = local_gnn_types
        self.global_model_types = global_model_types
        
        self.filtered_local_gnn_types = [gnn_type for gnn_type in local_gnn_types if gnn_type != 'None']
        self.filtered_global_model_types = [global_model_type for global_model_type in global_model_types if global_model_type != 'None']
        self.local_len = len(self.local_gnn_types)
        self.global_len = len(self.global_model_types)
        
        self.local_weights = nn.Parameter(torch.ones(self.local_len))
        self.global_weights = nn.Parameter(torch.ones(self.global_len))
        self.register_parameter('local_weights', self.local_weights)
        self.register_parameter('global_weights', self.global_weights)

        self.log_attn_weights = log_attn_weights

        if log_attn_weights and not all(
                global_model_type in ['Transformer', 'BiasedTransformer'] for global_model_type in global_model_types):
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_types}' global attention model."
            )

        # Local message-passing model.
        self.local_models = nn.ModuleList()
        self.local_gnn_with_edge_attr = []

        for local_gnn_type in local_gnn_types:
            if local_gnn_type == "None":
                self.local_models.append(None)
                self.local_gnn_with_edge_attr.append(False)
            elif local_gnn_type == "GCN":
                self.local_gnn_with_edge_attr.append(False)
                self.local_models.append(pygnn.GCNConv(dim_h, dim_h))
            elif local_gnn_type == 'GIN':
                self.local_gnn_with_edge_attr.append(False)
                gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                       self.activation(),
                                       Linear_pyg(dim_h, dim_h))
                self.local_models.append(pygnn.GINConv(gin_nn))

                # MPNNs supporting also edge attributes.
            elif local_gnn_type == 'GENConv':
                self.local_gnn_with_edge_attr.append(True)
                self.local_models.append(pygnn.GENConv(dim_h, dim_h))
            elif local_gnn_type == 'GINE':
                self.local_gnn_with_edge_attr.append(True)
                gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                       self.activation(),
                                       Linear_pyg(dim_h, dim_h))
                if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                    self.local_models.append(GINEConvESLapPE(gin_nn))
                else:
                    self.local_models.append(pygnn.GINEConv(gin_nn))
            elif local_gnn_type == 'GAT':
                self.local_gnn_with_edge_attr.append(True)
                self.local_models.append(pygnn.GATConv(in_channels=dim_h,
                                                 out_channels=dim_h // num_heads,
                                                 heads=num_heads,
                                                 edge_dim=dim_h))
            elif local_gnn_type == 'PNA':
                # Defaults from the paper.
                # aggregators = ['mean', 'min', 'max', 'std']
                # scalers = ['identity', 'amplification', 'attenuation']
                self.local_gnn_with_edge_attr.append(True)
                aggregators = ['mean', 'max', 'sum']
                scalers = ['identity']
                deg = torch.from_numpy(np.array(pna_degrees))
                self.local_models.append(pygnn.PNAConv(dim_h, dim_h,
                                                 aggregators=aggregators,
                                                 scalers=scalers,
                                                 deg=deg,
                                                 edge_dim=min(128, dim_h),
                                                 towers=1,
                                                 pre_layers=1,
                                                 post_layers=1,
                                                 divide_input=False))
            elif local_gnn_type == 'CustomGatedGCN':
                self.local_gnn_with_edge_attr.append(True)
                self.local_models.append(GatedGCNLayer(dim_h, dim_h,
                                                 dropout=dropout,
                                                 residual=True,
                                                 act=act,
                                                 equivstable_pe=equivstable_pe))
            else:
                raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_types = local_gnn_types

        # Global attention transformer-style model.
        self.self_attns = nn.ModuleList()
        for global_model_type in global_model_types:
            if global_model_type == 'None':
                self.self_attns.append(None)
            elif global_model_type in ['Transformer', 'BiasedTransformer']:
                self.self_attns.append(torch.nn.MultiheadAttention(
                    dim_h, num_heads, dropout=self.attn_dropout, batch_first=True))
                # self.global_model = torch.nn.TransformerEncoderLayer(
                #     d_model=dim_h, nhead=num_heads,
                #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
                #     layer_norm_eps=1e-5, batch_first=True)
            elif global_model_type == 'Performer':
                self.self_attns.append(SelfAttention(
                    dim=dim_h, heads=num_heads,
                    dropout=self.attn_dropout, causal=False))
            elif global_model_type == "BigBird":
                bigbird_cfg.dim_hidden = dim_h
                bigbird_cfg.n_heads = num_heads
                bigbird_cfg.dropout = dropout
                self.self_attns.append(SingleBigBirdLayer(bigbird_cfg))
            else:
                raise ValueError(f"Unsupported global x-former model: "
                                 f"{global_model_type}")
            self.global_model_types = global_model_types

            if self.layer_norm and self.batch_norm:
                raise ValueError("Cannot apply two types of normalization together")
        
        # # mix all local gnn outputs
        # self.local_gnn_mlp = nn.Sequential(
        #     nn.Linear(dim_h * len(local_gnn_types), dim_h * len(local_gnn_types)),
        #     nn.ReLU(),
        #     nn.Linear(dim_h * len(local_gnn_types), dim_h)
        # )
       
        # # mix all global model outputs
        # self.global_model_mlp = nn.Sequential(
        #     nn.Linear(dim_h * len(global_model_types), dim_h * len(global_model_types)),
        #     nn.ReLU(),
        #     nn.Linear(dim_h * len(global_model_types), dim_h)
        # )

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection
        h_out_list = []

        if len(self.local_gnn_types) >= 1:
            # Local MPNN with edge attributes.
            h_local_list = []
            for i in range(len(self.local_models)):
                local_gnn_type = self.local_gnn_types[i]
                if local_gnn_type == 'None':
                    h_local_list.append(h)
                    continue
                local_model = self.local_models[i]
                local_gnn_with_edge_attr = self.local_gnn_with_edge_attr[i]
                if local_gnn_type == 'CustomGatedGCN':
                    es_data = None
                    if self.equivstable_pe:
                        es_data = batch.pe_EquivStableLapPE
                    test_batch = Batch(batch=batch,
                                        x=h,
                                        edge_index=batch.edge_index,
                                        edge_attr=batch.edge_attr,
                                        pe_EquivStableLapPE=es_data)
                    local_out = local_model(test_batch)
                    # local_out = local_model(Batch(batch=batch,
                    #                               x=h,
                    #                               edge_index=batch.edge_index,
                    #                               edge_attr=batch.edge_attr,
                    #                               pe_EquivStableLapPE=es_data))
                    # GatedGCN does residual connection and dropout internally.
                    h_local = local_out.x
                    batch.edge_attr = local_out.edge_attr
                else:
                    if local_gnn_with_edge_attr:
                        if self.equivstable_pe:
                            h_local = local_model(h,
                                                  batch.edge_index,
                                                  batch.edge_attr,
                                                  batch.pe_EquivStableLapPE)
                        else:
                            h_local = local_model(h,
                                                  batch.edge_index,
                                                  batch.edge_attr)
                    else:
                        h_local = local_model(h, batch.edge_index)
                    h_local = self.dropout_local(h_local)
                    h_local = h_in1 + h_local  # Residual connection.
                h_local_list.append(h_local)
            
            if len(self.filtered_local_gnn_types) >= 1:
                local_weights_reshaped = torch.softmax(self.local_weights, dim=0).view(self.local_len, 1, 1)
                h_local_list_stacked = torch.stack(h_local_list)
                average_h_local = local_weights_reshaped * h_local_list_stacked
                average_h_local = sum(average_h_local)
            else:
                average_h_local = torch.stack(h_local_list).mean(dim=0)
            
            
            if self.layer_norm:
                average_h_local = self.norm1_local(average_h_local, batch.batch)
            if self.batch_norm:
                average_h_local = self.norm1_local(average_h_local)
            h_out_list.append(average_h_local)

        if len(self.global_model_types) >= 1:
            h_attn_list = []
            # Multi-head attention.
            for i in range(len(self.self_attns)):
                global_model_type = self.global_model_types[i]
                if global_model_type == 'None':
                    h_attn_list.append(h)
                    continue
                self_attn = self.self_attns[i]
                h_dense, mask = to_dense_batch(h, batch.batch)
                if global_model_type == 'Transformer':
                    h_attn = self._sa_block(self_attn, h_dense, None, ~mask)[mask]
                elif global_model_type == 'BiasedTransformer':
                    # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                    h_attn = self._sa_block(self_attn, h_dense, batch.attn_bias, ~mask)[mask]
                elif global_model_type == 'Performer':
                    h_attn = self_attn(h_dense, mask=mask)[mask]
                elif global_model_type == 'BigBird':
                    h_attn = self_attn(h_dense, attention_mask=mask)
                else:
                    raise RuntimeError(f"Unexpected {global_model_type}")
                h_attn_list.append(h_attn)

            # Average local representations.
            if len(self.filtered_global_model_types) >= 1:
                global_weights_reshaped = torch.softmax(self.global_weights, dim=0).view(self.global_len, 1, 1)
                h_attn_list_stacked = torch.stack(h_attn_list)
                average_h_attn = global_weights_reshaped * h_attn_list_stacked
                average_h_attn = sum(average_h_attn)
            else:
                average_h_attn = torch.stack(h_attn_list).mean(dim=0)

            average_h_attn = self.dropout_attn(average_h_attn)
            average_h_attn = h_in1 + average_h_attn  # Residual connection.
            
            if self.layer_norm:
                average_h_attn = self.norm1_attn(average_h_attn, batch.batch)
            if self.batch_norm:
                average_h_attn = self.norm1_attn(average_h_attn)
            h_out_list.append(average_h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        if len(h_out_list) == 0:
            h = torch.zeros_like(h_in1)  # Assuming h_in1 has the correct shape
        elif len(h_out_list) == 1:
            h = h_out_list[0]
        else:
            h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch
    def _sa_block(self, module, x, attn_mask, key_padding_mask):
        """
        Self-attention block.
        """
        if not self.log_attn_weights:
            x = module(x, x, x,
                       attn_mask=attn_mask,
                       key_padding_mask=key_padding_mask,
                       need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = module(x, x, x,
                          attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask,
                          need_weights=True,
                          average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """
        Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_types={self.local_gnn_types}, ' \
            f'global_model_types={self.global_model_types}, ' \
            f'heads={self.num_heads}'
        return s

