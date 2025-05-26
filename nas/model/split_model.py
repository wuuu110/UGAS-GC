import copy
import itertools
import math
import os
import random
import numpy as np
from sklearn.cluster import SpectralClustering
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from nas.layer.supernet_layer import SupernetLayer
from nas.model.grad_tracker import GradTracker
from sklearn.metrics.pairwise import cosine_similarity
from torch.autograd import grad
from nas.utils import dim_in_and_out
from torch_geometric.graphgym.config import cfg

from nas.model.layer_type import LayerType
from nas.model.subnet_model import SubnetModel
from nas.train.nas_trainer import NasTrainer
from nas.train.parallel_trainer import ParallelTrainer
from torch_geometric.graphgym.utils.comp_budget import params_count

class SplitModel():
    def __init__(self, model, split_type='grad_contrib', sim_type='cosine', loaders=None):

        self.model = model
        self.split_type = split_type
        self.sim_type = sim_type
        self.before_split_modules = []

        # [epoch, layer of model, [layer_module, layer_module]]
        self.modules_similarity = []

        self.get_before_split_modules()
        if split_type == 'grad_contrib':
            self.grad_tracker = GradTracker(model, self.before_split_modules)
        else:
            raise NotImplementedError(f"Split type {split_type} i s not implemented")

    def get_before_split_modules(self):
        """
        Get all modules of the model before split.
        """
        for layer in self.model.layers:
            layer_modules = []
            for local_model in layer.local_models:
                if local_model is not None:
                    layer_modules.append(local_model)
            for self_attn in layer.self_attns:
                if self_attn is not None:
                    layer_modules.append(self_attn)
            self.before_split_modules.append(layer_modules)
    
    def get_split_model_params(self, top_k=2):
        """
        Get the layer type params after split.
        The layer type params is final split result.
        """

        if self.split_type == 'grad_contrib':
            self.save_similarity()
            self.modules_similarity = np.mean(np.array(self.modules_similarity), axis=0).tolist()
            # method: ['stoer_wagner', 'karger', 'spectral', 'metis', 'flow']
            cut_results = self.min_cut(top_k, method='stoer_wagner')
            params = self._cut_results_to_params(cut_results)
        return params
    
    def save_similarity(self):
        """
        Save the similarity matrix of modules.
        The similarity matrix is saved in `self.modules_similarity`.
        """
        file_path = os.path.join(cfg.run_dir, 'similarity')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, f'{self.split_type}_similarity.npy')

        data = np.array(self.modules_similarity)
        np.save(file_name, data)
        print(f"[*] Save similarity matrix to {file_name}")
        # Save the similarity matrix to a file.
    
    def append_modules_similarity(self):
        """
        The function is called in `custom_train`. Calculate the similarity matrix after every epoch.
        Append the similarity matrix to `self.modules_similarity`.
        Accumulate all similarity matrixs in `self.modules_similarity` when calculating the final split result.
        """
        if self.split_type == 'grad_contrib':
            modules_similarity = self._grad_contrib_modules_similarity()
        else:
            raise NotImplementedError(f"Split type {self.split_type} is not implemented")
        self.modules_similarity.append(modules_similarity)

    def _grad_contrib_modules_similarity(self):
        """
        Calculate the similarity matrix of modules based on gradient contribution.
        """
        modules_similarity = [np.ones((len(self.before_split_modules[0]), len(self.before_split_modules[0])))]
        for i in range(len(self.before_split_modules) - 1):
            similarity_matrixs = []
            first_layer_modules = self.before_split_modules[i]
            second_layer_modules = self.before_split_modules[i + 1]
            for module1 in first_layer_modules:
                grad_contribs = [
                    self._module_grad_contrib(module1, module2)
                    for module2 in second_layer_modules
                ]
                similarity_matrixs.append(cosine_similarity(grad_contribs))
            similarity_matrixs = sum(similarity_matrixs) / len(similarity_matrixs)
            modules_similarity.append(similarity_matrixs)
        return modules_similarity
        # gc use both _grad_contrib_modules_similarity Func and append_modules_similarity Func
        # lead to return type: [epoch, cnt(Layers), choices_inLayer, choices_inLayer]

    def _module_grad_contrib(self, module1, module2):
        """
        params:
        - module1: the module of the first layer.
        - module2: the module of the next layer.
        return:
        - grad_contrib: The gradient contribution of module2 to module1.

        Calculate the gradient contribution of module1 to module2.
        `grad_tracker` is an instance of `GradTracker`, which is used to track the gradient of the model.
        The `grad_tracker` updates its data after each forward and backward pass.
        """
        module1_output = self.grad_tracker.module_output[module1]
        module2_output = self.grad_tracker.module_output[module2]
        module2_grad_output = self.grad_tracker.module_grad_output[module2]
    
        grad_contrib = grad(
            module2_output,
            module1_output,
            grad_outputs=module2_grad_output,
            retain_graph=True
        )[0]

        return grad_contrib.reshape(-1).cpu().detach().numpy()
    
    def _karger_min_cut(self, G, n_runs=100):
        """Karger随机最小割算法"""
        min_cut = float('inf')
        best_partition = None
        
        for _ in range(n_runs):
            g = G.copy()
            nodes = list(g.nodes())
            
            while len(nodes) > 2:
                u, v = np.random.choice(nodes, 2, replace=False)
                if not g.has_edge(u, v):
                    continue
                
                neighbors = list(g.neighbors(v))
                for neighbor in neighbors:
                    if neighbor != u:
                        weight = g[v][neighbor]['weight']
                        if g.has_edge(u, neighbor):
                            g[u][neighbor]['weight'] += weight
                        else:
                            g.add_edge(u, neighbor, weight=weight)
                g.remove_node(v)
                nodes.remove(v)
            
            cut_edges = list(g.edges(data=True))
            current_cut = sum(edge[2]['weight'] for edge in cut_edges)
            
            if current_cut < min_cut:
                min_cut = current_cut
                best_partition = (list(g.neighbors(nodes[0])), [nodes[0]])

        return min_cut, best_partition

    def _spectral_clustering(self, similarity_matrix):
        adj_matrix = np.array(similarity_matrix)
        np.fill_diagonal(adj_matrix, 0)
        
        sc = SpectralClustering(n_clusters=2, affinity='precomputed', 
                               assign_labels='discretize')
        labels = sc.fit_predict(adj_matrix)
        
        partition = [np.where(labels == 0)[0].tolist(),
                    np.where(labels == 1)[0].tolist()]
        
        cut_value = 0
        for i in partition[0]:
            for j in partition[1]:
                if j > i:  
                    cut_value += similarity_matrix[i][j]
        
        return cut_value, partition


    def min_cut(self, top_k=1, method='stoer_wagner', **kwargs):
        effective_len = sum(1 for item in self.modules_similarity if len(item) > 1)
        if top_k < 0 or top_k > effective_len:
            raise ValueError(f"top_k should be in range [0, {effective_len}]")
        
        cut_results = []
        
        for k in range(len(self.modules_similarity)):
            similarity_matrix = self.modules_similarity[k]
            if len(similarity_matrix) == 1:
                continue
                
            G = nx.Graph()
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix)):
                    adj_weight = similarity_matrix[i][j] + 1
                    G.add_edge(i, j, weight=adj_weight)
            
            if method == 'stoer_wagner':
                cut_value, partition = nx.stoer_wagner(G)
            elif method == 'karger':
                n_runs = kwargs.get('n_runs', 100)
                cut_value, partition = self._karger_min_cut(G, n_runs)
            elif method == 'spectral':
                cut_value, partition = self._spectral_clustering(similarity_matrix)
            elif method == 'metis':
                cut_value, partition = self._metis_partition(G)
            elif method == 'flow':
                cut_value, partition = self._flow_based_cut(G)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            partition = [sorted(p) for p in partition]
            corrected_value = cut_value - len(partition[0])*len(partition[1])
            
            cut_results.append({
                'layer_id': k,
                'cut_value': corrected_value,
                'partition': partition
            })
        
        cut_results = sorted(cut_results, key=lambda x: x['cut_value'])[:top_k]
        return sorted(cut_results, key=lambda x: x['layer_id'])

    def _cut_results_to_params(self, cut_results):
        """
        params:
        - cut_results: the split results of the model.
        return:
        - models_params: the split model params.

        Convert the split results to the split model params.
        """
        cur = 0
        models_params = [[]]
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            if 'None' in layer.local_gnn_types:
                layer.local_gnn_types.remove('None')
            if 'None' in layer.global_model_types:
                layer.global_model_types.remove('None')
            
            if cur < len(cut_results) and i == cut_results[cur]['layer_id']:
                partition = cut_results[cur]['partition']
                models_params = [copy.deepcopy(model_params) for model_params in models_params for _ in range(len(partition))]

                for j in range(len(models_params)):
                    model_params = models_params[j]
                    p = partition[j % len(partition)]
                    split_local_gnn_types = []
                    split_global_model_types = []
                    for idx in p:
                        if idx < len(layer.local_gnn_types):
                            split_local_gnn_types.append(layer.local_gnn_types[idx])
                        else:
                            split_global_model_types.append(layer.global_model_types[idx - len(layer.local_gnn_types)])
                    layer_params = [split_local_gnn_types] + [split_global_model_types]
                    model_params.append(layer_params)
                cur += 1
            else:
                for model_params in models_params:
                    layer_params = [layer.local_gnn_types] + [layer.global_model_types]
                    model_params.append(layer_params)
        return models_params
    
    def random_split_params(self, top_k):
        cut_results = []
        layer_indices = list(range(len(self.model.layers)))
        selected_indices = random.sample(layer_indices, top_k)

        for idx in selected_indices:
            layer = self.model.layers[idx]
            module_num = len(layer.local_gnn_types) + len(layer.global_model_types)
            partition_1 = random.sample(range(module_num), random.randint(1, module_num - 1))
            partition_2 = list(set(range(module_num)) - set(partition_1))
            partition = [sorted(partition_1), sorted(partition_2)]

            cut_results.append({
                'layer_id': idx,
                'partition': partition
            })
        cut_results = sorted(cut_results, key=lambda x: x['layer_id'])
        return self._cut_results_to_params(cut_results)
    
    def _count_nonlinear_units(self, module):
        """
        Count the number of nonlinear units in the given module.
        
        Args:
            module (nn.Module): The module to count nonlinear units in.
        
        Returns:
            int: The number of nonlinear units in the module.
        """
        nonlinear_units = 0
        for submodule in module.modules():
            if isinstance(submodule, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.SELU, nn.GELU)):
                nonlinear_units += 1
        return nonlinear_units
    
    def _end(self):
        """
        Release the hooks.
        """
        if self.split_type == 'grad_contrib':
            self.grad_tracker.remove_hooks()