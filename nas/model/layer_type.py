import copy
import os
import random
import pickle
from torch_geometric.graphgym.config import cfg

from nas.layer.supernet_layer import SupernetLayer

class LayerType():
    def __init__(self, space: list=None, param: list=None):
        if space != None:
            self.space = space
        else:
            self.space = [cfg.nas.supernet_layer_types] * cfg.nas.layer_num
        
        if param != None:
            self.param = param
        else:
            self.param = self._random_param()
        self.param = self.with_none()
    
    def _random_param(self):
        if self.space == None:
            raise ValueError("LayerType space is not defined")
        
        param = []
        for layer_space in self.space:
            local_gnn_type = [random.sample(layer_space[0], random.randint(0, len(layer_space[0])))]
            global_model_type = [random.sample(layer_space[1], random.randint(0, len(layer_space[1])))]
            param.append(local_gnn_type + global_model_type)
        return param
    
    def without_none(self):
        if self.param == None or self.space == None:
            raise ValueError("LayerType param or space is not defined")
        
        assert len(self.param) == len(self.space)
        
        new_param = [
            [
                [
                    local_gnn_type 
                    for local_gnn_type in layer_param[0]
                    if local_gnn_type != 'None' and local_gnn_type in self.param[i][0]
                ],
                [
                    global_model_type
                    for global_model_type in layer_param[1]
                    if global_model_type != 'None' and global_model_type in self.param[i][1]
                ]
            ] for i, layer_param in enumerate(self.space)
        ]
        return new_param
    
    def with_none(self):
        if self.param == None or self.space == None:
            raise ValueError("LayerType param or space is not defined")
        
        assert len(self.param) == len(self.space)
        
        new_param = [
            [
                [
                    local_gnn_type if local_gnn_type in self.param[i][0] else 'None' 
                    for local_gnn_type in layer_param[0]
                ],
                [
                    global_model_type if global_model_type in self.param[i][1] else 'None' 
                    for global_model_type in layer_param[1]
                ]
            ] for i, layer_param in enumerate(self.space)
        ]
        return new_param
    
    def crossover(self, other):
        if self.param == None or self.space == None or other.param == None:
            raise ValueError("LayerType param or space is not defined")
        
        if self.space != other.space:
            raise ValueError("LayerType spaces are not the same")
        
        new_param_1, new_param_2 = copy.deepcopy(self.param), copy.deepcopy(other.param)
        for i in range(len(new_param_1)):
            if random.random() < cfg.nas.ga.crossover_rate:
                new_param_1[i][0], new_param_2[i][0] = self._exchange(new_param_1[i][0], new_param_2[i][0])
                new_param_1[i][1], new_param_2[i][1] = self._exchange(new_param_1[i][1], new_param_2[i][1])
        
        return LayerType(space=self.space, param=new_param_1), LayerType(space=self.space, param=new_param_2)
    
    def mutate(self):
        new_param = copy.deepcopy(self.param)
        for i in range(len(new_param)):
            if random.random() < cfg.nas.ga.mutation_rate:
                new_param[i][0] = self._overturn(new_param[i][0], self.space[i][0])
                new_param[i][1] = self._overturn(new_param[i][1], self.space[i][1])
        return LayerType(space=self.space, param=new_param)
                
    def _exchange(self, list1, list2):
        assert len(list1) == len(list2)
        index = random.randint(0, len(list1))
        new_list1 = list1[:index] + list2[index:]
        new_list2 = list2[:index] + list1[index:]
        return new_list1, new_list2
    
    def _overturn(self, list, space):
        assert len(list) == len(space)
        index = random.sample(range(len(list)), random.randint(0, len(list)))
        new_list = copy.deepcopy(list)
        for idx in index:
            if space[idx] == 'None':
                continue
            if new_list[idx] == 'None':
                new_list[idx] = space[idx]
            else:
                new_list[idx] = 'None'                
        return new_list
    
    def is_parent(self, other):
        if len(self.param) > len(other.param):
            return False
        
        for i in range(len(self.param)):
            for j in range(len(self.param[i][0])):
                if (j >= len(other.param[i][0])) or (self.param[i][0][j] != 'None' and other.param[i][0][j] == 'None'):
                    return False
            for j in range(len(self.param[i][1])):
                if (j >= len(other.param[i][1])) or (self.param[i][1][j] != 'None' and other.param[i][1][j] == 'None'):
                    return False
        return True
    
    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.param, f)
            print(f"Successfully saved to {path}")
        except Exception as e:
            print(f"Failed to save to {path}: {e}")
    
    def load_model(self, model):
        self.param = []
        for layer in model.layers:
            if isinstance(layer, SupernetLayer):
                self.param.append([layer.local_gnn_types, layer.global_model_types])
        return self.param
    
    def load_path(self, path):
        with open(path, 'rb') as f:
            self.param = pickle.load(f)
        return self.param
    
