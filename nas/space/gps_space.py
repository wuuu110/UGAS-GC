import random

from nas.space.base_space import BaseSpace


class GpsSpace(BaseSpace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = {'local_gnn_types': ['GCN', 'GAT', 'GINE', 'GENConv'],
                       'global_model_types': ['Transformer', 'Performer']}

    def get_full_params(self):
        return self.params

    def get_random_params(self, threshold=0.5):
        random_params = self.params.copy()
        for key in random_params:
            for i in range(len(random_params[key])):
                random_params[key][i] = 'None' if random.random() < threshold else random_params[key][i]
        return random_params

    def get_full_params_list(self, num):
        return [self.params.copy() for _ in range(num)]

    def get_random_params_list(self, num, threshold=0.5):
        return [self.get_random_params(threshold) for _ in range(num)]
    