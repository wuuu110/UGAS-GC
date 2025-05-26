from concurrent.futures import ThreadPoolExecutor
import logging
import random

import numpy as np
import torch
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.config import cfg
from nas.model.layer_type import LayerType
from nas.model.subnet_model import SubnetModel
from nas.model.supernet_model import SupernetModel
from nas.train.default_model_train import DefaultModelTrain
from nas.train.nas_trainer import NasTrainer
from nas.train.parallel_trainer import ParallelTrainer
from nas.utils import dim_in_and_out, get_best_metric, get_best_metric_idx

class GeneticAlgorithm():
    def __init__(self, supernet, space_param, loaders):
        self.supernet = supernet
        
        self.enc_space = None
        if cfg.nas.encoder.enable == True:
            self.enc_space = cfg.nas.encoder.variable_types
        self.space_param = space_param
        self.loaders = loaders

        self.max_generations = cfg.nas.ga.max_generations
        self.population_size = cfg.nas.ga.population_size
        self.crossover_rate = cfg.nas.ga.crossover_rate
        self.mutation_rate = cfg.nas.ga.mutation_rate
        
        self.cur_generation = 0
        self.cur_model = 0
        
        self.population = []
        self.metric_values = []
        
    def search(self):
        # Initialize population
        self.population = self.init_pop()
        self.metric_values = self.eval_pop_metric(self.population)

        for k in range(self.max_generations):
            self.cur_generation = k + 1
            self.cur_model = 0
            logging.info(f'GA generation {k + 1} begin:')
            # Selection
            selected_population = self.selection(self.population, self.metric_values)

            # Crossover
            offspring_population = []
            offspring_metric_values = []
            for i in range(0, len(selected_population) - 1, 2):
                offspring1, offspring2 = selected_population[i].crossover(selected_population[i + 1])
                offspring_population.append(offspring1)
                offspring_population.append(offspring2)

            # Mutation
            for i in range(len(offspring_population)):
                offspring_population[i] = offspring_population[i].mutate()

            # Evaluate offspring population
            offspring_metric_values = self.eval_pop_metric(offspring_population)

            # Combine parent and offspring populations
            combined_population = self.population + offspring_population
            combined_metric_values = self.metric_values + offspring_metric_values

            # Select the best individuals from the combined population
            combined_indices = list(range(len(combined_population)))
            combined_indices.sort(key=lambda j: combined_metric_values[j], reverse=cfg.sort_fun == 'max')

            self.population = [combined_population[i] for i in combined_indices[:self.population_size]]
            self.metric_values = [combined_metric_values[i] for i in combined_indices[:self.population_size]]
            
            logging.info("GA generation %d/%d end.", k + 1, self.max_generations)
            for pop, metric_value in zip(self.population, self.metric_values):
                logging.info(f"individual: {pop.param}")
                logging.info(f"Metric value: {metric_value}")
            logging.info("Best individual in generation %d/%d:", k + 1, self.max_generations)
            logging.info(f"{self.population[get_best_metric_idx(self.metric_values)].param}")
            logging.info("Best metric value: %f", get_best_metric(self.metric_values))
            
        # Return the best solution
        best_index = self.metric_values.index(get_best_metric(self.metric_values))
        best_layer_type = self.population[best_index]
        best_metric = self.metric_values[best_index]
        return best_index, best_layer_type, best_metric

    def init_pop(self):
        return [LayerType(space=self.space_param) for _ in range(self.population_size)]

    def eval_pop_metric(self, population):
        parallel_trainer = []
        dim_in, dim_out = dim_in_and_out()
        for pop in population:
            self.cur_model += 1
            model = SubnetModel(dim_in, dim_out, pop).to(torch.device(cfg.accelerator))
            model.load_model_state(self.supernet.state_dict())
            parallel_trainer.append(
                NasTrainer(
                    f'ga_model_{self.cur_generation}_{self.cur_model}', 
                    model, 
                    self.loaders
                )
            )
        metric_best = ParallelTrainer(parallel_trainer).run()
        return metric_best

    def selection(self, population, metric_values):
        if cfg.sort_fun == 'min':
            fitness_values = metric_values
        else:
            fitness_values = [1 / (metric + 1e-6) for metric in metric_values]
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        selected_indices = np.random.choice(len(population), size=self.population_size, p=probabilities)
        return [population[i] for i in selected_indices]
    
    def trans_params(self, params):
        return params[0][0], params[1:]