from .. import optimization as op
from .. import ea as ea
from .. import nn as nn
from .. import sampling as sp
from .. import util as ut
from abc import ABC, abstractmethod
from keras.layers.recurrent import LSTM
import numpy as np


class TimeSeriesMAERandSampProblem(op.Problem):
    """ Mean Absolute Error Random Sampling RNN Problem
    """
    def __init__(self,
                 data,
                 targets,
                 verbose=0,
                 x_features=None,
                 y_features=None,
                 num_samples=30,
                 min_layers=1,
                 max_layers=1,
                 min_neurons=1,
                 max_neurons=1,
                 min_look_back=1,
                 max_look_back=1,
                 sampler=sp.MAERandomSampling,
                 nn_builder_class=nn.RNNBuilder,
                 **kwargs):
        super().__init__(data,
                         targets,
                         verbose,
                         **kwargs)
        if x_features is None:
            self.x_features = data.columns
        else:
            self.x_features = x_features
        if y_features is None:
            self.y_features = data.columns
        else:
            self.y_features = y_features
        self.num_samples = num_samples
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.min_look_back = min_look_back
        self.max_look_back = max_look_back
        self.builder = nn_builder_class()
        if not issubclass(sampler, sp.RandomSamplingFit):
            raise Exception("'sampler' is not valid")
        self.sampler = sampler()

    def evaluate(self,
                 solution):
        if solution.is_evaluated():
            if self.verbose > 1:
                print('Solution already evaluated')
            return
        model, layers, look_back = self.decode_solution(solution)
        df_x, df_y = ut.chop_data(self.data,
                                  self.x_features,
                                  self.y_features,
                                  look_back)
        results = self.sampler.fit(model,
                                   self.num_samples,
                                   df_x,
                                   df_y,
                                   **self.kwargs)
        if self.verbose > 1:
            print({'layers': layers,
                   'look_back': look_back,
                   'results': results})
        for target in self.targets:
            solution.set_fitness(target,
                                 results[target])

    def next_solution(self):
        solution = op.Solution(self.targets,
                               ['architecture'])
        num_layers = np.random.randint(low=self.min_layers,
                                       high=(self.max_layers + 1))
        layers = np.random.randint(low=self.min_neurons,
                                   high=(self.max_neurons + 1),
                                   size=num_layers)
        look_back = np.random.randint(low=self.min_look_back,
                                      high=(self.max_look_back + 1),
                                      size=1)
        solution.set_encoded('architecture',
                             np.concatenate((look_back, layers)).tolist())
        return solution

    def validate_solution(self,
                          solution):
        encoded = solution.get_encoded('architecture')
        # look back
        if len(encoded) < 1:
            encoded.append(self.min_look_back)
        if encoded[0] < self.min_look_back:
            encoded[0] = self.min_look_back
        elif encoded[0] > self.max_look_back:
            encoded[0] = self.max_look_back
        elif not isinstance(encoded[0], int):
            encoded[0] = int(encoded[0])
        # layers
        while (len(encoded) - 1) < self.min_layers:
            encoded.append(self.min_neurons)
        while (len(encoded) - 1) > self.max_layers:
            encoded.pop()
        for i in range(1, len(encoded)):
            if encoded[i] > self.max_neurons:
                encoded[i] = self.max_neurons
            elif encoded[i] < self.min_neurons:
                encoded[i] = self.min_neurons
            elif not isinstance(encoded[i], int):
                encoded[i] = int(encoded[i])

    def decode_solution(self,
                        solution):
        layers = ([len(self.x_features)] +
                  solution.get_encoded('architecture')[1:] +
                  [len(self.y_features)])
        look_back = solution.get_encoded('architecture')[0]
        model = self.builder.build_model(layers,
                                         verbose=self.verbose,
                                         **self.kwargs)
        return model, layers, look_back

    def solution_as_result(self,
                           solution):
        solution_desc = {}
        model, layers, look_back = self.decode_solution(solution)
        solution_desc['config'] = str(model.get_config())
        solution_desc['layers'] = layers
        solution_desc['look_back'] = look_back
        solution_desc['fitness'] = solution.fitness
        return model, solution_desc


class MuPlusLambda(ea.EABase):
    """ (Mu+Lambda) basic algorithm
    """
    def __init__(self,
                 problem,
                 seed=None,
                 verbose=0):
        super().__init__(problem,
                         seed,
                         verbose)
        # We add the default parameter values
        self.params.update({'p_mutation_i': 0.1,
                            'p_mutation_e': 0.1,
                            'mutation_scale_factor': 2})

    def mutate(self,
               solution):
        ea.gaussianMutation(solution.get_encoded('architecture'),
                            self.params['p_mutation_i'],
                            self.params['mutation_scale_factor'])
        ea.uniformLengthMutation(solution.get_encoded('architecture'),
                                 self.params['p_mutation_e'])

    def select(self,
               population):
        return ea.binaryTournament(population)

    def replace(self,
                population,
                offspring):
        return ea.elitistPlusReplacement(population,
                                         offspring)
