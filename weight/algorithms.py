import rnn as nn
import util as ut
from optimizer import BaseOptimizer
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
#@article{DEAP_JMLR2012,
#    author    = " F\'elix-Antoine Fortin and Fran\c{c}ois-Michel {De Rainville} and Marc-Andr\'e Gardner and Marc Parizeau and Christian Gagn\'e ",
#    title     = { {DEAP}: Evolutionary Algorithms Made Easy },
#    pages    = { 2171--2175 },
#    volume    = { 13 },
#    month     = { jul },
#    year      = { 2012 },
#    journal   = { Journal of Machine Learning Research }
#}
from deap import creator, base, tools, algorithms


#########################################################################################################################
def random_uniform(size, low=-1.0, high=1.0):
    return np.random.uniform(low=low, high=high, size=size)

def random_normal(size, loc=0.0, scale=1.0):
    return np.random.normal(loc=loc, scale=scale, size=size)

def random_normal_narrow(size, loc=0.0, scale=0.05):
    return np.random.normal(loc=loc, scale=scale, size=size)

#########################################################################################################################
class RandomSearch(BaseOptimizer):

    def __init__(self, data, config, cache, seed=1234):
        super().__init__(data, config, cache, seed)
        self.kernel_init = config.kernel_init_func
        self.recurrent_init = config.recurrent_init_func
        self.bias_init = config.bias_init_func

    # Override
    def _validate_config(self, config):
        if config.min_layers is None or config.min_layers < 1:
            return False
        if config.max_layers is None or config.max_layers < config.min_layers:
            return False
        if config.max_evals is None or config.max_evals < 1:
            return False
        if config.min_neurons is None or config.min_neurons < 1:
            return False
        if config.max_neurons is None or config.max_neurons < config.min_neurons:
            return False
        if config.targets is None or len(config.targets) < 1:
            return False
        if config.metrics is None or len(config.metrics) < 1:
            return False
        if config.params_neuron is None or config.params_neuron < 1:
            return False
        if config.min_look_back is None or config.min_look_back < 1:
            return False
        if config.max_look_back is None or config.max_look_back < config.min_look_back:
            return False
        if config.kernel_init_func is None:
            return False
        if config.kernel_init_func is None:
            return False
        if config.kernel_init_func is None:
            return False
        return True

    # Override
    def _run_algorithm(self, stats, hall_of_fame):
        creator.create("FitnessMulti", base.Fitness, weights=self.config.targets)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("individual", self._init_individual, clazz=creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual)
        # Initialize the logger
        logbook = tools.Logbook()
        evals = 0
        pop = toolbox.population(n=1)
        # Initialize population
        while evals < self.config.max_evals:
            print("-- Evaluations %i --" % evals)
            # Evaluate the entire population
            pop[:] = toolbox.population(n=1)
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
                print(ind.fitness)
            if hall_of_fame is not None:
                hall_of_fame.update(pop)
            # Gather the stats
            record = stats.compile(pop)
            print(record)
            evals += 1
            logbook.record(evaluations=evals, gen=evals, **record)
        return pop, logbook, hall_of_fame

    # Override
    def _decode_solution(self, encoded_solution):
        decoded = {}
        decoded['layers'] = encoded_solution[0]
        decoded['look_back'] = encoded_solution[1]
        decoded['weights'] = encoded_solution[2:]
        return decoded

    def _evaluate_individual(self, individual):
        metrics_dict = self._evaluate_solution(individual)
        return [metrics_dict[x] for x in self.config.metrics]

    def _init_individual(self, clazz):
        solution = list()
        # First, we define the architecture (how many layers and neurons per layer)
        ranges = [(self.config.min_neurons, self.config.max_neurons+1)] * np.random.randint(self.config.min_layers, high=self.config.max_layers+1)
        layers = [self.layer_in] + [np.random.randint(*p) for p in ranges] + [self.layer_out]
        solution.append(layers)
        # Then, the look back
        solution.append( np.random.randint(self.config.min_look_back, self.config.max_look_back+1 ) )
        # Input dim (implicit when initializing first hidden layer) and hidden layers
        for i in range(len(layers)-2):
            # Kernel weights
            solution.append( self.kernel_init( size=(layers[i], layers[i+1]*self.config.params_neuron) ) )
            # Recurrent weights
            solution.append( self.recurrent_init( size=(layers[i+1], layers[i+1]*self.config.params_neuron) ) )
            # Bias
            solution.append( self.bias_init( size=layers[i+1]*self.config.params_neuron) )
        # Output dim
        # Dense weights
        solution.append( self.kernel_init( size=(layers[-2], layers[-1] ) ) )
        # Bias
        solution.append( self.bias_init( size=layers[-1]) )
        return clazz(solution) 

#########################################################################################################################

class RandomSearchSpecificArch(RandomSearch):

    def __init__(self, data, config, cache, seed=1234):
        super().__init__(data, config, cache, seed)
        self.kernel_init = config.kernel_init_func
        self.recurrent_init = config.recurrent_init_func
        self.bias_init = config.bias_init_func

    # Override
    def _validate_config(self, config):
        if config.architecture is None or len(config.architecture) < 1:
            return False
        if config.max_evals is None or config.max_evals < 1:
            return False
        if config.targets is None or len(config.targets) < 1:
            return False
        if config.metrics is None or len(config.metrics) < 1:
            return False
        if config.params_neuron is None or config.params_neuron < 1:
            return False
        if config.min_look_back is None or config.min_look_back < 1:
            return False
        if config.max_look_back is None or config.max_look_back < config.min_look_back:
            return False
        if config.kernel_init_func is None:
            return False
        if config.kernel_init_func is None:
            return False
        if config.kernel_init_func is None:
            return False
        return True

    # Override
    def _init_individual(self, clazz):
        solution = list()
        # First, we define the architecture (how many layers and neurons per layer)        
        layers = [self.layer_in] + self.config.architecture + [self.layer_out]
        solution.append(layers)
        # Then, the look back
        solution.append( np.random.randint(self.config.min_look_back, self.config.max_look_back+1 ) )
        # Input dim (implicit when initializing first hidden layer) and hidden layers
        for i in range(len(layers)-2):
            # Kernel weights
            solution.append( self.kernel_init( size=(layers[i], layers[i+1]*self.config.params_neuron) ) )
            # Recurrent weights
            solution.append( self.recurrent_init( size=(layers[i+1], layers[i+1]*self.config.params_neuron) ) )
            # Bias
            solution.append( self.bias_init( size=layers[i+1]*self.config.params_neuron) )
        # Output dim
        # Dense weights
        solution.append( self.kernel_init( size=(layers[-2], layers[-1] ) ) )
        # Bias
        solution.append( self.bias_init( size=layers[-1]) )
        return clazz(solution) 

#########################################################################################################################



