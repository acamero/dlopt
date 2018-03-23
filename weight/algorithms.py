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

def glorot_uniform(size):
    limit = np.sqrt(6 / (size[0]+size[1]))
    return random_uniform(size, low=-limit, high=limit)

def orthogonal(size, gain=1.0):
    """Modification of the original keras code"""
    num_rows = 1
    for dim in size[:-1]:
        num_rows *= dim
    num_cols = size[-1]
    flat_shape = (num_rows, num_cols)
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(size)
    return gain * q[:size[0], :size[1]]


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
class HybridBase(BaseOptimizer):

    @abstractmethod
    def _mate(self, ind1, ind2):
        raise NotImplemented()

    @abstractmethod
    def _mutate(self, ind):
        raise NotImplemented()

    @abstractmethod
    def _select(self, individuals, k):
        raise NotImplemented()  

    @abstractmethod
    def _replace(self, pop, offspring):
        raise NotImplemented() 

    @abstractmethod
    def _auto_adjust(self, logbook):
        raise NotImplemented() 

    def _validate_config(self, config):
        """
        pop_size: population size
        cx_prob: crossover probability
        mut_prob: mutation probability
        max_evals: maximum number of evaluations
        offspring_size: size of the offspring
        targets: list of targets (min -1.0 or max 1.0), e.g. [-1.0, -1.0]
        metrics: list of metrics ('mean_mse', 'mean_mae', 'mse_dtl', 'mae_dtl', 
            'trainable_vars', 'train_time_dtl', 'mean_train_time'), e.g. ['mean_mse', 'trainable_vars']
        """
        if config.pop_size is None or config.pop_size < 1:
            return False
        if config.max_evals is None or config.max_evals < 1:
            return False
        if config.offspring_size is None or config.offspring_size < 1:
            return False
        if config.restarts is None or config.restarts < 1:
            return False
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

    def _evaluate_individual(self, individual):
        metrics_dict = self._evaluate_solution(individual)
        return [metrics_dict[x] for x in self.config.metrics]

    def _train_bp(self, individual):
        #TODO Train and update the solution with the new weights
        pass

    def _sample_architecture(self, individual):
        #TODO
        decoded = self._decode_solution(individual)
        print('Evaluate: ' + str(decoded['layers']) + ' ...')
        pass

    # Override
    def _decode_solution(self, encoded_solution):
        decoded = {}
        decoded['layers'] = encoded_solution[0]
        decoded['look_back'] = encoded_solution[1]
        decoded['weights'] = encoded_solution[2:]
        return decoded

    def _init_individual(self, clazz):
        solution = list()
        # First, we define the architecture (how many layers and neurons per layer)
        ranges = [(self.config.min_neurons, self.config.max_neurons+1)] * np.random.randint(self.config.min_layers, high=self.config.max_layers+1)
        layers = [self.layer_in] + [np.random.randint(*p) for p in ranges] + [self.layer_out]
        solution.append(layers)
        # Then, the look back
        solution.append( np.random.randint(self.config.min_look_back, self.config.max_look_back+1 ) )
        solution.extend( self._generate_weights(layers) )
        return clazz(solution) 

    def _validate_individual(self, individual):
        for i in range(len(individual[0])):
            if individual[0][i] < self.config.min_neurons:
                individual[0][i] = self.config.min_neurons
            if individual[0][i] > self.config.max_neurons:
                individual[0][i] = self.config.max_neurons
        if individual[1] < self.config.min_look_back:
           individual[1] = self.config.min_look_back
        if individual[1] > self.config.max_look_back:
           individual[1] = self.config.max_look_back
        #note that the weights are not fixed (in regard to these changes)!

    def _generate_weights(self, layers):
        weights = list()
        # Input dim (implicit when initializing first hidden layer) and hidden layers
        for i in range(len(layers)-2):
            # Kernel weights
            weights.append( self.kernel_init( size=(layers[i], layers[i+1]*self.config.params_neuron) ) )
            # Recurrent weights
            weights.append( self.recurrent_init( size=(layers[i+1], layers[i+1]*self.config.params_neuron) ) )
            # Bias
            weights.append( self.bias_init( size=layers[i+1]*self.config.params_neuron) )
        # Output dim
        # Dense weights
        weights.append( self.kernel_init( size=(layers[-2], layers[-1] ) ) )
        # Bias
        weights.append( self.bias_init( size=layers[-1]) )
        return weights

    def _run_algorithm(self, stats, hall_of_fame):
        # First, we initialize the framework
        creator.create("FitnessMulti", base.Fitness, weights=self.config.targets)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("individual", self._init_individual, clazz=creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual)
        toolbox.register("train", self._train_bp)
        toolbox.register("sample", self._sample_architecture)
        toolbox.register("mate", self._mate)
        toolbox.register("mutate", self._mutate)
        toolbox.register("select", self._select)
        toolbox.register("replace", self._replace)
        # Initialize the logger
        logbook = tools.Logbook()
        for i in range(self.config.restarts):
            # Get a group of candidates from an evolutionary process
            pop, logbook = self._micro_eval(toolbox, logbook)
            # Train the candidates using BP
            map(toolbox.train, pop)
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            if hall_of_fame is not None:
                hall_of_fame.update(pop)
        return pop, logbook, hall_of_fame

    def _micro_eval(self, toolbox, logbook):
        pop = toolbox.population(n=self.config.pop_size)
        # Evaluate the entire population
        fitnesses = list(map(toolbox.sample, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # Gather the stats
        record = stats.compile(pop)
        print(record)
        evals = self.config.pop_size
        logbook.record(evaluations=evals, gen=0, **record)
        g = 1
        # Begin the evolution
        while evals < self.config.max_evals:
            print("-- Generation %i --" % g)
            # Select the next generation individuals
            if self.config.offspring_size > 1:
                offspring = toolbox.select(pop, self.config.offspring_size)
            else:
                offspring = toolbox.select(pop, 2)
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)
            if self.config.offspring_size == 1:  
                offspring = [offspring[0]]
            for mutant in offspring:
                toolbox.mutate(mutant)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.sample, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # Replacement
            pop[:] = toolbox.replace(pop, offspring)
            # Gather the stats
            record = stats.compile(pop)
            print(record)
            evals = evals + self.config.offspring_size
            logbook.record(evaluations=evals, gen=g, **record)
            # The algorithm might adjust the parameters
            self._auto_adjust(logbook)
            g += 1
        return pop, logbook
