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
class EBase(BaseOptimizer):

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
        if config.targets is None or len(config.targets) < 1:
            return False
        if config.metrics is None or len(config.metrics) < 1:
            return False
        return True

    def _evaluate_individual(self, individual):
        metrics_dict = self._evaluate_solution(individual)
        return [metrics_dict[x] for x in self.config.metrics]

    def _decode_solution(self, encoded_solution):
        # 'drop-out' in [0,99]
        # 'look back' in [1, config.max_look_back]
        # 'rnn_arch' list (layer_in, layer_1, ..., layer_n, layer_out)
        # layer_in = len(config.x_features)
        # (layer_1, ...,layer_n), layer_i in [1, config.max_neurons], n in [1, config.max_layers]
        # layer_out = len(config.y_features)
        decoded = {}
        decoded['drop_out'] = float(encoded_solution[0]/100)
        decoded['look_back'] = encoded_solution[1]
        decoded['rnn_arch'] = [self.layer_in] + encoded_solution[2:] + [self.layer_out]
        return decoded

    def _init_individual(self, clazz):
        # range [min,max)
        ranges = [(0,100), (1, self.config.max_look_back+1)]
        ranges = ranges + [(1, self.config.max_neurons+1)] * np.random.randint(1, high=self.config.max_layers+1)
        return clazz(np.random.randint(*p) for p in ranges) 

    def _validate_individual(self, individual):
        if individual[0] < 0:
           individual[0] = 0
        if individual[0] > 99:
           individual[0] = 99
        if individual[1] < 0:
           individual[1] = 0
        if individual[1] > self.config.max_look_back:
           individual[1] = self.config.max_look_back
        for i in range(len(individual[2:])):
            if individual[i+2] < 1:
                individual[i+2] = 1
            if individual[i+2] > self.config.max_neurons:
                individual[i+2] = self.config.max_neurons
        while len(individual) > self.config.max_layers + 2:
            individual.pop()
        if len(individual) < 3:
            individual.append(1)

    def _run_algorithm(self, stats, hall_of_fame):
        # First, we initialize the framework
        creator.create("FitnessMulti", base.Fitness, weights=self.config.targets)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("individual", self._init_individual, clazz=creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual)
        toolbox.register("mate", self._mate)
        toolbox.register("mutate", self._mutate)
        toolbox.register("select", self._select)
        toolbox.register("replace", self._replace)
        # Initialize the logger
        logbook = tools.Logbook()
        # Initialize population
        pop = toolbox.population(n=self.config.pop_size)
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        if hall_of_fame is not None:
            hall_of_fame.update(pop)
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
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # Replacement
            pop[:] = toolbox.replace(pop, offspring)
            # Gather the stats
            record = stats.compile(pop)
            print(record)
            evals = evals + self.config.offspring_size
            if hall_of_fame is not None:
                hall_of_fame.update(pop)
            logbook.record(evaluations=evals, gen=g, **record)
            # The algorithm might adjust the parameters
            self._auto_adjust(logbook)
            g += 1
        return pop, logbook, hall_of_fame

         

#########################################################################################################################
class EliteSpxUniform(EBase):

    def _validate_config(self, config):
        if config.mut_max_step is None or config.mut_max_step < 0:
            return False
        if config.cx_prob is None or config.cx_prob > 1 or config.cx_prob < 0:
            return False
        if config.mut_prob is None or config.mut_prob > 1 or config.mut_prob < 0:
            return False
        if config.mut_x_prob is None or config.mut_x_prob > 1 or config.mut_x_prob < 0:
            return False
        return super()._validate_config(config)

    def _mate(self, ind1, ind2):
        if np.random.rand() < self.config.cx_prob:
            tools.cxOnePoint(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values

    def _mutate(self, ind):
        # Mutate the inner values
        for i in range(len(ind)):
            if np.random.rand() < self.config.mut_prob:
                # We are always moving at least one step forward
                step = np.max([1, np.random.randint(0, self.config.mut_max_step)])
                if np.random.rand() < 0.5:
                    step = -1 * step
                ind[i] = ind[i] + step
                del ind.fitness.values
        # Add or remove layers
        if np.random.rand() < self.config.mut_x_prob:
            i = np.random.randint(2, len(ind))
            # with the same probaility we add or delete a layer
            if np.random.rand() > 0.5:
                ind.insert(i, ind[i])
            elif len(ind) > 3:
                ind.pop(i)
            del ind.fitness.values
        if not ind.fitness.valid:
            self._validate_individual(ind)

    def _select(self, individuals, k):
        return tools.selTournament(individuals, k, 2)  

    def _replace(self, pop, offspring):
        return tools.selBest(pop, self.config.pop_size - self.config.offspring_size) + offspring

    def _auto_adjust(self, logbook):
        pass



#########################################################################################################################
class MuPLambdaSpxUniform(EBase):

    def _validate_config(self, config):
        if config.mut_max_step is None or config.mut_max_step < 0:
            return False
        if config.mut_prob is None or config.mut_prob > 1 or config.mut_prob < 0:
            return False
        if config.mut_x_prob is None or config.mut_x_prob > 1 or config.mut_x_prob < 0:
            return False
        return super()._validate_config(config)

    def _mate(self, ind1, ind2):
        pass

    def _mutate(self, ind):
        # Mutate the inner values
        for i in range(len(ind)):
            if np.random.rand() < self.config.mut_prob:
                # We are always moving at least one step forward
                step = np.max([1, np.random.randint(0, self.config.mut_max_step)])
                if np.random.rand() < 0.5:
                    step = -1 * step
                ind[i] = ind[i] + step
                del ind.fitness.values
        # Add or remove layers
        if np.random.rand() < self.config.mut_x_prob:
            i = np.random.randint(2, len(ind))
            # with the same probaility we add or delete a layer
            if np.random.rand() > 0.5:
                ind.insert(i, ind[i])
            elif len(ind) > 3:
                ind.pop(i)
            del ind.fitness.values
        if not ind.fitness.valid:
            self._validate_individual(ind)

    def _select(self, individuals, k):
        return tools.selTournament(individuals, k, 2)  

    def _replace(self, pop, offspring):
        return tools.selBest(pop + offspring, self.config.pop_size)

    def _auto_adjust(self, logbook):
        pass



#########################################################################################################################
class SelfAdjMuPLambdaSpxUniform(MuPLambdaSpxUniform):

    def _auto_adjust(self, logbook):
        means = logbook.select("avg")
        if len(means) > 1:
            diffs = []
            for i in range(len(self.config.targets)):            
                diff = means[-1][i] - means[-2][i]
                if self.config.targets[i] < 0 and diff <= 0:
                    diffs.append(1)
                elif self.config.targets[i] > 0 and diff >= 0:
                    diffs.append(1)
                else:
                    diffs.append(-1)        
            if np.sum(diffs) > 0:
                # We are improving (on average)
                self.config.mut_prob = self.config.mut_prob * 1.5
                self.config.mut_x_prob = self.config.mut_x_prob * 1.5
            else:
                self.config.mut_prob = self.config.mut_prob / 4
                self.config.mut_x_prob = self.config.mut_x_prob / 4
        # print(self.config.mut_prob, self.config.mut_x_prob)


