import numpy as np
from . import optimization as op
from abc import ABC, abstractmethod
from copy import deepcopy
from . import util as ut
import gc


def gaussianMutation(encoded,
                     p_mutation,
                     mutation_scale_factor=1):
    """ Element-wise Gaussian mutation
    Performs a Gaussian mutation on the i-th encoded variable
    with a probability p_mutation.
    """
    for i in range(len(encoded)):
        if np.random.rand() < p_mutation:
            encoded[i] += np.random.normal(
                scale=mutation_scale_factor)


def uniformMutation(encoded,
                    p_mutation,
                    mutation_max_step=1):
    """ Element-wise uniform mutation
    Performs a uniform step mutation on the i-th encoded variable
    with a probability p_mutation.
    """
    for i in range(len(encoded)):
        if np.random.rand() < p_mutation:
            step = np.max([1, np.random.randint(0, mutation_max_step)])
            if np.random.rand() < 0.5:
                step = -1 * step
            encoded[i] += step


def uniformLengthMutation(encoded,
                          p_mutation):
    """ With p_mutation probability copy/delete an encoded variable
    """
    if np.random.rand() < p_mutation:
        position = np.random.randint(0,
                                     len(encoded))
        if np.random.rand() < 0.5:
            encoded.pop(position)
        else:
            encoded.insert(position,
                           encoded[position])


def binaryTournament(population):
    """ Binary tournament. Selects two solutions from the population
    and compare them. Returns the fittest one.
    """
    positions = np.random.randint(low=0,
                                  high=len(population),
                                  size=2)
    if population[positions[0]].comparedTo(
            population[positions[1]]) > 0:
        return deepcopy(population[positions[0]])
    else:
        return deepcopy(population[positions[1]])


def elitistPlusReplacement(population,
                           offspring):
    """ Generates a new population (of the same size as the
    original population), selecting the fittest solution from
    the population and the offspring.
    """
    temporal = population + offspring
    temporal.sort(reverse=True)
    del temporal[len(population):]
    gc.collect()
    return temporal


class EABase(op.ModelOptimization):
    """ Evolutionary Algorithm base class.
    An initial population (of size equal to population_size) is
    evaluated using the problem's fitness function. Then, an offspring
    (of size equal to offspring_size) is created using the 'select'
    and 'mutate' functions, and evaluated. A new population is
    generated using the 'replace' criteria. This process is
    repeated for 'max_eval' number of times (number of solution
    evaluations).
    If 'max_restart' is greater than 0, the evolutionary process
    is restarted (once 'max_eval' evaluations are done). The new
    evolutionary process recieves 'migration_population_size'
    solutions from the previous process.
    """
    params = {'population_size': 2,
              'offspring_size': 1,
              'max_eval': 1,
              'max_restart': 0,
              'migration_population_size': 1}

    def __init__(self,
                 problem,
                 seed=None,
                 verbose=0):
        super().__init__(problem,
                         seed,
                         verbose)

    @abstractmethod
    def mutate(self,
               solution):
        raise Exception

    @abstractmethod
    def select(self,
               population):
        raise Exception

    @abstractmethod
    def replace(self,
                population,
                offspring):
        raise Exception

    def call_on_generation(self,
                           population):
        pass

    def call_on_restart(self,
                        population):
        pass

    def _go_one_step(self,
                     seed_population):
        population = (
            seed_population[:self.params['migration_population_size']] +
            [self.problem.next_solution()
             for _ in range(self.params['population_size'] -
                            self.params['migration_population_size'])])
        [self.problem.evaluate(solution) for solution in population]
        evaluations = len(population)
        generation = 0
        if self.verbose:
            ut.display_top(limit=10)
            print("Generation " + str(generation))
            for p in population:
                print("fitness:", p.fitness,
                      "encoded:", p.encoded)
        while evaluations < self.params['max_eval']:
            offspring = [self.select(population)
                         for _ in range(self.params['offspring_size'])]
            [self.mutate(x) for x in offspring]
            [x.clear_fitness() for x in offspring]
            [self.problem.validate_solution(x) for x in offspring]
            [self.problem.evaluate(x) for x in offspring]
            if self.verbose:
                print("Offspring " + str(generation))
                for o in offspring:
                    print("fitness:", o.fitness,
                          "solution:", o.encoded)
            population = self.replace(population,
                                      offspring)
            evaluations += len(offspring)
            generation += 1
            self.call_on_generation(population)
            if self.verbose:
                print(str(evaluations) + " evaluations")
                print("Generation " + str(generation))
                for p in population:
                    print("fitness:", p.fitness,
                          "solution:", p.encoded)
        return population

    def optimize(self,
                 **kwargs):
        self.params.update(kwargs)
        population = [self.problem.next_solution()
                      for _ in range(self.params['migration_population_size'])]
        restart = 0
        while restart <= self.params['max_restart']:
            if self.verbose and restart > 0:
                print("Restart " + str(restart))
            population = self._go_one_step(population)
            restart += 1
            self.call_on_restart(population)
        model, solution_desc = self.problem.solution_as_result(population[0])
        return model, solution_desc
