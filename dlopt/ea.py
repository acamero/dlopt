import numpy as np
from . import optimization as op
from abc import ABC, abstractmethod
from copy import deepcopy


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
    return temporal[0:len(population)]


class EABase(op.ModelOptimization):
    params = {'population_size': 1,
              'offspring_size': 1,
              'max_eval': 1}

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

    def optimize(self,
                 **kwargs):
        self.params.update(kwargs)
        population = [self.problem.next_solution()
                      for _ in range(self.params['population_size'])]
        [self.problem.evaluate(solution) for solution in population]
        evaluations = len(population)
        if self.verbose:
            print("Initial population evaluated")
        while evaluations < self.params['max_eval']:
            offspring = [self.select(population)
                         for _ in range(self.params['offspring_size'])]
            [self.mutate(x) for x in offspring]
            [self.problem.validate_solution(x) for x in offspring]
            [self.problem.evaluate(x) for x in offspring]
            pop = self.replace(population,
                               offspring)
            evaluations += len(offspring)
            if self.verbose:
                print(str(evaluations) + " evaluations")
        return population[0]
