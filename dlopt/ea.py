import numpy as np
from . import optimization as op
from abc import ABC, abstractmethod


def gaussianMutation(solution,
                     p_mutation,
                     mutation_scale_factor=1):
    """" Elemtn-wise Gaussian mutation
    Performs a Gaussian mutation on the i-th encoded variable
    with a probability p_mutation.
    """"
    for i in range(len(solution.encoded)):
        if np.random.rand() < p_mutation:
            solution.encoded[i] += np.random.normal(
                scale=mutation_scale_factor)


def uniformLengthMutation(solution,
                          p_mutation):
    """ With p_mutation probability copy/delete an encoded variable
    """
    if np.random.rand() < p_mutation:
        position = np.random.randint(0,
                                     len(solution.encoded))
        if np.random.rand() < 0.5:
            solution.encoded.pop(position)
        else:
            solution.encoded.insert(position,
                                    solution.encoded[position])


def binaryTournament(population):
    positions = np.random.randint(low=0,
                                  high=len(population),
                                  size=2)
    if population[positions[0]].comparedTo(
            population[positions[1]]) > 0:
        return deepcopy(population[position[0]])
    else:
        return deepcopy(population[position[1]])


def elitistPlusReplacement(population,
                           offspring):
    temporal = population + offspring
    temporal.sort(reverse=True)
    return temporal[0:len(population)]


class EABase(op.ModelOptimization):
    params = {'population_size': 1,
              'offspring_size': 1,
              'max_eval': 1}

    def __init__(self,
                 problem,
                 seed=None):
        super().__init__(problem,
                         seed)

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
        while evaluations < self.params['max_eval']:
            offspring = [self.select(population)
                         for _ in range(self.params['offspring_size'])]
            [self.mutate(x) for x in offspring]
            [self.problem.validate_solution(x) for x in offspring]
            [self.problem.evaluate(x) for x in offspring]
            pop = self.replace(population,
                               offspring)
            evaluations += len(offspring)
        return population


class MuPlusLambda(EABase):
    """ (Mu+Lambda) basic algorithm
    """
    def __init__(self,
                 problem,
                 seed=None):
        super().__init__(problem,
                         seed)
        self.params.update({'p_mutation_i': 0.1,
                            'p_mutation_e': 0.1,
                            'mutation_scale_factor': 2})

    def mutate(self,
               solution):
        gaussianMutation(solution,
                         self.params['p_mutation_i'],
                         self.params['mutation_scale_factor'])
        uniformCopyDeleteMutation(solution,
                                  self.params['p_mutation_e'])

    def select(self,
               population):
        return binaryTournament(population)

    def replace(self,
                population,
                offspring):
        return elitistPlusReplacement(population,
                                      offspring)
