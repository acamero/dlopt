import tensorflow as tf
import numpy as np
import random as rd
from . import util as ut
from abc import ABC, abstractmethod
from copy import deepcopy


class ModelOptimization(ABC):
    """ Base class for archtiecture/weights optimization
    of artificial neural networks
    """
    def __init__(self,
                 problem,
                 seed=None):
        if seed is not None:
            np.random.seed(seed)
            rd.seed(seed)
            tf.set_random_seed(seed)
        if not issubclass(problem,
                          Problem):
            raise Exception("A valid problem must be provided)
        self.problem = problem

    @abstractmethod
    def optimize(self,
                 **kwargs):
        raise Exception("'optimize' is not implemented")


class Problem(ABC):
    """ Problem base class
    """
    def __init__(self,
                 target):
        self.target = tuple(target)

    @abstractmethod
    def evaluate(self,
                 solution):
        raise Exception

    @abstractmethod
    def next_solution(self):
        raise Exception

    @abstractmethod
    def validate_solution(self):
        raise Exception


class Solution(object):
    """ Solution
    target: tuple of objective targets (-1: minimize, 1: maximize)
    fitness: list of evaluated values for each objective
    encoded: solution represented as a list of variables
    """
    def __init__(self,
                 target):
        """ target: list of targets. -1: minimize, 1 maximize
        """
        self.target = tuple(target)
        self.fitness = None
        self.encoded = None

    def comparedTo(self,
                   solution):
        """ Compare the fitness of this solution (a) to the
        fitness of the inputed solution (b).
        Returns:
        < 0  if (b) fitness values are 'better' than (a)
        0    if a==b
        > 0  if (a) fitness values are 'better' than (b)
        Exception if (a) and (b) are not comparables
        """
        if self.target != solution.target:
            raise Exception("Solutions are not comparables")
        if len(self.fitness) != len(solution.fitness):
            raise Exception("Solutions are not comparables")
        differences = np.subtract(self.fitness,
                                  solution.fitness)
        differences = np.multiply(differences,
                                  self.target)
        a_up = sum(differences > 0)
        b_up = sum(differences < 0)
        return (a_up - b_up)

    def __eq__(self,
               solution):
        if self.comparedTo(solution) == 0:
            return True
        else:
            return False

    def __lt__(self,
               solution):
        if self.comparedTo(solution) < 0:
            return True
        else:
            return False

    def __gt__(self,
               solution):
        if self.comparedTo(solution) > 0:
            return True
        else:
            return False

    def __le__(self,
               solution):
        if self.comparedTo(solution) <= 0:
            return True
        else:
            return False

    def __ge__(self,
               solution):
        if self.comparedTo(solution) >= 0:
            return True
        else:
            return False
