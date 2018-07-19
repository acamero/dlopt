import tensorflow as tf
import numpy as np
import random as rd
from . import util as ut
from abc import ABC, abstractmethod
from copy import deepcopy
from keras.utils import Sequence


class ModelOptimization(ABC):
    """ Base class for archtiecture/weights optimization
    of artificial neural networks
    """
    def __init__(self,
                 problem,
                 seed=None,
                 verbose=0):
        if seed is not None:
            np.random.seed(seed)
            rd.seed(seed)
            tf.set_random_seed(seed)
        if not isinstance(problem,
                          Problem):
            raise Exception("A valid problem must be provided")
        self.problem = problem
        self.verbose = verbose

    @abstractmethod
    def optimize(self,
                 **kwargs):
        """ Optimize a problem and returns:
        model            keras moodel
        solution_desc    dictionary that at least contains:
                config       model description as text (keras config)
                fitness      fitness of the solution
                         and optionally:
                y_predicted  predicted values
                *            more text/numerical information
        """
        raise Exception("'optimize' is not implemented")


class Dataset(object):
    """ Dataset
    training data
    validation data
    testing data
    """
    training_data = None
    validation_data = None
    testing_data = None
    input_dim = None
    output_dim = None
    x_features = None
    y_features = None

    def __init__(self,
                 training_data,
                 valitation_data,
                 testing_data,
                 x_features,
                 y_features):
        self.x_features = x_features
        self.y_features = y_features
        self.input_dim = len(x_features)
        self.output_dim = len(y_features)
        self.set_training(training_data)
        self.set_validation(valitation_data)
        self.set_testing(testing_data)

    def set_training(self,
                     data):
        if not isinstance(data,
                          Sequence):
            raise Exception("The dataset must implement keras.utils.Sequence")
        self.training_data = data
        self._validate_dim(data)

    def set_validation(self,
                       data):
        if not isinstance(data,
                          Sequence):
            raise Exception("The dataset must implement keras.utils.Sequence")
        self.validation_data = data
        self._validate_dim(data)

    def set_testing(self,
                    data):
        if not isinstance(data,
                          Sequence):
            raise Exception("The dataset must implement keras.utils.Sequence")
        self.testing_data = data
        self._validate_dim(data)

    def _validate_dim(self,
                      data):
        x, y = data.__getitem__(0)
        if self.input_dim != int(x.shape[-1]):
            raise Exception("Input dimension mismatch")
        if self.output_dim != int(y.shape[-1]):
            raise Exception("Output dimension mismatch")


class Problem(ABC):
    """ Problem base class
    """
    def __init__(self,
                 dataset,
                 targets,
                 verbose=0,
                 **kwargs):
        for t in targets:
            if np.abs(targets[t]) != 1:
                raise Exception("Target must be 1 or -1")
        self.targets = targets
        if not isinstance(dataset,
                          Dataset):
            raise Exception("The dataset must implement Dataset")
        self.dataset = dataset
        self.kwargs = kwargs
        self.verbose = verbose

    @abstractmethod
    def evaluate(self,
                 solution):
        """ Assign a set of fitness values to the
        given solution
        """
        raise Exception

    @abstractmethod
    def next_solution(self):
        """ Generates a random solution
        """
        raise Exception

    @abstractmethod
    def validate_solution(self,
                          solution):
        """ Validates and corrects (if necessary)
        the solution.
        """
        raise Exception

    @abstractmethod
    def decode_solution(self,
                        solution):
        """ Decodes the solution into a model
        Returns: model, _
        """
        raise Exception

    @abstractmethod
    def solution_as_result(self,
                           solution):
        """ Decodes the solution in the format required
        by 'ModelOptimization.optimize'
        """
        raise Exception


class Solution(object):
    """ Solution
    """
    def __init__(self,
                 targets,
                 variable_names):
        """ Input:
        target: dict of targets. -1: minimize, 1 maximize
            e.g. {'p': 1, 'mean': 1}
        variable_names: list
        """
        self.targets = targets
        self.fitness = {}
        self.encoded = {}
        for variable_name in variable_names:
            self.encoded[variable_name] = None

    def get_encoded(self,
                    variable_name):
        """ Returns the 'encoded_variable'
        """
        return self.encoded[variable_name]

    def set_encoded(self,
                    variable_name,
                    value):
        self.encoded[variable_name] = value

    def get_fitness(self,
                    target):
        if target in self.targets:
            return self.fitness[target]
        else:
            raise Exception("'target' does not match")

    def set_fitness(self,
                    target,
                    value):
        if target in self.targets:
            self.fitness[target] = value
        else:
            raise Exception("'target' does not match")

    def clear_fitness(self):
        for target in self.targets:
            self.fitness = {}

    def is_evaluated(self):
        if len(self.fitness) < len(self.targets):
            return False
        for target in self.targets:
            if target not in self.fitness:
                return False
        return True

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
        if len(set(self.targets.items()) &
               set(solution.targets.items())) != len(self.targets):
            raise Exception("Solutions are not comparables")
        if len(self.fitness) != len(solution.fitness):
            raise Exception("Solutions are not comparables")
        differences = []
        for t in self.fitness:
            differences.append(self.fitness[t] * self.targets[t] -
                               solution.fitness[t] * solution.targets[t])
        differences = np.array(differences)
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
