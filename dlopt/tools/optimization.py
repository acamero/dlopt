from .. import optimization as op
from .. import util as ut
from .. import nn as nn
from . import base as b
import pandas as pd
import numpy as np


class Optimizer(b.ActionBase):
    """ Optimization utility

    Execute the optimization of an ANN architecture using the defined
    algorithm

    Mandatory parameters:
    algorithm_class
    problem_class
    data_loader_class and data_loader_params
    targets
    model_filename

    Optional:
    output_logger_class and output_logger_params
    trainer_class and trainer_params
    **any other set of params supported by the classes/functions
    """
    def __init__(self,
                 seed=1234,
                 verbose=0):
        super().__init__(seed, verbose)

    def _is_valid_config(self,
                         **config):
        if 'algorithm_class' in config:
            if not issubclass(config['algorithm_class'],
                              op.ModelOptimization):
                return False
        else:
            return False
        if 'problem_class' in config:
            if not issubclass(config['problem_class'],
                              op.Problem):
                return False
        else:
            return False
        if 'data_loader_class' in config:
            if not issubclass(config['data_loader_class'],
                              ut.DataLoader):
                return False
            if 'data_loader_params' not in config:
                return False
        else:
            return False
        if 'targets' not in config:
            return False
        if 'model_filename' not in config:
            return False
        if 'output_logger_class' in config:
            if not issubclass(config['output_logger_class'],
                              ut.OutputLogger):
                return False
            if 'output_logger_params' not in config:
                return False
        return True

    def do_action(self,
                  **kwargs):
        if not self._is_valid_config(**kwargs):
            raise Exception('The configuration is not valid')
        if 'output_logger_class' in kwargs:
            self._set_output(kwargs['output_logger_class'],
                             kwargs['output_logger_params'])
        data_loader = kwargs['data_loader_class']()
        data_loader.load(**kwargs['data_loader_params'])
        problem = kwargs['problem_class'](data_loader.dataset,
                                          verbose=self.verbose,
                                          **kwargs)
        optimizer = kwargs['algorithm_class'](problem,
                                              seed=self.seed,
                                              verbose=self.verbose)
        model, solution_desc = optimizer.optimize(**kwargs)
        self._output(**solution_desc)
        if self.verbose:
            print(solution_desc)
        model.save(kwargs['model_filename'])
