from .. import optimization as op
from .. import util as ut
from .. import nn as nn
from . import base as b
import pandas as pd
import numpy as np
import datetime
import time


class RecurrentTraining(b.ActionBase):
    """ Recurrent training utility

    Training (and prediction) using a recurrent model

    Mandatory parameters:
    data_loader_class and data_loader_params
    architecture
    x_features
    y_features
    look_back
    nn_builder_class
    dropout
    epochs
    

    Optional:
    output_logger_class and output_logger_params
    **any other set of params supported by the classes/functions
    """
    def __init__(self,
                 seed=1234,
                 verbose=0):
        super().__init__(seed, verbose)

    def _is_valid_config(self,
                         **config):
        if 'data_loader_class' in config:
            if not issubclass(config['data_loader_class'],
                              ut.DataLoader):
                return False
            if 'data_loader_params' not in config:
                return False
        else:
            return False
        if 'architecture' not in config:
            return False
        if 'train_epochs' not in config:
            return False
        if 'train_dropout' not in config:
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
        dataset = data_loader.dataset
        builder = kwargs['nn_builder_class']()
        model = builder.build_model(kwargs['architecture'],
                                    verbose=self.verbose,
                                    **kwargs)
        if 'look_back' in kwargs:
            dataset.training_data.look_back = kwargs['look_back']
            dataset.validation_data.look_back = kwargs['look_back']
            dataset.testing_data.look_back = kwargs['look_back']
            if self.verbose:
                print("Look back updated")
        metrics, pred = self._train(model,
                                    dataset,
                                    kwargs['train_dropout'],
                                    kwargs['train_epochs'],
                                    **kwargs)
        solution_desc = {}
        solution_desc['y_predicted'] = pred.tolist()
        solution_desc['testing_metrics'] = metrics
        self._output(**solution_desc)
        if self.verbose:
            print(solution_desc)

    def _train(self,
               model,
               dataset,
               dropout,
               epochs,
               **kwargs):
        start = time.time()
        trainer = kwargs['nn_trainer_class'](verbose=self.verbose,
                                        **kwargs)
        trainer.load_from_model(model)
        trainer.init_weights(ut.random_uniform,
                             low=-0.5,
                             high=0.5)
        trainer.add_dropout(dropout)        
        trainer.train(dataset.training_data,
                      validation_dataset=dataset.validation_data,
                      epochs=epochs,
                      **kwargs)
        metrics, pred = trainer.evaluate(dataset.testing_data,
                                         **kwargs)
        del trainer
        evaluation_time = time.time() - start
        metrics['evaluation_time'] = evaluation_time
        if self.verbose:
            print(metrics)
        return metrics, pred
