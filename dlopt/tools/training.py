from .. import optimization as op
from .. import util as ut
from .. import nn as nn
from . import base as b
from .. import sampling as sp
import pandas as pd
import numpy as np
import datetime
import time
import gc


class RecurrentTraining(b.ActionBase):
    """ Recurrent training utility

    Training (and prediction) using a recurrent model

    Mandatory parameters:
    data_loader_class and data_loader_params
    architecture or listing_class
    x_features
    y_features
    look_back
    nn_builder_class
    dropout
    train_epochs
    

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
        if ('architecture' not in config
                and 'listing_class' not in config):
            return False
        if 'train_epochs' not in config:
            return False
        if 'output_logger_class' in config:
            if not issubclass(config['output_logger_class'],
                              ut.OutputLogger):
                return Fals
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
        builder = kwargs['nn_builder_class']

        architectures = None
        if 'listing_class' in kwargs:
            listing = kwargs['listing_class']()
            architectures = listing.list_architectures(
                **kwargs['listing_params'])
        else:
            architectures = []
            architectures.append(kwargs['architecture'])

        if architectures is None:
            raise Exception('No architectures found')

        layer_in = dataset.input_dim
        layer_out = dataset.output_dim

        if 'look_back' not in kwargs:
            look_back = [dataset.training_data.look_back]
        elif isinstance(kwargs['look_back'], list):
            look_back = kwargs['look_back']
        else:
            look_back = [kwargs['look_back']]

        for lb in look_back:
            dataset.training_data.look_back = lb
            dataset.validation_data.look_back = lb
            dataset.testing_data.look_back = lb
            if self.verbose:
                print("Look back updated ", lb)
            for architecture in architectures:
                layers = [layer_in] + architecture + [layer_out]
                model = builder.build_model(
                    layers,
                    verbose=self.verbose,
                    **kwargs)
                if 'train_dropout' in kwargs:
                    model = builder.add_dropout(model, kwargs['train_dropout'])
                builder.init_weights(
                    model,
                    ut.random_uniform,
                    low=-0.5,
                    high=0.5)
        
                metrics, pred, tr_metrics = self._train(
                    model,
                    dataset,
                    kwargs['train_epochs'],
                    **kwargs)
                solution_desc = {}
                solution_desc['architecture'] = layers
                solution_desc['look_back'] = kwargs['look_back']
                solution_desc['y_predicted'] = pred.tolist()
                solution_desc['testing_metrics'] = metrics
                solution_desc['training_metrics'] = tr_metrics
                if hasattr(data_loader, 'inverse_transform'):
                    pred_real = data_loader.inverse_transform(dataset.testing_data, pred)
                    if isinstance(pred_real, np.ndarray):
                        solution_desc['y_real_predicted'] = pred_real.tolist()
                    else:
                        solution_desc['y_real_predicted'] = pred_real.values.tolist()
                self._output(**solution_desc)
                if self.verbose:
                    print(solution_desc)
                del model
                del solution_desc
                gc.collect()

    def _train(self,
               model,
               dataset,
               epochs,
               **kwargs):
        start = time.time()
        trainer = kwargs['nn_trainer_class'](
            seed=self.seed,
            verbose=self.verbose,
            **kwargs)
        trainer.load_from_model(model)                
        tr_metrics = trainer.train(
                dataset.training_data,
                validation_dataset=dataset.validation_data,
                epochs=epochs,
                **kwargs)
        metrics, pred = trainer.evaluate(dataset.testing_data,
                                         **kwargs)
        del trainer
        evaluation_time = time.time() - start
        metrics['evaluation_time'] = evaluation_time
        return metrics, pred, tr_metrics
