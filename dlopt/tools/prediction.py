from .. import optimization as op
from .. import util as ut
from .. import nn as nn
from . import base as b
import pandas as pd
import numpy as np


class RecurrentPrediction(b.ActionBase):
    """ Recurrent prediction utility

    Predict using a recurrent model

    Mandatory parameters:
    data_loader_class and data_loader_params
    model_filename
    x_features
    y_features
    look_back

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
        if 'prediction_func' not in config:
            return False
        if 'data_loader_class' in config:
            if not issubclass(config['data_loader_class'],
                              ut.DataLoader):
                return False
            if 'data_loader_params' not in config:
                return False
        else:
            return False
        if 'model_filename' not in config:
            return False
        if 'x_features' not in config:
            return False
        if 'y_features' not in config:
            return False
        if 'look_back' not in config:
            return False
        if 'data_split_point' not in config:
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
        data = data_loader.load(**kwargs['data_loader_params'])
        if self.verbose:
            print('Split row',
                  data.iloc(kwargs['data_split_point']))
        model = nn.model_from_file(kwargs['model_filename'])
        pred, y = kwargs['prediction_func'](model,
                                            data[:kwargs['data_split_point']],
                                            data[kwargs['data_split_point']:],
                                            kwargs['x_features'],
                                            kwargs['y_features'],
                                            kwargs['look_back'])
        df_pred = pd.DataFrame(pred,
                               columns=kwargs['y_features'])
        self._output(df=df_pred)
        if self.verbose:
            print(df_pred)
