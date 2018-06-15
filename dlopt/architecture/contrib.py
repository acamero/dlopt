from .. import optimization as op
from .. import ea as ea
from .. import nn as nn
from .. import sampling as sp
from .. import util as ut
from . import problems as pr
from abc import ABC, abstractmethod
from keras.layers.recurrent import LSTM
import numpy as np


class TimeSeriesHybridMRSProblem(pr.TimeSeriesMAERandSampProblem):
    """ Mean Absolute Error Random Sampling RNN Problem
    """
    def __init__(self,
                 data,
                 targets,
                 verbose=0,
                 x_features=None,
                 y_features=None,
                 num_samples=30,
                 min_layers=1,
                 max_layers=1,
                 min_neurons=1,
                 max_neurons=1,
                 min_look_back=1,
                 max_look_back=1,
                 sampler=sp.MAERandomSampling,
                 nn_builder_class=nn.RNNBuilder,
                 nn_trainer_class=nn.TrainGradientBased,
                 training_split=0.8,
                 nn_metric=ut.mae_loss,
                 **kwargs):
        super().__init__(data,
                         targets,
                         verbose,
                         x_features,
                         y_features,
                         num_samples,
                         min_layers,
                         max_layers,
                         min_neurons,
                         max_neurons,
                         min_look_back,
                         max_look_back,
                         sampler,
                         nn_builder_class,
                         **kwargs)
        self.nn_trainer_class = nn_trainer_class
        self.training_split = training_split
        self.nn_metric = nn_metric

    def solution_as_result(self,
                           solution):
        solution_desc = {}
        model, layers, look_back = self.decode_solution(solution)
        solution_desc['layers'] = layers
        solution_desc['look_back'] = look_back
        solution_desc['fitness'] = solution.fitness
        nn_metric, pred = self._train(model,
                                      look_back)
        solution_desc['testing_metric'] = nn_metric
        solution_desc['y_predicted'] = pred.tolist()
        solution_desc['config'] = str(model.get_config())
        return model, solution_desc

    def _train(self,
               model,
               look_back):
        trainer = self.nn_trainer_class(verbose=self.verbose,
                                        **self.kwargs)
        trainer.load_from_model(model)
        split = int(self.training_split * len(self.data))
        df_x, df_y = ut.chop_data(self.data[:split],
                                  self.x_features,
                                  self.y_features,
                                  look_back)
        trainer.train(df_x,
                      df_y,
                      **self.kwargs)
        pred, y = nn.predict_on_predictions(model,
                                            self.data[:split],
                                            self.data[split:],
                                            self.x_features,
                                            self.y_features,
                                            look_back)
        metric = self.nn_metric(pred,
                                y)
        return metric, pred
