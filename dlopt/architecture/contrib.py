from .. import optimization as op
from .. import ea as ea
from .. import nn as nn
from .. import sampling as sp
from .. import util as ut
from . import problems as pr
from abc import ABC, abstractmethod
from keras.layers.recurrent import LSTM
import numpy as np
import time
import gc


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
                 validation_split=0.8,
                 nn_metric_func=ut.mae_loss,
                 dropout=0.5,
                 epochs=10,
                 batch_size=5,
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
        self.validation_split = validation_split
        self.nn_metric = nn_metric_func
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size

    def solution_as_result(self,
                           solution):
        solution_desc = {}
        model, layers, look_back = self.decode_solution(solution)
        solution_desc['layers'] = layers
        solution_desc['look_back'] = look_back
        solution_desc['fitness'] = solution.fitness
        nn_metric, pred = self._train(model,
                                      look_back,
                                      self.dropout,
                                      self.epochs)
        solution_desc['testing_metric'] = nn_metric
        solution_desc['y_predicted'] = pred.tolist()
        solution_desc['config'] = str(model.get_config())
        return model, solution_desc

    def _train(self,
               model,
               look_back,
               dropout,
               epochs):
        start = time.time()
        trainer = self.nn_trainer_class(verbose=self.verbose,
                                        **self.kwargs)
        trainer.load_from_model(model)
        trainer.init_weights(ut.random_uniform,
                             low=-0.5,
                             high=0.5)
        trainer.add_dropout(dropout)
        split = int(self.training_split * len(self.data))
        validation_split = int(self.validation_split * split)
        train_dataset = nn.TimeSeriesData(
            self.data[:validation_split],
            self.x_features,
            self.y_features,
            look_back)
        validation_dataset = nn.TimeSeriesData(
            self.data[validation_split:split],
            self.x_features,
            self.y_features,
            look_back)
        trainer.train(train_dataset,
                      validation_dataset=validation_dataset,
                      epochs=epochs,
                      **self.kwargs)
        del train_dataset
        del validation_dataset
        pred, y = nn.predict(model,
                             self.data[:split],
                             self.data[split:],
                             self.x_features,
                             self.y_features,
                             look_back)
        metric = self.nn_metric(pred,
                                y)
        del df_x
        del df_y
        del trainer
        gc_out = gc.collect()
        if self.verbose > 1:
            print("GC collect", gc_out)
        if self.verbose:
            print(self.nn_metric,
                  metric)
        return metric, pred


class SelfAdjMuPLambdaUniform(ea.EABase):
    """ (Mu+Lambda) basic algorithm
    """
    def __init__(self,
                 problem,
                 seed=None,
                 verbose=0):
        super().__init__(problem,
                         seed,
                         verbose)
        # We add the default parameter values
        self.params.update({'p_mutation_i': 0.1,
                            'p_mutation_e': 0.1,
                            'mutation_max_step': 2})
        self.last_avgs = {}

    def mutate(self,
               solution):
        ea.uniformMutation(solution.get_encoded('architecture'),
                           self.params['p_mutation_i'],
                           self.params['mutation_max_step'])
        ea.uniformLengthMutation(solution.get_encoded('architecture'),
                                 self.params['p_mutation_e'])

    def select(self,
               population):
        return ea.binaryTournament(population)

    def replace(self,
                population,
                offspring):
        return ea.elitistPlusReplacement(population,
                                         offspring)

    def call_on_generation(self,
                           population):
        super().call_on_generation(population)
        avgs = {}
        for target in population[0].targets:
            avgs[target] = np.mean([sol.get_fitness(target)
                                    for sol in population])
        if self.verbose > 1:
            print("Mutation parameters before tuning",
                  self.params['p_mutation_i'],
                  self.params['p_mutation_e'])
        if len(self.last_avgs) > 0:
            diffs = []
            for target in population[0].targets:
                diff = avgs[target] - self.last_avgs[target]
                if ((population[0].targets[target] < 0 and diff <= 0) or
                        (population[0].targets[target] > 0 and diff <= 0)):
                    diffs.append(1)
                else:
                    diffs.append(-1)
                    diffs.append(-1)
            if np.sum(diffs) > 0:
                # We are improving (on average)
                self.params['p_mutation_i'] = self.params['p_mutation_i'] * 1.5
                self.params['p_mutation_e'] = self.params['p_mutation_e'] * 1.5
            else:
                self.params['p_mutation_i'] = self.params['p_mutation_i'] / 4
                self.params['p_mutation_e'] = self.params['p_mutation_e'] / 4
        self.last_avgs = avgs
        # gc_out = gc.collect()
        if self.verbose > 1:
            # print("GC collect", gc_out)
            print("Averages:", str(avgs))
            print("Mutation parameters after tuning",
                  self.params['p_mutation_i'],
                  self.params['p_mutation_e'])


class TimeSeriesTrainProblem(op.Problem):
    """ Optimize an RNN architecture based on the results
    of pre-trained networks
    """
    def __init__(self,
                 data,
                 targets,
                 verbose=0,
                 x_features=None,
                 y_features=None,
                 min_layers=1,
                 max_layers=1,
                 min_neurons=1,
                 max_neurons=1,
                 min_look_back=1,
                 max_look_back=1,
                 train_epochs=10,
                 test_epochs=10,
                 nn_builder_class=nn.RNNBuilder,
                 nn_trainer_class=nn.TrainGradientBased,
                 training_split=0.8,
                 validation_split=0.8,
                 dropout=0.5,
                 batch_size=32,
                 **kwargs):
        super().__init__(data,
                         targets,
                         verbose,
                         **kwargs)
        if x_features is None:
            self.x_features = data.columns
        else:
            self.x_features = x_features
        if y_features is None:
            self.y_features = data.columns
        else:
            self.y_features = y_features
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.min_look_back = min_look_back
        self.max_look_back = max_look_back
        self.builder = nn_builder_class()
        self.dropout = dropout
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.nn_trainer_class = nn_trainer_class
        self.training_split = training_split
        self.validation_split = validation_split
        self.builder = nn_builder_class()
        self.batch_size = batch_size

    def evaluate(self,
                 solution):
        if solution.is_evaluated():
            if self.verbose > 1:
                print('Solution already evaluated')
            return
        model, layers, look_back = self.decode_solution(solution)
        results, _, _ = self._train(model,
                                    look_back,
                                    self.dropout,
                                    self.train_epochs)
        if self.verbose > 1:
            print({'layers': layers,
                   'look_back': look_back,
                   'results': results})
        for target in self.targets:
            solution.set_fitness(target,
                                 results[target])

    def next_solution(self):
        solution = op.Solution(self.targets,
                               ['architecture'])
        num_layers = np.random.randint(low=self.min_layers,
                                       high=(self.max_layers + 1))
        layers = np.random.randint(low=self.min_neurons,
                                   high=(self.max_neurons + 1),
                                   size=num_layers)
        look_back = np.random.randint(low=self.min_look_back,
                                      high=(self.max_look_back + 1),
                                      size=1)
        solution.set_encoded('architecture',
                             np.concatenate((look_back, layers)).tolist())
        return solution

    def validate_solution(self,
                          solution):
        encoded = solution.get_encoded('architecture')
        # look back
        if len(encoded) < 1:
            encoded.append(self.min_look_back)
        if encoded[0] < self.min_look_back:
            encoded[0] = self.min_look_back
        elif encoded[0] > self.max_look_back:
            encoded[0] = self.max_look_back
        elif not isinstance(encoded[0], int):
            encoded[0] = int(encoded[0])
        # layers
        while (len(encoded) - 1) < self.min_layers:
            encoded.append(self.min_neurons)
        while (len(encoded) - 1) > self.max_layers:
            encoded.pop()
        for i in range(1, len(encoded)):
            if encoded[i] > self.max_neurons:
                encoded[i] = self.max_neurons
            elif encoded[i] < self.min_neurons:
                encoded[i] = self.min_neurons
            elif not isinstance(encoded[i], int):
                encoded[i] = int(encoded[i])

    def decode_solution(self,
                        solution):
        layers = ([len(self.x_features)] +
                  solution.get_encoded('architecture')[1:] +
                  [len(self.y_features)])
        look_back = solution.get_encoded('architecture')[0]
        model = self.builder.build_model(layers,
                                         verbose=self.verbose,
                                         **self.kwargs)
        return model, layers, look_back

    def solution_as_result(self,
                           solution):
        solution_desc = {}
        model, layers, look_back = self.decode_solution(solution)
        solution_desc['layers'] = layers
        solution_desc['look_back'] = look_back
        solution_desc['fitness'] = solution.fitness
        metrics, pred, pred_on_preds = self._train(model,
                                                   look_back,
                                                   self.dropout,
                                                   self.test_epochs)
        solution_desc['testing_metrics'] = metrics
        solution_desc['y_predicted'] = pred.tolist()
        solution_desc['y_predicted_on_predictions'] = pred_on_preds.tolist()
        solution_desc['config'] = str(model.get_config())
        return model, solution_desc

    def _train(self,
               model,
               look_back,
               dropout,
               epochs):
        start = time.time()
        trainer = self.nn_trainer_class(verbose=self.verbose,
                                        **self.kwargs)
        trainer.load_from_model(model)
        trainer.init_weights(ut.random_uniform,
                             low=-0.5,
                             high=0.5)
        trainer.add_dropout(dropout)
        split = int(self.training_split * len(self.data))
        # df_x, df_y = ut.chop_data(self.data[:split],
        #                          self.x_features,
        #                          self.y_features,
        #                          look_back)
        validation_split = int(self.validation_split * split)
        train_dataset = nn.TimeSeriesData(
            self.data[:validation_split],
            self.x_features,
            self.y_features,
            look_back,
            self.batch_size)
        validation_dataset = nn.TimeSeriesData(
            self.data[validation_split:split],
            self.x_features,
            self.y_features,
            look_back,
            self.batch_size)
        trainer.train(train_dataset,
                      validation_dataset=validation_dataset,
                      epochs=epochs,
                      **self.kwargs)
        del train_dataset
        del validation_dataset
        pred_on_preds, y = nn.predict_on_predictions(model,
                                                     self.data[:split],
                                                     self.data[split:],
                                                     self.x_features,
                                                     self.y_features,
                                                     look_back)
        pred, y = nn.predict(model,
                             self.data[:split],
                             self.data[split:],
                             self.x_features,
                             self.y_features,
                             look_back)
        del trainer
        metrics = {}
        metrics['mae'] = ut.mae_loss(pred,
                                     y)
        metrics['mse'] = ut.mse_loss(pred,
                                     y)
        metrics['mae_on_preds'] = ut.mae_loss(pred_on_preds,
                                              y)
        metrics['mse_on_preds'] = ut.mse_loss(pred_on_preds,
                                              y)
        evaluation_time = time.time() - start
        metrics['evaluation_time'] = evaluation_time
        gc_out = gc.collect()
        if self.verbose > 1:
            print("GC collect", gc_out)
        if self.verbose:
            print(metrics)
        return metrics, pred, pred_on_preds
