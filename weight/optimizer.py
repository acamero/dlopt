import tensorflow as tf
import numpy as np
import random as rd
import rnn as nn
import util as ut
import pandas as pd
import argparse
from abc import ABC, abstractmethod
import hashlib
#@article{DEAP_JMLR2012,
#    author    = " F\'elix-Antoine Fortin and Fran\c{c}ois-Michel {De Rainville} and Marc-Andr\'e Gardner and Marc Parizeau and Christian Gagn\'e ",
#    title     = { {DEAP}: Evolutionary Algorithms Made Easy },
#    pages    = { 2171--2175 },
#    volume    = { 13 },
#    month     = { jul },
#    year      = { 2012 },
#    journal   = { Journal of Machine Learning Research }
#}
from deap import creator, base, tools, algorithms


# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
import os
os.environ['PYTHONHASHSEED'] = '0'


#########################################################################################################################
class BaseOptimizer(ABC):

    def __init__(self, data, config, cache, seed=1234):
        if not self._validate_config(config):
            print('The configuration is not valid')
            raise 
        self.data = data
        self.config = config
        self.cache = cache
        self.layer_in = len(config.x_features)
        self.layer_out = len(config.y_features)
        np.random.seed(seed)
        rd.seed(seed)
        tf.set_random_seed(seed)

    @abstractmethod
    def _validate_config(self, config):
        raise NotImplemented()
    
    @abstractmethod
    def _run_algorithm(self, stats, hall_of_fame):
        raise NotImplemented()

    @abstractmethod
    def _decode_solution(self, encoded_solution):
        raise NotImplemented()

    def _evaluate_solution(self, encoded_solution):
        decoded = self._decode_solution(encoded_solution)
        print('Evaluate: ' + str(decoded['layers']) + ' ...')
        model_hash = hashlib.sha224(str(decoded['look_back']).encode('UTF-8') + str(decoded['weights']).encode('UTF-8')).hexdigest()
        metrics = self.cache.upsert_cache(model_hash, None)
        if metrics is None:
            rnn_solution = nn.RNNBuilder(decoded['layers'], decoded['weights'])
            y_predicted = rnn_solution.predict(self.data[self.config.x_features], decoded['look_back'])
            y_gt = self.data[self.config.y_features].values[decoded['look_back']:,:]
            mse = ut.mse_loss(y_predicted, y_gt)
            mae = ut.mae_loss(y_predicted, y_gt)
            metrics = { 'trainable_params':int(rnn_solution.trainable_params),
                        'num_hidden_layers':int(rnn_solution.hidden_layers),
                        'layers':'-'.join(map(str, decoded['layers'])), 
                        'mse':mse, 
                        'mae':mae, 
                        'num_hidden_neurons':int(np.sum(decoded['layers'][1:-1])),
                        'look_back':int(decoded['look_back'])
                        }
            del rnn_solution
            self.cache.upsert_cache(model_hash, metrics)
        else:
            print('Metrics load from cache')
        print(metrics)
        return metrics

    def optimize(self, hof_size=1):
        """ Start the optimization
        hof_size: the number of individuals to retain in the hall of fame (best individuals seen)
        """
        # Initialize statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("med", np.median, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        hall_of_fame = tools.HallOfFame(hof_size, similar=self._eq)
        pop, logbook, hall_of_fame = self._run_algorithm(stats, hall_of_fame)
        return pop, logbook, hall_of_fame

    def _eq(self, a, b):
        eq = True
        for i in range(len(a.fitness.values)):
            if a.fitness.values[i] != b.fitness.values[i]:
                eq = False
                break
        return eq
   

#########################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--seed',
          type=int,
          default=1,
          help='Random seed.'
    )
    parser.add_argument(
          '--hof',
          type=int,
          default=1,
          help='Hall of fame size.'
    )
    parser.add_argument(
          '--config',
          type=str,
          default='config.json',
          help='Experiment configuration file path (json format).'
    )   
    parser.add_argument(
          '--nocache', 
          action='store_true'
    ) 
    FLAGS, unparsed = parser.parse_known_args()
    config = ut.Config()
    # Load the configuration
    config.load_from_file(FLAGS.config)
    print(config)
    # Load the data
    reader =config.data_reader_class()
    data = reader.load_data( config.data_folder )
    data = pd.concat([data['train'],data['test']])
    # Load the cache
    if not FLAGS.nocache:
        cache = ut.FitnessCache()
    else:
        cache = ut.NoCache()
    cache.load_from_file(config.cache_file) 
    # Select the optimization algorithm
    optimizer = config.optimizer_class(data, config, cache, seed=FLAGS.seed)
    # And look for an optimal RNN
    pop, logbook, hof = optimizer.optimize(FLAGS.hof)
    log_df = pd.DataFrame(data=logbook)
    log_df.to_csv(config.results_folder + config.config_name + '-' + str(FLAGS.seed) + '-log.csv', sep=';', encoding='utf-8')    
    try:
        np.set_printoptions(threshold=np.inf)
        with open(config.results_folder + config.config_name + '-' + str(FLAGS.seed) + '-sol.csv','w') as f:
            for sol in hof:
                f.write(str(sol) + ';' + str(sol.fitness.values) + '\n')
                #print('sol=' + str(sol) + ';fitness=' + str(sol.fitness.values))
        f.close()
    except IOError:
        print('Unable to store the hall of fame')
    cache.save_to_file(config.cache_file)

