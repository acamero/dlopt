import tensorflow as tf
import numpy as np
import random as rd
import rnn as nn
import util as ut
import pandas as pd
import argparse
from abc import ABC, abstractmethod
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
import algorithms


#########################################################################################################################
class BaseOptimizer(ABC):

    def __init__(self, data_dict, config, cache, seed=1234):
        if not self._validate_config(config):
            print('The configuration is not valid')
            raise 
        self.cache = cache
        self.data_dict = data_dict
        self.config = config
        self.layer_in = len(config.x_features)
        self.layer_out = len(config.y_features)
        self.min_delta = config.min_delta
        self.patience = config.patience
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
        model_name = '-'.join(map(str, decoded['rnn_arch'])) + '.'
        model_name = model_name + str(decoded['look_back']) + '.' 
        model_name = model_name + str(decoded['drop_out']) 
        print(model_name)
        model_file_name = self.config.models_folder + model_name + '.hdf5'
        train_metrics = self.cache.upsert_cache(model_name, None)
        if train_metrics is None:
            trainer = nn.TrainRNN(rnn_arch=decoded['rnn_arch'], 
                    drop_out=decoded['drop_out'], model_file=model_file_name, 
                    new=True, min_delta = self.min_delta, patience = self.patience)
            train_metrics = trainer.train(self.data_dict, 
                    x_features=self.config.x_features, 
                    y_features=self.config.y_features,                     
                    epoch=self.config.epoch, 
                    val_split=self.config.val_split, 
                    batch_size=self.config.batch_size, 
                    look_back=decoded['look_back']) # Note that the look_back value is got from the solution
            self.cache.upsert_cache(model_name, train_metrics)
        else:
            print("Metrics load from cache")
        print(decoded, train_metrics)
        return train_metrics

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
        hall_of_fame = tools.HallOfFame(hof_size)
        pop, logbook, hall_of_fame = self._run_algorithm(stats, hall_of_fame)
        return pop, logbook, hall_of_fame
   



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
          '--merge',
          choices=['inner', 'outer'],
          default='outer',
          help='Merge training/testing dataset using inner or outer criteria'
    )
    parser.add_argument(
          '--config',
          type=str,
          default='config.json',
          help='Experiment configuration file path (json format).'
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    config = ut.Config()
    config.load_from_file(FLAGS.config)
    reader =config.data_reader_class()
    inner = False
    if FLAGS.merge == 'inner':
        inner = True
    data_dict = reader.load_data( config.data_folder, inner )
    cache = ut.FitnessCache()
    cache.load_from_file(config.cache_file) 
    optimizer = config.optimizer_class(data_dict, config, cache, seed=FLAGS.seed)
    pop, logbook, hof = optimizer.optimize(FLAGS.hof)
    log_df = pd.DataFrame(data=logbook)
    log_df.to_csv(config.results_folder + config.config_name + '-' + str(FLAGS.seed) + '-log.csv', sep=';', encoding='utf-8')    
    try:
        with open(config.results_folder + config.config_name + '-' + str(FLAGS.seed) + '-sol.csv','w') as f:
            for sol in hof:
                f.write(str(sol) + ';' + str(sol.fitness.values))
                print('sol=' + str(sol) + ';fitness=' + str(sol.fitness.values))
        f.close()
    except IOError:
        print('Unable to store the hall of fame')
    cache.save_to_file(config.cache_file)

