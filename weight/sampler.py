import tensorflow as tf
import numpy as np
import random as rd
import rnn as nn
import util as ut
import pandas as pd
import argparse
import hashlib
import os
os.environ['PYTHONHASHSEED'] = '0'

#########################################################################################################################
def random_uniform(size, low=-1.0, high=1.0):
    return np.random.uniform(low=low, high=high, size=size)

def random_normal(size, loc=0.0, scale=1.0):
    return np.random.normal(loc=loc, scale=scale, size=size)

def random_normal_narrow(size, loc=0.0, scale=0.05):
    return np.random.normal(loc=loc, scale=scale, size=size)

def glorot_uniform(size):
    limit = np.sqrt(6 / (size[0]+size[1]))
    return random_uniform(size, low=-limit, high=limit)

def orthogonal(size, gain=1.0):
    """Modification of the original keras code"""
    num_rows = 1
    for dim in size[:-1]:
        num_rows *= dim
    num_cols = size[-1]
    flat_shape = (num_rows, num_cols)
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(size)
    return gain * q[:size[0], :size[1]]

#########################################################################################################################
class Sampler(object):

    def __init__(self, data, config, seed=1234):
        if not self._validate_config(config):
            print('The configuration is not valid')
            raise 
        if config.merge_data:
            self.data = pd.concat([data['train'],data['test']])
            print("Testing and training data merged")
        else:
            self.data = data['test']
        self.config = config
        self.layer_in = len(config.x_features)
        self.layer_out = len(config.y_features)
        np.random.seed(seed)
        rd.seed(seed)
        tf.set_random_seed(seed)
        self.output_file = config.results_folder + config.config_name + '-' + str(seed) + '-sol.csv'

    def _validate_config(self, config):
        if config.merge_data is None:
            return False
        if config.samples is None or config.samples < 1:
            return False
        if config.min_neurons is None or config.min_neurons < 1:
            return False
        if config.max_neurons is None or config.max_neurons < config.min_neurons:
            return False        
        if config.min_layers is None or config.min_layers < 1:
            return False
        if config.max_layers is None or config.max_layers < config.min_layers:
            return False       
        if config.params_neuron is None or config.params_neuron < 1:
            return False
        if config.min_look_back is None or config.min_look_back < 1:
            return False
        if config.max_look_back is None or config.max_look_back < config.min_look_back:
            return False
        if config.kernel_init_func is None:
            return False
        if config.kernel_init_func is None:
            return False
        if config.kernel_init_func is None:
            return False
        if config.results_folder is None:
            return False
        if config.config_name is None:
            return False
        return True

    def _generate_weights(self, layers):
        weights = list()
        # Input dim (implicit when initializing first hidden layer) and hidden layers
        for i in range(len(layers)-2):
            # Kernel weights
            weights.append( self.config.kernel_init_func( size=(layers[i], layers[i+1]*self.config.params_neuron) ) )
            # Recurrent weights
            weights.append( self.config.recurrent_init_func( size=(layers[i+1], layers[i+1]*self.config.params_neuron) ) )
            # Bias
            weights.append( self.config.bias_init_func( size=layers[i+1]*self.config.params_neuron) )
        # Output dim
        # Dense weights
        weights.append( self.config.kernel_init_func( size=(layers[-2], layers[-1] ) ) )
        # Bias
        weights.append( self.config.bias_init_func( size=layers[-1]) )
        return weights

    
    def _sample_architecture(self, layers, look_back):
        rnn_solution = nn.RNNBuilder(layers)
        maes = list()
        for i in range(self.config.samples):
            weights = self._generate_weights(layers)
            rnn_solution.update_weights( weights )
            y_predicted = rnn_solution.predict(self.data[self.config.x_features], look_back)
            y_gt = self.data[self.config.y_features].values[look_back:,:]
            mae = ut.mae_loss(y_predicted, y_gt)
            maes.append(mae)
        del rnn_solution
        mean = np.mean(maes)
        sd = np.std(maes)
        metrics = {'mean':mean, 'sd':sd, 'maes':maes, 'arch':layers, 'look_back':look_back}
        return metrics

    def sample(self):
        init_patch = [self.config.min_neurons]
        self._rec_sample(patch=init_patch)

    def _rec_sample(self, patch=[], layer=0):        
        if len(patch) < self.config.max_layers:
            tmp = patch + [self.config.min_neurons]
            self._rec_sample(patch=tmp, layer=(layer+1))
        if patch[layer] < self.config.max_neurons:
            tmp = patch.copy()
            tmp[layer] = tmp[layer] + 1
            self._rec_sample(patch=tmp, layer=layer)
        if len(patch) >= self.config.min_layers:
            for look_back in range(self.config.min_look_back, self.config.max_look_back):
                layers = [self.layer_in] + patch + [self.layer_out]
                print('arch: ' + str(layers) + ' lb:' + str(look_back))
                metrics = self._sample_architecture(layers, look_back)
                print('mean:' + str(metrics['mean']) + ' sd:' + str(metrics['sd']) )
                self._save_metrics(metrics)

    def _save_metrics(self, metrics):
        try:
            np.set_printoptions(threshold=np.inf)
            with open(self.output_file, 'a') as f:
                f.write(str(metrics)+'\n')
            f.close()
        except IOError:
            print('Unable to store the metrics')
   

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
          '--config',
          type=str,
          default='config.json',
          help='Experiment configuration file path (json format).'
    )   
    FLAGS, unparsed = parser.parse_known_args()
    config = ut.Config()
    # Load the configuration
    config.load_from_file(FLAGS.config)
    print(config)
    # Load the data
    reader =config.data_reader_class()
    data = reader.load_data( config.data_folder )
    # Select the optimization algorithm
    sampler = Sampler(data, config, seed=FLAGS.seed)
    sampler.sample()

