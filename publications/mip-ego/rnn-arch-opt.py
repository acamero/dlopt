import os
import numpy as np
import time
import argparse
import sys

#import our package, the surrogate model and the search space classes
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

from sin import SinDataLoader
from dlopt.nn import RNNBuilder as nn_builder_class
from dlopt.nn import TrainGradientBased as nn_trainer_class
from dlopt import sampling as samp
from dlopt import util as ut


#TODO: move the params to a configutarion file
#and a maximum look back
data_loader_params = {'freq': 1,
                      'start': 0,
                      'stop': 100,
                      'step': 0.1,
                      'x_features': ['sin'],
                      'y_features': ['sin'],
                      'training_ratio' : 0.8,
                      'validation_ratio' : 0.2,
                      'batch_size': 5}
etc_params = {'num_samples': 100,
              'truncated_lower': 0.0,
              'truncated_upper': 2.0,
              'threshold': 0.01,
              'model_filename': 'rnn-arch-opt-best_sin.hdf5',
              'dropout': 0.5,
              'epochs': 10,
              'dense_activation': 'sigmoid'}
opt_params = {'max_hl': 3, #max n of hidden layers
              'max_nn': 100, #max n of nn per layer
              'max_lb': 30, #max look back
              'max_eval': 3,
              'max_iter': 100,
              'n_init_samples': 2,
              'data_loader_class': 'sin.SinDataLoader'}


def decode_solution(x, input_dim, output_dim, **kwargs):
  global verbose
  print(x)
  cells = dict(filter(lambda elem: elem[0].startswith('cells_per_layer_'), x.items()))
  cells = sorted(cells.items())
  layers = dict(filter(lambda elem: elem[0].startswith('layer_'), x.items()))
  layers = sorted(layers.items())
  hidden = []
  for c, l in zip(cells, layers):
    if l[1] == 'Y':
      hidden.append(c[1])

  if len(hidden) == 0:
    return None, None
  architecture = [input_dim] + hidden + [output_dim]
  print(architecture)
  model = nn_builder_class.build_model(architecture,
                                       verbose=verbose,
                                       **kwargs)
  look_back = x['look_back']
  return model, look_back


nn_eval = 1
# The "black-box" objective function
def obj_func(x):
  global nn_eval
  global dataset
  global etc_params
  global random_seed
  print("### " + str(nn_eval) + " ######################################")
  nn_eval += 1
  model, look_back = decode_solution(x, dataset.input_dim, dataset.output_dim, **etc_params)
  if model is None:
    print("{'log_p': -10000, 'warning': 'null architecture'}")
    return -10000
  #return np.random.rand()    
  sampler = samp.MAERandomSampling(random_seed)
  #TODO copy the dataset before changing the look_back param   
  dataset.testing_data.look_back = look_back
  metrics = sampler.fit(model=model,
                        data=dataset.testing_data,
                        **etc_params)
  print(metrics)
  return metrics['log_p']


#Gradien-based NN optimization
def train_solution(x, dataset, **kwargs):
  model, look_back = decode_solution(x, dataset.input_dim, dataset.output_dim)
  if model is None:
    print("Imposible to train a null model")
    return None
  start = time.time()
  trainer = nn_trainer_class(verbose=verbose,
                             **kwargs)
  if 'dropout' in kwargs:
    model = nn_builder_class.add_dropout(model,
                                         kwargs['dropout'])
  nn_builder_class.init_weights(model,
                                ut.random_uniform,
                                low=-0.5,
                                high=0.5)
  trainer.load_from_model(model)
  dataset.training_data.look_back = look_back
  dataset.validation_data.look_back = look_back
  dataset.testing_data.look_back = look_back
  trainer.train(dataset.training_data,
                validation_dataset=dataset.validation_data,                  
                **kwargs)
  metrics, pred = trainer.evaluate(dataset.testing_data,
                                   **kwargs)
  evaluation_time = time.time() - start
  metrics['evaluation_time'] = evaluation_time
  return model, metrics, pred



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed',
                      type=int,
                      default=31081984,
                      help='Random seed (default=31081984).')
  parser.add_argument('--verbose',
                      type=int,
                      default=0,
                      help='Verbose level. 0=silent, 1=verbose, 2=debug.')
  flags, unparsed = parser.parse_known_args()
  random_seed = flags.seed
  verbose = flags.verbose
  #Load the data
  #TODO when using DLOPT config this is not necessary
  data_loader = ut.load_class_from_str(opt_params['data_loader_class'])()
  #instead, use just...
  #data_loader = opt_params['data_loader_class']()
  data_loader.load(**data_loader_params)
  dataset = data_loader.dataset
  #Define the search space
  cells_per_layer = OrdinalSpace([1, opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
  look_back = OrdinalSpace([2, opt_params['max_lb']], 'look_back')
  #TODO: I am adding the NominalSpace and ContinuousSpace because the search space is expecting
  #at least two different space types. Moreover, if one type is missing I got an error: 
  #"TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'"
  #And also because I need to test different numbers of hidden layers
  layer = NominalSpace(['Y', 'N'], 'layer') * opt_params['max_hl']
  garbage = ContinuousSpace([0., 1.], 'garbage')

  #TODO: remove garbage
  search_space = cells_per_layer * layer * look_back * garbage

  #next we define the surrogate model and the optimizer.
  model = RandomForest(levels=search_space.levels)
  opt = mipego(search_space,
               obj_func, model, 
               minimize=False,
               max_eval=opt_params['max_eval'],
               max_iter=opt_params['max_iter'],
               infill='EI',       #Expected improvement as criteria
               n_init_sample=opt_params['n_init_samples'],  #We start with 10 initial samples
               n_point=1,         #We evaluate every iteration 1 time
               n_job=1,           #  with 1 process (job).
               optimizer='MIES',  #We use the MIES internal optimizer.
               verbose=True,
               log_file="rnn-arch-opt_sin_" + str(random_seed) +".log",
               random_seed=random_seed)

  print("### Begin Optimization ######################################")
  incumbent, stop_dict = opt.run()
  print(stop_dict)
  x = incumbent.to_dict()
  # x = {'cells_per_layer_0': 12, 'cells_per_layer_1': 20, 'cells_per_layer_2': 76, 'layer_0': 'N', 'layer_1': 'N', 'layer_2': 'Y', 'look_back': 5, 'garbage': 0.11725690809327188}
  print("Best solution: " + str(x))
  print("### End Optimization ######################################")

  print("### Start Training ######################################")
  etc_params['model_filename'] = etc_params['model_filename'].replace('.hdf5',
                                                                      '_' + str(random_seed) + '.hdf5')
  model, metrics, pred = train_solution(x, dataset, **etc_params)
  print(metrics)
  print(pred)
  print("### End Training ######################################")
