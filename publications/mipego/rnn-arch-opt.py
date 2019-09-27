import os
import numpy as np
import time
import argparse
import sys

#import our package, the surrogate model and the search space classes
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

from dlopt.nn import RNNBuilder as nn_builder_class
from dlopt.nn import TrainGradientBased as nn_trainer_class
from dlopt import sampling as samp
from dlopt import util as ut

from problems import get_problems


#TODO: move the params to a configutarion file
data_loader_params = {} # passed to the data loader
etc_params = {} # sampler and training params
opt_params = {} # problem and mip-ego params
decoder = None


class SolutionDecoder(object):

  def __init__(self,
               solution_decoder,
               repair=True,
               verbose=False):
    self.solution_decoder = solution_decoder
    self.repair = repair
    self.verbose = verbose

  def _repair(self,
             hidden):
    repaired = []
    for h in hidden:
      if h != 0:
        repaired.append(h)
    return repaired

  def _isvalid(self,
               hidden):
    if len(hidden) == 0:
      return False
    for h in hidden:
      if h == 0:
        return False
    return True 

  def decode_solution(self, x, input_dim, output_dim, **kwargs):
    print(x)
    hidden = self.solution_decoder(x)
    if self.repair:
      hidden = self._repair(hidden)
    for h in reversed(hidden):
      if h == 0:
        hidden.pop()
    if not self._isvalid(hidden):
      return None, None, None
    architecture = [input_dim] + hidden + [output_dim]
    print(architecture)
    model = nn_builder_class.build_model(architecture,
                                         verbose=self.verbose,
                                         **kwargs)
    look_back = x['look_back']
    solution_id = str(architecture) + '+' + str(look_back)
    return model, look_back, solution_id


def decode_solution_flag(x):
  cells = dict(filter(lambda elem: elem[0].startswith('cells_per_layer_'), x.items()))
  cells = sorted(cells.items())
  layers = dict(filter(lambda elem: elem[0].startswith('layer_'), x.items()))
  layers = sorted(layers.items())
  hidden = []
  for c, l in zip(cells, layers):
    if l[1] == 'Y':
      hidden.append(c[1])
    else:
      hidden.append(0)
  return hidden


def decode_solution_size(x):
  cells = dict(filter(lambda elem: elem[0].startswith('cells_per_layer_'), x.items()))
  cells = sorted(cells.items())
  size = x['size']  
  hidden = []
  for c in cells[:size]:
    hidden.append(c[1])
  for c in cells[size:]:
    hidden.append(0)
  return hidden


def decode_solution_plain(x):
  cells = dict(filter(lambda elem: elem[0].startswith('cells_per_layer_'), x.items()))
  cells = sorted(cells.items())
  hidden = []
  for c in cells:
    hidden.append(c[1])
  return hidden


nn_eval = 1
lookup = {}
# The "black-box" objective function
def obj_func(x):
  global nn_eval
  global dataset
  global etc_params
  global random_seed
  global lookup
  global decoder
  print("### " + str(nn_eval) + " ######################################")
  nn_eval += 1
  model, look_back, solution_id = decoder.decode_solution(x, dataset.input_dim, dataset.output_dim, **etc_params)
  if model is None:
    print("{'log_p': -10000, 'warning': 'null architecture'}")
    return -10000
  if solution_id in lookup:    
    print("# Already computed solution")
    return lookup[solution_id]
  sampler = samp.MAERandomSampling(random_seed)
  #TODO copy the dataset before changing the look_back param   
  dataset.testing_data.look_back = look_back
  metrics = sampler.fit(model=model,
                        data=dataset.testing_data,
                        **etc_params)
  print(metrics)
  lookup[solution_id] = metrics['log_p']
  return metrics['log_p']


#Gradien-based NN optimization
def train_solution(x, dataset, **kwargs):
  global decoder
  model, look_back, solution_id = decoder.decode_solution(x, dataset.input_dim, dataset.output_dim, **kwargs)
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
  problems = get_problems()
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed',
                      type=int,
                      default=31081984,
                      help='Random seed (default=31081984).')
  parser.add_argument('--verbose',
                      type=int,
                      default=0,
                      help='Verbose level. 0=silent, 1=verbose, 2=debug.')
  parser.add_argument('--problem',
                      type=str,
                      default='test',
                      help='Available problems: ' + str(problems.keys()) )
  parser.add_argument('--encoding',
                      type=str,
                      default='flag',
                      help='Available encodings: flag, size, plain')
  parser.add_argument('--norepair',
                      dest='repair',
                      action='store_false',
                      help='Available encodings: flag, size, plain')
  flags, unparsed = parser.parse_known_args()
  random_seed = flags.seed
  verbose = flags.verbose
  print("Problem: " + flags.problem)
  data_loader_params = problems[flags.problem]['data_loader_params']
  etc_params = problems[flags.problem]['etc_params']
  opt_params = problems[flags.problem]['opt_params']
  #Load the data
  #TODO when using DLOPT config this is not necessary
  data_loader = ut.load_class_from_str(opt_params['data_loader_class'])()
  #instead, use just...
  #data_loader = opt_params['data_loader_class']()
  data_loader.load(**data_loader_params)
  dataset = data_loader.dataset
  
  #Define the search space
  search_space = None
  model = None
  print("Encoding: " + flags.encoding)
  print("Repair: " + str(flags.repair))
  if flags.encoding == 'flag':
    cells_per_layer = OrdinalSpace([opt_params['min_nn'], opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    layer = NominalSpace(['Y', 'N'], 'layer') * opt_params['max_hl']
    search_space = cells_per_layer * layer * look_back
    #assign the right decode function
    decoder = SolutionDecoder(solution_decoder=decode_solution_flag, repair=flags.repair, verbose=verbose)
    model = RandomForest(levels=search_space.levels)
  elif flags.encoding == 'size':
    cells_per_layer = OrdinalSpace([opt_params['min_nn'], opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    size = OrdinalSpace([1, opt_params['max_hl']], 'size')
    # size = NominalSpace(list(range(1, opt_params['max_hl']+1)), 'size')
    search_space = cells_per_layer * size * look_back
    decoder = SolutionDecoder(solution_decoder=decode_solution_size, repair=flags.repair, verbose=verbose)
    model = RandomForest()
    # model = RandomForest(levels=search_space.levels)
  elif flags.encoding == 'plain':
    #TODO the lower bound for the number of neurons has to be set
    cells_per_layer = OrdinalSpace([0, opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    search_space = cells_per_layer * look_back
    decoder = SolutionDecoder(solution_decoder=decode_solution_plain, repair=flags.repair, verbose=verbose)
    model = RandomForest()
  else:
    raise Exception("Invalid encoding")
  
  opt = mipego(search_space,
               obj_func,
               model, 
               minimize=False,
               max_eval=opt_params['max_eval'],
               max_iter=opt_params['max_iter'],
               infill='EI',       #Expected improvement as criteria
               n_init_sample=opt_params['n_init_samples'],  #We start with 10 initial samples
               n_point=1,         #We evaluate every iteration 1 time
               n_job=1,           #  with 1 process (job).
               optimizer='MIES',  #We use the MIES internal optimizer.
               verbose=True,
               log_file=etc_params['model_filename'].replace('.hdf5',
                                                             '_' + str(random_seed) + '.log'),
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
  if hasattr(data_loader, 'inverse_transform'):
    pred_real = data_loader.inverse_transform(dataset.testing_data, pred)
    print("Real (inversed) predicted values")
    print(pred_real)
  print("### End Training ######################################")
