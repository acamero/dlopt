import os
import numpy as np
import time
import argparse
import sys
import copy
import gc

#import our package, the surrogate model and the search space classes
from BayesOpt import BO
#from mipego import mipego
from BayesOpt.Surrogate import RandomForest
#from mipego.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, ProductSpace
#from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

from dlopt.nn import RNNBuilder as nn_builder_class
from dlopt.nn import TrainGradientBased as nn_trainer_class
from dlopt import sampling as samp
from dlopt import util as ut

from problems import get_problems


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


class MRSProblem(object):

  DEFAULT_FITNESS = -10000

  def __init__(self,
               solution_decoder,
               dataset,
               repair=True,
               random_seed=0,
               verbose=False,
               **etc_params):
    self.solution_decoder = solution_decoder
    self.dataset = dataset
    self.etc_params = etc_params
    self.repair = repair
    self.verbose = verbose
    self.random_seed = random_seed
    self.nn_eval = 0
    self.lookup = {}

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

  def penalty(self,
              x):
    penalty = 0
    hidden = self.solution_decoder(x)
    architecture = [self.dataset.input_dim]
    last_h = -1
    for ix, h in enumerate(hidden):
      if h > 0:
        penalty += ix - last_h - 1
        last_h = last_h + 1
        architecture += [h]

    architecture += [self.dataset.output_dim]
    look_back = x['look_back']
    solution_id = str(architecture) + '+' + str(look_back)    
    if (solution_id in self.lookup 
        or last_h == -1):
      penalty = (len(hidden) * (len(hidden) + 1)) / 2 
    return penalty 

  def _decode_solution(self,
                       x):
    print(x)
    hidden = self.solution_decoder(x)
    if self.repair:
      hidden = self._repair(hidden)
    for h in reversed(hidden):
      if h == 0:
        hidden.pop()
    if not self._isvalid(hidden):
      return None, None, None
    architecture = [self.dataset.input_dim] + hidden + [self.dataset.output_dim]
    print(architecture)
    model = nn_builder_class.build_model(architecture,
                                         verbose=self.verbose,
                                         **self.etc_params)
    look_back = x['look_back']
    solution_id = str(architecture) + '+' + str(look_back)
    return model, look_back, solution_id

  def mrs_fit(self,
              x):
    self.nn_eval += 1
    print("### " + str(self.nn_eval) + " ######################################")
    # K.clear_session()
    # gc.collect()
    model, look_back, solution_id = self._decode_solution(x)
    if model is None:
      print("{'log_p': "+ str(self.DEFAULT_FITNESS) + ", 'warning': 'null architecture'}")
      return self.DEFAULT_FITNESS
    if solution_id in self.lookup:    
      print("# Already computed solution")
      return self.lookup[solution_id]
    sampler = samp.MAERandomSampling(self.random_seed)
    #TODO copy the dataset before changing the look_back param   
    self.dataset.testing_data.look_back = look_back
    metrics = sampler.fit(model=model,
                          data=self.dataset.testing_data,
                          **self.etc_params)
    print(metrics)
    self.lookup[solution_id] = copy.copy(metrics['log_p'])
    return self.lookup[solution_id]

  def train_solution(self,
                     x):
    model, look_back, solution_id = self._decode_solution(x)
    if model is None:
      print("Imposible to train a null model")
      return None
    start = time.time()
    trainer = nn_trainer_class(verbose=verbose,
                               **self.etc_params)
    if 'dropout' in self.etc_params:
      model = nn_builder_class.add_dropout(model,
                                           self.etc_params['dropout'])
    nn_builder_class.init_weights(model,
                                  ut.random_uniform,
                                  low=-0.5,
                                  high=0.5)
    trainer.load_from_model(model)
    self.dataset.training_data.look_back = look_back
    self.dataset.validation_data.look_back = look_back
    self.dataset.testing_data.look_back = look_back
    trainer.train(self.dataset.training_data,
                validation_dataset=self.dataset.validation_data,                  
                **self.etc_params)
    metrics, pred = trainer.evaluate(self.dataset.testing_data,
                                     **self.etc_params)
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
                      default=1,
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
  parser.add_argument('--warmdata',
                      type=str,
                      default=None,
                      help='Warm start data filename')
  parser.add_argument('--constraint',
                      dest='constraint',
                      action='store_true',
                      help='Add constraints to the EI')
  flags, unparsed = parser.parse_known_args()
  verbose = flags.verbose
  print("Seed: " + str(flags.seed))
  print("Problem: " + flags.problem)
  data_loader_params = problems[flags.problem]['data_loader_params']
  etc_params = problems[flags.problem]['etc_params']
  opt_params = problems[flags.problem]['opt_params']
  etc_params['model_filename'] = etc_params['model_filename'].replace('.hdf5',
                                                                      '_' + str(flags.seed) + '.hdf5')
  etc_params['log_filename'] = etc_params['log_filename'].replace('.log',
                                                                  '_' + str(flags.seed) + '.log')
  if 'data_filename' not in etc_params:
    etc_params['data_filename'] = etc_params['log_filename'].replace('.log',
                                                                     '.csv')
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
    search_space = ProductSpace(ProductSpace(cells_per_layer, layer), look_back)
    # mipego -> search_space = cells_per_layer * layer * look_back
    #assign the right decode function
    solution_decoder=decode_solution_flag
    model = RandomForest(levels=search_space.levels)
  elif flags.encoding == 'size':
    cells_per_layer = OrdinalSpace([opt_params['min_nn'], opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    size = OrdinalSpace([1, opt_params['max_hl']], 'size')
    # size = NominalSpace(list(range(1, opt_params['max_hl']+1)), 'size')
    search_space = ProductSpace(ProductSpace(cells_per_layer, size), look_back)
    # mipego -> search_space = cells_per_layer * size * look_back
    solution_decoder=decode_solution_size
    model = RandomForest()
    # model = RandomForest(levels=search_space.levels)
  elif flags.encoding == 'plain':
    #TODO the lower bound for the number of neurons has to be set
    cells_per_layer = OrdinalSpace([0, opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    search_space = ProductSpace(cells_per_layer, look_back)
    # mipego -> search_space = cells_per_layer * look_back
    solution_decoder=decode_solution_plain
    model = RandomForest()
  else:
    raise Exception("Invalid encoding")

  print("Warm start data: " + str(flags.warmdata))

  mrs_problem = MRSProblem(solution_decoder,
                           dataset,
                           repair=flags.repair,
                           random_seed=flags.seed,
                           verbose=flags.verbose,
                           **etc_params)
  
  print("Constraint: " + str(flags.constraint))
  constraint_eq_func = mrs_problem.penalty if flags.constraint else None
  #TODO pass mrs_problem.mrs_fit as the obj_func
  #opt = mipego(
  opt = BO(
      search_space,
      mrs_problem.mrs_fit,
      model, 
      minimize=False,
      eq_func=constraint_eq_func,
      max_eval=opt_params['max_eval'],
      max_iter=opt_params['max_iter'],
      infill='EI',       #Expected improvement as criteria
      n_init_sample=opt_params['n_init_samples'],  #We start with 10 initial samples
      n_point=1,         #We evaluate every iteration 1 time
      n_job=1,           #  with 1 process (job).
      optimizer='MIES',  #We use the MIES internal optimizer.
      eval_type='dict',  #To get the solution, as well as the var_name
      verbose=flags.verbose,
      log_file=etc_params['log_filename'],
      random_seed=flags.seed,
      data_file=etc_params['data_filename'],
      warm_data=flags.warmdata)  # mipego -> warm_data_file=flags.warmdata)

  print("### Begin Optimization ######################################")
  incumbent_list, fitness, stop_dict = opt.run()
  print(stop_dict)
  x = opt.xopt.to_dict()
  print("Best solution: " + str(x) + ", Fitness: " + str(fitness))
  print("### End Optimization ######################################")
  print("### Start Training ######################################")    
  model, metrics, pred = mrs_problem.train_solution(x)
  print("### End Training ######################################")
  print(model.summary())
  print(metrics)
  print(pred)
  if hasattr(data_loader, 'inverse_transform'):
    pred_real = data_loader.inverse_transform(dataset.testing_data, pred)
    print("Real (inversed) predicted values")
    print(pred_real)
