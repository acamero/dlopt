import os
import numpy as np
import time
import argparse
import sys
import copy
import gc

from sklearn.metrics import mean_absolute_error, mean_squared_error

from BayesOpt import BO
from BayesOpt.base import Solution
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, ProductSpace


##########################################
# Problems
MAX_EVALS = 100
N_INIT_SAMPLES = 5
X = range(0,30)
MAX_DIM = 10 # i.e., up to x^(MAX_DIM-1)

Y0 = [(0.82) for x in X]
Y1 = [(-0.13*x + 0.82) for x in X]
Y2 = [(0.57*x**2 - 0.13*x + 0.82) for x in X]
Y3 = [(-0.16*x**3 + 0.57*x**2 - 0.13*x + 0.82) for x in X]
Y4 = [(-0.99*x**4 - 0.16*x**3 + 0.57*x**2 - 0.13*x + 0.82) for x in X]

Y = [Y0, Y1, Y2, Y3, Y4]

##########################################
# Encodings
def decode_solution_flag(x):
  coeffs = dict(filter(lambda elem: elem[0].startswith('coeffs_'), x.items()))
  coeffs = sorted(coeffs.items())
  dims = dict(filter(lambda elem: elem[0].startswith('dims_'), x.items()))
  dims = sorted(dims.items())
  poly = []
  for c, l in zip(coeffs, dims):
    if l[1] == 'Y':
      poly.append(c[1])
    else:
      poly.append(0)
  return poly


def decode_solution_size(x):
  coeffs = dict(filter(lambda elem: elem[0].startswith('coeffs_'), x.items()))
  coeffs = sorted(coeffs.items())
  dim = x['dim']  
  poly = []
  for c in coeffs[:dim]:
    poly.append(c[1])
  for c in coeffs[dim:]:
    poly.append(0)
  return poly


def decode_solution_plain(x):
  coeffs = dict(filter(lambda elem: elem[0].startswith('coeffs_'), x.items()))
  coeffs = sorted(coeffs.items())
  poly = []
  for c in coeffs:
    poly.append(c[1])
  return poly


##########################################
# BO Problems
class PolyFitProblem(object):

  def __init__(
        self,
        solution_decoder,
        x,
        Y,
        error_fn,
        random_seed=0,
        verbose=False):
    self.solution_decoder = solution_decoder
    self.Y = Y
    self.x = x
    self.error_fn = error_fn
    self.verbose = verbose
    self.random_seed = random_seed
    self.nn_eval = 0

  def _poly_to_str(
        self,
        poly):
    _str = ""
    for ix, c in enumerate(poly):
      if c > 0:
        _str = _str + "+" + str(c) + "x^" + str(ix)
      elif c < 0:
        _str = _str + str(c) + "x^" + str(ix)
    return _str

  def _decode_solution(
        self,
        x):
    if self.verbose: print(x)
    poly = self.solution_decoder(x)
    if self.verbose: print(self._poly_to_str(poly))
    fn = lambda x: [c * x**ix for ix, c in enumerate(poly)]
    poly_fn = lambda x : np.sum(fn(x))
    return poly_fn

  def fit(
        self,
        x):
    self.nn_eval += 1
    print("### " + str(self.nn_eval) + " ######################################")
    poly_fn = self._decode_solution(x)
    _Y = []
    for _x in self.x:
      _Y.append(poly_fn(_x))
    print(_Y)
    return self.error_fn(self.Y, _Y)


##########################################
def iterate_BO_cycle(
      search_space,
      problem,
      model, 
      max_eval,
      verbose,
      log_file,
      random_seed,
      data_file,
      warm_data):
  opt = BO(
      search_space,
      problem,
      model, 
      minimize=True,
      max_eval=max_eval,
      infill='EI',
      n_init_sample=N_INIT_SAMPLES,
      n_point=1,
      n_job=1,
      optimizer='MIES',
      eval_type='dict',
      verbose=verbose,
      log_file=log_file,
      random_seed=random_seed,
      data_file=data_file,
      warm_data=warm_data)
  best_sol_as_list, fitness, stop_dict = opt.run()
  if verbose: print(stop_dict)
  best_sol = opt.xopt.to_dict()
  return best_sol, fitness, opt.data
    

##########################################
def augment_decrease_data(
      data,
      augment=True):
  if data is not None and len(data) > 0: 
    original_dim = len(data[0].tolist())
    if original_dim == 1 and augment is False:
      return data, 1
    elif original_dim == MAX_DIM and augment:
      return data, MAX_DIM
  else:
    return data, None
  new_data = None
  new_dim = 0
  if augment:
    new_dim = original_dim + 1
    for _sol in data:
      coeffs = _sol.tolist()
      coeffs.append(0.0)
      var_name = _sol.var_name.tolist()
      var_name.append('coeffs_' + str(original_dim))
      temp = Solution(
          coeffs,
          var_name=var_name,
          n_eval=_sol.n_eval,
          fitness=_sol.fitness)
      if new_data is None:
        new_data = temp
      else:
        new_data = new_data + temp
  else:
    new_dim = original_dim - 1
    for _sol in data:
      coeffs = _sol.tolist()
      if coeffs[-1] == 0.0:
        coeffs.pop()
        var_name = _sol.var_name.tolist()
        var_name.pop()
        temp = Solution(
            coeffs,
            var_name=var_name,
            n_eval=_sol.n_eval,
            fitness=_sol.fitness)
        if new_data is None:
          new_data = temp
        else:
          new_data = new_data + temp
  return new_data, new_dim


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed',
                      type=int,
                      default=31081984,
                      help='Random seed (default=31081984).')
  parser.add_argument('--verbose',
                      type=int,
                      default=1,
                      help='Verbose level. 0=silent, 1=verbose, 2=debug.')
  parser.add_argument('--degree',
                      type=int,
                      default=0,
                      help='Available degrees from 0 up to ' + str(len(Y)-1))
  parser.add_argument('--encoding',
                      type=str,
                      default='flag',
                      help='Available encodings: flag, size, plain')
  parser.add_argument('--error',
                      type=str,
                      default='mse',
                      help='Available error functions: mae, mse')
  parser.add_argument('--strategy',
                      type=str,
                      default='variable',
                      help='Available strategies: variable or augm')
  
  flags, unparsed = parser.parse_known_args()
  verbose = flags.verbose
  print("Seed: " + str(flags.seed))
  print("Problem degree: " + str(flags.degree))
  print("Error metric: " + flags.error)
  error_fn = mean_squared_error
  if flags.error == 'mae':
    error_fn = mean_absolute_error

  if flags.strategy == 'variable':
    search_space = None
    model = None
    print("Encoding: " + flags.encoding)
    if flags.encoding == 'flag':
      coeffs = ContinuousSpace([-1, 1], 'coeffs') * MAX_DIM
      dims = NominalSpace(['Y', 'N'], 'dims') * MAX_DIM
      search_space = ProductSpace(coeffs, dims)
      solution_decoder = decode_solution_flag
      #model = RandomForest()
      model = RandomForest(levels=search_space.levels)
    elif flags.encoding == 'size':
      coeffs = ContinuousSpace([-1, 1], 'coeffs') * MAX_DIM
      dim = OrdinalSpace([1, MAX_DIM], 'dim')
      search_space = ProductSpace(coeffs, dim)
      solution_decoder = decode_solution_size
      model = RandomForest()
      # model = RandomForest(levels=search_space.levels)
    elif flags.encoding == 'plain':   
      coeffs = ContinuousSpace([-1, 1], 'coeffs') * MAX_DIM
      search_space = coeffs
      solution_decoder = decode_solution_plain
      model = RandomForest()
    else:
      raise Exception("Invalid encoding " + flags.encoding)

    problem = PolyFitProblem(
        solution_decoder,
        X,
        Y[flags.degree],
        error_fn,
        random_seed=flags.seed,
        verbose=flags.verbose)
    filename = flags.strategy + "_" + flags.error + "_" + flags.encoding + "_deg" + str(flags.degree) + "_" + str(flags.seed)
    print("### Begin Optimization " + filename + " ######################################")
    best_sol, fitness, _ = iterate_BO_cycle(
        search_space,
        problem.fit,
        model,         
        max_eval=MAX_EVALS,
        verbose=flags.verbose,
        log_file=filename + ".log",
        random_seed=flags.seed,
        data_file=filename + ".data",
        warm_data=None)
    print("Best solution: {\"filename\": \"" + filename + "\", \"solution\": " + str(best_sol) + ", \"fitness\": " + str(fitness) + "}")
    print("### End Optimization " + filename + " ######################################")
  elif flags.strategy == 'augm':
    evals_per_cycle = np.ceil(MAX_EVALS / MAX_DIM)
    remainder = MAX_EVALS
    warm_data = None
    problem = PolyFitProblem(
        decode_solution_plain,
        X,
        Y[flags.degree],
        error_fn,
        random_seed=flags.seed,
        verbose=flags.verbose)
    filename = flags.strategy + "_" + flags.error + "_plain_deg" + str(flags.degree) + "_" + str(flags.seed)
    dim = 1
    mean_fitness = np.Inf
    best_sol = None
    fitness = None
    print("### Begin Optimization " + filename + " ######################################")
    for ix in range(MAX_DIM):
      if remainder < evals_per_cycle:
        evals_per_cycle = remainder
      search_space = ContinuousSpace([-1, 1], 'coeffs') * (dim)
      model = RandomForest()
      print("### Begin cycle " + filename + "_ix" + str(ix) + "_dim" + str(dim) + " ######################################")
      best_sol, fitness, data = iterate_BO_cycle(
        search_space,
        problem.fit,
        model,         
        max_eval=evals_per_cycle,
        verbose=flags.verbose,
        log_file=filename + "_ix" + str(ix) + "_dim" + str(dim) + ".log",
        random_seed=flags.seed,
        data_file=filename + "_ix" + str(ix) + "_dim" + str(dim) + ".data",
        warm_data=warm_data)
      print("Temporal best solution ("+ str(ix) + "): " +
          str(best_sol) + ", Fitness: " + str(fitness) +
          ", Mean Fitness: " + str(np.mean(data.fitness)))
      print("### End cycle " + filename + "_ix" + str(ix) + "_dim" + str(dim) + " ######################################")
      remainder = remainder - evals_per_cycle
      if remainder <= 0: break
      #TODO augment/decrease dim in data
      augment = True
      if np.mean(data.fitness) > mean_fitness:
        augment = False
      mean_fitness = np.mean(data.fitness)
      warm_data, dim = augment_decrease_data(data, augment=augment)
    print("Best solution: {\"filename\": \"" + filename + "\", \"solution\": " + str(best_sol) + ", \"fitness\": " + str(fitness) + "}")
    print("### End Optimization " + filename + " ######################################")
