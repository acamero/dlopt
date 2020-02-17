import os
import numpy as np
import time
import argparse
import sys
import copy
import gc

from sklearn.metrics import mean_absolute_error, mean_squared_error

from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, ProductSpace


##########################################
# Problems
MAX_EVALS = 100
N_INIT_SAMPLES = 10
X = range(0,100)
MAX_DIM = 5 # i.e., up to x^(MAX_DIM-1)

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
  coeffs = dict(filter(lambda elem: elem[0].startswith('coeff_'), x.items()))
  coeffs = sorted(coeffs.items())
  dim = x['dim']  
  poly = []
  for c in coeffs[:dim]:
    poly.append(c[1])
  for c in coeffs[dim:]:
    poly.append(0)
  return poly


def decode_solution_plain(x):
  coeffs = dict(filter(lambda elem: elem[0].startswith('coeff_'), x.items()))
  coeffs = sorted(coeffs.items())
  poly = []
  for c in coeffs:
    poly.append(c[1])
  return poly


##########################################
# BO Problems
class PolyFitProblem(object):

  def __init__(self,
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

  def _poly_to_str(self,
                   poly):
    _str = ""
    for ix, c in enumerate(poly):
      if c > 0:
        _str = _str + "+" + str(c) + "x^" + str(ix)
      elif c < 0:
        _str = _str + str(c) + "x^" + str(ix)
    return _str

  def _decode_solution(self,
                       x):
    if verbose: print(x)
    poly = self.solution_decoder(x)
    if verbose: print(self._poly_to_str(poly))
    fn = lambda x: [c * x**ix for ix, c in enumerate(poly)]
    poly_fn = lambda x : np.sum(fn(x))
    return poly_fn

  def fit(self,
              x):
    self.nn_eval += 1
    print("### " + str(self.nn_eval) + " ######################################")
    poly_fn = self._decode_solution(x)    
    _Y = []
    for _x in self.x:
      _Y.append(poly_fn(_x))
    return self.error_fn(self.Y, _Y)


    

    



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
  
  #Define the search space
  search_space = None
  model = None
  print("Encoding: " + flags.encoding)
  if flags.encoding == 'flag':
    coeffs = ContinuousSpace([-1, 1], 'coeffs') * MAX_DIM
    dims = NominalSpace(['Y', 'N'], 'dims') * MAX_DIM
    search_space = ProductSpace(coeffs, dims)
    #assign the right decode function
    solution_decoder=decode_solution_flag
    model = RandomForest(levels=search_space.levels)
  elif flags.encoding == 'size':
    coeffs = ContinuousSpace([-1, 1], 'coeffs') * MAX_DIM
    dim = OrdinalSpace([1, MAX_DIM], 'dim')
    search_space = ProductSpace(coeffs, dim)
    solution_decoder=decode_solution_size
    model = RandomForest()
    # model = RandomForest(levels=search_space.levels)
  elif flags.encoding == 'plain':   
    coeffs = ContinuousSpace([-1, 1], 'coeffs') * MAX_DIM
    search_space = coeffs
    solution_decoder=decode_solution_plain
    model = RandomForest()
  else:
    raise Exception("Invalid encoding " + flags.encoding)

  error_fn = mean_squared_error
  if flags.error == 'mae':
    error_fn = mean_absolute_error

  problem = PolyFitProblem(
      solution_decoder,
      X,
      Y[flags.degree],
      error_fn,
      random_seed=flags.seed,
      verbose=flags.verbose)
  
  filename = flags.strategy + "_" + flags.error + "_" + flags.encoding + "_" + str(flags.degree) + "_" + str(flags.seed)
  if flags.strategy == 'variable':
    opt = BO(
        search_space,
        problem.fit,
        model, 
        minimize=True,
        max_eval=MAX_EVALS,
        infill='EI',
        n_init_sample=N_INIT_SAMPLES,
        n_point=1,
        n_job=1,
        optimizer='MIES',
        eval_type='dict',
        verbose=flags.verbose,
        log_file=filename + ".log",
        random_seed=flags.seed,
        data_file=filename + ".data")

    print("### Begin Optimization ######################################")
    incumbent_list, fitness, stop_dict = opt.run()
    print(stop_dict)
    best_sol = opt.xopt.to_dict()
    print("Best solution: " + str(best_sol) + ", Fitness: " + str(fitness))
    print("### End Optimization ######################################")
