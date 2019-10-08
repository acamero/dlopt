import os
import numpy as np
import time
import argparse
import sys
import json
from itertools import product

from problems import get_problems

DEFAULT_FITNESS = -10000
SEP = ','

if __name__ == '__main__':
  problems = get_problems()
  parser = argparse.ArgumentParser()
  parser.add_argument('--encoding',
                      type=str,
                      default='flag',
                      help='Available encodings: flag, size, plain')
  parser.add_argument('--problem',
                      type=str,
                      default='test',
                      help='Available problems: ' + str(problems.keys()) )
  parser.add_argument('--format',
                      type=str,
                      default='csv',
                      help='Format of the file: csv, json' )
  flags, unparsed = parser.parse_known_args()

  print("Problem: " + flags.problem)
  data_loader_params = problems[flags.problem]['data_loader_params']
  etc_params = problems[flags.problem]['etc_params']
  opt_params = problems[flags.problem]['opt_params']

  print("Encoding: " + flags.encoding)

  invalid_solutions = []
  if flags.encoding == 'flag':
    # cells_per_layer = OrdinalSpace([opt_params['min_nn'], opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    # look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    # layer = NominalSpace(['Y', 'N'], 'layer') * opt_params['max_hl']
    # search_space = cells_per_layer * layer * look_back
    _layer_product = list(product('YN', repeat=opt_params['max_hl']))
    layer_product = []
    for layers in _layer_product:
      prev_n = False
      append = False
      for ix, layer in enumerate(layers):
        if layer == 'N':
          if ix == 0:
             append = True
             break
          prev_n = True
        elif prev_n:
          append = True
          break
      if append:
        layer_product.append(layers)
    del _layer_product
    cells_product = list(product([opt_params['min_nn'], opt_params['max_nn']], repeat=opt_params['max_hl']))
    lb_product = [opt_params['min_lb'], opt_params['max_lb']]
    invalid_space = list(product(cells_product, lb_product, layer_product))
    for inv in invalid_space:
      solution = {}      
      for ix, cells in enumerate(inv[0]):
        solution['cells_per_layer_' + str(ix)] = cells      
      for ix, layer in enumerate(inv[2]):
        solution['layer_' + str(ix)] = layer
      solution['look_back'] = inv[1]
      solution['n_eval'] = 1
      solution['fitness'] = DEFAULT_FITNESS
      invalid_solutions.append(solution)
  elif flags.encoding == 'size':
    #cells_per_layer = OrdinalSpace([opt_params['min_nn'], opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    #look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    #size = OrdinalSpace([1, opt_params['max_hl']], 'size')
    #search_space = cells_per_layer * size * look_back
    pass
  elif flags.encoding == 'plain':
    #TODO the lower bound for the number of neurons has to be set
    #cells_per_layer = OrdinalSpace([0, opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    #look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    #search_space = cells_per_layer * look_back
    pass
  else:
    raise Exception("Invalid encoding")


  filename = 'data_' + flags.problem + '_'+ flags.encoding 
  if flags.format == 'json':
    filename += '.json'
  elif flags.format == 'csv':
    filename += '.csv'
  
  if len(invalid_solutions) == 0:
    print("No invalid solutions")
    exit()

  with open(filename, 'w') as outfile:
    if flags.format == 'json':
      json.dump(invalid_solutions, outfile)
    elif flags.format == 'csv':
      outfile.write(SEP.join([str(x) for x in invalid_solutions[0].keys()]))
      for solution in invalid_solutions:
        outfile.write("\n")
        outfile.write(SEP.join([str(x) for x in invalid_solutions[0].values()]))
    else:
      raise Exception("Invalid output format")
    print(str(len(invalid_solutions)) + " invalid solutions added to " + filename)
