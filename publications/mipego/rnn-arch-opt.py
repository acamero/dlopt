import os
import numpy as np
import time
import argparse
import sys

#import our package, the surrogate model and the search space classes
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

# from sin import SinDataLoader
from dlopt.nn import RNNBuilder as nn_builder_class
from dlopt.nn import TrainGradientBased as nn_trainer_class
from dlopt import sampling as samp
from dlopt import util as ut


#TODO: move the params to a configutarion file
data_loader_params = {} # passed to the data loader
etc_params = {} # sampler and training params
opt_params = {} # problem and mip-ego params

problems = {}
problems['test'] = {}
problems['test']['data_loader_params'] = {
    'freq': 1,
    'start': 0,
    'stop': 100,
    'step': 0.1,
    'x_features': ['sin'],
    'y_features': ['sin'],
    'training_ratio' : 0.8,
    'validation_ratio' : 0.2,
    'batch_size': 5}
problems['test']['etc_params'] = {
    'num_samples': 30,
    'truncated_lower': 0.0,
    'truncated_upper': 2.0,
    'threshold': 0.01,
    'model_filename': 'rnn-arch-opt-best_test.hdf5',
    'dropout': 0.5,
    'epochs': 5,
    'dense_activation': 'tanh'}
problems['test']['opt_params'] = {
    'max_hl': 3, #max n of hidden layers
    'min_nn': 1,
    'max_nn': 100, #max n of nn per layer
    'min_lb': 2,
    'max_lb': 30, #max look back
    'max_eval': 5,
    'max_iter': 100,
    'n_init_samples': 2,
    'data_loader_class': 'loaders.SinDataLoader'}

problems['sin'] = {}
problems['sin']['data_loader_params'] = {
    'freq': 1,
    'start': 0,
    'stop': 100,
    'step': 0.1,
    'x_features': ['sin'],
    'y_features': ['sin'],
    'training_ratio' : 0.8,
    'validation_ratio' : 0.2,
    'batch_size': 5}
problems['sin']['etc_params'] = {
    'num_samples': 100,
    'truncated_lower': 0.0,
    'truncated_upper': 2.0,
    'threshold': 0.01,
    'model_filename': 'rnn-arch-opt-best_sin.hdf5',
    'dropout': 0.5,
    'epochs': 100,
    'dense_activation': 'tanh'}
problems['sin']['opt_params'] = {
    'max_hl': 3, #max n of hidden layers
    'min_nn': 1,
    'max_nn': 100, #max n of nn per layer
    'min_lb': 2,
    'max_lb': 30, #max look back
    'max_eval': 100,
    'max_iter': 100,
    'n_init_samples': 10,
    'data_loader_class': 'loaders.SinDataLoader'}

problems['waste'] = {}
problems['waste']['data_loader_params'] = {
    "filename": "../../data/waste/rubbish-2013.csv",
    "batch_size" : 5,
    "training_ratio": 0.8,
    "validation_ratio": 0.2,
    "x_features": ["C-A100", "C-A107", "C-A108", "C-A109", "C-A11", "C-A110", "C-A111", "C-A112", "C-A113", "C-A115", 
               "C-A117", "C-A119", "C-A12", "C-A120", "C-A121", "C-A122", "C-A123", "C-A124", "C-A125", "C-A126", 
               "C-A127", "C-A128", "C-A129", "C-A13", "C-A130", "C-A132", "C-A133", "C-A134", "C-A135", "C-A136", 
               "C-A137", "C-A139", "C-A14", "C-A140", "C-A141", "C-A142", "C-A144", "C-A146", "C-A147", "C-A148", 
               "C-A15", "C-A151", "C-A155", "C-A156", "C-A16", "C-A160", "C-A163", "C-A164", "C-A166", "C-A167", 
               "C-A168", "C-A169", "C-A17", "C-A170", "C-A171", "C-A172", "C-A173", "C-A174", "C-A175", "C-A176", 
               "C-A177", "C-A178", "C-A179", "C-A18", "C-A181", "C-A183", "C-A184", "C-A185", "C-A187", "C-A189", 
               "C-A191", "C-A192", "C-A193", "C-A194", "C-A195", "C-A196", "C-A197", "C-A198", "C-A199", "C-A2", 
               "C-A20", "C-A204", "C-A206", "C-A207", "C-A208", "C-A21", "C-A210", "C-A211", "C-A212", "C-A213", 
               "C-A214", "C-A215", "C-A216", "C-A218", "C-A22", "C-A220", "C-A221", "C-A223", "C-A225", "C-A227", 
               "C-A228", "C-A232", "C-A235", "C-A236", "C-A237", "C-A238", "C-A239", "C-A241", "C-A242", "C-A244", 
               "C-A248", "C-A249", "C-A25", "C-A250", "C-A251", "C-A252", "C-A254", "C-A255", "C-A256", "C-A257", 
               "C-A258", "C-A259", "C-A26", "C-A260", "C-A261", "C-A266", "C-A27", "C-A270", "C-A271", "C-A272", 
               "C-A273", "C-A274", "C-A277", "C-A279", "C-A28", "C-A280", "C-A282", "C-A283", "C-A284", "C-A286", 
               "C-A287", "C-A288", "C-A289", "C-A290", "C-A294", "C-A299", "C-A3", "C-A300", "C-A302", "C-A303", 
               "C-A304", "C-A305", "C-A308", "C-A309", "C-A31", "C-A310", "C-A312", "C-A313", "C-A316", "C-A317", 
               "C-A318", "C-A319", "C-A32", "C-A321", "C-A322", "C-A324", "C-A328", "C-A329", "C-A35", "C-A36", 
               "C-A37", "C-A38", "C-A39", "C-A40", "C-A41", "C-A44", "C-A46", "C-A47", "C-A49", "C-A51", 
               "C-A52", "C-A54", "C-A55", "C-A56", "C-A57", "C-A59", "C-A6", "C-A61", "C-A62", "C-A63", 
               "C-A64", "C-A65", "C-A67", "C-A68", "C-A69", "C-A7", "C-A70", "C-A73", "C-A74", "C-A76", 
               "C-A77", "C-A78", "C-A79", "C-A8", "C-A80", "C-A81", "C-A83", "C-A84", "C-A85", "C-A86", 
               "C-A89", "C-A9", "C-A90", "C-A93", "C-A96", "C-A98", "C-A99"],
    "y_features": ["C-A100", "C-A107", "C-A108", "C-A109", "C-A11", "C-A110", "C-A111", "C-A112", "C-A113", "C-A115", 
               "C-A117", "C-A119", "C-A12", "C-A120", "C-A121", "C-A122", "C-A123", "C-A124", "C-A125", "C-A126", 
               "C-A127", "C-A128", "C-A129", "C-A13", "C-A130", "C-A132", "C-A133", "C-A134", "C-A135", "C-A136", 
               "C-A137", "C-A139", "C-A14", "C-A140", "C-A141", "C-A142", "C-A144", "C-A146", "C-A147", "C-A148", 
               "C-A15", "C-A151", "C-A155", "C-A156", "C-A16", "C-A160", "C-A163", "C-A164", "C-A166", "C-A167", 
               "C-A168", "C-A169", "C-A17", "C-A170", "C-A171", "C-A172", "C-A173", "C-A174", "C-A175", "C-A176", 
               "C-A177", "C-A178", "C-A179", "C-A18", "C-A181", "C-A183", "C-A184", "C-A185", "C-A187", "C-A189", 
               "C-A191", "C-A192", "C-A193", "C-A194", "C-A195", "C-A196", "C-A197", "C-A198", "C-A199", "C-A2", 
               "C-A20", "C-A204", "C-A206", "C-A207", "C-A208", "C-A21", "C-A210", "C-A211", "C-A212", "C-A213", 
               "C-A214", "C-A215", "C-A216", "C-A218", "C-A22", "C-A220", "C-A221", "C-A223", "C-A225", "C-A227", 
               "C-A228", "C-A232", "C-A235", "C-A236", "C-A237", "C-A238", "C-A239", "C-A241", "C-A242", "C-A244", 
               "C-A248", "C-A249", "C-A25", "C-A250", "C-A251", "C-A252", "C-A254", "C-A255", "C-A256", "C-A257", 
               "C-A258", "C-A259", "C-A26", "C-A260", "C-A261", "C-A266", "C-A27", "C-A270", "C-A271", "C-A272", 
               "C-A273", "C-A274", "C-A277", "C-A279", "C-A28", "C-A280", "C-A282", "C-A283", "C-A284", "C-A286", 
               "C-A287", "C-A288", "C-A289", "C-A290", "C-A294", "C-A299", "C-A3", "C-A300", "C-A302", "C-A303", 
               "C-A304", "C-A305", "C-A308", "C-A309", "C-A31", "C-A310", "C-A312", "C-A313", "C-A316", "C-A317", 
               "C-A318", "C-A319", "C-A32", "C-A321", "C-A322", "C-A324", "C-A328", "C-A329", "C-A35", "C-A36", 
               "C-A37", "C-A38", "C-A39", "C-A40", "C-A41", "C-A44", "C-A46", "C-A47", "C-A49", "C-A51", 
               "C-A52", "C-A54", "C-A55", "C-A56", "C-A57", "C-A59", "C-A6", "C-A61", "C-A62", "C-A63", 
               "C-A64", "C-A65", "C-A67", "C-A68", "C-A69", "C-A7", "C-A70", "C-A73", "C-A74", "C-A76", 
               "C-A77", "C-A78", "C-A79", "C-A8", "C-A80", "C-A81", "C-A83", "C-A84", "C-A85", "C-A86", 
               "C-A89", "C-A9", "C-A90", "C-A93", "C-A96", "C-A98", "C-A99"]}
problems['waste']['etc_params'] = {
    'num_samples': 100,
    'truncated_lower': 0.0,
    'truncated_upper': 1.0,
    'threshold': 0.01,
    'model_filename': 'rnn-arch-opt-best_waste.hdf5',
    'dropout': 0.5,
    'epochs': 100,
    'dense_activation': 'sigmoid'}
problems['waste']['opt_params'] = {
    'max_hl': 8, #max n of hidden layers
    'min_nn': 10,
    'max_nn': 300, #max n of nn per layer
    'min_lb': 2,
    'max_lb': 30, #max look back
    'max_eval': 100,
    'max_iter': 100,
    'n_init_samples': 10,
    'data_loader_class': 'loaders.RubbishDataLoader'}


def decode_solution(x, input_dim, output_dim, **kwargs):
  raise Exception("Function decode_solution is not defined")
  # return model, look_back, solution_id

def decode_solution_flag(x, input_dim, output_dim, **kwargs):
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
    return None, None, None
  architecture = [input_dim] + hidden + [output_dim]
  print(architecture)
  model = nn_builder_class.build_model(architecture,
                                       verbose=verbose,
                                       **kwargs)
  look_back = x['look_back']
  solution_id = str(architecture) + '+' + str(look_back)
  return model, look_back, solution_id


def decode_solution_size(x, input_dim, output_dim, **kwargs):
  global verbose
  print(x)
  cells = dict(filter(lambda elem: elem[0].startswith('cells_per_layer_'), x.items()))
  cells = sorted(cells.items())
  size = x['size']  
  hidden = []
  for c in cells[:size]:
    hidden.append(c[1])

  if len(hidden) == 0:
    return None, None, None
  architecture = [input_dim] + hidden + [output_dim]
  print(architecture)
  model = nn_builder_class.build_model(architecture,
                                       verbose=verbose,
                                       **kwargs)
  look_back = x['look_back']
  solution_id = str(architecture) + '+' + str(look_back)
  return model, look_back, solution_id


nn_eval = 1
lookup = {}
# The "black-box" objective function
def obj_func(x):
  global nn_eval
  global dataset
  global etc_params
  global random_seed
  global lookup
  print("### " + str(nn_eval) + " ######################################")
  nn_eval += 1
  model, look_back, solution_id = decode_solution(x, dataset.input_dim, dataset.output_dim, **etc_params)
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
  model, look_back, solution_id = decode_solution(x, dataset.input_dim, dataset.output_dim, **kwargs)
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
  parser.add_argument('--problem',
                      type=str,
                      default='test',
                      help='Available problems: ' + str(problems.keys()) )
  parser.add_argument('--encoding',
                      type=str,
                      default='flag',
                      help='Available encodings: flag, size' )
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
  print("Encoding: " + flags.encoding)
  if flags.encoding == 'flag':
    cells_per_layer = OrdinalSpace([opt_params['min_nn'], opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    layer = NominalSpace(['Y', 'N'], 'layer') * opt_params['max_hl']
    search_space = cells_per_layer * layer * look_back
    #assign the right decode function
    decode_solution = decode_solution_flag
  elif flags.encoding == 'size':
    cells_per_layer = OrdinalSpace([opt_params['min_nn'], opt_params['max_nn']], 'cells_per_layer') * opt_params['max_hl']
    look_back = OrdinalSpace([opt_params['min_lb'], opt_params['max_lb']], 'look_back')
    #size = OrdinalSpace([1, opt_params['max_hl']], 'size')
    size = NominalSpace(list(range(1, opt_params['max_hl']+1)), 'size')
    search_space = cells_per_layer * size * look_back
    #assign the right decode function
    decode_solution = decode_solution_size
  else:
    raise Exception("Invalid encoding")



  #next we define the surrogate model and the optimizer.
  model = RandomForest(levels=search_space.levels)
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
  print("### End Training ######################################")
