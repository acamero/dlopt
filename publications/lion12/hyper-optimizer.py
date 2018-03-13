import tensorflow as tf
import util as ut
import scipy as sp
from hyperopt import hp, fmin, tpe
import numpy as np
import argparse



data_dict = {}
tests = []

# define an objective function
def objective(args):    
    # print( args )
    config = args['config']
    for attr in args:
        if attr != 'config':
            config.set(attr, args[attr])
    print("Evaluate " + str(config) )
    cache = ut.FitnessCache()
    optimizer = config.optimizer_class(data_dict, config, cache, seed=args['seed'])
    pop, logbook, hof = optimizer.optimize(args['hof'])
    sol = hof[0]
    print("Fitness " + str(sol.fitness.values) )
    tests.append( str(config) + ";" + str(sol.fitness.values) )
    return sol.fitness.values[0]


#######################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--seed',
          type=int,
          default=1,
          help='Random seed.'
    )    
    parser.add_argument(
          '--merge',
          choices=['inner', 'outer'],
          default='outer',
          help='Merge training/testing dataset using inner or outer criteria'
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
    FLAGS, unparsed = parser.parse_known_args()
    # Configure the hyperparametrization
    config = ut.Config()
    config.load_from_file(FLAGS.config)
    reader =config.data_reader_class()
    inner = False
    if FLAGS.merge == 'inner':
        inner = True
    data_dict = reader.load_data( config.data_folder, inner )
    # Configure the search space
    space = hp.choice('algorithm_type', [
            {
                'config': config,
                'seed': FLAGS.seed,
                'hof': FLAGS.hof,
                'mut_max_step': hp.quniform('mut_max_step', 1, 10, 1),
                'cx_prob': hp.uniform('cx_prob', 0, 1),
                'mut_prob': hp.uniform('mut_prob', 0, 1),
                'mut_x_prob': hp.uniform('mut_x_prob', 0, 1),
                'pop_size': 1 + hp.randint('pop_size', 10),
                'offspring_size': 1 + hp.randint('offspring_size', 10),
                'batch_size': 10 + hp.randint('batch_size', 100)
            }
            ])
    best = fmin(objective, space, algo=tpe.suggest, max_evals=3)
    print(best)
    try:
        with open(config.results_folder + config.config_name + '-' + str(FLAGS.seed) + '-sol.txt','w') as f:
            f.write(str(best))
        f.close()
    except IOError:
        print('Unable to store the solution')
    try:
        with open(config.results_folder + config.config_name + '-' + str(FLAGS.seed) + '-log.txt','w') as f:
            for test in tests:
                f.write(test + '\n')
        f.close()
    except IOError:
        print('Unable to store the log')
