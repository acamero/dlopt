import argparse
import sys
from pathlib import Path
import dlopt.tools
from dlopt import util as ut

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed.')
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Configuration file.')
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='Verbose level. 0=silent, 1=verbose, 2=debug.')
    flags, unparsed = parser.parse_known_args()
    config = ut.Config()
    config.load_from_file(flags.config)
    if not config.has('action_class'):
        raise Exception('No action class set in the configuration')
    action = config.action_class(flags.seed,
                                 flags.verbose)
    action.do_action(**config.as_dict())
