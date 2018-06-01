import argparse
import sys
from pathlib import Path


if __name__ == '__main__' and __package__ is None:
    _file = Path(__file__).resolve()
    parent, top = _file.parent, _file.parents[2]
    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:
        pass
    import dlopt.tools
    __package__ = 'dlopt.tools'
    from .. import util as ut
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
    flags, unparsed = parser.parse_known_args()
    config = ut.Config()
    config.load_from_file(flags.config)
    if not config.has('action_class'):
        raise Exception('No action class set in the configuration')
    data = None
    action = config.action_class(data,
                                 config)
    action.do_action()
