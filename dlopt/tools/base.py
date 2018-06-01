from abc import ABC, abstractmethod
from .. import util as ut


class ActionBase(object):
    """ Base class for the tools defined in this package
    """
    def __init__(self,
                 data,
                 config,
                 seed):
        if not isinstance(config, ut.Config):
            raise Exception('The configuration is not valid')
        self.config = config
        self.seed = seed
        self.data = data

    @abstractmethod
    def do_action(self,
                  *args):
        raise NotImplemented('do_action is not implemented')
