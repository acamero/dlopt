from abc import ABC, abstractmethod
from .. import util as ut


class ActionBase(object):
    """ Base class for the tools defined in this package
    """
    _output_logger = None

    def __init__(self,
                 seed,
                 verbose):
        self.seed = seed
        self.verbose = verbose

    @abstractmethod
    def do_action(self,
                  **kwargs):
        raise NotImplemented('do_action is not implemented')

    def _output(self,
                **kwargs):
        if self._output_logger is None:
            print(str(kwargs))
        else:
            self._output_logger.output(**kwargs)

    def _set_output(self,
                    output_logger_class,
                    output_logger_params):
        self._output_logger = output_logger_class(**output_logger_params)
