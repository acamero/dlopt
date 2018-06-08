from .. import optimization as op
from abc import ABC, abstractmethod


class ArchitectureProblem(op.Problem):
    """ Problem base class
    """
    def __init__(self,
                 target,
                 min_layers=1,
                 max_layers=1,
                 min_neurons=1,
                 max_neurons=1):
        super().__init__(target)
        restrictions = {}
        restrictions['min_layers'] = min_layers
        restrictions['max_layers'] = max_layers
        restrictions['min_neurons'] = min_neurons
        restrictions['max_neurons'] = max_neurons

    @abstractmethod
    def evaluate(self,
                 solution):
        raise Exception

    def next_solution(self):
        op.Solution(target)

    @abstractmethod
    def validate_solution(self):
        raise Exception
