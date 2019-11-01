from dlopt import util as ut
from dlopt import sampling as sp
import json


class RunBasedListing(sp.ArchitectureListing):
    """ Generate all architecture configurations given a
    set of restrictions
    """
    restrictions = {
        'solutions_file': None,
        'decode_func': None,
        'repair': None}

    def __init__(self):
        pass

    def _repair(self,
               hidden):
        repaired = []
        for h in hidden:
            if h != 0:
                repaired.append(h)
        return repaired

    def _isvalid(self,
                 hidden):
        if len(hidden) == 0:
            return False
        for h in hidden:
            if h == 0:
                return False
        return True

    def list_architectures(self,
                           **kwargs):
        self.restrictions.update(kwargs)
        architectures = []
        decoder = ut.load_class_from_str(self.restrictions['decode_func'])
        with open(self.restrictions['solutions_file'], "r") as fp:
            for cnt, line in enumerate(fp):
                solution = eval(line)
                hidden = decoder(solution)
                if self.restrictions['repair']:
                    hidden = self._repair(hidden)
                for h in reversed(hidden):
                    if h == 0:
                        hidden.pop()
                if self._isvalid(hidden):
                    architectures.append(hidden)
        return architectures


if __name__ == '__main__':
    lst = RunBasedListing()
    archs = lst.list_architectures(
        solutions_file="manual/eunite.c.size.31081001.solutions.out",
        decode_func="rnn-arch-opt.decode_solution_size")
    print(archs)
    archs = lst.list_architectures(
        solutions_file="manual/eunite.c.flag.31081001.solutions.out",
        decode_func="rnn-arch-opt.decode_solution_flag",
        repair=True)
    print(archs)
