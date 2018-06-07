import dlopt.util as ut
import pandas as pd
import numpy as np


class SinDataLoader(ut.DataLoader):
    """ Generate a sinusoidal wave
    """
    params = {'freq': 1,
              'start': 0,
              'stop': 10,
              'step': 0.1}
    def load(self,
             **kwargs):
        self.params.update(kwargs)
        sin = np.sin(2 *
                     np.pi *
                     self.params['freq'] *
                     np.arange(start=self.params['start'], 
                               stop=self.params['stop'], 
                               step=self.params['step']))
        return pd.DataFrame(data=sin,
                            columns=["sin"])
