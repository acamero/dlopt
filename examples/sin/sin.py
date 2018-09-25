import dlopt.util as ut
from dlopt.nn import TimeSeriesDataset
from dlopt.optimization import Dataset
import pandas as pd
import numpy as np


class SinDataLoader(ut.DataLoader):
    """ Generate a sinusoidal wave
    """
    params = {'freq': 1,
              'start': 0,
              'stop': 10,
              'step': 0.1,
              'x_features': ['sin'],
              'y_features': ['sin'],
              'training_ratio' : 0.8,
              'validation_ratio' : 0.2,
              'batch_size': 5}

    def load(self,
             **kwargs):
        self.params.update(kwargs)
        sin = np.sin(2 *
                     np.pi *
                     self.params['freq'] *
                     np.arange(start=self.params['start'], 
                               stop=self.params['stop'], 
                               step=self.params['step']))
        df = pd.DataFrame(data=sin,
                          columns=['sin'])
        split = int(self.params['training_ratio'] * df.shape[0])
        validation_split = int((1 - self.params['validation_ratio']) * split)
        training_data = TimeSeriesDataset(df[:validation_split],
                                          self.params['x_features'],
                                          self.params['y_features'],
                                          batch_size=self.params['batch_size'])
        validation_data = TimeSeriesDataset(df[validation_split:split],
                                            self.params['x_features'],
                                            self.params['y_features'],
                                            batch_size=self.params['batch_size'])
        testing_data = TimeSeriesDataset(df[split:],
                                         self.params['x_features'],
                                         self.params['y_features'],
                                         batch_size=self.params['batch_size'])
        self.dataset = Dataset(training_data,
                               validation_data,
                               testing_data,
                               input_dim=len(self.params['x_features']),
                               output_dim=len(self.params['y_features']))
