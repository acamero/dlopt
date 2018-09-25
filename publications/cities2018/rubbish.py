import dlopt.util as ut
from dlopt.nn import TimeSeriesDataset
from dlopt.optimization import Dataset
import pandas as pd
import numpy as np
from datetime import datetime

class RubbishDataLoader(ut.DataLoader):
    """ Load the rubbish dataset
    filename    file containing the data
    scaler      sklearn scaler funtion to be used with the data
                e.g. MinMaxScaler
    """
    params = {'filename': None,
              'scaler_class': None,
              'x_features' : None,
              'y_features' : None,
              'batch_size' : None,
              'training_ratio' : None,
              'validation_ratio' : None,
              'training_split_row' : None}
    scaler = None

    def load(self,
             **kwargs):
        self.params.update(kwargs)
        if self.params['filename'] is None:
            raise Exception("A 'filename' must be provided")
        if self.params['x_features'] is None:
            raise Exception("A 'x_features' list must be provided")
        if self.params['y_features'] is None:
            raise Exception("A 'y_features' list must be provided")
        if self.params['batch_size'] is None:
            raise Exception("A 'batch_size' must be provided")
        if (self.params['training_ratio'] is None
                and self.params['training_split_row'] is None):
            raise Exception("A 'training_ratio' or 'training_split row must be provided")
        if self.params['validation_ratio'] is None:
            raise Exception("A 'validation_ratio' must be provided")
        df = pd.read_csv(self.params['filename'])
        df = df.set_index(pd.to_datetime(df['date'].values))
        df.drop(['date'], axis=1, inplace=True)
        df = df / 100
        if self.params['scaler_class'] is not None:
            scaler_class = ut.load_class_from_str(self.params['scaler_class'])
            self.scaler = scaler_class()
            df = pd.DataFrame(self.scaler.fit_transform(df),
                              columns=df.columns)
        if self.params['training_ratio'] is not None:
            split = int(self.params['training_ratio'] * df.shape[0])
        else:
            split = self.params['training_split_row']
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

    def inverse_transform(self,
                          df):
        if self.scaler is None:
            return df
        else:
            inversed = pd.DataFrame(self.scaler.inverse_transform(df),
                                    columns=df.columns)
            return inversed
