import dlopt.util as ut
from dlopt.nn import TimeSeriesDataset
from dlopt.optimization import Dataset
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


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


class AppliancesEnergyDataLoader(ut.DataLoader):
    """ Load the Appliances energy prediction Data Set
    file    path to the data file
    """
    params = {'filename': None,
              'scaler_class': None,
              'x_features' : None,
              'y_features' : None,
              'batch_size' : None,
              'training_ratio' : None,
              'validation_ratio' : None,
              'training_split_row' : None,
              'precompute': False}

    def load(self,
             **kwargs):
        self.params.update(kwargs)
        if self.params['filename'] is None:
            raise Exception("A 'file' must be provided")        
        if self.params['batch_size'] is None:
            raise Exception("A 'batch_size' must be provided")
        df = pd.read_csv(self.params['filename'])
        df = df.set_index(pd.to_datetime(df['date'].values))
        df.drop(['date'], axis=1, inplace=True)
        # Remove random variables added in the dataset
        # df.drop(['rv1', 'rv2'], axis=1, inplace=True)
        # Preprocess data
        self.scaler = None
        if self.params['scaler_class'] is not None:
            scaler_class = ut.load_class_from_str(self.params['scaler_class'])
            self.scaler = scaler_class()
        if self.scaler is not None:
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
                                          batch_size=self.params['batch_size'],
                                          precompute=bool(self.params['precompute']))
        validation_data = TimeSeriesDataset(df[validation_split:split],
                                            self.params['x_features'],
                                            self.params['y_features'],
                                            batch_size=self.params['batch_size'],
                                            precompute=bool(self.params['precompute']))
        testing_data = TimeSeriesDataset(df[split:],
                                         self.params['x_features'],
                                         self.params['y_features'],
                                         batch_size=self.params['batch_size'],
                                         precompute=bool(self.params['precompute']))
        self.dataset = Dataset(training_data,
                               validation_data,
                               testing_data,
                               input_dim=len(self.params['x_features']),
                               output_dim=len(self.params['y_features']))

    def inverse_transform(self,
                          dataset,
                          df):
        if self.scaler is None:
            return df
        df_pred = None
        if isinstance(pred, np.ndarray):
            df_pred = pd.DataFrame(pred, columns=dataset.y_features)
        elif isinstance(pred, pd.DataFrame):
            df_pred = pred
        else:
            raise Exception("Please provide a valid 'pred' (numpy ndarray or pandas DF)")
        _df = dataset.df.tail(df_pred.shape[0])
        _df =_df.drop(dataset.y_features, axis=1)
        _df = pd.concat([_df, df_pred.set_index(_df.index)], axis=1)
        inversed = pd.DataFrame(
            self.scaler.inverse_transform(_df[list(set(dataset.x_features + dataset.y_features))]),
            columns=list(set(dataset.x_features + dataset.y_features)))
        return inversed[dataset.y_features]


class CoalDataLoader(ut.DataLoader):
    """ Load the coal dataset
    filename    file containing the data
    """
    params = {'filename': None,
              'scaler_class': None,
              'x_features' : None,
              'y_features' : None,
              'batch_size' : None,
              'training_ratio' : None,
              'validation_ratio' : None,
              'training_split_row' : None}

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


class ParkingDFDataReader(ut.DataLoader):
    """ Load the birmingham parking occupancy dataset
    filename    file containing the data
    """
    params = {'training_filename': None,
              'testing_filename': None,
              'x_features' : None,
              'y_features' : None,
              'batch_size' : None,
              'validation_ratio' : None}

    def load(self,
             **kwargs):
        self.params.update(kwargs)
        if self.params['training_filename'] is None:
            raise Exception("A 'training_filename' must be provided")
        if self.params['testing_filename'] is None:
            raise Exception("A 'testing_filename' must be provided")
        if self.params['x_features'] is None:
            raise Exception("A 'x_features' list must be provided")
        if self.params['y_features'] is None:
            raise Exception("A 'y_features' list must be provided")
        if self.params['batch_size'] is None:
            raise Exception("A 'batch_size' must be provided")
        if self.params['validation_ratio'] is None:
            raise Exception("A 'validation_ratio' must be provided")


        df_tr = pd.read_csv(self.params['training_filename'], sep = ',')
        df_tr = df_tr.set_index(df_tr['Datetime'].values)
        df_tr.drop(['Datetime'], axis=1, inplace=True)        
        print(df_tr.describe())
        df_ts = pd.read_csv(self.params['testing_filename'], sep = ',')
        df_ts = df_ts.set_index(df_ts['Datetime'].values)
        df_ts.drop(['Datetime'], axis=1, inplace=True)
        print(df_ts.describe())
        validation_split = int((1 - self.params['validation_ratio']) * df_tr.shape[0])
        training_data = TimeSeriesDataset(df_tr[:validation_split],
                                          self.params['x_features'],
                                          self.params['y_features'],
                                          batch_size=self.params['batch_size'])
        validation_data = TimeSeriesDataset(df_tr[validation_split:],
                                            self.params['x_features'],
                                            self.params['y_features'],
                                            batch_size=self.params['batch_size'])
        testing_data = TimeSeriesDataset(df_ts,
                                         self.params['x_features'],
                                         self.params['y_features'],
                                         batch_size=self.params['batch_size'])
        self.dataset = Dataset(training_data,
                               validation_data,
                               testing_data,
                               input_dim=len(self.params['x_features']),
                               output_dim=len(self.params['y_features']))
