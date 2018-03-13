import numpy as np
import os
import pandas as pd
import glob
import json
from abc import ABC, abstractmethod
from dateutil.parser import parse

############################################################################################################
def mse_loss(y_predict, y):
    return np.mean(np.square(y_predict - y)) 

def mae_loss(y_predict, y):
    return np.mean(np.abs(y_predict - y)) 


############################################################################################################
class DataReader(ABC):
    @abstractmethod
    def load_data(self, data_path, inner=False):
        raise NotImplemented()

class ParkingDFDataReader(DataReader):
    """
    """
    def load_data(self, data_path, inner=False):
        dfs = {}
        train_file = 'training_norm_outer_df.csv'
        test_file = 'testing_norm_outer_df.csv'
        if inner:
            train_file = 'training_norm_inner_df.csv'
            test_file = 'testing_norm_inner_df.csv'
        temp_df = pd.read_csv(data_path + train_file, sep = ',')
        temp_df = temp_df.set_index(temp_df['Datetime'].values)
        temp_df.drop(['Datetime'], axis=1, inplace=True)        
        dfs['train'] = temp_df
        temp_df = pd.read_csv(data_path + test_file, sep = ',')
        temp_df = temp_df.set_index(temp_df['Datetime'].values)
        temp_df.drop(['Datetime'], axis=1, inplace=True)
        dfs['test'] = temp_df
        return dfs



class ParkingDataReader(DataReader):
    """
    """
    def load_data(self, data_path, inner=False):
        dfs = {}
        dfs['train'] = self._load_training(data_path, inner)
        dfs['test'] = self._load_testing(data_path, inner)
        return dfs
#
    def _load_training(self, data_path, inner):
        _file = data_path + 'training.csv'
        df_raw = pd.read_csv(_file, sep = ',')
        return self._transform_df(df_raw, inner)
#
    def _load_testing(self, data_path, inner):
        _file = data_path + 'testing.csv'
        df_raw = pd.read_csv(_file, sep = ',')
        df_raw['Timestamp'] = df_raw['Timestamp'].map(self._round_time)
        return self._transform_df(df_raw, inner)
#
    def _transform_df(self, df_raw, inner):
        system_code_numbers = pd.unique(df_raw['SystemCodeNumber'])
        df = None
        for code in system_code_numbers:
            temp_df = df_raw.loc[df_raw['SystemCodeNumber']==code][['Percentage','Date','Timestamp']]
            temp_df['timestamp'] = temp_df['Date'] + ' ' + temp_df['Timestamp'].map(self._convert_time)
            temp_df['timestamp'] = temp_df['timestamp'].map(parse)
            temp_df['Percentage'] = temp_df['Percentage'] / 100
            temp_df = temp_df.set_index(temp_df['timestamp'].values)
            temp_df.drop(['Date','Timestamp', 'timestamp'], axis=1, inplace=True)
            temp_df.columns = [code]
            if df is None:
                df = temp_df.copy()
            elif inner:
                df = pd.merge(df, temp_df, how='inner', left_index=True, right_index=True)
            else:
                df = pd.merge(df, temp_df, how='outer', left_index=True, right_index=True)
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')
        df['weekday'] = df.index.weekday
        df['time'] = df.index.hour + df.index.minute /60
        return df
#
    def _convert_time(self, formatted_time):
        if formatted_time%1 == 0:
            str_time = str(int(formatted_time)) + ':00:00'
        else:        
            str_time = str(int(formatted_time)) + ':30:00'
        return str_time
#
    def _round_time(self, _time):
        return round( _time * 2 ) / 2



############################################################################################################
class FitnessCache(object):

    _CACHE = {}

    def load_from_file(self, filename):        
        try:
            with open(filename, 'r') as f:
                f_str = f.read()
                #print(f_str)
                self._CACHE = json.loads(f_str)
                print(str(len(self._CACHE)) + ' entries loaded into the cache memory')
            f.close()
        except IOError:
            print('Unable to load the cache')

    def upsert_cache(self, config, fitness):
        if fitness:
            self._CACHE[str(config)] = fitness
            return self._CACHE[str(config)]
        elif str(config) in self._CACHE:
            return self._CACHE[str(config)]
        return None

    def save_to_file(self, filename):
        dj = json.dumps(self._CACHE)
        try:
            with open(filename,'w') as f:
                f.write(str(dj))
            f.close()
            print(str(len(self._CACHE)) + ' cache entries saved')
        except IOError:
            print('Unable to store the cache')


############################################################################################################     
class Config(object):

    def __init__(self):
        # Default params
        self.config_name = 'default'
        self.data_folder = '../data/'
        self.models_folder = 'models/'
        self.results_folder = 'results/'
        self.cache_file = 'cache.json'
        self.optimizer_class = None
        self.data_reader_class = None
        self.x_features = []
        self.y_features = []
        self.test_split = 0.2
        self.epoch = 100
        self.val_split = 0.3
        self.batch_size = 512
        self.max_look_back = 100
        self.max_neurons = 16
        self.max_layers = 16
        self.min_delta = 0.0001
        self.patience = 50
    
    def __str__(self):
        return str(self.__dict__)
    
    def is_equal(self, to):
        if str(to)==str(self):
            return True
        return False

    def get(self, attr):
        return getattr(self, attr)

    def set(self, attr, value):
        setattr(self, attr, value)
    
    def load_from_file(self, filename):
        json_config = {}
        try:
            with open(filename, 'r') as f:
                f_str = f.read()
                json_config = json.loads(f_str)
            f.close()
        except IOError:
            print('Unable to load the configuration file')
        # Update the configuration using the data in the file
        for key in json_config:
            if key == 'optimizer_class' and json_config[key] != '':
                if json_config[key].count('.') > 0:
                    module_name = json_config[key][0:json_config[key].rfind('.')]
                    class_name = json_config[key][json_config[key].rfind('.')+1:]
                    self.optimizer_class = getattr( __import__(module_name), class_name )
                else:
                    self.optimizer_class = locals()[json_config[key]]
            elif key == 'data_reader_class' and json_config[key] != '':
                if json_config[key].count('.') > 0:
                    module_name = json_config[key][0:json_config[key].rfind('.')]
                    class_name = json_config[key][json_config[key].rfind('.')+1:]
                    self.data_reader_class = getattr( __import__(module_name), class_name )
                else:
                    self.data_reader_class = locals()[json_config[key]]
            else:
                setattr(self, key, json_config[key])

