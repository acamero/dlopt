import numpy as np
import os
import pandas as pd
import glob
import json
from abc import ABC, abstractmethod
from dateutil.parser import parse
from datetime import date, datetime

############################################################################################################
def mse_loss(y_predict, y):
    return np.mean(np.square(y_predict - y)) 

def mae_loss(y_predict, y):
    return np.mean(np.abs(y_predict - y)) 


############################################################################################################
class DataReader(ABC):
    @abstractmethod
    def load_data(self, data_path):
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

class SinDataReader(DataReader):
    def load_data(self, data_path):
        dfs = {}
        try:
            with open(data_path, 'r') as f:
                f_str = f.read()
                params = json.loads(f_str)                
            f.close()
            sin = np.sin( 2 * np.pi * params["freq"] * 
                                    np.arange(start=params["training"]["start"], 
                                    stop=params["training"]["stop"], 
                                    step=params["training"]["step"] ) )
            dfs['train'] = pd.DataFrame(data=sin, columns=["sin"])
            sin = np.sin( 2 * np.pi * params["freq"] *
                                    np.arange(start=params["testing"]["start"], 
                                    stop=params["testing"]["stop"], 
                                    step=params["testing"]["step"] ) )
            dfs['test'] = pd.DataFrame(data=sin, columns=["sin"])
        except IOError:
            print('Unable to load the cache')
        return dfs

class FFSinDataReader(DataReader):
    def load_data(self, data_path):
        dfs = {}
        try:
            with open(data_path, 'r') as f:
                f_str = f.read()
                params = json.loads(f_str)                
            f.close()
            amplitudes = params["amplitudes"]
            fundamental = params["fundamental"]
            x = np.arange(start=params["training"]["start"], 
                                    stop=params["training"]["stop"], 
                                    step=params["training"]["step"] ) 
            dfs['train'] = pd.DataFrame( pd.DataFrame( self.oboe_dtl(x, amplitudes, fundamental) ).sum() / np.sum(amplitudes), columns=["ff"])
            dfs['train'] = dfs['train'].rename(columns={0:"ff"})
            x = np.arange(start=params["testing"]["start"], 
                                    stop=params["testing"]["stop"], 
                                    step=params["testing"]["step"] ) 
            dfs['test'] = pd.DataFrame( pd.DataFrame( self.oboe_dtl(x, amplitudes, fundamental) ).sum() / np.sum(amplitudes), columns=["ff"])
        except IOError:
            print('Unable to load the cache')
        return dfs
    def oboe_dtl(self, x, amplitudes, fundamental):
        for _ix, _amp in enumerate(amplitudes):
            yield( _amp * np.sin(2 * np.pi * (_ix+1) * fundamental * x) )


class HouseholdDataReader(DataReader):
    _sub = '_mean_norm'
    def load_data(self, data_path):
         dfs = {}
         dfs['train'] = pd.read_csv(data_path + 'household_train' + self._sub + '.csv', sep=",")
         dfs['test'] = pd.read_csv(data_path + 'household_test' + self._sub + '.csv', sep=",")
         return dfs

class HouseholdRawDataReader(DataReader):
    _year = 2000 # dummy leap year to allow input X-02-29 (leap day)
    _seasons = [(0, (date(_year,  1,  1),  date(_year,  3, 20))),    # winter
           (1, (date(_year,  3, 21),  date(_year,  6, 20))),         # spring
           (2, (date(_year,  6, 21),  date(_year,  9, 22))),         # summer
           (3, (date(_year,  9, 23),  date(_year, 12, 20))),         # autumn
           (0, (date(_year, 12, 21),  date(_year, 12, 31)))]         # winter
    def __init__(self, split=0.8):
        self._split = split
    def _get_season(self, now):
        now = pd.to_datetime(now)
        if isinstance(now, datetime):
            now = now.date()
        now = now.replace(year=self._year)
        return next(season for season, (start, end) in self._seasons if start <= now <= end)
    def load_data(self, data_path):
         df = pd.read_csv(data_path, sep=";", na_values={'?'})
         #df = df.fillna(df.mean(axis=0, numeric_only=True))
         df = df.fillna(method='pad')
         df = df.fillna(method='bfill')
         df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
         df = df.set_index(df['Datetime'].values)
         df.drop(['Date','Time', 'Datetime'], axis=1, inplace=True)
         df['Week_day'] = df.index.weekday
         df['Time'] = df.index.hour + df.index.minute /60
         df['Season'] = list(map(self._get_season, df.index.values))
         df = (df - df.mean()) / (df.max() - df.min())
         pos = int(df.shape[0] * self._split)
         dfs = {}
         dfs['train'] = df.ix[0:pos]
         dfs['test'] = df.ix[pos:]
         return dfs
        

############################################################################################################
class FitnessCache(object):

    _cache = {}

    def load_from_file(self, filename):        
        try:
            with open(filename, 'r') as f:
                f_str = f.read()
                self._cache = json.loads(f_str)
                print(str(len(self._cache)) + ' entries loaded into the cache memory')
            f.close()
        except IOError:
            print('Unable to load the cache')

    def upsert_cache(self, config, fitness):
        if fitness:
            self._cache[str(config)] = fitness
            return self._cache[str(config)]
        elif str(config) in self._cache:
            return self._cache[str(config)]
        return None

    def save_to_file(self, filename):
        dj = json.dumps(self._cache)
        try:
            with open(filename,'w') as f:
                f.write(str(dj))
            f.close()
            print(str(len(self._cache)) + ' cache entries saved')
        except IOError:
            print('Unable to store the cache')

class NoCache(object):
    def load_from_file(self, filename):
        pass
    def upsert_cache(self, config, fitness):
        return None
    def save_to_file(self, filename):
        try:
            with open(filename,'w') as f:
                f.write("")
            f.close()
        except IOError:
            print('Unable to store the cache')

############################################################################################################     
class Config(object):

    def __init__(self):
        # Default params
        self.config_name = 'default'
        self.data_folder = '../data/'
        self.mode_folder = 'models/'
        self.results_folder = 'results/'
        self.optimizer_class = None
        self.data_reader_class = None
        self.cache_file = 'cache.json'
        self.x_features = []
        self.y_features = []        
        self.max_look_back = 100
        self.max_neurons = 16
        self.max_layers = 16
        self.blind = False
    
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

    def has(self, attr):
        return hasattr(self, attr)
    
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
            if (key.endswith("_class") or key.endswith("_func")) and json_config[key] != '':
                if json_config[key].count('.') > 0:
                    module_name = json_config[key][0:json_config[key].rfind('.')]
                    class_name = json_config[key][json_config[key].rfind('.')+1:]
                    setattr(self, key, getattr( __import__(module_name), class_name ) )
                else:
                    setattr(self, key, locals()[json_config[key]] )            
            else:
                setattr(self, key, json_config[key])

