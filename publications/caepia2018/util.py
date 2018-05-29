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

#class ElectricityLoadDataReader(DataReader):
#    def load_data(self, data_path):
        

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

