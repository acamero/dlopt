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


class SepChannelsDataReader(DataReader):
    """
    Load data stored in separate channels. Each channel contains a Unix timestamp and a value, separed by a space.
    The name of each channel has to be in the format 'channel_<channel_number>.dat".
    Channels labels are stored in a 'labels.dat' file with the following
    format: "<channel_number> <channel_name>"
    """
    def load_data(self, data_path):
        dirList = os.listdir(data_path) 
        labels = {}
        dfs = {}
        for _dir in dirList:
            if os.path.isdir(data_path + _dir) == True:
                label_file = data_path + _dir + "/labels.dat"
                labels[_dir] = {}
                with open(label_file) as f:
                    for line in f:
                        splitted_line = line.split(' ')
                        labels[_dir][int(splitted_line[0])] = splitted_line[0] + '_' + splitted_line[1].strip() 
                print(labels)
                dfs[_dir] = self._read_merge_data(data_path, _dir, labels)                
        #
        return dfs

    def _read_merge_data(self, path, _dir, labels):
        _file = path + _dir + '/channel_{}.dat'.format(1)
        df = pd.read_table(_file, sep = ' ', names = ['unix_time', labels[_dir][1]], 
                                       dtype = {'unix_time': 'int64', labels[_dir][1]:'float64'})
        num_apps = len(glob.glob(path + _dir + '/channel*'))
        for i in range(2, num_apps + 1):
            _file = path + _dir + '/channel_{}.dat'.format(i)
            data = pd.read_table(_file, sep = ' ', names = ['unix_time', labels[_dir][i]], 
                                       dtype = {'unix_time': 'int64', labels[_dir][i]:'float64'})
            df = pd.merge(df, data, how = 'inner', on = 'unix_time')
        df['timestamp'] = df['unix_time'].astype("datetime64[s]")
        df = df.set_index(df['timestamp'].values)
        df.drop(['unix_time','timestamp'], axis=1, inplace=True)
        return df


############################################################################################################
class FitnessCache(object):

    _CACHE = {}

    def load_from_file(self, filename):        
        try:
            with open(filename, 'r') as f:
                f_str = f.read()
                print(f_str)
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
        self.mode_folder = 'models/'
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
        self.min_delta = 0.000001
        self.patience = 10
    
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

