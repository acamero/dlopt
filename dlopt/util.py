import numpy as np
import json
import importlib


def random_uniform(size,
                   **kwargs):
    low = -1.0
    high = 1.0
    if kwargs is not None:
        if 'low' in kwargs:
            low = kwargs['low']
        if 'high' in kwargs:
            high = kwargs['high']
    return np.random.uniform(low=low,
                             high=high,
                             size=size)


def random_normal(size,
                  **kwargs):
    loc = 0.0
    scale = 1.0
    if kwargs is not None:
        if 'loc' in kwargs:
            low = kwargs['loc']
        if 'scale' in kwargs:
            high = kwargs['scale']
    return np.random.normal(loc=loc,
                            scale=scale,
                            size=size)


def glorot_uniform(size):
    limit = np.sqrt(6 / (size[0] + size[1]))
    return random_uniform(size,
                          low=-limit,
                          high=limit)


def mse_loss(y_predict,
             y):
    return np.mean(np.square(y_predict - y))


def mae_loss(y_predict,
             y):
    return np.mean(np.abs(y_predict - y))


def chop_data(df, x_features, y_features, look_back):
    len_data = df.shape[0]
    x = np.array([df[x_features].values[i:i+look_back]
                  for i in range(len_data-look_back)]).reshape(-1,
                                                               look_back,
                                                               len(x_features))
    y = df[y_features].values[look_back:, :]
    return x, y


def orthogonal(size,
               **kwargs):
    """Modification of the original keras code"""
    gain = 1.0
    if kwargs is not None:
        if 'gain' in kwargs:
            low = kwargs['gain']
    num_rows = 1
    for dim in size[:-1]:
        num_rows *= dim
    num_cols = size[-1]
    flat_shape = (num_rows, num_cols)
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(size)
    return gain * q[:size[0], :size[1]]


class Config(object):
    """ Configuration class loaded from a file
    """
    def __init__(self):
        # Default params
        self.config_name = 'default'

    def __str__(self):
        return str(self.__dict__)

    def is_equal(self,
                 to):
        if str(to) == str(self):
            return True
        return False

    def get(self,
            attr):
        return getattr(self, attr)

    def set(self,
            attr,
            value):
        setattr(self, attr, value)

    def has(self,
            attr):
        return hasattr(self, attr)

    def load_from_file(self,
                       filename):
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
            if ((key.endswith("_class") or key.endswith("_func"))
                    and json_config[key] != ''):
                if json_config[key].count('.') > 0:
                    module_name = json_config[key][
                        0:json_config[key].rfind('.')]
                    class_name = json_config[key][
                        json_config[key].rfind('.')+1:]
                    setattr(self,
                            key,
                            getattr(importlib.import_module(module_name),
                                    class_name))
                else:
                    setattr(self, key, locals()[json_config[key]])
            else:
                setattr(self, key, json_config[key])
