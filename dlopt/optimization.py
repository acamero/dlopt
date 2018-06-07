import tensorflow as tf
import numpy as np
import random as rd
from . import util as ut
from abc import ABC, abstractmethod


class ModelOptimization(ABC):
    """ Base class for archtiecture/weights optimization
    of artificial neural networks
    """
    def __init__(self,
                 seed=1234):
        np.random.seed(seed)
        rd.seed(seed)
        tf.set_random_seed(seed)
