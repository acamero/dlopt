import tensorflow as tf
import numpy as np
import random as rd
import gc
import copy
from . import util as ut
from scipy.stats import truncnorm
from abc import ABC, abstractmethod
import time
from sklearn.metrics import log_loss
from keras.models import model_from_json


class RandomSampling(object):
    """ Perform a random sampling using a given metric"""
    def __init__(self,
                 seed=None):        
        if seed is not None:
            np.random.seed(seed)
            rd.seed(seed)
            tf.random.set_seed(seed)

    def sample(self,
               model,
               init_function,
               num_samples,
               data,
               metric_function,
               save_best=False,
               minimize=True,
               **kwargs):
        best_weights = None
        best_metric_val = None
        sampled_metrics = list()
        model.compile(optimizer='sgd', loss=metric_function, metrics=[metric_function])
        data._precompute()
        for i in range(num_samples):
            weights = self._generate_weights(model, init_function, **kwargs)
            model.set_weights(weights)
            metric = model.evaluate_generator(data)[1]
            sampled_metrics.append(copy.copy(metric))
            if save_best:
                if (best_metric_val is None
                        or ((minimize and metric < best_metric_val)
                             or (not minimize and metric > best_metric_val))):
                    best_metric_val = metric
                    del best_weights
                    gc.collect()
                    best_weights = weights
        gc.collect()
        return sampled_metrics, best_weights

    def _generate_weights(self,
                          model,
                          init_function,
                          **kwargs):
        weights = list()
        for w in model.weights:
            weights.append(init_function(w.shape, **kwargs))
        return weights


class ArchitectureListing(ABC):
    """ Abstract class for representing the listing of an
    architectural search space
    """
    @abstractmethod
    def list_architectures(self,
                           **kwargs):
        """ Generate a list of architectures given a set of
        restrictions
        """
        raise NotImplemented()


class FullSpaceListing(ArchitectureListing):
    """ Generate all architecture configurations given a
    set of restrictions
    """
    restrictions = {
        'init_arch': None,
        'min_neurons': 1,
        'max_neurons': 1,
        'min_layers': 1,
        'max_layers': 1,
        'pace': 1}

    def __init__(self):
        pass

    def list_architectures(self,
                           **kwargs):
        self.restrictions.update(kwargs)
        if self.restrictions['init_arch'] is not None:
            init_patch = (self.restrictions['init_arch']
                          + [self.restrictions['min_neurons']])
            init_layer = len(init_patch) - 1
        else:
            init_patch = [self.restrictions['min_neurons']]
            init_layer = 0
        return self._recursive(patch=init_patch,
                               layer=init_layer)

    def _recursive(self,
                   patch,
                   layer):
        architectures = []
        if len(patch) < self.restrictions['max_layers']:
            tmp = patch + [self.restrictions['min_neurons']]
            architectures += self._recursive(patch=tmp,
                                             layer=(layer+1))
        if ((patch[layer] + self.restrictions['pace']) <= self.restrictions['max_neurons']):
            tmp = patch.copy()
            tmp[layer] = tmp[layer] + self.restrictions['pace']
            architectures += self._recursive(patch=tmp,
                                             layer=layer)
        if len(patch) >= self.restrictions['min_layers']:
            architectures.append(patch)
        return architectures


class RandomSamplingFit(ABC):
    def __init__(self,
                 seed=None):
        self.sampler = RandomSampling(seed)

    @abstractmethod
    def fit(self,
            model,
            num_samples,
            data,
            **kwargs):
        raise Exception("'fit' is not implemented")


class MAERandomSampling(RandomSamplingFit):
    """ Perform a MAE random sampling.

    The sampling uses MAE as the metric, random_normal initialization, and
    fits a truncated normal distribution to the samples.
    """
    def fit(self,
            model,
            num_samples,
            data,
            truncated_lower=0.0,
            truncated_upper=2.0,
            threshold=0.01,
            **kwargs):
        start = time.time()
        samples, _ = self.sampler.sample(
            model,
            ut.random_normal,
            num_samples,
            data,
            'mae',
            **kwargs)
        """
        The standard form of this distribution is a standard normal truncated
        to the range [a, b] — notice that a and b are defined over the domain
        of the standard normal. To convert clip values for a specific mean and
        standard deviation, use:

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        """
        mean = np.mean(samples)
        std = np.std(samples)
        a = (truncated_lower - mean) / std
        b = (truncated_upper - mean) / std
        p_threshold = truncnorm.cdf(threshold,
                                    a,
                                    b,
                                    loc=mean,
                                    scale=std)
        if p_threshold == 0:
            # log_p = np.finfo(float).min
            # Some software, e.g., mipego, have trouble dealing with long floats...
            log_p = np.log(1e-300)
        else:
            log_p = np.log(p_threshold)
        sampling_time = time.time() - start
        return {'p': p_threshold,
                'log_p': log_p,
                'mean': mean,
                'std': std,
                'sampling_time': sampling_time,
                'samples': samples}
