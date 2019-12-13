from .. import sampling as samp
from .. import util as ut
from .. import nn as nn
from . import base as b
import pandas as pd
import numpy as np
import gc


class TimeSeriesMAERandomSampler(b.ActionBase):
    """ MAE Random Sampler

    Perform a MAE random sampling over the list of architectures or over a
    search space definition passed in the Configuration.

    Mandatory parameters:
    architectures or (listing_class and listing_params)
    data_loader_class and data_loader_params
    min_look_back
    max_look_back
    nn_builder_class
    num_samples

    Optional:
    output_logger_class and output_logger_params
    **any other set of params supported by the classes/functions
    """
    def __init__(self,
                 seed=1234,
                 verbose=0):
        super().__init__(seed, verbose)

    def _is_valid_config(self,
                         **config):
        if ('architectures' not in config and
                'listing_class' not in config):
            return False
        if 'listing_class' in config:
            if not issubclass(config['listing_class'],
                              samp.ArchitectureListing):
                return False
            if 'listing_params' not in config:
                return False
        if 'data_loader_class' in config:
            if not issubclass(config['data_loader_class'],
                              ut.DataLoader):
                return False
            if 'data_loader_params' not in config:
                return False
        else:
            return False
        if (('min_look_back' not in config
                or config['min_look_back'] < 1)
                and 'look_back' not in config):
            return False
        if (('max_look_back' not in config
                or config['max_look_back'] < config['min_look_back'])
                and 'look_back' not in config):
            return False
        if ('nn_builder_class' not in config or
                not issubclass(config['nn_builder_class'],
                               nn.NNBuilder)):
            return False
        if ('num_samples' not in config or
                config['num_samples'] < 1):
            return False
        return True

    def do_action(self,
                  **kwargs):
        if not self._is_valid_config(**kwargs):
            raise Exception('The configuration is not valid')
        if ('output_logger_class' in kwargs and
                'output_logger_params' in kwargs):
            self._set_output(kwargs['output_logger_class'],
                             kwargs['output_logger_params'])
        data_loader = kwargs['data_loader_class']()
        data_loader.load(**kwargs['data_loader_params'])
        dataset = data_loader.dataset
        layer_in = dataset.input_dim
        layer_out = dataset.output_dim
        architectures = None
        if 'listing_class' in kwargs:
            listing = kwargs['listing_class']()
            architectures = listing.list_architectures(
                **kwargs['listing_params'])
        else:
            architectures = kwargs['architectures']
        if architectures is None:
            raise Exception('No architectures found')
        nn_builder = kwargs['nn_builder_class']
        # do the sampling
        if ('min_look_back' in kwargs 
                and 'max_look_back' in kwargs):
            look_back_list = range(kwargs['min_look_back'],
                                   kwargs['max_look_back']+1)
        elif ('look_back' in kwargs
                and isinstance(kwargs['look_back'], list)):
            look_back_list = kwargs['look_back']
        elif ('look_back' in kwargs
                and isinstance(kwargs['look_back'], int)):
            look_back_list = [kwargs['look_back']]
        else:
            raise Exception("Please provide a valid look back configuration")

        for look_back in look_back_list:
            dataset.testing_data.look_back = look_back
            if self.verbose:
                print("Look back updated", look_back)
            for architecture in architectures:
                # Build the network
                layers = [layer_in] + architecture + [layer_out]
                model = nn_builder.build_model(layers,
                                               verbose=self.verbose,
                                               **kwargs)
                sampler = samp.MAERandomSampling(self.seed)
                metrics = sampler.fit(model=model,
                                      data=dataset.testing_data,
                                      **kwargs)
                results = {}
                results['metrics'] = metrics
                results['architecture'] = layers
                results['look_back'] = look_back
                self._output(**results)
                if self.verbose:
                    print(results)
                del results
                del model
                del sampler
                gc.collect()


class CategoricalSeqRandomSampler(b.ActionBase):
    """ Random Sampler

    Perform a random sampling over the list of architectures or over a
    search space definition passed in the Configuration.

    Mandatory parameters:
    architectures or (listing_class and listing_params)
    data_loader_class and data_loader_params
    min_look_back
    max_look_back
    nn_builder_class
    num_samples
    weights_init_func

    Optional:
    output_logger_class and output_logger_params
    **any other set of params supported by the classes/functions
    """
    def __init__(self,
                 seed=1234,
                 verbose=0):
        super().__init__(seed, verbose)

    def _is_valid_config(self,
                         **config):
        if ('architectures' not in config and
                'listing_class' not in config):
            return False
        if 'listing_class' in config:
            if not issubclass(config['listing_class'],
                              samp.ArchitectureListing):
                return False
            if 'listing_params' not in config:
                return False
        if 'data_loader_class' in config:
            if not issubclass(config['data_loader_class'],
                              ut.DataLoader):
                return False
            if 'data_loader_params' not in config:
                return False
        else:
            return False
        if ('min_look_back' not in config or
                config['min_look_back'] < 1):
            return False
        if ('max_look_back' not in config or
                config['max_look_back'] < config['min_look_back']):
            return False
        if ('nn_builder_class' not in config or
                not issubclass(config['nn_builder_class'],
                               nn.NNBuilder)):
            return False
        if ('num_samples' not in config or
                config['num_samples'] < 1):
            return False
        if 'weights_init_func' not in config:
            return False
        if 'random_samp_metric_func' not in config:
            return False
        return True

    def do_action(self,
                  **kwargs):
        if not self._is_valid_config(**kwargs):
            raise Exception('The configuration is not valid')
        if ('output_logger_class' in kwargs and
                'output_logger_params' in kwargs):
            self._set_output(kwargs['output_logger_class'],
                             kwargs['output_logger_params'])
        data_loader = kwargs['data_loader_class']()
        data_loader.load(**kwargs['data_loader_params'])
        dataset = data_loader.dataset
        layer_in = dataset.input_dim
        layer_out = dataset.output_dim
        architectures = None
        if 'listing_class' in kwargs:
            listing = kwargs['listing_class']()
            architectures = listing.list_architectures(
                **kwargs['listing_params'])
        else:
            architectures = kwargs['architectures']
        if architectures is None:
            raise Exception('No architectures found')
        nn_builder = kwargs['nn_builder_class']
        for architecture in architectures:
            # Build the network
            layers = [layer_in] + architecture + [layer_out]
            model = nn_builder.build_model(layers,
                                           verbose=self.verbose,
                                           **kwargs)
            # do the sampling
            for look_back in range(kwargs['min_look_back'],
                                   kwargs['max_look_back']+1):
                sampler = samp.RandomSampling(self.seed)
                dataset.testing_data.look_back = look_back
                samples = sampler.sample(
                    model=model,
                    init_function=kwargs['weights_init_func'],
                    data=dataset.testing_data,
                    metric_function=kwargs['random_samp_metric_func'],
                    **kwargs)
                results = {}
                results['metrics'] = samples
                results['architecture'] = layers
                results['look_back'] = look_back
                self._output(**results)
                if self.verbose:
                    print(results)
            del model
