from .. import sampling as samp
from .. import util as ut
from .. import nn as nn
from . import base as b


class MAERandomSampler(b.ActionBase):
    """ MAE Random Sampler

    Perform a MAE random sampling over the list of architectures or over a
    search space definition passed in the Configuration.

    architectures or architecture_listing
    samples
    min_look_back
    max_look_back
    """
    def __init__(self,
                 data,
                 config,
                 seed=1234):
        super().__init__(data, config, seed)
        if not self._is_valid_config(config):
            raise Exception('The configuration is not valid')
        self.layer_in = len(config.x_features)
        self.layer_out = len(config.y_features)

    def _is_valid_config(self,
                         config):
        if (not config.has('architectures') and
                not config.has('listing_class')):
            return False
        if config.has('listing_class'):
            if not issubclass(config.listing_class,
                              samp.ArchitectureListing):
                print(type(config.listing_class))
                return False
            if not config.has('listing_restrictions'):
                return False
        if (not config.has('samples') or
                config.samples < 1):
            return False
        if (not config.has('min_look_back') or
                config.min_look_back < 1):
            return False
        if (not config.has('max_look_back') or
                config.max_look_back < config.min_look_back):
            return False
        if (not config.has('nn_builder_class') or
                not issubclass(config.nn_builder_class,
                               nn.NNBuilder)):
            return False
        return True

    def do_action(self,
                  *args):
        architectures = None
        if self.config.has('listing_class'):
            listing = self.config.listing_class()
            architectures = listing.list_architectures(
                self.config.listing_restrictions)
        else:
            architectures = self.config.architectures
        if architectures is None:
            raise Exception('No architectures passed')
        for architecture in architectures:
            # Build the network
            nn_builder = self.config.nn_builder_class()
            layers = [self.layer_in] + architecture + [self.layer_out]
            model = nn_builder.build_model(layers, verbose=1)
            # do the sampling
            # store the results
