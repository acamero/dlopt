import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import datetime
import time
import math
from . import util as ut
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model, Sequence, to_categorical
from keras import backend as K


def model_from_file(model_filename):
    return load_model(model_filename)


class NNBuilder(ABC):
    """ Artificial Neural Network base builder
    """
    model = None

    @abstractmethod
    def build_model(self,
                    layers,
                    **kwargs):
        raise NotImplemented()

    def model_to_png(self,
                     out_file,
                     shapes=True):
        plot_model(self.model,
                   to_file=out_file,
                   show_shapes=shapes)


class RNNBuilder(NNBuilder):
    """ Recurrent neural network builder
    """
    def build_model(self,
                    layers,
                    cell=LSTM,
                    weights=None,
                    dense_activation='tanh',
                    embedding=None,
                    verbose=0,
                    **kwargs):
        """ Architecture from layers array:
        lstm layers <- len(layers) - 2
        input dim <- layers[0]
        output dim <- layers[-1]
        """
        if embedding and len(layers) <= 3:
            raise Exception("Provide more than 3 layers when using embedding")
        self.model = Sequential()
        for i in range(len(layers) - 2):
            if (embedding
               and i == 0
               and len(layers) > 3):
                self.model.add(Embedding(input_dim=layers[i],
                                         output_dim=layers[i+1],
                                         embeddings_initializer='zeros'))
            else:
                self.model.add(cell(
                    # Keras API 2
                    input_shape=(None, layers[i]),
                    units=layers[i+1],
                    # Keras API 1
                    # input_dim=layers[i],
                    # output_dim=layers[i+1],
                    kernel_initializer='zeros',
                    recurrent_initializer='zeros',
                    bias_initializer='zeros',
                    # Uncomment to use last batch state to init
                    # next training step.
                    # Specify shuffle=False when calling fit()
                    # batch_size=batch_size, stateful=True,
                    # return_sequences=True if i < (len(layers) - 3)
                    #                       else False))
                    return_sequences=True if i < (len(layers) - 3) else False))
        self.model.add(Dense(layers[-1],
                       activation=dense_activation,
                       kernel_initializer='zeros',
                       bias_initializer='zeros'))
        if weights:
            self.model.set_weights(weights)
        self.trainable_params = int(np.sum(
              [K.count_params(p) for p in set(self.model.trainable_weights)]))
        if verbose > 1:
            self.model.summary()
        return self.model


class TrainNN(ABC):
    """ Training class
    """
    model = None

    def __init__(self,
                 verbose=0,
                 **kwargs):
        self.verbose = verbose

    @abstractmethod
    def train(self,
              train_dataset,
              validation_dataset=None,
              **kwargs):
        raise NotImplemented()

    @abstractmethod
    def evaluate(self,
                 train_dataset,
                 **kwargs):
        raise NotImplemented()

    def load_from_file(self,
                       model_filename):
        if isinstance(model_filename, str):
            self.model = load_model(model_filename)
        else:
            raise TypeError()

    def load_from_model(self,
                        model):
        self.model = model


class TrainGradientBased(TrainNN):
    """ Train an artificial neural network
    """
    def __init__(self,
                 model_filename="trained-model.hdf5",
                 optimizer=Adam(lr=5e-5),
                 monitor='val_loss',
                 min_delta=1e-5,
                 patience=50,
                 metrics=['mae', 'mse', 'msle', 'mape'],
                 verbose=0,
                 **kwargs):
        super().__init__(verbose=verbose,
                         **kwargs)
        self.checkpointer = ModelCheckpoint(filepath=model_filename,
                                            verbose=verbose,
                                            save_best_only=True)
        self.optimizer = optimizer
        self.early_stopping = EarlyStopping(monitor=monitor,
                                            min_delta=min_delta,
                                            patience=patience,
                                            verbose=verbose,
                                            mode='auto')
        self.metrics = metrics

    def add_dropout(self,
                    dropout):
        x = Sequential()
        for layer in self.model.layers[:-1]:
            x.add(layer)
            x.add(Dropout(dropout))
        x.add(self.model.layers[-1])
        self.model = x
        if self.verbose > 1:
            self.model.summary()

    def init_weights(self,
                     init_function,
                     **kwargs):
        weights = list()
        for w in self.model.weights:
            weights.append(init_function(w.shape, **kwargs))
        self.model.set_weights(weights)

    def train(self,
              train_dataset,
              validation_dataset=None,
              validation_steps=None,
              epochs=100,
              steps_per_epoch=None,
              loss='mean_squared_error',
              **kwargs):
        self.model.compile(loss=loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)
        self.trainable_count = int(np.sum(
            [K.count_params(p) for p in set(self.model.trainable_weights)]))
        start = time.time()
        if self.verbose:
            print('Start training (', start, ')')
        verb = 0
        if self.verbose > 1:
            verb = 1
        elif self.verbose == 1:
            verb = 2
        self.model.fit_generator(train_dataset,
                                 validation_data=validation_dataset,
                                 validation_steps=validation_steps,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 callbacks=[self.early_stopping,
                                            self.checkpointer],
                                 verbose=verb)
        train_time = time.time() - start
        if self.verbose:
            print('Finish trainning. Total time: ', train_time)
        return {'trainable_vars': self.trainable_count,
                'training_time': train_time}

    def evaluate(self,
                 train_dataset,
                 **kwargs):
        values = self.model.evaluate_generator(train_dataset)
        metrics_dict = dict(zip(self.metrics, values[1:]))
        prediction = self.model.predict_generator(train_dataset)
        return metrics_dict, prediction


class TimeSeriesDataset(Sequence):
    """ Implements the Sequence iterator for a time series
    """
    def __init__(self,
                 df,
                 x_features,
                 y_features,
                 look_back=1,
                 batch_size=5):
        self.df = df
        self.x_features = x_features
        self.y_features = y_features
        self.look_back = look_back
        self.batch_size = batch_size

    def __getitem__(self,
                    index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        begin = index*self.batch_size
        if (index + 1) == self.__len__():
            # In the last batch we add all the remaining data
            end = self.df.shape[0] - self.look_back
        else:
            end = (index+1)*self.batch_size
        x = np.array([self.df[self.x_features].values[i:i+self.look_back]
                      for i in range(begin, end)])
        y = self.df[self.y_features].values[
            (begin + self.look_back):(end + self.look_back), :]
        return x, y

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        # We round to the floor, therefore we "skip" some data.
        # As a work around, the remainder is added to the last batch
        return int(np.floor(
            (self.df.shape[0] - self.look_back) / self.batch_size))

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass


class CategoricalSeqDataset(Sequence):
    """ Implements the Sequence iterator for a sequence of tokens
    """
    def __init__(self,
                 tokens_sequence,
                 num_classes,
                 look_back=1,
                 batch_size=5):
        self.tokens_sequence = tokens_sequence
        self.num_classes = num_classes
        self.look_back = look_back
        self.batch_size = batch_size

    def __getitem__(self,
                    index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        begin = index*self.batch_size
        if (index + 1) == self.__len__():
            # In the last batch we add all the remaining data
            end = len(self.tokens_sequence) - self.look_back
        else:
            end = (index+1)*self.batch_size
        x = np.array([self.tokens_sequence[i:i+self.look_back]
                      for i in range(begin, end)])
        y = np.array(self.tokens_sequence[
            (begin + self.look_back):(end + self.look_back)])
        y = y[:, np.newaxis]
        y = to_categorical(y, self.num_classes)
        return x, y

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        # We round to the floor, therefore we "skip" some data.
        # As a work around, the remainder is added to the last batch
        return int(np.floor(
            (len(self.tokens_sequence) - self.look_back) / self.batch_size))

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass
