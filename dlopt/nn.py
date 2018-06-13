import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import datetime
import time
import math
from . import util as ut
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K


def predict_on_predictions(model,
                           train_df,
                           test_df,
                           x_features,
                           y_features,
                           look_back):
    """ Predict the test interval using the training dataset and the
    predictions already made by the network"""
    X_test = train_df[x_features].values[-look_back:]
    X_test = X_test.reshape(-1, look_back, len(x_features))
    append_features = list(filter(lambda x: x not in y_features, x_features))
    pred = np.empty((0, len(y_features)), int)
    for i in range(test_df.shape[0]):
        pred = np.append(pred,
                         model.predict(X_test),
                         axis=0)
        X_test = X_test[0][1:]
        x_append = np.concatenate((pred[i],
                                   test_df[append_features].values[i]))
        X_test = np.append(X_test,
                           x_append.reshape((1, len(x_features))),
                           axis=0)
        X_test = X_test.reshape(-1, look_back, len(x_features))
    return pred


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
                    verbose=0,
                    **kwargs):
        # self.hidden_layers=len(layers) - 2
        # self.layers=layers
        # self.input_dim=layers[0]
        # self.output_dim=layers[-1]
        self.model = Sequential()
        for i in range(len(layers) - 2):
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
                # Uncomment to use last batch state to init next training step.
                # Specify shuffle=False when calling fit()
                # batch_size=batch_size, stateful=True,
                return_sequences=True if i < (len(layers) - 3) else False))
        self.model.add(Dense(layers[-1],
                       activation=dense_activation,
                       kernel_initializer='zeros',
                       bias_initializer='zeros'))
        if weights:
            self.model.set_weights(weights)
        self.trainable_params = int(np.sum(
              [K.count_params(p) for p in set(self.model.trainable_weights)]))
        if verbose:
            self.model.summary()
        return self.model


class TrainNN(object):
    """ Train an artificial neural network
    """
    model = None

    def __init__(self,
                 file_name='model.hdf5',
                 optimizer=Adam(lr=5e-5),
                 monitor='val_loss',
                 min_delta=0.0001,
                 patience=50,
                 verbose=0):
        self.checkpointer = ModelCheckpoint(filepath=file_name,
                                            verbose=verbose,
                                            save_best_only=True)
        self.optimizer = optimizer
        self.early_stopping = EarlyStopping(monitor=monitor,
                                            min_delta=min_delta,
                                            patience=patience,
                                            verbose=verbose,
                                            mode='auto')
        self.verbose = verbose

    def load_from_file(self,
                       file_name):
        if isinstance(file_name, str):
            self.model = load_model(file_name)
        else:
            raise TypeError()

    def load_from_model(self,
                        model):
        self.model = model

    def add_drop_out(self,
                     drop_out):
        x = Sequential()
        for layer in self.model.layers[:-1]:
            x.add(layer)
            x.add(Dropout(drop_out))
        x.add(self.model.layers[-1])
        self.model = x

    def train(self,
              x_df,
              y_df,
              epochs=100,
              validation_split=0.3,
              batch_size=20,
              loss='mean_squared_error',
              shuffle=False):
        self.model.compile(loss=loss,
                           optimizer=self.optimizer)
        self.trainable_count = int(np.sum(
            [K.count_params(p) for p in set(self.model.trainable_weights)]))
        start = time.time()
        if self.verbose:
            print('Start training (', start, ')')
        history = self.model.fit(x_df,
                                 y_df,
                                 batch_size=batch_size,
                                 verbose=self.verbose,
                                 nb_epoch=epochs,
                                 validation_split=validation_split,
                                 callbacks=[self.early_stopping,
                                            self.checkpointer],
                                 shuffle=shuffle)
        train_time = time.time() - start
        if self.verbose:
            print('Finish trainning. Total time: ', train_time)
        return {'trainable_vars': self.trainable_count,
                'training_time': train_time}
