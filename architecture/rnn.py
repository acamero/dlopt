import numpy as np
import pandas as pd
import util as ut
import datetime
import time
import math
import warnings
warnings.filterwarnings("ignore")
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

# Configuramos la sesión de TensorFlow para que el sistema escoja el número óptimo de hebras
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)))


############################################################################################################
class TrainRNN(object):

    def __init__(self, rnn_arch=[2,16,32,64,1], drop_out=0.3,
            model_file="lstm_model.hdf5", new=True, min_delta = 0.0001, patience = 50):
        """Train a RNN using the input data
        rnn_arch: list containing the number of neurons per layer (the number of hidden layers
            is defined implicitly)
        drop_out: drop out used in the RNN
        model_file: name of the file where the model is saved
        new: if True, a new model is created, otherwise an existent model is used        
        """        
        if new:
            self.model = self._build_lstm_model(rnn_arch, drop_out)
            adam = Adam(lr = 5e-5)
            self.model.compile(loss='mean_squared_error', optimizer=adam)
        else:
            self.model = load_model(model_file)
        self.checkpointer = ModelCheckpoint(filepath=model_file, verbose=0, save_best_only=True)
        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=0, mode='auto')
        self.trainable_count = int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
        
    def _process_data(self, df, x_features, y_features, look_back):
        len_data = df.shape[0]
        x = np.array( [df[x_features].values[i:i+look_back] 
                for i in range(len_data - look_back)] ).reshape(-1,look_back, len(x_features))
        y = df[y_features].values[look_back:,:]
        return x,y

    def _build_lstm_model(self, layers, drop_out):
        model = Sequential()
        for i in range(len(layers) - 2):
            model.add(LSTM(
                    input_dim=layers[i],
                    output_dim=layers[i+1], 
                    #stateful=True,
                    return_sequences= True if i < len(layers) - 3 else False ))
            model.add(Dropout(drop_out))
    
        model.add(Dense(layers[-1]))
        model.summary()
        return model

    def train(self, dfs, x_features, y_features, test_split=0.2,
            epoch=100, val_split=0.3, batch_size=512, look_back=50):
        """Train a RNN using the input data
        dfs: a dictionary of data frames containing the data
        x_features: column names of the data frames used as input
        y_features: column names of the data frames used as output
        test_split: ratio of the data used to test
        epoch: number of times that an epoch is going to be run for training
        val_split: ratio of the training data used to validate
        batch_size: size of the training batch
        look_back: number of times back in time used to train the RNN
        """
        mses = {}
        maes = {}
        train_times = {}
        for _dir in dfs:
            print("Training data: " + str(_dir))
            test = int(dfs[_dir].shape[0] * test_split)
            mse_loss_lstm, mae_loss_lstm, train_time = self._train_on_data(dfs[_dir][test:], 
                    dfs[_dir][:test], x_features, y_features,
                    epoch, val_split, batch_size, look_back)
            mses[_dir] = mse_loss_lstm
            maes[_dir] = mae_loss_lstm
            train_times[_dir] = train_time
        mean_mse = np.mean( list(mses.values() ) )
        mean_mae = np.mean( list(maes.values() ) )
        mean_train_time = np.mean( list(train_times.values() ) )
        return {'mean_mse':mean_mse, 'mean_mae':mean_mae,
            'mse_dtl':mses, 'mae_dtl':maes, 
            'trainable_vars':self.trainable_count,
            'train_time_dtl':train_times, 'mean_train_time':mean_train_time}

    def _train_on_data(self, train_set, test_set, x_features, y_features,
            epoch, val_split, batch_size, look_back):
        X_train, y_train = self._process_data(train_set, 
                x_features, y_features, look_back)
        X_test, y_test = self._process_data(test_set, 
                x_features, y_features, look_back)
        #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)        
        start = time.time()
        print('Start training. Time: ', start)        
        hist_lstm = self.model.fit(
                    X_train,
                    y_train, 
                    batch_size=batch_size,
                    verbose=2,
                    nb_epoch=epoch,
                    validation_split=val_split,
                    callbacks=[self.early_stopping,  self.checkpointer],
                    shuffle=False)
        train_time = time.time() - start
        print('Finish trainning. Time: ', train_time)
        # Test the RNN
        # model = load_model(self.model_file)
        pred_lstm = self.model.predict(X_test)
        mse_loss_lstm = ut.mse_loss(pred_lstm, y_test)
        mae_loss_lstm = ut.mae_loss(pred_lstm, y_test)
        print('Mean square error on test set: ', mse_loss_lstm)
        print('Mean absolute error on the test set: ', mae_loss_lstm)
        return [mse_loss_lstm, mae_loss_lstm, train_time]






############################################################################################################



