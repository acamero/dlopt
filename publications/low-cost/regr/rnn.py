import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import time
import math
import warnings
import util as ut
warnings.filterwarnings("ignore")
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

# Configuramos la sesión de TensorFlow para que el sistema escoja el número óptimo de hebras
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)))


############################################################################################################
def decode_arch(weights_dict):
    arch = []
    arch.append( len(weights_dict[0]['kernel']) )
    for i in range(len(weights_dict)):
        if 'recurrent' in weights_dict[i].keys():
            arch.append( len(weights_dict[i]['recurrent']) )
        elif 'dense'  in weights_dict[i].keys():
            arch.append( len(weights_dict[i]['dense'][0]) )
    return arch

def get_weights_array(weights_dict):
    weights_array = []
    for i in range(len(weights_dict)):
        if 'kernel' in weights_dict[i].keys():
            weights_array.append( weights_dict[i]['kernel'] )
        if 'recurrent' in weights_dict[i].keys():
            weights_array.append( weights_dict[i]['recurrent'] )
        if 'dense' in weights_dict[i].keys():
            weights_array.append( weights_dict[i]['dense'] )
        if 'bias' in weights_dict[i].keys():
            weights_array.append( weights_dict[i]['bias'] )
    return weights_array
        


############################################################################################################
class RNNBuilder(object):

    def __init__(self, layers, weights=None, dense_activation='tanh'):      
        self.model = self._build_model( layers, dense_activation)
        if weights:
            self.model.set_weights( weights )
        self.trainable_params = int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
        self.model.summary()
        #self.model_to_png("model.png")

    def _build_model(self, layers, dense_activation):
        self.hidden_layers = len(layers) - 2
        self.layers = layers
        self.input_dim = layers[0]
        self.output_dim = layers[-1]
        model = Sequential()
        for i in range(len(layers) - 2):
            model.add(
                    LSTM(
                    #SimpleRNN(
                    input_dim=layers[i],
                    output_dim=layers[i+1],
                    kernel_initializer='zeros', 
                    recurrent_initializer='zeros',
                    bias_initializer='zeros',
                    # Uncomment to use last batch state to init next training step.
                    # Specify shuffle=False when calling fit() 
                    #batch_size=batch_size, stateful=True,
                    return_sequences= True if i < len(layers) - 3 else False )
                    )
        model.add(Dense(layers[-1], activation=dense_activation, kernel_initializer='zeros', bias_initializer='zeros'))        
        return model

    def update_weights(self, weights):
        self.model.set_weights( weights )

    def predict(self, df_X, look_back):        
        len_data = len(df_X)
        X = np.array( [df_X.values[i:i+look_back] 
                    for i in range(len_data - look_back)] ).reshape(-1,look_back, self.input_dim)
        return self.model.predict(X)

    def predict_blind(self, train_set, test_set, x_features, y_features, look_back):
        X_test = train_set[x_features].values[-look_back:] 
        X_test = X_test.reshape(-1,look_back, len(x_features))        
        append_features = list(filter(lambda x: x not in y_features, x_features))
        pred_lstm = np.empty( (0, len(y_features)) , int)    
        # Add the predicted values
        for i in range(test_set.shape[0]):        
            pred_lstm = np.append( pred_lstm, self.model.predict(X_test), axis=0)
            X_test = X_test[0][1:]
            x_append = np.concatenate( (pred_lstm[i], test_set[append_features].values[i]) )
            X_test = np.append( X_test, x_append.reshape((1, len(x_features)) ), axis=0)
            X_test = X_test.reshape(-1,look_back, len(x_features))  
        return pred_lstm

    def model_to_png(self, out_file, shapes=True):
        plot_model( self.model, to_file=out_file, show_shapes=shapes)

class BPTrainRNN(object):

    def __init__(self, rnn_arch=[2,16,32,64,1], drop_out=0.3,
            model_file="lstm_model.hdf5", new=True, min_delta = 0.0001, patience = 50, dense_activation='tanh'):
        """Train a RNN using the input data
        rnn_arch: list containing the number of neurons per layer (the number of hidden layers
            is defined implicitly)
        drop_out: drop out used in the RNN
        model_file: name of the file where the model is saved
        new: if True, a new model is created, otherwise an existent model is used        
        """        
        if new:
            self.model = self._build_lstm_model(rnn_arch, drop_out, dense_activation)
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

    def _build_lstm_model(self, layers, drop_out, dense_activation):
        model = Sequential()
        for i in range(len(layers) - 2):
            model.add(LSTM(
                    input_dim=layers[i],
                    output_dim=layers[i+1], 
                    #stateful=True,
                    return_sequences= True if i < len(layers) - 3 else False ))
            model.add(Dropout(drop_out))
    
        model.add(Dense(layers[-1], activation=dense_activation))
        model.summary()
        return model

    def train(self, dfs, x_features, y_features, epoch=100, val_split=0.3, batch_size=512, look_back=50):
        """Train a RNN using the input data
        dfs: a dictionary of data frames containing the data
        x_features: column names of the data frames used as input
        y_features: column names of the data frames used as output
        epoch: number of times that an epoch is going to be run for training
        val_split: ratio of the training data used to validate
        batch_size: size of the training batch
        look_back: number of times back in time used to train the RNN
        """
        mse_loss_lstm, mae_loss_lstm, train_time = self._train_on_data(dfs['train'], 
                    dfs['test'], x_features, y_features,
                    epoch, val_split, batch_size, look_back)
        return {'mse':mse_loss_lstm, 'mae':mae_loss_lstm,
            'trainable_vars':self.trainable_count,'train_time':train_time}

    def _train_on_data(self, train_set, test_set, x_features, y_features,
            epoch, val_split, batch_size, look_back, blind=True):
        X_train, y_train = self._process_data(train_set, 
                x_features, y_features, look_back)
        start = time.time()
        print('Start training. Time: ', start)          
        hist_lstm = self.model.fit(
                    X_train,
                    y_train, 
                    batch_size=batch_size,
                    verbose=0,
                    nb_epoch=epoch,
                    validation_split=val_split,
                    callbacks= [self.early_stopping,  self.checkpointer],
                    shuffle=False)
        train_time = time.time() - start
        print('Finish trainning. Time: ', train_time)
        # Test the RNN
        if blind:
            pred_lstm = self._predict_blind(train_set, test_set, x_features, y_features, look_back)
            y_test = test_set[y_features].values[:,:]        
        else:
            pred_lstm = self._predict_update(train_set, test_set, x_features, y_features, look_back)
            y_test = test_set[y_features].values[look_back:,:]
        mse_loss_lstm = ut.mse_loss(pred_lstm, y_test)
        mae_loss_lstm = ut.mae_loss(pred_lstm, y_test)
        print('Mean square error on test set: ', mse_loss_lstm)
        print('Mean absolute error on the test set: ', mae_loss_lstm)
        return [mse_loss_lstm, mae_loss_lstm, train_time]

    def _predict_blind(self, train_set, test_set, x_features, y_features, look_back):
        X_test = train_set[x_features].values[-look_back:] 
        X_test = X_test.reshape(-1,look_back, len(x_features))        
        append_features = list(filter(lambda x: x not in y_features, x_features))
        pred_lstm = np.empty( (0, len(y_features)) , int)    
        # Add the predicted values
        for i in range(test_set.shape[0]):        
            pred_lstm = np.append( pred_lstm, self.model.predict(X_test), axis=0)
            X_test = X_test[0][1:]
            x_append = np.concatenate( (pred_lstm[i], test_set[append_features].values[i]) )
            X_test = np.append( X_test, x_append.reshape((1, len(x_features)) ), axis=0)
            X_test = X_test.reshape(-1,look_back, len(x_features))  
        return pred_lstm

    def _predict_update(self, train_set, test_set, x_features, y_features, look_back):
        X_test, y_test = self._process_data(test_set, x_features, y_features, look_back)
        pred_lstm = self.model.predict(X_test)
        return pred_lstm



############################################################################################################

if __name__ == '__main__':
    simple = False
    look_back = 1
    layer_in = 30
    layer_out = 28
    df_X = pd.DataFrame(np.linspace(0,1,100*layer_in).reshape(100,layer_in))
    # Random net
    min_neurons = 62
    max_neurons = 62
    min_layers = 8
    max_layers = 8    
    params_neuron = 4
    ranges = [(min_neurons, max_neurons+1)] * np.random.randint(min_layers, high=max_layers+1)
    layers = [layer_in] + [np.random.randint(*p) for p in ranges] + [layer_out]
    weights = list()
    for i in range(len(layers)-2):
        # Kernel weights
        weights.append( np.random.uniform(low=-1.0, high=1.0, size=(layers[i], layers[i+1]*params_neuron) ) )
        # Recurrent weights
        weights.append( np.random.uniform(low=-1.0, high=1.0, size=(layers[i+1], layers[i+1]*params_neuron) ) )
        # Bias
        weights.append( np.random.uniform(low=-1.0, high=1.0, size=layers[i+1]*params_neuron) )
    # Output dim
    # Dense weights
    weights.append( np.random.uniform(low=-1.0, high=1.0, size=(layers[-2], layers[-1] ) ) )
    # Bias
    weights.append( np.random.uniform(low=-1.0, high=1.0, size=layers[-1]) )
    
    # Simple
    weights_s = {}
    weights_s[0] = {}
    weights_s[0]['kernel'] = np.array([ 
                               [.1,.1,.1,.1] 
                           ], dtype='f')
    weights_s[0]['recurrent'] = np.array([
                                  [.2,.2,.2,.2]
                              ], dtype='f')
    weights_s[0]['bias'] = np.array([.0,.0,.0,.0], dtype='f')
    weights_s[1] = {}
    weights_s[1]['dense'] = np.array([ [1.] ], dtype='f')
    weights_s[1]['bias'] = np.array([.0], dtype='f')
    # Build a RNN based on the weights
    if simple:
        layers_s = decode_arch(weights_s)    
        weights = get_weights_array(weights_s)    
    print(layers)
    #print(weights)
    rnn_handler = RNNBuilder(layers, weights)
    # Get an image of the model
    #rnn_handler.model_to_png('models/model.png')
    # Predict the next value of the series
    y = rnn_handler.predict( df_X, look_back = look_back)
    print(y)
    

