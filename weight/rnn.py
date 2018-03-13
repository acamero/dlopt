import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import time
import math
import warnings
warnings.filterwarnings("ignore")
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
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

    def __init__(self, layers, weights):      
        self.model = self._build_model( layers )
        self.model.set_weights( weights )
        self.trainable_params = int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))

    def _build_model(self, layers):
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
        model.add(Dense(layers[-1], kernel_initializer='zeros', bias_initializer='zeros'))        
        return model

    def predict(self, df_X, look_back):        
        len_data = len(df_X)
        X = np.array( [df_X.values[i:i+look_back] 
                    for i in range(len_data - look_back)] ).reshape(-1,look_back, self.input_dim)
        return self.model.predict(X)

    def model_to_png(self, out_file, shapes=True):
        plot_model( self.model, to_file=out_file, show_shapes=shapes)




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
    

