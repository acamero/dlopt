import tensorflow as tf
from keras.models import load_model
import util as ut
import argparse
import numpy as np
import pandas as pd
import os.path
import rnn as nn

#########################################################################################################################
def predict_blind(model_file, data_dict, x_features, y_features, look_back):
    print("Predict blind")
    model = load_model(model_file)
    # Get the last slinding window (train and test sets are correlative)
    X_test = data_dict['train'][x_features].values[-decoded['look_back']:] 
    X_test = X_test.reshape(-1,look_back, len(x_features))
    y_test = data_dict['test'][y_features].values[:,:]
    append_features = list(filter(lambda x: x not in y_features, x_features))
    pred_lstm = np.empty( (0,y_test.shape[1]), int)    
    # Add the predicted values
    for i in range(y_test.shape[0]):        
        pred_lstm = np.append( pred_lstm, model.predict(X_test), axis=0)
        X_test = X_test[0][1:]
        x_append = np.concatenate( (pred_lstm[i], data_dict['test'][append_features].values[i]) )
        X_test = np.append( X_test, x_append.reshape((1, len(x_features)) ), axis=0)
        X_test = X_test.reshape(-1,look_back, len(x_features))
    mse_loss_lstm = ut.mse_loss(pred_lstm, y_test)
    mae_loss_lstm = ut.mae_loss(pred_lstm, y_test)
    print('Mean square error on test set: ', mse_loss_lstm)
    print('Mean absolute error on the test set: ', mae_loss_lstm)
    pred_df = pd.DataFrame( pred_lstm )
    pred_df.columns = config.y_features    
    pred_df = pred_df.set_index(data_dict['test'].index.values)
    return pred_df, mse_loss_lstm, mae_loss_lstm

def predict(model_file, data_dict, x_features, y_features, look_back):
    print("Predict")
    model = load_model(model_file)
    df_X = data_dict['test'][x_features]
    len_data = len(df_X)
    X = np.array( [df_X.values[i:i+look_back] 
                    for i in range(len_data - look_back)] ).reshape(-1,look_back, len(x_features))
    Y = model.predict(X)
    y_gt = data_dict['test'][config.y_features].values[look_back:,:]
    mse_loss_lstm = ut.mse_loss(Y, y_gt)
    mae_loss_lstm = ut.mae_loss(Y, y_gt)
    print('Mean square error on test set: ', mse_loss_lstm)
    print('Mean absolute error on the test set: ', mae_loss_lstm)
    pred_df = pd.DataFrame( Y )
    pred_df.columns = config.y_features    
    pred_df = pred_df.set_index(data_dict['test'].index.values[look_back:])
    return pred_df, mse_loss_lstm, mae_loss_lstm


def decode_solution(encoded_solution, layer_in, layer_out):
    # 'drop-out' in [0,99]
    # 'look back' in [1, config.max_look_back]
    # 'rnn_arch' list (layer_in, layer_1, ..., layer_n, layer_out)
    # layer_in = len(config.x_features)
    # (layer_1, ...,layer_n), layer_i in [1, config.max_neurons], n in [1, config.max_layers]
    # layer_out = len(config.y_features)        
    decoded = {}
    decoded['drop_out'] = float(encoded_solution[0]/100)
    decoded['look_back'] = encoded_solution[1]
    decoded['rnn_arch'] = [layer_in] + encoded_solution[2:] + [layer_out]
    return decoded


#########################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--config',
          type=str,
          default='config.json',
          help='Experiment configuration file path (json format).'
    )
    parser.add_argument(
          '--blind',
           action='store_true',
           help='Predict based in the output of the network'
    )        
    FLAGS, unparsed = parser.parse_known_args()
    config = ut.Config()
    config.load_from_file(FLAGS.config)
    reader =config.data_reader_class()
    data_dict = reader.load_data( config.data_folder )
    layer_in = len(config.x_features)
    layer_out = len(config.y_features)
    decoded = decode_solution( config.solution, layer_in, layer_out )
    model_name = '-'.join(map(str, decoded['rnn_arch'])) + '.'
    model_name = model_name + str(decoded['look_back']) + '.' 
    model_name = model_name + str(decoded['drop_out']) 
    model_file = config.models_folder + model_name + '.hdf5'
    # verify if the model already exists
    if not os.path.isfile(model_file):
        trainer = nn.BPTrainRNN(rnn_arch=decoded['rnn_arch'], 
                drop_out=decoded['drop_out'], model_file=model_file, 
                new=True, min_delta = config.min_delta, patience = config.patience,
                dense_activation = config.dense_activation)
        train_metrics = trainer.train(data_dict, 
                x_features=config.x_features, 
                y_features=config.y_features,                     
                epoch=config.epoch, 
                val_split=config.val_split, 
                batch_size=config.batch_size, 
                look_back=decoded['look_back'])
    # predict 
    if FLAGS.blind:
        pred_df, mse_loss_lstm, mae_loss_lstm = predict_blind( model_file, data_dict, 
                config.x_features, config.y_features, decoded['look_back'])
    else:
        pred_df, mse_loss_lstm, mae_loss_lstm = predict( model_file, data_dict, 
                config.x_features, config.y_features, decoded['look_back'])
    pred_df.to_csv(config.results_folder + model_name + '.csv' )
    mse = ut.mse_loss(pred_df, data_dict['test'])
    mse.loc['Mean'] = np.mean(mse)
    mse.loc['Max'] = np.max(mse)
    mse.loc['Min'] = np.min(mse)
    mse.loc['Sdev'] = np.std(mse)
    mae = ut.mae_loss(pred_df, data_dict['test'])
    mae.loc['Mean'] = np.mean(mae)
    mae.loc['Max'] = np.max(mae)
    mae.loc['Min'] = np.min(mae)
    mae.loc['Sdev'] = np.std(mae)

    mse.to_csv(config.results_folder + model_name + '-mse.csv')
    mae.to_csv(config.results_folder + model_name + '-mae.csv')

