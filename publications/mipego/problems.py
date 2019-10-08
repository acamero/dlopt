from sklearn.preprocessing import MinMaxScaler, StandardScaler

problems = {}
problems['test'] = {}
problems['test']['data_loader_params'] = {
    'freq': 1,
    'start': 0,
    'stop': 100,
    'step': 0.1,
    'x_features': ['sin'],
    'y_features': ['sin'],
    'training_ratio' : 0.8,
    'validation_ratio' : 0.2,
    'batch_size': 5}
problems['test']['etc_params'] = {
    'num_samples': 30,
    'truncated_lower': 0.0,
    'truncated_upper': 2.0,
    'threshold': 0.01,
    'model_filename': 'rnn-arch-opt-best_test.hdf5',
    'log_filename': 'rnn-arch-opt-best_test.log',
    'dropout': 0.5,
    'epochs': 5,
    'dense_activation': 'tanh'}
problems['test']['opt_params'] = {
    'max_hl': 3, #max n of hidden layers
    'min_nn': 1,
    'max_nn': 100, #max n of nn per layer
    'min_lb': 2,
    'max_lb': 30, #max look back
    'max_eval': 5,
    'max_iter': 100,
    'n_init_samples': 2,
    'data_loader_class': 'loaders.SinDataLoader'}

problems['sin'] = {}
problems['sin']['data_loader_params'] = {
    'freq': 1,
    'start': 0,
    'stop': 100,
    'step': 0.1,
    'x_features': ['sin'],
    'y_features': ['sin'],
    'training_ratio' : 0.8,
    'validation_ratio' : 0.2,
    'batch_size': 5}
problems['sin']['etc_params'] = {
    'num_samples': 100,
    'truncated_lower': 0.0,
    'truncated_upper': 2.0,
    'threshold': 0.01,
    'model_filename': 'rnn-arch-opt-best_sin.hdf5',
    'log_filename': 'rnn-arch-opt-best_sin.log',
    'dropout': 0.5,
    'epochs': 100,
    'dense_activation': 'tanh'}
problems['sin']['opt_params'] = {
    'max_hl': 3, #max n of hidden layers
    'min_nn': 1,
    'max_nn': 100, #max n of nn per layer
    'min_lb': 2,
    'max_lb': 30, #max look back
    'max_eval': 100,
    'max_iter': 100,
    'n_init_samples': 10,
    'data_loader_class': 'loaders.SinDataLoader'}

problems['waste'] = {}
problems['waste']['data_loader_params'] = {
    "filename": "../../data/waste/rubbish-2013.csv",
    "batch_size" : 5,
    "training_ratio": 0.8,
    "validation_ratio": 0.2,
    "x_features": ["C-A100", "C-A107", "C-A108", "C-A109", "C-A11", "C-A110", "C-A111", "C-A112", "C-A113", "C-A115", 
               "C-A117", "C-A119", "C-A12", "C-A120", "C-A121", "C-A122", "C-A123", "C-A124", "C-A125", "C-A126", 
               "C-A127", "C-A128", "C-A129", "C-A13", "C-A130", "C-A132", "C-A133", "C-A134", "C-A135", "C-A136", 
               "C-A137", "C-A139", "C-A14", "C-A140", "C-A141", "C-A142", "C-A144", "C-A146", "C-A147", "C-A148", 
               "C-A15", "C-A151", "C-A155", "C-A156", "C-A16", "C-A160", "C-A163", "C-A164", "C-A166", "C-A167", 
               "C-A168", "C-A169", "C-A17", "C-A170", "C-A171", "C-A172", "C-A173", "C-A174", "C-A175", "C-A176", 
               "C-A177", "C-A178", "C-A179", "C-A18", "C-A181", "C-A183", "C-A184", "C-A185", "C-A187", "C-A189", 
               "C-A191", "C-A192", "C-A193", "C-A194", "C-A195", "C-A196", "C-A197", "C-A198", "C-A199", "C-A2", 
               "C-A20", "C-A204", "C-A206", "C-A207", "C-A208", "C-A21", "C-A210", "C-A211", "C-A212", "C-A213", 
               "C-A214", "C-A215", "C-A216", "C-A218", "C-A22", "C-A220", "C-A221", "C-A223", "C-A225", "C-A227", 
               "C-A228", "C-A232", "C-A235", "C-A236", "C-A237", "C-A238", "C-A239", "C-A241", "C-A242", "C-A244", 
               "C-A248", "C-A249", "C-A25", "C-A250", "C-A251", "C-A252", "C-A254", "C-A255", "C-A256", "C-A257", 
               "C-A258", "C-A259", "C-A26", "C-A260", "C-A261", "C-A266", "C-A27", "C-A270", "C-A271", "C-A272", 
               "C-A273", "C-A274", "C-A277", "C-A279", "C-A28", "C-A280", "C-A282", "C-A283", "C-A284", "C-A286", 
               "C-A287", "C-A288", "C-A289", "C-A290", "C-A294", "C-A299", "C-A3", "C-A300", "C-A302", "C-A303", 
               "C-A304", "C-A305", "C-A308", "C-A309", "C-A31", "C-A310", "C-A312", "C-A313", "C-A316", "C-A317", 
               "C-A318", "C-A319", "C-A32", "C-A321", "C-A322", "C-A324", "C-A328", "C-A329", "C-A35", "C-A36", 
               "C-A37", "C-A38", "C-A39", "C-A40", "C-A41", "C-A44", "C-A46", "C-A47", "C-A49", "C-A51", 
               "C-A52", "C-A54", "C-A55", "C-A56", "C-A57", "C-A59", "C-A6", "C-A61", "C-A62", "C-A63", 
               "C-A64", "C-A65", "C-A67", "C-A68", "C-A69", "C-A7", "C-A70", "C-A73", "C-A74", "C-A76", 
               "C-A77", "C-A78", "C-A79", "C-A8", "C-A80", "C-A81", "C-A83", "C-A84", "C-A85", "C-A86", 
               "C-A89", "C-A9", "C-A90", "C-A93", "C-A96", "C-A98", "C-A99"],
    "y_features": ["C-A100", "C-A107", "C-A108", "C-A109", "C-A11", "C-A110", "C-A111", "C-A112", "C-A113", "C-A115", 
               "C-A117", "C-A119", "C-A12", "C-A120", "C-A121", "C-A122", "C-A123", "C-A124", "C-A125", "C-A126", 
               "C-A127", "C-A128", "C-A129", "C-A13", "C-A130", "C-A132", "C-A133", "C-A134", "C-A135", "C-A136", 
               "C-A137", "C-A139", "C-A14", "C-A140", "C-A141", "C-A142", "C-A144", "C-A146", "C-A147", "C-A148", 
               "C-A15", "C-A151", "C-A155", "C-A156", "C-A16", "C-A160", "C-A163", "C-A164", "C-A166", "C-A167", 
               "C-A168", "C-A169", "C-A17", "C-A170", "C-A171", "C-A172", "C-A173", "C-A174", "C-A175", "C-A176", 
               "C-A177", "C-A178", "C-A179", "C-A18", "C-A181", "C-A183", "C-A184", "C-A185", "C-A187", "C-A189", 
               "C-A191", "C-A192", "C-A193", "C-A194", "C-A195", "C-A196", "C-A197", "C-A198", "C-A199", "C-A2", 
               "C-A20", "C-A204", "C-A206", "C-A207", "C-A208", "C-A21", "C-A210", "C-A211", "C-A212", "C-A213", 
               "C-A214", "C-A215", "C-A216", "C-A218", "C-A22", "C-A220", "C-A221", "C-A223", "C-A225", "C-A227", 
               "C-A228", "C-A232", "C-A235", "C-A236", "C-A237", "C-A238", "C-A239", "C-A241", "C-A242", "C-A244", 
               "C-A248", "C-A249", "C-A25", "C-A250", "C-A251", "C-A252", "C-A254", "C-A255", "C-A256", "C-A257", 
               "C-A258", "C-A259", "C-A26", "C-A260", "C-A261", "C-A266", "C-A27", "C-A270", "C-A271", "C-A272", 
               "C-A273", "C-A274", "C-A277", "C-A279", "C-A28", "C-A280", "C-A282", "C-A283", "C-A284", "C-A286", 
               "C-A287", "C-A288", "C-A289", "C-A290", "C-A294", "C-A299", "C-A3", "C-A300", "C-A302", "C-A303", 
               "C-A304", "C-A305", "C-A308", "C-A309", "C-A31", "C-A310", "C-A312", "C-A313", "C-A316", "C-A317", 
               "C-A318", "C-A319", "C-A32", "C-A321", "C-A322", "C-A324", "C-A328", "C-A329", "C-A35", "C-A36", 
               "C-A37", "C-A38", "C-A39", "C-A40", "C-A41", "C-A44", "C-A46", "C-A47", "C-A49", "C-A51", 
               "C-A52", "C-A54", "C-A55", "C-A56", "C-A57", "C-A59", "C-A6", "C-A61", "C-A62", "C-A63", 
               "C-A64", "C-A65", "C-A67", "C-A68", "C-A69", "C-A7", "C-A70", "C-A73", "C-A74", "C-A76", 
               "C-A77", "C-A78", "C-A79", "C-A8", "C-A80", "C-A81", "C-A83", "C-A84", "C-A85", "C-A86", 
               "C-A89", "C-A9", "C-A90", "C-A93", "C-A96", "C-A98", "C-A99"]}
problems['waste']['etc_params'] = {
    'num_samples': 100,
    'truncated_lower': 0.0,
    'truncated_upper': 1.0,
    'threshold': 0.01,
    'model_filename': 'rnn-arch-opt-best_waste.hdf5',
    'log_filename': 'rnn-arch-opt-best_waste.log',
    'dropout': 0.5,
    'epochs': 1000,
    'dense_activation': 'sigmoid'}
problems['waste']['opt_params'] = {
    'max_hl': 8, #max n of hidden layers
    'min_nn': 10,
    'max_nn': 300, #max n of nn per layer
    'min_lb': 2,
    'max_lb': 30, #max look back
    'max_eval': 100,
    'max_iter': 100,
    'n_init_samples': 10,
    'data_loader_class': 'loaders.RubbishDataLoader'}

problems['eunite'] = {}
problems['eunite']['data_loader_params'] = {
    'training_filename': '../../data/eunite/eunite.training.csv',
    'testing_filename': '../../data/eunite/eunite.testing.csv',
    'x_features': ["X00.30","X01.00","X01.30","X02.00","X02.30","X03.00","X03.30","X04.00",
                   "X04.30","X05.00","X05.30","X06.00","X06.30","X07.00","X07.30","X08.00",
                   "X08.30","X09.00","X09.30","X10.00","X10.30","X11.00","X11.30","X12.00",
                   "X12.30","X13.00","X13.30","X14.00","X14.30","X15.00","X15.30","X16.00",
                   "X16.30","X17.00","X17.30","X18.00","X18.30","X19.00","X19.30","X20.00",
                   "X20.30","X21.00","X21.30","X22.00","X22.30","X23.00","X23.30","X24.00",
                   "MaxLoads","Temperature","Holiday","Weekday"],
    'y_features': ["MaxLoads"],
    'validation_ratio' : 0.2,
    'batch_size': 5,
    'max_look_back': 30,
    'scaler_fn': StandardScaler}
problems['eunite']['etc_params'] = {
    'num_samples': 100,
    'truncated_lower': 0.0,
    'truncated_upper': 100.0,
    'threshold': 0.01,
    'model_filename': 'rnn-arch-opt-best_eunite.hdf5',
    'log_filename': 'rnn-arch-opt-best_eunite.log',
    'dropout': 0.5,
    'epochs': 1000,
    'dense_activation': 'linear'}
problems['eunite']['opt_params'] = {
    'max_hl': 8, #max n of hidden layers
    'min_nn': 10,
    'max_nn': 100, #max n of nn per layer
    'min_lb': 2,
    'max_lb': 30, #max look back
    'max_eval': 100,
    'max_iter': 100,
    'n_init_samples': 10,
    'data_loader_class': 'loaders.EuniteDataLoader'}

def get_problems():
  return problems
