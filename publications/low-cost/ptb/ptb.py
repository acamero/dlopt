import dlopt.util as ut
from dlopt.optimization import Dataset
from dlopt.nn import CategoricalSeqDataset
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import os


class PTBDataLoader(ut.DataLoader):
    """ Load the Penn Tree Bank dataset
    folder    folder that contains the raw data    
    """
    params = {'folder': None,
              'batch_size' : None}

    def __init__(self):
        self.tokenizer = Tokenizer()

    def _read_file(self,
                   filename):
        corpus = None
        with open(filename, 'r') as _file:
            corpus = _file.read().replace('\n', '<eos>')
        if corpus is None:
            raise Exception("Problem reading " + filename)
        return corpus

    def load(self,
             **kwargs):
        self.params.update(kwargs)
        if self.params['folder'] is None:
            raise Exception("A 'folder' must be provided")        
        if self.params['batch_size'] is None:
            raise Exception("A 'batch_size' must be provided")
        training_corpus = self._read_file(os.path.join(
            self.params['folder'], 'ptb.train.txt'))
        testing_corpus = self._read_file(os.path.join(
            self.params['folder'], 'ptb.test.txt'))
        valid_corpus = self._read_file(os.path.join(
            self.params['folder'], 'ptb.valid.txt'))
        self.tokenizer.fit_on_texts([training_corpus,
                                     testing_corpus,
                                     valid_corpus])
        vocab_size = len(self.tokenizer.word_index) + 1
        training_seq = self.tokenizer.texts_to_sequences([training_corpus])[0]
        training_data = CategoricalSeqDataset(training_seq, vocab_size,
            batch_size=self.params['batch_size'])
        validation_seq = self.tokenizer.texts_to_sequences([valid_corpus])[0]
        validation_data = CategoricalSeqDataset(validation_seq, vocab_size,
            batch_size=self.params['batch_size'])
        testing_seq = self.tokenizer.texts_to_sequences([testing_corpus])[0]
        testing_data = CategoricalSeqDataset(testing_seq, vocab_size,
            batch_size=self.params['batch_size'])
        self.dataset = Dataset(training_data,
                               validation_data,
                               testing_data,
                               input_dim=vocab_size,
                               output_dim=vocab_size)
