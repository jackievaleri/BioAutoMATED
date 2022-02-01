import functools
import itertools
import math
import os
from os.path import splitext
import pickle
import re
import random
import sys

import numpy as np
import pandas as pd
import seaborn as sns
#from datacleaner_updated import *


class One_Hot_Encoder:
    """
    The One_Hot_Encoder class provides functions to encode a string over a
    given alphabet into an integer matrix of shape (len(string), len(alphabet))
    where each row represents a position in the string and each column
    represents a character from the alphabet. Each row has exactly one 1 at the
    matching alphabet character and consists of 0s otherwise.
    """

    def __init__(self, alphabet):
        """ Initialize the object with an alphabet.
        
        Parameters
        ----------
        alphabet : str
            The alphabet that will be used for encoding/decoding (e.g. "ACGT").
        """
        self.alphabet = alphabet
        self.table = {symbol: i for i, symbol in enumerate(alphabet)}
        self.table_rev = {v: k for k, v in self.table.items()}
    
    def encode(self, sequence):
        """ Encode a sequence into a one-hot integer matrix.
        
        The sequence should only contain characters from the alphabet provided to __init__.
        Parameters
        ----------
        sequence : str
            The sequence that should be encoded.
        Returns
        -------
        one_hot: numpy.ndarray
            A numpy array with shape (len(sequence), len(alphabet)).
        """
        one_hot = np.zeros((len(sequence), len(self.table)), np.uint8)
        one_hot[np.arange(len(sequence)), np.fromiter(map(self.table.__getitem__, sequence), np.uint32)] = 1
        return one_hot

    def decode(self, one_hot):
        """ Decode a one-hot integer matrix into the original sequence.
        Parameters
        ----------
        one_hot : numpy.ndarray
            A one-hot matrix (e.g. as created by the encode function).
        Returns
        -------
        sequence: str
            The sequence that is represented by the one-hot matrix.
        """
        return ''.join(map(self.table_rev.__getitem__, np.argmax(one_hot, axis=1)))

# function to one-hot encode a cleaned sequence 
def ohe_sequence(processed_seq, sequence_type):
    """ One Hot Encodes a processed sequence as per its alphabet. Input must already be clean (using data-cleaner)"""
    # pre-defined dictionary that was used during data-cleaning, includes the unique pad character
    alph = Alphabet(sequence_type)
    alph_letters = sorted(list(alph.extended_letters()))
    ohe = One_Hot_Encoder(alph_letters)
    ohe_seq = np.transpose(ohe.encode(processed_seq).astype('float32')) # shape = (num_letters, len_seq)
    return ohe_seq

# function to take df of cleaned sequences and output matrices of the correct shape for 2d cnn
def prepare_2DCNN_input(df_processed, sequence_type):
    """ converts input sequences into 2D images suitable for 2D CNN training using deepswarm."""
    input_seqs = df_processed['sequence'].values.tolist() # do not change the name!
    scrambled_seqs = df_processed['scrambled_sequence'].values.tolist() # do not change the name!
    
    ohe_input_seqs = np.array([ohe_sequence(s, sequence_type) for s in input_seqs]) # shape = (num_seqs, num_letters, len_seq)
    ohe_scrambled_seqs = np.array([ohe_sequence(s, sequence_type) for s in scrambled_seqs]) # shape = (num_seqs, num_letters, len_seq)
    
    shapes = ohe_input_seqs.shape
    ohe_2d_input = ohe_input_seqs.reshape(shapes[0], shapes[2], shapes[1], 1) # shape = (num_seqs, len_seq, num_letters, 1) for 2D CNN
    ohe_2d_scrambled = ohe_scrambled_seqs.reshape(shapes[0], shapes[2], shapes[1], 1) # shape = (num_seqs, len_seq, num_letters, 1) for 2D CNN
    
    return ohe_2d_input, ohe_2d_scrambled

def process_oneinput(clean_seq, sequence_type):
    """ This function takes a clean sequence and returns a 2D CNN numpy array."""
    seq_ohe = ohe_sequence(clean_seq, sequence_type) # (num_letters, len_seq)
    s1 = np.swapaxes(seq_ohe, 0, 1) # (len_seq, num_letters)
    s2 = np.expand_dims(s1, axis = 0) # (1, len_seq, num_letters); 1 because num_seqs == 1
    seq_2dcnn = np.expand_dims(s2, axis = -1) # (1, len_seq, num_letters, 1) expands to 2D CNN input shape for deepswarm and autokeras
    return seq_2dcnn

def predict_oneinput(clean_seq, sequence_type, keras_model):
    """ Takes a clean sequence and predicts its corresponding class given a trained keras model."""
    seq_2dcnn = process_oneinput(clean_seq, sequence_type)
    proba_ = keras_model.predict(seq_2dcnn)
    pred_cls = np.argmax(proba_, axis = None, out = None)
    return pred_cls