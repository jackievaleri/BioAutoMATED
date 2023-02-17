#!/usr/bin/env python
# coding: utf-8

############## PART 1: IMPORT STATEMENTS ##############

# import system libraries
import abc
import os
from os.path import splitext
import sys
import shutil
import math
import pickle
import itertools
import numpy as np
import random
import pandas as pd
from time import time
from Bio.Seq import Seq
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import Image
import pickle
import warnings
warnings.filterwarnings("ignore")

# import sklearn libs
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

# import tensorflow libs
import tensorflow as tf 

# import Keras
from keras import optimizers, applications, regularizers
from keras.models import Sequential, load_model
from keras.models import model_from_json, load_model
from keras.layers import Activation, Conv1D, Conv2D, Reshape, BatchNormalization, Dropout, Flatten, Dense, merge, Input, Lambda, InputLayer, Convolution2D, MaxPooling1D, MaxPooling2D, ZeroPadding2D, Bidirectional, concatenate
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.layers import BatchNormalization
import autokeras

############## PART 2: GLOBAL VARIABLES AND GENERIC FUNCTIONS ##############

# library containing list of glycoletters curated from SweetTalk
path = './main_classes/'
with open(path + 'glycoletter_lib.pkl', 'rb') as file:
    glycoletter_lib = pickle.load(file)

# constants related to the allowed alphabets: nucleotide, protein, or glycan seqs
NUCLEIC_ACID = 'nucleic_acid'
PROTEIN = 'protein'
GLYCAN = 'glycan'

# this is the dictionary of allowed letters/words for each biological sequence-type
ALPHABETS = {
    NUCLEIC_ACID: list('ATCG'),
    PROTEIN: list('ARNDCEQGHILKMFPSTWYVUO'), # Include U, O (Selenocysteine, Pyrrolysine)
    GLYCAN: glycoletter_lib
}

# this is the dictionary of allowed substitutions to convert non-standard letters into the standard alphabet
SUBSTITUTIONS = {
    NUCLEIC_ACID: {
            'R' : 'AG',
            'Y' : 'CT',
            'S' : 'GC',
            'W' : 'AT',
            'K' : 'GT',
            'M' : 'AC',
            'B' : 'CGT',
            'D' : 'AGT',
            'H' : 'ACT',
            'V' : 'ACG',
    },
    PROTEIN: {
            'B' : 'RN',
            'Z' : 'EQ',
            'J' : 'LI',
    },
}

GAP_LETTERS = {
    NUCLEIC_ACID: 'N',
    PROTEIN: 'X',
    GLYCAN: 'X'
}

# these are allowed augmentation types
AUGMENTATION_TYPES = ['none', 'complement', 'reverse_complement', 'both_complements']

def fill(sequence, sequence_type):
    """replace gaps (e.g. "-") with the corresponding gap letter.
    Parameters
    ----------
    sequence : str representing one nucleic acid or protein sequence
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    
    Returns
    -------
    filled_seq : str representing no-gapped one nucleic acid or protein sequence
    """ 
        
    if sequence_type == 'glycan': # not done for glycans
        
        return(sequence)
    else:
        # takes the upper-case for all nucleic_acid and protein alphabets
        seq = sequence.upper()
        filled_seq = seq.replace('-', GAP_LETTERS[sequence_type])
        filled_seq = seq.replace(' ', GAP_LETTERS[sequence_type])
        filled_seq = seq.replace('.', GAP_LETTERS[sequence_type])
        filled_seq = seq.replace('*', GAP_LETTERS[sequence_type])
        
        return(filled_seq)

def pad(sequence, length, sequence_type):
    """pad or truncate sequence to desired length using the appropriate gap letter
    Parameters
    ----------
    sequence : str (if nucleic_acid or protein; else list of strs for glycans) representing one sequence of any type
    length : int representing the length to which a sequence should be padded or truncated to
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    
    Returns
    -------
    seq : str (if nucleic_acid or protein; else list of strs for glycans) representing length-standardized sequence
    """ 

    if sequence_type != 'glycan':
        seq = sequence.upper()
    else:
        seq = sequence
            
    len_seq = len(seq)
    delta = length - len_seq
        
    if delta >= 0:
        # pads using the gap letter token
        if sequence_type != 'glycan':
            padding = ''.join(GAP_LETTERS[sequence_type] for _ in range(delta))
            seq += padding
        else:
            padding = [GAP_LETTERS[sequence_type]] * delta
            seq.extend(padding)
        
        return(seq)
    else:
        seq = sequence[0:length]
        
        return(seq)

def checkValidity(df_data_input, sequence_type):
    """check to make sure that letters are within defined alphabet
    Parameters
    ----------
    df_data_input : list of sequences
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'

    Returns
    -------
    None
    """ 

    valid_letters = ALPHABETS[sequence_type]
    if sequence_type == 'glycan':
        all_letters = [item for sublist in df_data_input for item in sublist] # flatten list of lists
    else:
        all_letters = ''.join(df_data_input)
    valid = [x in valid_letters for x in all_letters]
    valid = list(set(valid))

    if all(valid): 
        print('Confirmed: All sequence characters are in alphabet')
    else:
        letts = list(set([all_letters[i] for i, x in enumerate(valid) if not x]))
        print('Warning: Unknown letter(s) "{0}" found in sequence'.format(', '.join(letts)))
        for lett in letts:
            badseqs = [s for s in df_data_input if lett in s]
            print('Example of bad letter ' + lett + ': ' + str(badseqs[0]))
        # this is actually handled in one_hot_encode function - just printed here :)
        if sequence_type != 'glycan':
            for lett in letts:
                if lett in SUBSTITUTIONS[sequence_type]:
                   print('Replacing ' + lett + ' with substitution : ' + ', '.join(SUBSTITUTIONS[sequence_type][lett]))
                   print('Setting all substitutions to 1 in one-hot encoded representation...')

def makeComplement(seq, rev = False):
    """make complement or reverse complement of a sequence
    Parameters
    ----------
    seq : str representing one nucleic acid sequence
    rev : bool indicating reverse complement or regular complement

    Returns
    -------
    s : str representing one complementary / reverse complementary nucleic acid sequence
    """

    s = Seq(seq)
    if rev:
        
        return(s.reverse_complement())
    else:
        
        return(s.complement())

def augmentData(df_data_input, df_data_output, sequence_type, augment_data):
    """augment data if desired
    Parameters
    ----------
    df_data_input : list of nucleic acid sequences
    df_data_output : pandas Series object of target values
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    augment_data : str, either 'none', 'complement', 'reverse_complement', or 'both_complements'

    Returns
    -------
    df_data_input : list representing augmented nucleic acid sequences
    df_data_output : pandas DataFrame object of target values
    """

    if augment_data in AUGMENTATION_TYPES:
        if (sequence_type != NUCLEIC_ACID) & (augment_data != 'none'):
            print("Augmentation is only possible for sequence_type='nucleic_acid'. Setting augmentation_type to 'none'.'")
        more = []
        more_output = []
        if augment_data == 'complement':
            more = [makeComplement(s) for s in df_data_input]
            more_actual = [] # these few lines are about making sure our new sequences are unique- not in real train set
            more_output = []
            for x,y in zip(more,list(df_data_output)):
                if x not in df_data_input:
                    more_actual.append(x)
                    more_output.append(y)
            more = more_actual
            print('Confirmed: Augmentation of type=' + str(augment_data) + ' concluded.')
        elif augment_data == 'reverse_complement':
            more = [makeComplement(s, rev = True) for s in df_data_input]
            more_actual = []
            more_output = []
            for x,y in zip(more,list(df_data_output)):
                if x not in df_data_input:
                    more_actual.append(x)
                    more_output.append(y)
            more = more_actual
            print('Confirmed: Augmentation of type=' + str(augment_data) + ' concluded.')
        elif augment_data == 'both_complements':
            more = [makeComplement(s, rev = True) for s in df_data_input]
            even_more = [makeComplement(s) for s in df_data_input]
            more.extend(even_more)
            more_output = list(df_data_output)
            even_more_output = list(df_data_output)
            more_output.extend(even_more_output)
            more_actual = []
            more_output_actual = []
            for x,y in zip(more,more_output):
                if x not in df_data_input:
                    more_actual.append(x)
                    more_output_actual.append(y)
            more = more_actual
            more_output = more_output_actual
            print('Confirmed: Augmentation of type=' + str(augment_data) + ' concluded.')        
        else: # equals none
            print('Confirmed: No data augmentation requested')
        df_data_input.extend(more)
        df_data_output = list(df_data_output)
        df_data_output.extend(more_output)
        df_data_output = pd.DataFrame(df_data_output) # keep as pandas df - input will be one hot encoded but this is df

        return(df_data_input, df_data_output)
    else:
        
        raise ValueError('augmentation_type "{0}" is invalid. Valid values are: {1}'.format(augment_data, ', '.join(AUGMENTATION_TYPES)))
    
def scramble(seq, sequence_type):
    """scramble one sequence at a time to make scrambled control
    Parameters
    ----------
    seq : str (if nucleic_acid or protein; else list of strs for glycans) representing one sequence
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'

    Returns
    -------
    scrambled_sequence : str (if nucleic_acid or protein; else list of strs for glycans) representing scrambled sequence
    """

    if len(set(seq)) == 1: # only 1 letter long - don't shuffle
        
        return seq
    else:
        scrambled_sequence = seq
        while scrambled_sequence == seq:
            chars = list(seq)
            random.shuffle(chars)
            if sequence_type == 'glycan':
                scrambled_sequence = chars
            else:
                scrambled_sequence = ''.join(chars)
        
        return scrambled_sequence

def small_glymotif_find(s):
    """breaks down glycans into list of monomers. assumes that user provides glycans in a consistent universal notation
    Parameters
    ----------
    s : str representing one glycan sequence

    Returns
    -------
    b : str representing one glycan sequence separated by *
    """
    
    b = s.split('(')
    b = [k.split(')') for k in b]
    b = [item for sublist in b for item in sublist]
    b = [k.strip('[') for k in b]
    b = [k.strip(']') for k in b]
    b = [k.replace('[', '') for k in b]
    b = [k.replace(']', '') for k in b]
    b = '*'.join(b)
    
    return b

def process_glycans(glycan_list):
    """wrapper function to process list of glycans into glycoletters
    Parameters
    ----------
    glycan_list : list of glycan strs

    Returns
    -------
    glycan_motifs : list of list representations of glycans, made up of glycoletter strs
    """

    glycan_motifs = [small_glymotif_find(k) for k in glycan_list]
    glycan_motifs = [k.split('*')for k in glycan_motifs]
    
    return glycan_motifs

from contextlib import contextmanager
import sys, os
@contextmanager
def suppress_stdout(suppress):
    """helper function to suppress std out when the verbosity is 0
    Parameters
    ----------
    suppress : int representing 0=not verbose, 1=verbose

    Returns
    -------
    None
    """

    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr # DeepSwarm outputs red background text which gets read stderr not stdout, so handle this too
        if suppress == 0: # if no verbose
            sys.stdout = devnull
            sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

############## PART 3: AUTOML BACKEND ##############

class AutoMLBackend(object): 
    
    __metaclass__  = abc.ABCMeta
    
    def __init__(self, data_path, model_folder, output_folder, max_runtime, num_folds):
        """constructor for AutoMLBackend object"""

        self.data_path = data_path
        self.model_folder = model_folder
        self.output_folder = output_folder
        self.max_runtime = max_runtime
        self.num_folds = num_folds
        self.df_data_input, self.df_data_output, self.data = self.read_in_data()
    
    def get_available_gpus(self):
        """get number of available GPUs
        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        ngpus= len([x.name for x in local_device_protos if x.device_type == 'GPU'])
        #print('Available GPUs: '+ str(ngpus))
    
    def read_in_data(self):
        """read input data in csv, .xls, or .xlxs format
        Parameters
        ----------
        None

        Returns
        -------
        df_data_input : pandas Series object of sequence inputs
        df_data_output : pandas Series object of target values
        data : pandas DataFrame of original file
        """

        _, ext = splitext(self.data_path)
        if ext == '.csv':
            try:
                data = pd.read_csv(self.data_path)
            except:
                raise
        elif (ext == '.xls'):
            try:
                data = pd.read_excel(self.data_path)
            except:
                raise
        elif (ext == '.xlsx'):
            try:
                data = pd.read_excel(self.data_path, engine='openpyxl')
            except:
                raise
        else:
            raise ValueError('Unsupported data format. Please convert to csv, xls, or xlsx')
        
        try:
            df_data_input = data[self.input_col]
        except:
            raise ValueError('Input column ' + str(self.input_col) + ' does not exist in the data')
        try:
            df_data_output = data[self.target_col]
        except:
            raise ValueError('Target column ' + str(self.target_col) + ' does not exist in the data')
        
        return df_data_input, df_data_output, data

    def clean_input(df_data_input, df_data_output, pad_seqs, augment_data, sequence_type):
        """helper function to clean up input sequences can be used by any model - not bound to a class
        fill gaps, add padding, map non-standard letters, ensure all letters are valid, and augment if wanted

        Parameters
        ----------
        df_data_input : pandas Series object with sequences 
        df_data_output : pandas Series object with target values
        pad_seqs : str indicating pad_seqs method, either 'max', 'min', 'average'
        augment_data : str, either 'none', 'complement', 'reverse_complement', or 'both_complements'
        sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
        
        Returns
        -------
        df_data_input : list of sequences
        df_data_output : pandas DataFrame with target values
        scrambled_df_data_input : list of scrambled sequences
        alph : list representation of alphabet
        """

        if sequence_type not in ALPHABETS:
            raise ValueError('sequence_type "{0}" is invalid. Valid alphabet types are: {1}'.format(sequence_type, ', '.join(ALPHABETS)))

        alph = ALPHABETS[sequence_type]

        df_data_input = list(df_data_input)
        
        if sequence_type == 'glycan':
            df_data_input = process_glycans(df_data_input)

        checkValidity(df_data_input, sequence_type)

        # fill gaps
        df_data_input = [fill(seq, sequence_type) for seq in df_data_input]

        # fix Us to Ts in nucleic acids
        if sequence_type == 'nucleic_acid':
            df_data_input = [s.replace("U", "T") for s in df_data_input]
        
        ## pad to same length
        max_len = max([len(seq) for seq in df_data_input])
        min_len = min([len(seq) for seq in df_data_input])
        avg_len = int(np.average([len(seq) for seq in df_data_input]))

        if max_len == min_len:
            print("Confirmed: No need to pad or truncate, all sequences same length")
        elif pad_seqs == 'max':
            print("Padding all sequences to a length of " + str(max_len))
            df_data_input = [pad(seq, max_len, sequence_type) for seq in df_data_input]
        elif pad_seqs == 'min':
            print("Truncating all sequences to a length of " + str(min_len))
            df_data_input = [pad(seq, min_len, sequence_type) for seq in df_data_input]
        elif pad_seqs == 'average':
            print("Truncating all sequences to a length of " + str(avg_len))
            df_data_input = [pad(seq, avg_len, sequence_type) for seq in df_data_input]
        else:
            raise ValueError('padding method "{0}" is invalid and sequences are not same length. Valid padding methods are: max, min, average'.format(pad_seqs))

        # augment data if wanted
        df_data_input, df_data_output = augmentData(df_data_input, df_data_output, sequence_type, augment_data)

        # scramble all seqs
        scrambled_df_data_input = [scramble(seq, sequence_type) for seq in df_data_input]

        # for all sequence types, trim alph to represent only letters that appear
        trimmed_alph = []
        for letter in alph:
            for seq in df_data_input:
                if letter in seq:
                    trimmed_alph.append(letter)
                    break
        alph = trimmed_alph

        # lastly, shuffle seqs
        temp = list(zip(df_data_input,list(df_data_output.iloc[:,0]),scrambled_df_data_input))
        random.shuffle(temp)
        df_data_input, df_data_output, scrambled_df_data_input = zip(*temp)
        df_data_input=list(df_data_input)
        df_data_output=pd.DataFrame(df_data_output)
        scrambled_df_data_input=list(scrambled_df_data_input)

        return(df_data_input, df_data_output, scrambled_df_data_input, alph)

    @abc.abstractmethod
    def convert_input(self):
        """transform data input into proper format for the specific AutoML tool"""
    
    @abc.abstractmethod
    def transform_target(self): 
        """transform specified output into format for prediction"""
        
    @abc.abstractmethod
    def find_best_architecture(self, X, y): 
        """run AutoML tool on the data to find the best pipeline/topology for the given problem"""
        
    @abc.abstractmethod
    def train_architecture_kfold(self, X, y): 
        """robustly train the optimal pipeline/topology to get performance metrics on the user's data"""

    @abc.abstractmethod
    def run_system(self): 
        """run the overall system: process user data, find best architecture, compute metrics on the user's data, and train a final model for user's use"""

    def one_hot_encode_seq(sequence, sequence_type, alph):
        """encode a sequence into a one-hot integer matrix, adapted from pysster One_Hot_Encoder
        Parameters
        ----------
        sequence : str (if nucleic_acid or protein; else list of strs for glycans) representing one sequence
        sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
        alph : list representation of alphabet

        Returns
        -------
        one_hot : a numpy array with shape (len(sequence), len(alphabet))
        """
    
        one_hot = np.zeros((len(sequence), len(alph)), np.uint8)
        for i in range(len(sequence)):
            lett = sequence[i]
            if lett in alph:
                one_hot[i] = [1 if lett == l else 0 for l in alph]
            elif sequence_type != 'glycan' and lett in SUBSTITUTIONS[sequence_type]:
                subs = SUBSTITUTIONS[sequence_type][lett]
                one_hot[i] = [1 if lett in subs else 0 for l in alph]
            else:
                one_hot[i] = [0] * len(alph)
            one_hot = one_hot.astype('float32')
        
        return one_hot

    def onehot_seqlist(seq_list, sequence_type, alph, model_type):
        """one hot encode every member of a sequence, with specific model_type in mind
        provides both one hot and numeric outputs
        Parameters
        ----------
        seq_list : list of sequences to one-hot encode
        sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
        alph : list representation of alphabet
        model_type : str, one of 'deepswarm', 'autokeras', or 'tpot' 

        Returns
        -------
        oh_data_input : a numpy array with shape (len(sequence), len(alphabet))
        numerical_data_input : a numpy array of sequences converted to numerical inputs 
        """

        oh_data_input = [] # conventional one-hot encoding
        numerical_data_input = [] # deepswarm-formatted data

        for seq in seq_list:
            one_hot_seq = AutoMLBackend.one_hot_encode_seq(seq, sequence_type, alph)
            oh_data_input.append(np.transpose(one_hot_seq))               
            numerical_data_input.append(np.argmax((one_hot_seq), axis=1))
        if model_type == 'tpot':
            
            return(oh_data_input, numerical_data_input)
        else:
            # reformat data so it can be treated as an "image" by deep swarm
            seq_data = np.array(oh_data_input)
            seq_data = seq_data.reshape(seq_data.shape[0], seq_data.shape[2], seq_data.shape[1])

            # Deepswarm needs to expand dimensions for 2D CNN
            numerical_data_input = seq_data.reshape(seq_data.shape[0], seq_data.shape[1], seq_data.shape[2], 1)
            
            return(oh_data_input, numerical_data_input) # it looks weird but this is correct oh_data_input (not seq_data), numerical_data_input

    def reverse_onehot2seq(onehot_data, alph, sequence_type, numeric = False): 
        """function to reverse one hot or numeric input from DS or AK; can also be used to reverse one hot from TPOT
        be careful when using these reverse onehot functions - they are trickier to manipulate than it may seem
        Parameters
        ----------
        onehot_data : list of one-hot encoded sequences
        alph : list representation of alphabet
        sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
        numeric : bool, see notes below for usage

        Returns
        -------
        seqs : list of sequences
        """

        # use numeric = True with np.array of size (?, len(seq), len(alph), 1)
        # in other words, use numeric = True with onehot_seqlist model_type == 'deepswarm' or 'autokeras', output = 2nd output (numeric output)
        
        # use numeric = False with list where each entry is np.array of size (len(seq), len(alph))
        # in other words, use numeric = False with onehot_seqlist model_type == 'deepswarm' or 'autokeras', output = 1st output (onehot output)
        # numeric = False also works with TPOT onehot output 
        
        seqs = []
        alph =list(alph)
        if sequence_type != 'glycan':
            for idx,oh_seq in enumerate(onehot_data):
                if not numeric:
                    oh_seq = oh_seq.reshape(oh_seq.shape[0], oh_seq.shape[1])
                    numeric_seq = np.argmax(oh_seq,axis=1) 
                else:
                    oh_seq = oh_seq.reshape(oh_seq.shape[1], oh_seq.shape[0])
                    numeric_seq = oh_seq
                    numeric_seq = np.argmax(oh_seq,axis=0) 
                seqs.append(''.join(alph[nt_idx] for nt_idx in numeric_seq))
        else:
            for idx, oh_seq in enumerate(onehot_data):
                if not numeric:
                    oh_seq = oh_seq.reshape(oh_seq.shape[0], oh_seq.shape[1])
                    numeric_seq = np.argmax(oh_seq,axis=1) 
                else:
                    oh_seq = oh_seq.reshape(oh_seq.shape[1], oh_seq.shape[0])
                    numeric_seq = oh_seq
                    numeric_seq = np.argmax(oh_seq,axis=0) 
                newglyc = [alph[s] for s in numeric_seq]
                seqs.append(newglyc)

        return seqs 

    def reverse_tpot2seq(numerical_data, alph, sequence_type): 
        """function to reverse one hot or numeric input from TPOT-style, numeric input only
        be careful when using these reverse onehot functions - they are trickier to manipulate than it may seem
        Parameters
        ----------
        numerical_data : numpy array of sequences converted to numerical inputs
        alph : list representation of alphabet
        sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'

        Returns
        -------
        seqs : list of sequences
        """

        # use where each entry is np.array of size (len(seq))
        
        seqs = []
        alph =list(alph)
        if sequence_type != 'glycan':
            for idx,numeric_seq in enumerate(numerical_data):
                seqs.append(''.join(alph[nt_idx] for nt_idx in numeric_seq))
        else:
            for idx, numeric_seq in enumerate(numerical_data):
                newglyc = [alph[s] for s in numeric_seq]
                seqs.append(newglyc)
        
        return seqs 

    def generic_predict(oh_data_input, numerical_data_input, model_type, final_model_path, final_model_name):
        """generic prediction function using a trained model
        Parameters
        ----------
        oh_data_input : list of sequences converted to one-hot encoded matrix inputs 
        numerical_data_input : numpy array of sequences converted to numerical inputs
        model_type : str, one of 'deepswarm', 'autokeras', or 'tpot'
        final_model_path : str representing folder with final model
        final_model_name : str representing name of final model

        Returns
        -------
        y : numpy array of predicted values
        """

        if model_type == 'deepswarm':
            X = numerical_data_input
            # get sequences with help from https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
            with CustomObjectScope({'GlorotUniform': glorot_uniform(), 'BatchNormalizationV1': BatchNormalization()}): # , 'BatchNormalizationV1': BatchNormalization()
                try:
                    #print("using original model")
                    model = load_model(final_model_path+final_model_name)
                    model.load_weights(final_model_path+final_model_name)
                except Exception as e:
                    #print(e)
                    model = tf.keras.models.load_model(final_model_path+final_model_name)
                
        elif model_type == 'autokeras':
            X = np.array(oh_data_input)
            model = autokeras.utils.pickle_from_file(final_model_path+final_model_name)
        
        elif model_type == 'tpot':
            X = numerical_data_input
            with open(final_model_path+final_model_name, 'rb') as file:  
                model = pickle.load(file) 

        #predict y values
        y = model.predict(X)
        
        return(y)

    def print_stats(stats, output_file_path):
        """print stats to an output file, adapted from: https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
        Parameters
        ----------
        stats : list of performance metrics
        output_file_path : str representing file path to write to

        Returns
        -------
        None
        """

        with open(output_file_path,'a') as fh:
            # Pass the file handle in as a lambda function to make it callable
            for stat in stats:
                fh.write(stat+ '\n')

############## PART 4: AUTOML CLASSIFIER ##############

class AutoMLClassifier(AutoMLBackend): 
    
    def __init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, do_auto_bin, bin_threshold, input_col, target_col):
        """constructor for AutoMLClassifier, child of AutoMLBackend"""

        self.input_col = input_col
        self.target_col = target_col
        AutoMLBackend.__init__(self, data_path, model_folder, output_folder, max_runtime, num_folds)
        self.do_auto_bin = do_auto_bin
        self.bin_threshold= bin_threshold
    
    def transform_target(self, multiclass): 
        """transform target values if desired
        Parameters
        ----------
        multiclass : bool representing if multiclass classification or binary classification

        Returns
        -------
        transformed_output : numpy array of transformed target values
        data_transformer : sklearn.preprocessing Scaler method
        """

        if multiclass:
            col = self.df_data_output.iloc[:,0]
            transformed_output = col.astype("category").cat.codes
            transformed_output = list(transformed_output)
            data_transformer = None
        else:
            if self.do_auto_bin: 
                # first, transform to the uniform distribution 
                #data_transformer = preprocessing.QuantileTransformer(random_state=0)
                data_transformer = preprocessing.RobustScaler()
                transformed_output = data_transformer.fit_transform(self.df_data_output.values.reshape(-1, 1))
                # then, bin the data based on 
                binned_output = (transformed_output > 0.5).astype(int)
                # reform the data to match format needed to convert to categorical 
                min_max_scaler = preprocessing.MinMaxScaler()
                transformed_output = min_max_scaler.fit_transform(binned_output.reshape(-1, 1))
            else: 
                # user does not want data transfomed -- operate on the target distribution directly
                data_transformer = None
                if bin_threshold != None: 
                    # user has prior knowledge of data categories: use specified threshold
                    binned_output = (self.df_data_output > self.bin_threshold).astype(int)
                else: 
                    # default bin_threshold = the median of the data (handle class skewing)
                    bin_threshold = np.median(self.df_data_output.values)
                    binned_output = (self.df_data_output > self.bin_threshold).astype(int)
                # pull out the values from the data frame vector 
                binned_output = binned_output.values
                # transform to categorical (need to reshape to match desired format)
                min_max_scaler = preprocessing.MinMaxScaler()
                transformed_output = to_categorical(min_max_scaler.fit_transform(binned_output.reshape(-1, 1)))
        
        return transformed_output,  data_transformer
    
    def classification_performance_eval(self, report_path, Y, score_Y, file_tag = '', display=False):
        """perform evaluation of classification performance
        note: uses micro and macro averaging to evaluate the overall performance across all classes
               precision=PRE=(TP)/(TP+FP)
        
               In “micro averaging”, we’d calculate the performance, e.g., precision, from the individual 
               true positives, true negatives, false positives, and false negatives of the the k-class model:
               PRE_micro=(TP_1 + .... + TP_k) / (TP_1+ ... +TP_k + FP_1 + .... + FP_k)
               
               And in macro-averaging, we average the performances of each individual class:
               PRE_marco=/PRE_1+...+PRE_k)/(k)
        Parameters
        ----------
        report_path : str of file path to save graphs
        Y : numpy array of true classes of data/dataset (usually N_samples x 1 categorical vector)   
        score_Y : numpy array of probability scores of belonging to any of classess in tested data/dataset (usually a N_samples x N_class float vector with 0 to 1 values)
        file_tag : str of specific name to add to graph name
        display : integer corresponding to display boolean

        Returns
        -------
        deploy_test_metrics : list of metrics values
        deploy_test_metric_names : list of metrics names, i.e. ['auROC Class 0', 'auROC Class 1', ...]
        """

        # get number of samples and number of outputs
        # depending on whether data is in categorical form already or not 
        if self.multiclass and len(np.shape(score_Y)) < 2:
            Y = to_categorical(Y)
            (n_samples, num_classes) = np.shape(Y)
            score_Y = to_categorical(score_Y, num_classes = num_classes)

        elif len(np.shape(score_Y)) == 1: 
            (n_samples, ) = np.shape(score_Y)
            num_classes = 2

            # turn true class into numerical categorical in case it is a string
            Y = (to_categorical(Y, num_classes = num_classes))
            score_Y = (to_categorical(score_Y, num_classes = num_classes))
        else: 
            (n_samples, num_classes ) = np.shape(score_Y)
        
        # plot linewidth.
        lw = 2

        # compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        mcc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(Y[:,i], score_Y[:,i])
            mcc[i] = matthews_corrcoef(Y[:,i], score_Y[:,i].round())
            roc_auc[i] = auc(fpr[i], tpr[i])

        # compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), score_Y.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # compute macro-average ROC curve and ROC area
        # first aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

        # then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # finally average it and compute AUC
        mean_tpr /= num_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        import matplotlib
        from matplotlib import rc
        font = {'size'   : 16}
        matplotlib.rc('font', **font)
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            ## for Palatino and other serif fonts use:
            #rc('font',**{'family':'serif','serif':['Palatino']})
            #rc('text', usetex=True)

        # change font
        matplotlib.rcParams['font.sans-serif'] = "Arial"
        matplotlib.rcParams['font.family'] = "sans-serif"
        sns.set_style("whitegrid")

        # plot all ROC curves
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='cornflowerblue', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = itertools.cycle(['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy',  ])

        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(bbox_to_anchor=(1.4,0), loc="lower right", prop={'size': 10})
        plt.tight_layout()
        plt.savefig(report_path + 'classification_ROC_' + file_tag, dpi=300)
        if display:
            plt.show()

        # zoom in view of the upper left corner.
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
        plt.xlim(0, 0.3)
        plt.ylim(0.65, 1)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='cornflowerblue', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = itertools.cycle(['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy',  ])

        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.legend(bbox_to_anchor=(1.4,0), loc="lower right", prop={'size': 10})
        plt.tight_layout()
        plt.savefig(report_path + 'classification_zoomedROC_' + file_tag, dpi=300)
        if display:
            plt.show()

        deploy_test_metrics = []
        deploy_test_metric_names = []
        for i in range(num_classes):
            deploy_test_metrics.append(roc_auc[i])
            deploy_test_metric_names.append('auROC Class ' + str(i))
        
        deploy_test_metrics.append(roc_auc['micro'])
        deploy_test_metric_names.append('auROC Micro Avg')
        deploy_test_metrics.append(roc_auc['macro'])
        deploy_test_metric_names.append('auROC Macro Avg')

        for i in range(num_classes):
            deploy_test_metrics.append(mcc[i])
            deploy_test_metric_names.append('MCC Class ' + str(i))

        return deploy_test_metrics, deploy_test_metric_names
    
    def write_results(self, metric_names, avg_metric_folds, std_metric_folds, compiled_metrics, all_metric_folds, scrambled=False, subset='all'): 
        """write and report model evaluation metrics
        Parameters
        ----------
        metric_names : list of metric names
        avg_metric_folds : numpy array of metric averages across folds
        std_metric_folds : numpy array of metric std deviations across folds
        compiled_metrics : list of compiled metrics
        all_metric_folds : numpy array of metrics in all folds, in shape (metrics, number of folds)
        scrambled : boolean indicating if this is the scrambled control or not
        subset : str indicating if this is the entire dataset or part of the dataset during the data robustness test

        Returns
        -------
        results_path : str indicating the file path + file name of the results text file
        """

        if subset == 'all':
            if scrambled:
                results_filename = 'scrambled/' + subset + '_scrambled_control_results.txt'
            else:
                results_filename = subset + '_results.txt'
        else:
            if scrambled:
                results_filename = 'robustness/' + subset + '_scrambled_control_results.txt'
            else:
                results_filename = 'robustness/' + subset + '_results.txt'
        with open(self.output_folder + results_filename, "w") as f:
            for metric, compiled_metric, avg_metric_fold, std_metric_fold, all_metric_fold in zip(metric_names, compiled_metrics, avg_metric_folds, std_metric_folds, all_metric_folds):
                f.write(metric + ' (compiled): ' + str(compiled_metric) + '\n')
                f.write('avg ' +metric + ' over folds: ' + str(avg_metric_fold)+ '\n')
                f.write('standard deviation of ' + metric + ': ' + str(std_metric_fold)+ '\n')
                f.write('all values of ' + metric + ': ' + str(all_metric_fold) + '\n')
        results_path = self.output_folder + results_filename
        
        return results_path
        
############## PART 5: AUTOML REGRESSOR ##############

class AutoMLRegressor(AutoMLBackend): 
    
    def __init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, do_transform, input_col, target_col):
        """constructor for AutoMLRegressor, child of AutoMLBackend"""

        self.input_col = input_col
        self.target_col = target_col
        AutoMLBackend.__init__(self, data_path, model_folder, output_folder, max_runtime, num_folds)
        self.do_transform = do_transform
    
    def transform_target(self): 
        """transform output vector to desired distribution
        Parameters
        ----------
        None

        Returns
        -------
        transformed_output : numpy array of transformed target values
        data_transformer : sklearn.preprocessing Scaler method
        """

        data_transformer = preprocessing.RobustScaler()
        transformed_output = data_transformer.fit_transform(self.df_data_output.values.reshape(-1, 1))
        
        return transformed_output,  data_transformer
    
    def r2(self, x, y):
        """perform R2 metric
        Parameters
        ----------
        x : numpy array representing x-axis values
        y : numpy array representing y-axis values

        Returns
        -------
        r2 : float representing r2 value
        """
        
        return stats.pearsonr(x, y)[0] ** 2

    def regression_performance_eval(self,report_path, Y, pred_Y, file_tag = '', display=False):
        """perform evaluation of regression performance
        Parameters
        ----------
        report_path : str of file path to save graphs
        Y : numpy array of true values of data/dataset
        score_Y : numpy array of predicted values
        file_tag : str of specific name to add to graph name
        display : integer corresponding to display sboolean

        Returns
        -------
        deploy_test_metrics : list of metrics values
        """

        # get number of samples and number of outputs
        (n_samples, n_outputs) = np.shape(Y)

        # calculate difference between Predicted output and target output
        diff_Y = pred_Y - Y
        absDiff_Y = np.abs(diff_Y)

        # compute the absolute mean, absolute standard deviation prediction-target difference:
        ad_mean_Y = np.mean(absDiff_Y, axis=0) # Mean absolute difference
        ad_std = np.std(absDiff_Y, axis=0)     # Standard deviation of the Mean absolute difference 

        # initialize empty array for R2 calculation
        ad_r2 = np.zeros_like(ad_mean_Y)       # R2

        # create Graphs
        # R2 (Coefficient of Determination)
        index = 0 # because just one predictor
        ad_r2[index] = self.r2(pred_Y[:,index], Y[:,index])

        import matplotlib
        from matplotlib import rc
        font = {'size'   : 20}
        matplotlib.rc('font', **font)
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            ## for Palatino and other serif fonts use:
            #rc('font',**{'family':'serif','serif':['Palatino']})
            #rc('text', usetex=True)

        # change font
        matplotlib.rcParams['font.sans-serif'] = "Arial"
        matplotlib.rcParams['font.family'] = "sans-serif"
        sns.set_style("whitegrid")

        # Display Output Values
        x_tot=np.squeeze(pred_Y[:,index])
        y_tot=np.squeeze(Y[:,index])
        pearson = stats.pearsonr(x_tot, y_tot)[0]
        spearman = stats.spearmanr(x_tot, y_tot)[0]

        # NOTE: for sklearn r2 need to have y_true, y_pred order 
        #g = sns.jointplot(x_tot, y_tot, kind="reg", color="b", stat_func=self.r2) # stat_func argument deprecated
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
        g = sns.jointplot(x_tot, y_tot, kind="reg", color="cornflowerblue")
        g.ax_joint.text = self.r2
        g.plot_joint(plt.scatter, c="cornflowerblue", s=4, linewidth=1, marker=".", alpha=0.08)
        try:
        	g.plot_joint(sns.kdeplot, zorder=0, color="grey", n_levels=6, shade=False)
        except: # if discrete, then need hist instead of kde
        	g.plot_joint(sns.histplot, zorder=0, color="grey",alpha=0.05)
        g.ax_joint.collections[0].set_alpha(0)
        g.set_axis_labels("Predicted", "Experimental");

        # save the figure
        g.savefig(report_path +'regression_' + file_tag + ".png", bbox_inches='tight', dpi=300)
        if display:
            plt.show()
        ad_pearson = pearson
        ad_spearman = spearman
        
        # store model performance metrics for return   
        deploy_test_metrics = [ad_mean_Y[index], ad_std[index], ad_r2[index], ad_pearson, ad_spearman]
        
        return deploy_test_metrics

    # [Helper Function] Write and report model evaluation metrics
    def write_results(self, avg_r2_folds, std_r2_folds, compiled_r2, all_r2, scrambled = False, subset = 'all'): 
        """write and report model evaluation metrics
        note that these variable names are actually outdated, we handle all types of performance metrics not just r2
        Parameters
        ----------
        avg_r2_folds : numpy array of metric averages across folds
        std_r2_folds : numpy array of metric std deviations across folds
        compiled_r2 : list of compiled metrics
        all_r2 : numpy array of metrics in all folds, in shape (metrics, number of folds)
        scrambled : boolean indicating if this is the scrambled control or not
        subset : str indicating if this is the entire dataset or part of the dataset during the data robustness test

        Returns
        -------
        results_path : str indicating the file path + file name of the results text file
        """

        if subset == 'all':
            if scrambled:
                results_filename = 'scrambled/' + subset + '_scrambled_control_results.txt'
            else:
                results_filename = subset + '_results.txt'
        else:
            if scrambled:
                results_filename = 'robustness/' + subset + '_scrambled_control_results.txt'
            else:
                results_filename = 'robustness/' + subset + '_results.txt'

        with open(self.output_folder + results_filename, "w") as f:
            f.write('r2 (compiled): ' + str(compiled_r2) + '\n')
            f.write('avg r2 over folds: ' + str(avg_r2_folds)+ '\n')
            f.write('standard deviation of r2: ' + str(std_r2_folds)+ '\n')

            f.write('all MAE: ' + str(all_r2[0]) + '\n')
            f.write('std MAE: ' + str(all_r2[1]) + '\n')
            f.write('all r2: ' + str(all_r2[2]) + '\n')
            f.write('all pearson: ' + str(all_r2[3]) + '\n')
            f.write('all spearman: ' + str(all_r2[4]) + '\n')

        return self.output_folder + results_filename

############## END OF FILE ##############