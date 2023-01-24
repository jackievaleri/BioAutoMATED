#!/usr/bin/env python
# coding: utf-8

############## PART 1: IMPORT STATEMENTS ##############

# import generic functions
from BioSeqAutoML_generic_automl_classes import AutoMLBackend

# import system libraries
import os
import platform
import random
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tqdm import tqdm_notebook as tqdm
import math
import itertools
import numpy as np
import logomaker
from scipy.stats import sem, t
from scipy import mean
import pickle
import string

# import some keras and visualization libraries
import keras
from vis.visualization import visualize_saliency, visualize_activation
from vis.utils import utils
from keras import activations
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.layers import BatchNormalization
import autokeras

############## PART 2: FEATURE IMPORTANCE FUNCTIONS ##############

def plot_ft_importance(oh_data_input, final_model_path, final_model_name, plot_path, plot_name):
    """wrapper function for plotting in silico mutagenesis
    Parameters
    ----------
    oh_data_input : list of sequences converted to one-hot encoded matrix inputs 
    final_model_path : str representing folder with final model
    final_model_name : str representing name of final model
    plot_path : str representing folder where plots are to be located
    plot_name : str representing name of plot
    
    Returns
    -------
    None
    """ 
    seq_len = len(list(oh_data_input[0][0]))
    
    with open(final_model_path+final_model_name, 'rb') as file:  
        model = pickle.load(file)
    feature_names = [f"{i}" for i in range(seq_len)]
    try:
        importances = model.feature_importances_
    except:
        print("No feature importances can be computed from this model.")
        return()
    
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
    try:
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        forest_importances.plot.bar(yerr=std, ax=ax)
    except:
        forest_importances.plot.bar(ax=ax)

    ax.set_xlabel('Position', fontsize=20)
    ax.set_ylabel("Feature Importance", fontsize=20)
    plt.tick_params(length = 10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(0, seq_len, 10))
    ax.set_xticklabels([str(x) for x in np.arange(0, seq_len, 10)], fontsize = 20, rotation = 0)
    ax.set_yticklabels([str(np.round(x,2)) for x in ax.get_yticks()], fontsize = 20)

    plt.tight_layout()
    plt.savefig(plot_path+final_model_name.split('.pkl')[0] + plot_name)
    plt.savefig(plot_path+final_model_name.split('.pkl')[0] + plot_name.split('.png')[0] + '.svg')    

############## PART 3: ACTIVATION MAP AND SALIENCY MAP FUNCTIONS ##############

def plot_saliency_maps(numerical_data_input, oh_data_input, alph, final_model_path, final_model_name, plot_path, plot_name, sequence_type, interpret_params):
    """plot saliency maps with trained models
    adopted from code from Valeri, Collins, Ramesh et al. Nature Communications 2020
    Parameters
    ----------
    numerical_data_input : a numpy array of sequences converted to numerical inputs 
    oh_data_input : list of sequences converted to one-hot encoded matrix inputs 
    alph : list representation of alphabet
    final_model_path : str representing folder with final model
    final_model_name : str representing name of final model
    plot_path : str representing folder where plots are to be located
    plot_name : str representing name of plot
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    interpret_params : dict of extra interpretation parameters, with keys 'sample_number_class_activation_maps' (int), 'class_activation_grad_modifier' (str), 'class_activation_layer_index' (int);
        'sample_number_saliency_maps' (int), 'saliency_map_grad_modifier' (str), 'saliency_map_layer_index' (int), 'sample_number_mutagenesis' (int)

    Returns
    -------
    final_arr : numpy array of saliency values for plotting sequence logos
    plot_name : str representing combined folder + file name of plot
    alph : list representation of (potentially trimmed-down) alphabet
    seq_len : int representing length of a typical sequence
    """ 

    # defaults
    sample_number_saliency_maps = interpret_params.get('sample_number_saliency_maps', 100)
    saliency_map_grad_modifier = interpret_params.get('saliency_map_grad_modifier',None)
    saliency_map_layer_index = interpret_params.get('saliency_map_layer_index',-1)

    X = numerical_data_input
    seq_len = len(numerical_data_input[0])
    
    # look at saliency map for random samples of input sequences
    # modified code from keras-vis package github page - https://github.com/raghakot/keras-vis/blob/master/examples/mnist/attention.ipynb
    
    # import features with help from https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
    with CustomObjectScope({'GlorotUniform': glorot_uniform(), 'BatchNormalizationV1': BatchNormalization()}): # , 'BatchNormalizationV1': BatchNormalization()
        try:
            model = load_model(final_model_path+final_model_name)
        except:
            model = tf.keras.models.load_model(final_model_path+final_model_name)        
        model.load_weights(final_model_path+final_model_name)

    # note this takes a long time if you have a high number of sample_number_saliency_maps
    num_show = sample_number_saliency_maps
    grads = [] 
    # randomly draw samples (w/o replacement)
    rand_indices = np.random.choice(range(len(X)), num_show, replace=False)
    for idx in rand_indices:
        # layer idx = what layer to maximize/minimize (i.e. output node)
        # seed_input = the one-hot encoded sequence whose path thru the model we are tracing 
        # filter_indices = what layer are we computing saliency w/ r.t. (i.e. input layer)
        try:
            grad = visualize_saliency(model, layer_idx=saliency_map_layer_index, filter_indices=None, seed_input=X[idx], grad_modifier=saliency_map_grad_modifier)
        except:
            # re-initialize variables in the graph/model - sometimes needed with custom layers
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                grad = visualize_saliency(model, layer_idx=saliency_map_layer_index, filter_indices=None, seed_input=X[idx], grad_modifier=saliency_map_grad_modifier)

        grads.append(grad)
        
    # look at average saliency based on original nucleotide
    grads_over_letters_on = {c: np.zeros(seq_len) for c in alph}
    counts_per_positon_on = {c: np.zeros(seq_len) for c in alph}
    
    for sample_idx,idx in enumerate(rand_indices): 
        # have to use the OH data input, NOT the numerical data input (which is used for the model)
        # else the reverse onebhot2seq doesn't work
        onehot = oh_data_input[idx]
        onehot = np.transpose(onehot)
        orig = AutoMLBackend.reverse_onehot2seq([onehot], alph, sequence_type, numeric=False)[0]

        for position, nt in enumerate(orig): 
            grad_at_pos = grads[sample_idx][position][alph.index(nt)]
            grads_over_letters_on[nt][position] += grad_at_pos
            counts_per_positon_on[nt][position] += 1

    cmap = sns.color_palette("PuBu", n_colors = 20)
    final_arr = [grads_over_letters_on[letter]/counts_per_positon_on[letter] for letter in alph]
    
    # additional culling needed for glycans - if not, length 1027 is too long
    if sequence_type == 'glycan':
        final_arr = pd.DataFrame(final_arr)
        sums = final_arr.sum(axis=1)
        sums = [s > np.quantile(sums, 0.90) for s in sums]
        if sum(sums) > 26 or sum(sums) == 0: # get the top 26 characters
            sums = list(final_arr.sum(axis=1))
            sums.sort(reverse=True)
            cutoff = sums[np.minimum(len(sums)-1,25)]
            sums = final_arr.sum(axis=1) # need to get original, unsorted list back
            sums = [s > cutoff for s in sums]
        alph = [i for (i, v) in zip(alph, sums) if v]
        final_arr = final_arr.iloc[sums,:]
        
    # normalized by number of times a nt appeared at each position 
    import matplotlib
    from matplotlib import rc
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (seq_len / 4, np.maximum(len(alph) / 3.5, 3)), dpi = 300)
    plt.rcParams["figure.dpi"] = 300
        
    g = sns.heatmap(final_arr, cmap=cmap,  cbar_kws={"orientation": "vertical", "pad": 0.035, "fraction": 0.05})
    ax.tick_params(length = 10)
    plt.xticks(np.arange(0, seq_len, 10), np.arange(0, seq_len, 10), fontsize = 15, rotation = 0)
    plt.yticks(np.arange(0.5, len(alph)), np.arange(0.5, len(alph)), fontsize = 15, rotation = 0)
    g.set_yticklabels(alph, fontsize = 20, rotation = 0)
    g.set_xticklabels(g.get_xticklabels(), fontsize = 20)
    plt.xlabel('Position', fontsize = 20)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Normalized Saliency", rotation = 270, fontsize = 20)
    ax.collections[0].colorbar.ax.set_yticklabels(ax.collections[0].colorbar.ax.get_yticklabels())
    cbar.ax.get_yaxis().labelpad = 30
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length = 10, labelsize=15)

    plotnamefinal = plot_path + final_model_name.split('.h5')[0]  + plot_name
    print('Saliency map saved to ' + plotnamefinal)
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    plt.savefig(plotnamefinal.split('.png')[0] + '.svg')

    return(final_arr, plot_path + final_model_name.split('.h5')[0], alph, seq_len)

def plot_activation_maps(numerical_data_input, oh_data_input, alph, final_model_path, final_model_name, plot_path, plot_name, sequence_type, interpret_params):
    """plot activation maps with trained models
    adopted from code from Valeri, Collins, Ramesh et al. Nature Communications 2020
    Parameters
    ----------
    numerical_data_input : a numpy array of sequences converted to numerical inputs 
    oh_data_input : list of sequences converted to one-hot encoded matrix inputs 
    alph : list representation of alphabet
    final_model_path : str representing folder with final model
    final_model_name : str representing name of final model
    plot_path : str representing folder where plots are to be located
    plot_name : str representing name of plot
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    interpret_params : dict of extra interpretation parameters, with keys 'sample_number_class_activation_maps' (int), 'class_activation_grad_modifier' (str), 'class_activation_layer_index' (int);
        'sample_number_saliency_maps' (int), 'saliency_map_grad_modifier' (str), 'saliency_map_layer_index' (int), 'sample_number_mutagenesis' (int)

    Returns
    -------
    final_arr : numpy array of activation values for plotting sequence logos
    plot_name : str representing combined folder + file name of plot
    alph : list representation of (potentially trimmed-down) alphabet
    seq_len : int representing length of a typical sequence
    """ 

    # defaults
    sample_number_class_activation_maps = interpret_params.get('sample_number_class_activation_maps',100)
    class_activation_grad_modifier = interpret_params.get('class_activation_grad_modifier', None)
    class_activation_layer_index = interpret_params.get('class_activation_layer_index',-1)

    X = numerical_data_input
    seq_len = len(numerical_data_input[0])
    
    # look at activation map for random samples of input sequences
    # modified code from keras-vis package github page - https://github.com/raghakot/keras-vis/blob/master/examples/mnist/attention.ipynb
    
    # import features with help from https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
    with CustomObjectScope({'GlorotUniform': glorot_uniform(), 'BatchNormalizationV1': BatchNormalization()}): # BatchNormalization removed
        try:
            model = load_model(final_model_path+final_model_name)
        except:
            model = tf.keras.models.load_model(final_model_path+final_model_name)
        model.load_weights(final_model_path+final_model_name)

    # note this takes a long time if you have a high number of sample_number_saliency_maps
    num_show = sample_number_class_activation_maps
    grads = [] 
    # randomly draw samples (w/o replacement)
    rand_indices = np.random.choice(range(len(X)), num_show, replace=False)
    for idx in rand_indices:
        try:
            grad = visualize_activation(model, layer_idx=class_activation_layer_index, filter_indices=None, seed_input=X[idx], grad_modifier=class_activation_grad_modifier)
        except:
            # re-initialize variables in the graph/model - sometimes needed with custom layers
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                grad = visualize_saliency(model, layer_idx=class_activation_layer_index, filter_indices=None, seed_input=X[idx], grad_modifier=class_activation_grad_modifier)

        grads.append(grad)
        
    # look at average saliency based on original nucleotide
    grads_over_letters_on = {c: np.zeros(seq_len) for c in alph}
    counts_per_positon_on = {c: np.zeros(seq_len) for c in alph}

    for sample_idx,idx in enumerate(rand_indices): 
        # have to use the OH data input, NOT the numerical data input (which is used for the model)
        # else the reverse onebhot2seq doesn't work
        onehot = oh_data_input[idx]
        onehot = np.transpose(onehot)
        orig = AutoMLBackend.reverse_onehot2seq([onehot], alph, sequence_type, numeric = False)[0]

        for position, nt in enumerate(orig): 
            grad_at_pos = grads[sample_idx][position][alph.index(nt)]
            grads_over_letters_on[nt][position] += grad_at_pos
            counts_per_positon_on[nt][position] += 1

    cmap = sns.color_palette("PuBu", n_colors = 20)
    final_arr = [grads_over_letters_on[letter]/counts_per_positon_on[letter] for letter in alph]
    
    # additional culling needed for glycans - if not, length 1027 is too long
    if sequence_type == 'glycan':
        final_arr = pd.DataFrame(final_arr)
        sums = final_arr.sum(axis=1)
        sums = [s > np.quantile(sums, 0.90) for s in sums]
        if sum(sums) > 26 or sum(sums) == 0: # get the top 26 characters
            sums = list(final_arr.sum(axis=1))
            sums.sort(reverse=True)
            cutoff = sums[np.minimum(len(sums)-1,25)]
            sums = final_arr.sum(axis=1) # need to get original, unsorted list back
            sums = [s > cutoff for s in sums]
        alph = [i for (i, v) in zip(alph, sums) if v]
        final_arr = final_arr.iloc[sums,:]
    
    import matplotlib
    from matplotlib import rc
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (seq_len / 4,np.maximum(len(alph) / 3.5, 3)), dpi = 300)
    plt.rcParams["figure.dpi"] = 300
    
    g = sns.heatmap(final_arr, cmap=cmap,  cbar_kws={"orientation": "vertical", "pad": 0.035, "fraction": 0.05})
    ax.tick_params(length = 10)
    plt.xticks(np.arange(0, seq_len, 10), np.arange(0, seq_len, 10), fontsize = 15, rotation = 0)
    plt.yticks(np.arange(0.5, len(alph)), np.arange(0.5, len(alph)), fontsize = 15, rotation = 0)

    g.set_yticklabels(alph, fontsize = 20, rotation = 0)        
    g.set_xticklabels(g.get_xticklabels(), fontsize = 20)
    plt.xlabel('Position', fontsize = 20)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Activation  ", rotation = 270, fontsize = 20)
    ax.collections[0].colorbar.ax.set_yticklabels(ax.collections[0].colorbar.ax.get_yticklabels())
    cbar.ax.get_yaxis().labelpad = 30
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length = 10, labelsize=15)

    plotnamefinal = plot_path + final_model_name.split('.h5')[0]  + plot_name
    print('Activation map saved to ' + plotnamefinal)
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    plt.savefig(plotnamefinal.split('.png')[0] + '.svg')
    
    return(final_arr, plot_path + final_model_name.split('.h5')[0], alph, seq_len)

def plot_seqlogos(arr, alph, sequence_type, plot_path, plot_name, lenarr):
    """plots the sequence logos of activation or saliency maps from the trained model
    adopted from code from Valeri, Collins, Ramesh et al. Nature Communications 2020
    Parameters
    ----------
    arr : numpy array of (activation map or saliency map) values for plotting sequence logos
    alph : list representation of (potentially trimmed-down) alphabet
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    plot_path : str representing folder where plots are to be located
    plot_name : str representing name of plot
    len_arr : int representing length of a typical sequence

    Returns
    -------
    None
    """ 
    
    # the columns are the alphabet letters and the rows are position-wise relative activation/saliency
    if sequence_type == 'glycan':
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from matplotlib.pyplot import cm
        colors = cm.turbo(np.linspace(0,1,len(alph)))
        legend_elements = []
        alphabet_string = string.ascii_lowercase
        alphabet_list = list(alphabet_string)
        labels = []
        for i in range(len(alph)):
            labels.append(alphabet_list[i])
            legend_elements.append(Line2D([0], [0], marker='o', color=colors[i], label= alphabet_list[i] + ': ' + alph[i],
                                  markerfacecolor=colors[i], markersize=10))

        # create the figure
        fig, ax = plt.subplots(figsize = (2,10), dpi = 300)
        plt.gca().axison = False
        ax.legend(handles=legend_elements, loc='center')
        alph = labels
        plt.tight_layout()
        plt.savefig(plot_path + 'legend' + plot_name.split('.png')[0] + '.svg')
    
    nn_df = pd.DataFrame(data = arr).T
    nn_df.columns = alph # 1st row as the column names
    nn_df.index = range(len(nn_df))
    
    # create Logo object
    import matplotlib
    from matplotlib import rc
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (lenarr / 4,np.maximum(len(alph) / 3.5, 3)), dpi = 300)
    nn_df = nn_df.fillna(0)

    # here, we plot saliency values as weights and activation values as probabilities
    # this is a design choice and you could easily change this!
    if 'saliency' in plot_name:
        nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'weight')
    else:
        nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'probability')

    # set color scheme and make sequence logo using LogoMaker
    if sequence_type != 'glycan':
        if sequence_type == 'nucleic_acid':
            color_scheme = 'classic'
        elif sequence_type == 'protein':
            color_scheme = 'chemistry'
        nn_logo = logomaker.Logo(nn_df, ax = ax, color_scheme = color_scheme, stack_order = 'big_on_top', fade_below = True)

    else:
        nn_logo = logomaker.Logo(nn_df, ax = ax, color_scheme = dict(zip(alphabet_list, colors)), stack_order = 'big_on_top', fade_below = True)

    if 'saliency' in plot_name:
        nn_logo.ax.set_ylabel('Weight', fontsize = 25)
    else:        
        nn_logo.ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
        nn_logo.ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'], fontsize = 20)
        nn_logo.ax.set_ylabel('Probability', fontsize = 25)

    # style using Logo methods
    nn_logo.style_spines(visible = False)
    nn_logo.style_spines(spines = ['left'], visible = True)
    nn_logo.ax.set_xticks(np.arange(0, lenarr, 10))
    nn_logo.ax.set_xticklabels([str(x) for x in np.arange(0, lenarr, 10)], fontsize = 20)
    nn_logo.ax.set_xlabel('Position', fontsize = 25)

    plt.tight_layout()
    plt.savefig(plot_path + plot_name)
    plt.savefig(plot_path + plot_name.split('.png')[0] + '.svg')

############## PART 4: IN SILICO MUTAGENESIS FUNCTIONS ##############

def get_one_bp_mismatches(seq, smallalph, sequence_type):
    """finds all one bp mismatches - allegedly O(nk^2) where k = size of the sequence
    heavily borrowed code from https://codereview.stackexchange.com/questions/156490/finding-roughly-matching-genome-sequences-in-python-dictionary
    Parameters
    ----------
    seq : str (if nucleic_acid or protein; else list of strs for glycans) representing sequence
    smallalph : list representation of (potentially trimmed-down) alphabet
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'

    Returns
    -------
    mismatches : list of mutated sequences
    """ 

    mismatches = []
    for e,i in enumerate(seq):
        for b in smallalph:
            if sequence_type != 'glycan':
                new_seq = seq[:e] + b + seq[e+1:]
            else:
                new_seq = list(seq[:e])
                new_seq.extend([b])
                new_seq.extend(seq[e+1:])
            mismatches.append(new_seq)
    
    return mismatches

def get_list_of_mismatches_from_list(original_list, sequence_type, alph):
    """finds all one bp mismatches of sequences in a list of sequences
    Parameters
    ----------
    original_list : list of str (if nucleic_acid or protein; else list of strs for glycans) representing sequence
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    alph : list representation of (potentially trimmed-down) alphabet

    Returns
    -------
    final_list : list of lists of mutated sequences
    """ 
    
    final_list = []
    for seq_list in [get_one_bp_mismatches(seq, alph, sequence_type) for seq in original_list]:
        for mismatch in seq_list:
            final_list.append(mismatch)
    
    return final_list

def get_new_mismatch_seqs(seq_list, alph, sequence_type, model_type='deepswarm'):
    """reformats all mismatched sequences into appropriate input type for the specific model
    Parameters
    ----------
    seq_list : list of lists of mutated sequences
    alph : list representation of (potentially trimmed-down) alphabet
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    model_type : str, one of 'deepswarm', 'autokeras', or 'tpot' 
    
    Returns
    -------
    numerical_data_input : a numpy array of sequences converted to numerical inputs 
    oh_data_input : list of sequences converted to one-hot encoded matrix inputs 
    """ 

    oh_data_input = [] # conventional one-hot encoding
    numerical_data_input = [] # deepswarm-formatted data  
    
    for seq in seq_list:
        one_hot_seq = AutoMLBackend.one_hot_encode_seq(seq, sequence_type, alph)
        oh_data_input.append(np.transpose(one_hot_seq))               
        numerical_data_input.append(np.argmax((one_hot_seq), axis=1)) #To numerical for Deepswarm

    # reformat data so it can be treated as an "image" by deep swarm
    if model_type != 'tpot':
        seq_data = np.array(oh_data_input)
        seq_data = seq_data.reshape(seq_data.shape[0], seq_data.shape[2], seq_data.shape[1])

        # Deepswarm needs to expand dimensions for 2D CNN
        numerical_data_input = seq_data.reshape(seq_data.shape[0], seq_data.shape[1], seq_data.shape[2], 1)
    else: # TPOT special case
        numerical_data_input = np.array(numerical_data_input)
    
    return(numerical_data_input, oh_data_input)

def add_to_dataframe(predictions, data_df, model_type = 'deepswarm'):
    """add predictions of mutated sequences to a dataframe so it can be analyzed
    Parameters
    ----------
    predictions : numpy array of predicted values
    data_df : pandas DataFrame with mismatched sequences
    model_type : str, one of 'deepswarm', 'autokeras', or 'tpot' 
    
    Returns
    -------
    data_df : pandas DataFrame with mismatched sequences and their predicted values
    """ 

    if model_type == 'deepswarm':
        on_preds = predictions[:,0] # technically do not have to use one class or another as long as we are consistent
        
    if model_type == 'autokeras' or model_type == 'tpot':
        on_preds = predictions
    data_df['preds'] = on_preds.flatten()
    
    return(data_df)

def get_std_dev_at_each_bp(df, smallalph, num_seqs_to_test):
    """get the standard deviation across all of the 1-mers for each of the sequences (one 1-mer at a time)
    Parameters
    ----------
    df : pandas DataFrame with mismatched sequences and their predicted values
    smallalph : list representation of (potentially trimmed-down) alphabet
    num_seqs_to_test : int corresponding to sample_number_mutagenesis
    
    Returns
    -------
    all_val_std_devs : list of numpy array of size (num_seqs_to_test, seq_len) with all std deviations of predictions
    """ 

    seq_len = len(df.iloc[0,0]) # as an example, get seq len of the first sequence
    hardcode_num_base_seqs = num_seqs_to_test
    alph_len = len(smallalph)
    all_val_std_devs = np.zeros((hardcode_num_base_seqs, seq_len))
    curr_index_of_seqs = 0

    for row in range(0, hardcode_num_base_seqs):
        for col in range(0, seq_len):
            val_for_current_seqs = df.iloc[curr_index_of_seqs:(curr_index_of_seqs+alph_len),1]
            val_std_dev = np.std(list(val_for_current_seqs))   
            all_val_std_devs[row, col] = val_std_dev           
            curr_index_of_seqs = curr_index_of_seqs + alph_len    
    
    return([all_val_std_devs])

def get_means_and_bounds_of_stds(matrix):
    """get the means and conf intervals for std deviations in order to plot
    Parameters
    ----------
    matrix : list of numpy array of size (num_seqs_to_test, seq_len) with all std deviations of predictions
    
    Returns
    -------
    means : list of means of std deviations of predictions across the length of the sequence
    conf_int_lower_bound : list of lower confidence intervals for plotting
    conf_int_upper_bound : list of upper confidence intervals for plotting
    """ 
    
    means = []
    conf_int_lower_bound = []
    conf_int_upper_bound = []
    
    matrix = np.array(matrix)[0]

    for column in matrix.T:
        confidence = 0.95 # compute 95% confidence interval
        n = len(column)
        m = mean(column)
        std_err = sem(column)
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        means.append(m)
        conf_int_lower_bound.append(m-h)
        conf_int_upper_bound.append(m+h)
        
    return([means, conf_int_lower_bound, conf_int_upper_bound])

def get_matrix(listofseqs, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, plot_name, seq_len, model_type):
    """sample list of sequences, generate all mismatches, run mismatch seqs thru model, and generate matrix of std deviations of predictions
    Parameters
    ----------
    listofseqs : numpy array of sequences converted to one-hot encoded matrix inputs 
    num_seqs_to_test : int corresponding to sample_number_mutagenesis
    alph : list representation of alphabet
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    final_model_path : str representing folder with final model
    final_model_name : str representing name of final model
    plot_path : str representing folder where plots are to be located
    plot_name : str representing name of plot
    seq_len : int representing length of the sequence
    model_type : str, one of 'deepswarm', 'autokeras', or 'tpot' 
    
    Returns
    -------
    matrix_of_stds : list of numpy array of size (num_seqs_to_test, seq_len) with all std deviations of predictions
    """ 

    if num_seqs_to_test > len(listofseqs):
        num_seqs_to_test = len(listofseqs)
    
    # sample lists
    listofseqs = list(listofseqs)
    sample = random.sample(listofseqs, num_seqs_to_test)
    sample = [np.transpose(a) for a in sample]

    # get mismatched lists
    orig = AutoMLBackend.reverse_onehot2seq(sample, alph, sequence_type, numeric = False)
    smallalph = plot_rawseqlogos(listofseqs, alph, sequence_type, plot_path, plot_name, seq_len)

    mismatches = get_list_of_mismatches_from_list(orig, sequence_type, smallalph) 
    numerical_X, oh_X = get_new_mismatch_seqs(mismatches, alph, sequence_type, model_type)
    predictions = AutoMLBackend.generic_predict(oh_X, numerical_X, model_type, final_model_path, final_model_name)

    mismatchdf = pd.DataFrame()
    mismatchdf[0] = mismatches

    run_thru_model = add_to_dataframe(predictions, mismatchdf, model_type)

    # get std of predictions
    matrix_of_stds = get_std_dev_at_each_bp(run_thru_model, smallalph, num_seqs_to_test)
    
    return(matrix_of_stds)

def plot_mutagenesis(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, plot_path, plot_name, sequence_type, model_type, interpret_params):
    """wrapper function for plotting in silico mutagenesis
    Parameters
    ----------
    numerical_data_input : a numpy array of sequences converted to numerical inputs 
    oh_data_input : list of sequences converted to one-hot encoded matrix inputs 
    alph : list representation of alphabet
    numerical : list of numerical output values, if the target values could be converted to floats
    numericalbool : bool representing if the target output values could be converted to floats
    final_model_path : str representing folder with final model
    final_model_name : str representing name of final model
    plot_path : str representing folder where plots are to be located
    plot_name : str representing name of plot
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    model_type : str, one of 'deepswarm', 'autokeras', or 'tpot' 
    interpret_params : dict of extra interpretation parameters, with keys 'sample_number_class_activation_maps' (int), 'class_activation_grad_modifier' (str), 'class_activation_layer_index' (int);
        'sample_number_saliency_maps' (int), 'saliency_map_grad_modifier' (str), 'saliency_map_layer_index' (int), 'sample_number_mutagenesis' (int)    
    
    Returns
    -------
    None
    """ 

    num_seqs_to_test = interpret_params.get('sample_number_mutagenesis',50)

    # get top, worst, average seqs
    d = list(numerical)
    howmanyclasses = len(list(set(d)))
    seq_len = len(list(oh_data_input[0][0]))
    oh = np.array(oh_data_input)
    
    average_matrix_of_stds = []
    best_matrix_of_stds = []
    worst_matrix_of_stds = []
    classes = []
    classlabels = []

    if howmanyclasses < 2:
        average_matrix_of_stds = get_matrix(oh, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'raw_data_random_seq_logo', seq_len, model_type)
        
    elif howmanyclasses < 3:
        possvals= list(set(d))
        classlabels = possvals
        for poss in possvals:
            truthvals = [d1 == poss for d1 in d]
            classx = oh[truthvals]
            classx_matrix_of_stds = get_matrix(classx, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'raw_data_class' + str(poss) + '_seq_logo', seq_len, model_type)
            classes.append(classx_matrix_of_stds)

    else:
        if numericalbool:
            best = oh[d >= np.quantile(d, .9)]
            worst = oh[d < np.quantile(d, .1)]
            average = oh[[(x < np.quantile(d, .8) and x > np.quantile(d, 0.2)) for x in d]]

            best_matrix_of_stds = get_matrix(best, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'raw_data_best_seq_logo', seq_len, model_type)
            worst_matrix_of_stds = get_matrix(worst, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'raw_data_worst_seq_logo', seq_len, model_type)
            average_matrix_of_stds = get_matrix(average, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'raw_data_random_seq_logo', seq_len, model_type)
        else:
            possvals= list(set(d))
            classlabels = possvals
            for poss in possvals:
                truthvals = [d1 == poss for d1 in d]
                classx = oh[truthvals]
                classx_matrix_of_stds = get_matrix(classx, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'raw_data_class' + str(poss) + '_seq_logo', seq_len, model_type)
                classes.append(classx_matrix_of_stds)
            
    # plot mutagenesis
    import matplotlib
    from matplotlib import rc
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
    x = list(range(0, seq_len))
    palette = ['darkorange', 'grey', 'sandybrown', 'darkolivegreen', 'maroon', 'rosybrown', 'cornflowerblue','navy']

    # plot random
    if len(average_matrix_of_stds) > 0:
        means, conf_int_lower_bound, conf_int_upper_bound = get_means_and_bounds_of_stds(average_matrix_of_stds)
        plt.plot(x, means, 'v-', color='grey', label = 'Random', linewidth = 0.7, markeredgecolor = 'black',markersize=5, alpha = 0.8)
        plt.plot(x, means, '-', color = 'grey', linewidth = 0.7)
        plt.plot(x, conf_int_lower_bound, '--', color='grey', linewidth=0.5)
        plt.plot(x, conf_int_upper_bound, '--', color='grey', linewidth=0.5)
        ax.fill_between(x, conf_int_lower_bound, conf_int_upper_bound, color = 'grey', alpha = 0.2)

    # plot each class
    if len(classes) > 0:
        index = 0
        for classx in classes:
            means, conf_int_lower_bound, conf_int_upper_bound = get_means_and_bounds_of_stds(classx)
            plt.plot(x, means, '^-', color = palette[index % len(palette)], linewidth = 0.7, label = 'Class ' + str(classlabels[index]), markeredgecolor = 'black', markersize=5, alpha = 0.8)
            plt.plot(x, means, '-', color = palette[index % len(palette)], linewidth = 0.7)
            plt.plot(x, conf_int_lower_bound, '--', linewidth=0.5, color = palette[index % len(palette)])
            plt.plot(x, conf_int_upper_bound, '--', linewidth=0.5, color = palette[index % len(palette)])
            ax.fill_between(x, conf_int_lower_bound, conf_int_upper_bound, alpha = 0.2, color = palette[index % len(palette)])
            index = index + 1
    
    #plot worst
    if len(worst_matrix_of_stds) > 0:
        means, conf_int_lower_bound, conf_int_upper_bound = get_means_and_bounds_of_stds(worst_matrix_of_stds)
        plt.plot(x, means, '^-', color = 'tab:orange', linewidth = 0.7, label = 'Bottom 10%',markeredgecolor = 'black', markersize=5, alpha = 0.8)
        plt.plot(x, means, '-', color = 'tab:orange', linewidth = 0.7)
        plt.plot(x, conf_int_lower_bound, '--', linewidth=0.5, color = 'tab:orange')
        plt.plot(x, conf_int_upper_bound, '--', linewidth=0.5, color = 'tab:orange')
        ax.fill_between(x, conf_int_lower_bound, conf_int_upper_bound, color = 'tab:orange', alpha = 0.2)

    # plot best
    if len(best_matrix_of_stds) > 0:
        means, conf_int_lower_bound, conf_int_upper_bound = get_means_and_bounds_of_stds(best_matrix_of_stds)
        plt.plot(x, means, 'o-', label = 'Top 10%', linewidth = 0.7, markersize=5, markeredgecolor = 'black', color = 'tab:blue', alpha = 0.8)
        plt.plot(x, means, '-', color = 'tab:blue', linewidth = 0.7)
        plt.plot(x, conf_int_lower_bound, '--', linewidth=0.5,  color = 'tab:blue')
        plt.plot(x, conf_int_upper_bound, '--', linewidth=0.5,  color = 'tab:blue')
        ax.fill_between(x, conf_int_lower_bound, conf_int_upper_bound, color = 'tab:blue', alpha = 0.2)

    plt.legend(loc="upper left", markerscale = 1)
    ax.set_xlabel('Position', fontsize=20)
    ax.set_ylabel("Std Dev of Subunit Mismatch", fontsize=20)
    plt.tick_params(length = 10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(0, seq_len, 10))
    ax.set_xticklabels([str(x) for x in np.arange(0, seq_len, 10)], fontsize = 20)
    plt.tight_layout()
    plt.savefig(plot_path+final_model_name.split('.h5')[0] + plot_name)
    plt.savefig(plot_path+final_model_name.split('.h5')[0] + plot_name.split('.png')[0] + '.svg')

    print('In silico mutagenesis plot saved to ' + plot_path+final_model_name.split('.h5')[0] + plot_name)

############## PART 5: RAW SEQUENCE LOGO FUNCTIONS ##############

def plot_rawseqlogos(arr, fullalph, sequence_type, plot_path, plot_name, lenarr):
    """plots the raw experimental input sequences as sequence logos
    adopted from code from Valeri, Collins, Ramesh et al. Nature Communications 2020
    Parameters
    ----------
    arr : numpy array of sequences converted to one-hot encoded matrix inputs
    fullalph : list representation of alphabet
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    plot_path : str representing folder where plots are to be located
    plot_name : str representing name of plot
    len_arr : int representing length of a typical sequence

    Returns
    -------
    alph : list representation of (potentially trimmed-down) alphabet
    """ 

    # the columns are the alphabet letters and the rows are position-wise probabilities of 1-mer composiiton
    pfm = np.zeros((lenarr, len(fullalph)))
    for seq in arr:
        pfm = pfm + seq.T

    nn_df = pd.DataFrame(data = pfm)
    
    # additional culling needed for glycans - if not, length 1027 is too long
    if sequence_type == 'glycan':
        final_arr = pd.DataFrame(nn_df)
        sums = final_arr.sum(axis=0)
        sums = [s > np.quantile(sums, 0.90) for s in sums]
        if sum(sums) > 26 or sum(sums) == 0: # get the top 26 characters
            sums = list(final_arr.sum(axis=1))
            sums.sort(reverse=True)
            cutoff = sums[np.minimum(len(sums)-1,25)]
            sums = final_arr.sum(axis=1) # need to get original, unsorted list back
            sums = [s > cutoff for s in sums]
        alph = [i for (i, v) in zip(fullalph, sums) if v]
        actualalph = alph # so it doesn't get overwritten by the labels a, b, c, etc.
    
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from matplotlib.pyplot import cm
        colors = cm.turbo(np.linspace(0,1,len(alph)))
        legend_elements = []
        alphabet_string = string.ascii_lowercase + string.ascii_uppercase
        alphabet_list = list(alphabet_string)
        labels = []
        for i in range(len(alph)):
            labels.append(alphabet_list[i])
            legend_elements.append(Line2D([0], [0], marker='o', color=colors[i], label= alphabet_list[i] + ': ' + alph[i],
                                  markerfacecolor=colors[i], markersize=10))

        # create the figure
        import matplotlib
        from matplotlib import rc
        font = {'size'   : 20}
        matplotlib.rc('font', **font)
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        fig, ax = plt.subplots(figsize = (2,10), dpi = 300)
        plt.gca().axison = False
        ax.legend(handles=legend_elements, loc='center')
        plt.tight_layout()
        plt.savefig(plot_path + 'legend' + plot_name + '.png')
        plt.savefig(plot_path + 'legend' + plot_name + '.svg')

        nn_df.columns = fullalph # 1st row as the column names
        nn_df = nn_df[actualalph]
        nn_df.columns = labels # now make compatible with logomaker
    
    else:
        alph = fullalph 
        nn_df.columns = alph

    nn_df.index = range(len(nn_df))
    
    # create Logo object
    import matplotlib
    from matplotlib import rc
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (lenarr / 4,np.maximum(len(alph) / 3.5, 3)), dpi = 300)
    nn_df = nn_df.fillna(0)
    nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'probability')
    if sequence_type != 'glycan':
        if sequence_type == 'nucleic_acid':
            color_scheme = 'classic'
        elif sequence_type == 'protein':
            color_scheme = 'chemistry'
        nn_logo = logomaker.Logo(nn_df, ax = ax, color_scheme = color_scheme, stack_order = 'big_on_top', fade_below = True)
    else:
        nn_logo = logomaker.Logo(nn_df, ax = ax, color_scheme = dict(zip(alphabet_list, colors)), stack_order = 'big_on_top', fade_below = True)

    # style using Logo methods
    nn_logo.style_spines(visible = False)
    nn_logo.style_spines(spines = ['left'], visible = True)
    nn_logo.ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    nn_logo.ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'], fontsize = 20)
    nn_logo.ax.set_xticks(np.arange(0, lenarr, 10))
    nn_logo.ax.set_xticklabels([str(x) for x in np.arange(0, lenarr, 10)], fontsize = 20)
    nn_logo.ax.set_xlabel('Position', fontsize = 25)
    nn_logo.ax.set_ylabel('Probability', fontsize = 25)
    plt.tight_layout()
    plt.savefig(plot_path + plot_name + '.png')
    plt.savefig(plot_path + plot_name + '.svg')

    return(alph)

############## END OF FILE ##############