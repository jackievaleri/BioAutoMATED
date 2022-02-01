from CAML_generic_automl_classes import AutoMLBackend
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
import keras
import math
import itertools

# some visualization imports
# used keras-vis from git (pip3 install git+https://github.com/raghakot/keras-vis.git)
from vis.visualization import visualize_saliency, visualize_activation
from vis.utils import utils
from keras import activations
import logomaker

from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.layers import BatchNormalization
import numpy as np
import logomaker
from scipy.stats import sem, t
from scipy import mean
import autokeras
import pickle


def plot_saliency_maps(numerical_data_input, oh_data_input, alph, final_model_path, final_model_name, plot_path, plot_name, sequence_type, interpret_params):
    # defaults
    try:
        sample_number_saliency_maps = interpret_params['sample_number_saliency_maps']
    except:
        sample_number_saliency_maps = 100
    try:
        saliency_map_grad_modifier = interpret_params['saliency_map_grad_modifier']
    except:
        saliency_map_grad_modifier = None  
    try:
        saliency_map_layer_index = interpret_params['saliency_map_layer_index']
    except:
        saliency_map_layer_index = -1


    X = numerical_data_input
    seq_len = len(numerical_data_input[0])
    
    # look at saliency map for positive vs. negative classes
    # modified code from keras-vis package github page 
    # https://github.com/raghakot/keras-vis/blob/master/examples/mnist/attention.ipynb
    
    # https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
    with CustomObjectScope({'GlorotUniform': glorot_uniform(), 'BatchNormalizationV1': BatchNormalization()}): # , 'BatchNormalizationV1': BatchNormalization()
        model = load_model(final_model_path+final_model_name)
        model.load_weights(final_model_path+final_model_name)

    # NOTE: This takes a long time!
    num_show = sample_number_saliency_maps
    grads = [] 
    # randomly draw samples (w/o replacement)
    rand_indices = np.random.choice(range(len(X)), num_show, replace=False)
    for idx in rand_indices:
        # layer idx = what layer to maximize/minimize (i.e. output node)
        # seed_input = the one-hot encoded sequence whose path thru the model we are tracing 
        # filter_indices = what layer are we computing saliency w/ r.t. (i.e. input layer)
        grad = visualize_saliency(model, layer_idx=saliency_map_layer_index, filter_indices=None, seed_input=X[idx], grad_modifier=saliency_map_grad_modifier)
        grads.append(grad)
        
    # look at average saliency based on original nucleotide
    grads_over_letters_on = {c: np.zeros(seq_len) for c in alph}
    counts_per_positon_on = {c: np.zeros(seq_len) for c in alph}
    
    #print(grads_over_letters_on[nt][position])
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
        if sum(sums) > 26:
            sums = [s > np.quantile(sums, 0.95) for s in sums] # 0.05 * 1027 (max) = 51
        if sum(sums) > 26:
            sums = [s > np.quantile(sums, 0.98) for s in sums]  # 0.02 * 1027 (max) = 21 
        alph = [i for (i, v) in zip(alph, sums) if v]
        final_arr = final_arr.iloc[sums,:]
        
    # Normalized by number of times a nt appeared at each position 
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

    #plt.title('Saliency', fontsize = 25)
    #plt.show()
    plotnamefinal = plot_path + final_model_name.split('.h5')[0]  + plot_name
    print('Saliency map saved to ' + plotnamefinal)
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    plt.savefig(plotnamefinal.split('.png')[0] + '.svg')

    return(final_arr, plot_path + final_model_name.split('.h5')[0], alph, seq_len)

def plot_seqlogos(arr, alph, sequence_type, plot_path, plot_name, lenarr):
    """ Plots the sequence activation from the trained model. Input is an activation matrix"""
    # plot the activation nucleotide akin to saliency using logomaker
    # the columns are the alphabet letters and the rows are position-wise relative activation
    
    if sequence_type == 'glycan':
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from matplotlib.pyplot import cm
        import string
        colors = cm.turbo(np.linspace(0,1,len(alph)))
        legend_elements = []
        alphabet_string = string.ascii_lowercase
        alphabet_list = list(alphabet_string)
        labels = []
        for i in range(len(alph)):
            labels.append(alphabet_list[i])
            legend_elements.append(Line2D([0], [0], marker='o', color=colors[i], label= alphabet_list[i] + ': ' + alph[i],
                                  markerfacecolor=colors[i], markersize=10))

        # Create the figure
        #print(legend_elements)
        fig, ax = plt.subplots(figsize = (2,10), dpi = 300)
        plt.gca().axison = False
        ax.legend(handles=legend_elements, loc='center')
        alph = labels
        plt.tight_layout()
        plt.savefig(plot_path + 'legend' + plot_name)
        plt.savefig(plot_path + 'legend' + plot_name.split('.png')[0] + '.svg')
        #plt.show()
    
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

    if 'saliency' in plot_name:
        nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'weight')
        #nn_df = logomaker.transform_matrix(nn_df, from_type = 'probability', to_type = 'weight')
        # do nothing for saliency - already in probability format
    else:
        nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'probability')

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

    #plt.show()
    plt.tight_layout()
    plt.savefig(plot_path + plot_name)
    plt.savefig(plot_path + plot_name.split('.png')[0] + '.svg')


def plot_activation_maps(numerical_data_input, oh_data_input, alph, final_model_path, final_model_name, plot_path, plot_name, sequence_type, interpret_params):
     # defaults
    try:
        sample_number_class_activation_maps = interpret_params['sample_number_class_activation_maps']
    except:
        sample_number_class_activation_maps = 100
    try:
        class_activation_grad_modifier = interpret_params['class_activation_grad_modifier']
    except:
        class_activation_grad_modifier = None  
    try:
        class_activation_layer_index = interpret_params['class_activation_layer_index']
    except:
        class_activation_layer_index = -1


    X = numerical_data_input
    seq_len = len(numerical_data_input[0])
    
    # look at saliency map for positive vs. negative classes
    # modified code from keras-vis package github page 
    # https://github.com/raghakot/keras-vis/blob/master/examples/mnist/attention.ipynb
    
    # https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
    with CustomObjectScope({'GlorotUniform': glorot_uniform(), 'BatchNormalizationV1': BatchNormalization()}): # BatchNormalization removed
        model = load_model(final_model_path+final_model_name)
        model.load_weights(final_model_path+final_model_name)

    # NOTE: This takes a long time!
    num_show = sample_number_class_activation_maps
    grads = [] 
    # randomly draw samples (w/o replacement)
    rand_indices = np.random.choice(range(len(X)), num_show, replace=False)
    for idx in rand_indices:
        grad = visualize_activation(model, layer_idx=class_activation_layer_index, filter_indices=None, seed_input=X[idx], grad_modifier=class_activation_grad_modifier)
        grads.append(grad)
        
    # look at average saliency based on original nucleotide
    grads_over_letters_on = {c: np.zeros(seq_len) for c in alph}
    counts_per_positon_on = {c: np.zeros(seq_len) for c in alph}

    #print(grads_over_letters_on[nt][position])
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
        #final_arr = grads[i]
        #final_arr = final_arr.reshape(final_arr.shape[1], final_arr.shape[0])
    final_arr = [grads_over_letters_on[letter]/counts_per_positon_on[letter] for letter in alph]
    
    # additional culling needed for glycans - if not, length 1027 is too long
    if sequence_type == 'glycan':
        final_arr = pd.DataFrame(final_arr)
        sums = final_arr.sum(axis=1)
        sums = [s > np.quantile(sums, 0.90) for s in sums]
        if sum(sums) > 26:
            sums = [s > np.quantile(sums, 0.95) for s in sums] # 0.05 * 1027 (max) = 51
        if sum(sums) > 26:
            sums = [s > np.quantile(sums, 0.98) for s in sums]  # 0.02 * 1027 (max) = 21 
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

    #plt.title('Saliency', fontsize = 25)
    #plt.show()
    plotnamefinal = plot_path + final_model_name.split('.h5')[0]  + plot_name
    print('Activation map saved to ' + plotnamefinal)
    plt.tight_layout()
    plt.savefig(plotnamefinal)
    plt.savefig(plotnamefinal.split('.png')[0] + '.svg')
    return(final_arr, plot_path + final_model_name.split('.h5')[0], alph, seq_len)


 # in silico mutagenesis
 # heavily borrowed code from https://codereview.stackexchange.com/questions/156490/finding-roughly-matching-genome-sequences-in-python-dictionary
# allegedly O(nk^2) where k = size of the sequence
def get_one_bp_mismatches(seq, smallalph, sequence_type):
    #print('small alph: ' + str(smallalph))
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
    
    final_list = []
    for seq_list in [get_one_bp_mismatches(seq, alph, sequence_type) for seq in original_list]:
        for mismatch in seq_list:
            final_list.append(mismatch)
           # if len(final_list) > 10:
           #     break
    return final_list

def get_new_mismatch_seqs(seq_list, alph, sequence_type, model_type='deepswarm'):
    oh_data_input = [] # conventional one-hot encoding
    numerical_data_input = [] # deepswarm-formatted data  
    
    for seq in seq_list:
        one_hot_seq = AutoMLBackend.one_hot_encode_seq(seq, sequence_type, alph)
        oh_data_input.append(np.transpose(one_hot_seq))               
        numerical_data_input.append(np.argmax((one_hot_seq), axis=1)) #To numerical for Deepswarm

    # reformat data so it can be treated as an "image" by deep swarm
    #if model_type == 'deeepswarm'
    if model_type != 'tpot':
        seq_data = np.array(oh_data_input)
        seq_data = seq_data.reshape(seq_data.shape[0], seq_data.shape[2], seq_data.shape[1])
    #print('Number of Samples: ', seq_data.shape[0]) 
    #print('Typical sequence shape: ', seq_data.shape[1:])
    
        # Deepswarm needs to expand dimensions for 2D CNN
        numerical_data_input = seq_data.reshape(seq_data.shape[0], seq_data.shape[1], seq_data.shape[2], 1)
    else: #TPOT NOTE *****
        numerical_data_input = np.array(numerical_data_input)
        #print(numerical_data_input.shape)
    return(numerical_data_input, oh_data_input)

def add_to_dataframe(predictions, data_df, model_type = 'deepswarm'):
    if model_type == 'deepswarm':
        on_preds = predictions[:,0] # technically do not have to use one class or another as long as we are consistent
        
    if model_type == 'autokeras' or model_type == 'tpot':
        on_preds = predictions
    data_df['preds'] = on_preds.flatten()
    #print(data_df)
    return(data_df)

# can we get the standard deviation across all of the base pairs for each of the 100 sequences (one bp at a time)
def get_std_dev_at_each_bp(df, smallalph, num_seqs_to_test):
   # print(df)
    seq_len = len(df.iloc[0,0]) # as an example, get seq len of the first sequence
    #print('seq len: ' + str(seq_len))
    hardcode_num_base_seqs = num_seqs_to_test
    alph_len = len(smallalph)
   # print('alph len: ' + str(alph_len))
    all_val_std_devs = np.zeros((hardcode_num_base_seqs, seq_len))
    curr_index_of_seqs = 0
   # print(all_ON_std_devs)
    for row in range(0, hardcode_num_base_seqs):
        for col in range(0, seq_len):
            val_for_current_seqs = df.iloc[curr_index_of_seqs:(curr_index_of_seqs+alph_len),1]
            val_std_dev = np.std(list(val_for_current_seqs))   
            all_val_std_devs[row, col] = val_std_dev           
            curr_index_of_seqs = curr_index_of_seqs + alph_len    
    return([all_val_std_devs])

def get_means_and_bounds_of_stds(matrix):
    
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

def plot_rawseqlogos(arr, fullalph, sequence_type, plot_path, plot_name, lenarr):
    """ Plots the sequence activation from the trained model. Input is an activation matrix"""
    # plot the activation nucleotide akin to saliency using logomaker
    # the columns are the alphabet letters and the rows are position-wise relative activation
      
    pfm = np.zeros((lenarr, len(fullalph)))
   # print('full alph: ' + str(len(fullalph)))
   # print('pfm: ' + str(pfm.shape))
   # print('seq: ' + str(arr))
    for seq in arr:
        pfm = pfm + seq.T

    nn_df = pd.DataFrame(data = pfm)
    #print(nn_df)
    
    # additional culling needed for glycans - if not, length 1027 is too long
    if sequence_type == 'glycan':
        final_arr = pd.DataFrame(nn_df)
        sums = final_arr.sum(axis=0)
        sums = [s > np.quantile(sums, 0.90) for s in sums]
        if sum(sums) > 26:
            sums = [s > np.quantile(sums, 0.95) for s in sums] # 0.05 * 1027 (max) = 51
        if sum(sums) > 26:
            sums = [s > np.quantile(sums, 0.98) for s in sums]  # 0.02 * 1027 (max) = 21 
        alph = [i for (i, v) in zip(fullalph, sums) if v]
        actualalph = alph # so it doesn't get overwritten by the labels a, b, c, etc.
        #print(alph)
    
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from matplotlib.pyplot import cm
        import string
        colors = cm.turbo(np.linspace(0,1,len(alph)))
        legend_elements = []
        alphabet_string = string.ascii_lowercase + string.ascii_uppercase
        alphabet_list = list(alphabet_string)
        labels = []
        for i in range(len(alph)):
            labels.append(alphabet_list[i])
            legend_elements.append(Line2D([0], [0], marker='o', color=colors[i], label= alphabet_list[i] + ': ' + alph[i],
                                  markerfacecolor=colors[i], markersize=10))

        # Create the figure
        #print(legend_elements)
        import matplotlib
        from matplotlib import rc
        font = {'size'   : 20}
        matplotlib.rc('font', **font)
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        fig, ax = plt.subplots(figsize = (2,10), dpi = 300)
        plt.gca().axison = False
        ax.legend(handles=legend_elements, loc='center')
        #alph = labels
        plt.tight_layout()
        plt.savefig(plot_path + 'legend' + plot_name + '.png')
        plt.savefig(plot_path + 'legend' + plot_name + '.svg')
        #plt.show()
        
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
    #print(nn_df)
    nn_df = logomaker.transform_matrix(nn_df, from_type = 'counts', to_type = 'probability')
    #print(nn_df)
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
    #plt.show()
    plt.tight_layout()
    plt.savefig(plot_path + plot_name + '.png')
    plt.savefig(plot_path + plot_name + '.svg')

    return(alph)

def plot_mutagenesis(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, plot_path, plot_name, sequence_type, model_type, interpret_params):
    try:
        num_seqs_to_test = interpret_params['sample_number_mutagenesis']
    except:
        num_seqs_to_test = 50
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
   # print(average_matrix_of_stds)
    if howmanyclasses < 2:
        average_matrix_of_stds = get_matrix(oh, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'raw_data_random_seq_logo', seq_len, model_type)

        
    elif howmanyclasses < 3:

        possvals= list(set(d))
       # print(possvals)
        classlabels = possvals
       # print('here we are')
        #print(d)
        for poss in possvals:
          #  print(poss)
            truthvals = [d1 == poss for d1 in d]
            classx = oh[truthvals]
            classx_matrix_of_stds = get_matrix(classx, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'raw_data_class' + str(poss) + '_seq_logo', seq_len, model_type)
            classes.append(classx_matrix_of_stds)
           # print('class ' + str(poss))
           # print(classx_matrix_of_stds)
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
            
    # plot

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

    if len(classes) > 0:
        index = 0
       # print(classes)
        for classx in classes:
            means, conf_int_lower_bound, conf_int_upper_bound = get_means_and_bounds_of_stds(classx)
            #print('Class ' + str(classlabels[index]))
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

    #ax.set_title('Predictions', fontsize = 30)
    #legend_prop = fm.FontProperties(fname=fpath, size = 30)
    plt.legend(loc="upper left", markerscale = 1)
    ax.set_xlabel('Position', fontsize=20)
    ax.set_ylabel("Std Dev of 1 Monomer Mismatch", fontsize=20)

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
    #plt.show()
