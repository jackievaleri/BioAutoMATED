# import statements 
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
from pysster.One_Hot_Encoder import One_Hot_Encoder

import keras as keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.layers import BatchNormalization
from scipy.stats import sem, t
from scipy import mean

import logomaker
import matplotlib
import matplotlib.font_manager as fm
import os
from itertools import combinations_with_replacement

sys.path.insert(1, '../main_classes/')
from CAML_interpret_helpers import plot_rawseqlogos, get_one_bp_mismatches, get_new_mismatch_seqs
from CAML_generic_automl_classes import AutoMLBackend
from CAML_generic_automl_classes import process_glycans, checkValidity, fill, makeComplement
from CAML_constraints_for_design_helpers import *
from CAML_seqprop_helpers import *

from sklearn.decomposition import PCA
import itertools

import random
from keras.layers import Layer, InputSpec
from tensorflow.python.framework import ops

# kmer design functions

def get_all_possible_kmers(k, alph, sequence_type):
    kmerlist = []
    if sequence_type != 'glycan':
        for combo in combinations_with_replacement(alph, r=k):
            kmerlist.append(''.join(combo))
    else:
        for combo in combinations_with_replacement(alph, r=k):
            kmerlist.append(list(combo))
    return(kmerlist)

def get_kmer_mismatches(seq, alph, sequence_type, k):
    replacements = get_all_possible_kmers(k, alph, sequence_type)
    mismatches = []
    for e,i in enumerate(seq):
        if e < len(seq) - k:
            for b in replacements:
                if sequence_type != 'glycan':
                    new_seq = seq[:e] + b + seq[e+k:]
                else:
                    new_seq = list(seq[:e])
                    new_seq.extend([b])
                    new_seq.extend(seq[e+k:])
                mismatches.append(new_seq)
    return mismatches

def get_list_of_k_mismatches_from_list(original_list, sequence_type, alph, k, constraints, substitution_type):
    
    final_list = []
    
    if 'random' in substitution_type:
        for i in range(k):
            for seq_list in [get_one_bp_mismatches(seq, alph, sequence_type) for seq in original_list]:
                for mismatch in seq_list:
                    final_list.append(mismatch)
            if 'constrained' in substitution_type:
                final_list = [enforce_bio_constraints(seq, constraints) for seq in final_list]
            if sequence_type != 'glycan':
                original_list = list(set(final_list)) # deduplicate
            else: # since lists are unhashable must deduplicate another way
                final_list = [tuple(i) for i in final_list] # deduplicate
                original_list = list(set(final_list))
                original_list = [list(i) for i in original_list]

    elif 'blocked' in substitution_type:
        for seq_list in [get_kmer_mismatches(seq, alph, sequence_type, k) for seq in original_list]:
            for mismatch in seq_list:
                final_list.append(mismatch)
        if 'constrained' in substitution_type:
            final_list = [enforce_bio_constraints(seq, constraints) for seq in final_list]
        if sequence_type != 'glycan':
            original_list = list(set(final_list)) # deduplicate
        else: # since lists are unhashable must deduplicate another way
            final_list = [tuple(i) for i in final_list] # deduplicate
            original_list = list(set(final_list))
            original_list = [list(i) for i in original_list]
    return original_list

def add_kmer_to_dataframe(predictions, data_df, class_of_interest, model_type = 'deepswarm'):
    if model_type == 'deepswarm':
        on_preds = predictions[:,int(class_of_interest)]
        
    if model_type == 'autokeras' or model_type == 'tpot':
        on_preds = predictions
    data_df.columns = ['seqs']
    data_df['preds'] = on_preds.flatten()
    return(data_df)


def check_generation_method(substitution_type, seq_len, alph, k, seq_thresh = 150000):
    valid_types = ['random', 'blocked', 'constrained_random', 'constrained_blocked']
    # from George
    if substitution_type not in valid_types: 
        print('substitution_type "{0}" is invalid. Using random substitution. Valid values are: {1}'.format(substitution_type, str(valid_types)))
        substitution_type = 'random'
    
    alphlen = len(alph)
    original_k = k
    if 'blocked' in substitution_type:
        guestimate_num_seqs = pow(alphlen, k) * (seq_len - 1) 
        print(k, guestimate_num_seqs)
        if guestimate_num_seqs > seq_thresh:
            
            while guestimate_num_seqs > seq_thresh:
                k = k-1
                guestimate_num_seqs = pow(alphlen, k) * (seq_len - 1)
                print(k, guestimate_num_seqs)
            print('k = {0} is too large for alphabet size and sequence length. Setting k = {1}'.format(original_k, k))
    elif 'random' in substitution_type:
        guestimate_num_seqs = pow(alphlen * seq_len, k)
        print(k, guestimate_num_seqs)
        if guestimate_num_seqs > seq_thresh:
            while guestimate_num_seqs > seq_thresh:
                k = k-1
                guestimate_num_seqs = pow(alphlen * seq_len, k)
                print(k, guestimate_num_seqs)
            print('k = {0} is too large for alphabet size and sequence length. Setting k = {1}'.format(original_k, k))
    return(k)

def get_denovo_seqs_predictions(listofseqs, num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, plot_name, seq_len, model_type, k, substitution_type, class_of_interest, constraints):
    
    if num_seqs_to_test > len(listofseqs):
        num_seqs_to_test = len(listofseqs)
    # sample lists
    listofseqs = list(listofseqs)
    sample = random.sample(listofseqs, num_seqs_to_test)
    
    sample = [np.transpose(a) for a in sample]
    # get mismatched lists
    orig = AutoMLBackend.reverse_onehot2seq(sample, alph, sequence_type, numeric = False)
    smallalph = plot_rawseqlogos(listofseqs, alph, sequence_type, plot_path, plot_name, seq_len)

    origseqs = get_list_of_k_mismatches_from_list(orig, sequence_type, smallalph, 0, constraints, substitution_type)
    mismatches = get_list_of_k_mismatches_from_list(orig, sequence_type, smallalph, k, constraints, substitution_type)

    X_num, X_oh = get_new_mismatch_seqs(mismatches, alph, sequence_type, model_type)
    X_orig_num, X_orig_oh = get_new_mismatch_seqs(origseqs, alph, sequence_type, model_type)
    
    predictions = AutoMLBackend.generic_predict(X_oh, X_num, model_type, final_model_path, final_model_name)
    orig_preds = AutoMLBackend.generic_predict(X_orig_oh, X_orig_num, model_type, final_model_path, final_model_name)

    mismatchdf = pd.DataFrame()
    mismatchdf[0] = mismatches
    run_thru_model = add_kmer_to_dataframe(predictions, mismatchdf, class_of_interest, model_type)
    
    origdf = pd.DataFrame()
    origdf[0] = origseqs
    orig_run_model = add_kmer_to_dataframe(orig_preds, origdf, class_of_interest, model_type)
    
    return(run_thru_model, orig_run_model)

# plotting helper functions

# https://www.codespeedy.com/similarity-metrics-of-strings-in-python/
def get_cosine(s1, s2):
    s3 = 0
    for i,j in zip(s1, s2):
        if i==j:
            s3 += 1
        else:
            s3 += 0
    s3 = s3/float(len(s1))
    return(s3) 

def compute_pairwise_distances(df, sequence_type, plot_path, plot_name, alph, seq_len, model_type):
    #colors = ['navy', 'cornflowerblue', 'skyblue','darkorange', 'sandybrown','rosybrown', 'maroon', 'grey', 'darkolivegreen']
    colors = ['navy', 'cornflowerblue', 'darkorange']
    mutated_seqs = {}
    original_seqs = {}
    storm_seqs = {}
    for typeseq, typedf in df.groupby('method'):
        
        if len(typedf) > 100: # can't do pairwise cosine on too many at once
            typedf = typedf[typedf['preds'] >= np.quantile(typedf['preds'], 0.9)]

        seqs = list(typedf['seqs'])
        dists = []
        
        for i in range(len(seqs) - 1):
            j = i + 1
            dist = get_cosine(seqs[i], seqs[j])
            dists.append(dist)
        if 'Mutated' in typeseq:
            mutated_seqs[typeseq] = dists
        elif 'STORM' in typeseq:
            storm_seqs[typeseq] = dists
        else:
            original_seqs[typeseq] = dists
           
    # plot denovo seqs
    index = 0
    fig, ax = plt.subplots(figsize = (6,5), dpi = 300)
    for n, h in mutated_seqs.items():
        plt.hist(h, alpha = 1, label = n, color = colors[index % len(colors)], histtype = 'step')
        index = index + 1
    plt.legend(loc="upper left", markerscale = 1)
    ax.set_xlabel('Pairwise Cosine Distance')
    ax.set_ylabel("Number of Seqs in Top 10%")

    plt.tick_params(length = 5)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_path + 'plot_denovo' + plot_name)
    plt.savefig(plot_path + 'plot_denovo' + plot_name + '.svg')
    plt.show()
    
    # plot storm seqs
    if len(storm_seqs) > 0:
        index = 0
        fig, ax = plt.subplots(figsize = (6,5), dpi = 300)
        for n, h in storm_seqs.items():
            plt.hist(h, alpha = 1, label = n, color = colors[index % len(colors)], histtype = 'step')
            index = index + 1
        plt.legend(loc="upper left", markerscale = 1)
        ax.set_xlabel('Pairwise Cosine Distance')
        ax.set_ylabel("Number of Seqs in Top 10%")

        plt.tick_params(length = 5)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.savefig(plot_path + 'plot_storm' + plot_name)
        plt.savefig(plot_path + 'plot_storm' + plot_name + '.svg')
        plt.show()
    
    # plot original seqs
    index = 0
    fig, ax = plt.subplots(figsize = (6,5), dpi = 300)
    for n, h in original_seqs.items():
        plt.hist(h, alpha = 1, label = n, color = colors[(index+len(mutated_seqs.keys())) % len(colors)], histtype = 'step')
        index = index + 1
    plt.legend(loc="upper left", markerscale = 1)
    ax.set_xlabel('Pairwise Cosine Distance')
    ax.set_ylabel("Number of Seqs in Top 10%")

    plt.tick_params(length = 5)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_path + 'plot_original_seqs_control' + plot_name)
    plt.savefig(plot_path + 'plot_original_seqs_control' + plot_name + '.svg')
    plt.show()
    
    for typeseq, typedf in df.groupby('method'):
        if 'Original' in typeseq:
            continue # we already plotted these
        seqs = list(typedf['seqs'])
        seqs, _ = AutoMLBackend.onehot_seqlist(seqs, sequence_type, alph, model_type)
        plot_rawseqlogos(seqs, alph, sequence_type, plot_path, 'seq_logo_' + typeseq, seq_len)

# storm design functions


def run_gradient_ascent(input_seq, original_out, num_samples, final_model_path, seq_len, alph, target, class_of_interest, bio_constraints, transform):
    
    # build generator network
    if bio_constraints is not None:
        _, seqprop_generator = build_generator(seq_length=seq_len, n_sequences=num_samples, batch_normalize_pwm=False,init_sequences = [input_seq],
                                              sequence_templates=[bio_constraints], pwm_transform_func=transform)# batch_normalize_pwm=True) # sequence_templates = bio_constraints
    else:
        _, seqprop_generator = build_generator(seq_length=seq_len, n_sequences=num_samples, batch_normalize_pwm=False,init_sequences = [input_seq],
                                              sequence_templates=None, pwm_transform_func=transform)
    # build predictor network and hook it on the generator PWM output tensor
    _, seqprop_predictor = build_predictor(seqprop_generator, load_saved_predictor(final_model_path, seq_len, alph), final_model_path, n_sequences=num_samples, eval_mode='pwm')
    
    
    #Build Loss Model (In: Generator seed, Out: Loss function)
    _, loss_model = build_loss_model(seqprop_predictor, loss_func_mod, target, class_of_interest, load_saved_predictor(final_model_path, seq_len, alph)) 
    
    
    #Specify Optimizer to use
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    #Compile Loss Model (Minimize self)
    loss_model.compile(loss=lambda true, pred: pred, optimizer=opt)

    #Fit Loss Model
    #seed_input = np.reshape([X[0]], [1,59,4,1]) # any input toehold to be modified

    callbacks =[
                EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=0, mode='auto'),
                #SeqPropMonitor(predictor=seqprop_predictor)#, plot_every_epoch=True, track_every_step=True, )#cse_start_pos=70, isoform_start=target_cut, isoform_end=target_cut+1, pwm_start=70-40, pwm_end=76+50, sequence_template=sequence_template, plot_pwm_indices=[0])
            ]

    num_epochs=50
    train_history = loss_model.fit([], [0], epochs=num_epochs, steps_per_epoch=1000, callbacks=callbacks)

    #Retrieve optimized PWMs and predicted (optimized) target
    _, optimized_pwm, optimized_onehot, predicted_out = seqprop_predictor.predict(x=None, steps=1)
    
    predicted_out = predicted_out[:,int(class_of_interest)]
    print('Original Predicted Value:', original_out)
    print('New Predicted Value: ', predicted_out)
   
    return optimized_pwm, optimized_onehot, predicted_out

# from https://nbviewer.org/github/johli/seqprop/blob/master/examples/basic/seqprop_basic_sequence_transform.ipynb

def get_revcomp_transform_opt(rev_constraints) :
            
    def transform_func(pwm) :     
        for con in rev_constraints:
            if con[0] == 'reverse_complement':
                start1 = con[2]
                end1 = con[3]
                start2 = con[4]
                end2 = con[5]
                list_of_pos = []
                # a[1::-1]   # the first two items, reversed
                # a[:-3:-1]  # the last two items, reversed
                pwm = K.concatenate([pwm[..., :start2, :, :], K.reverse(K.concatenate([pwm[..., start1:end1+1, 1::-1, :], pwm[..., start1:end1+1, :-3:-1, :]],axis = 2), axes = 1),  pwm[..., end2+1:, :, :]], axis=-3)
        
        return(pwm)
    
    return transform_func

# wrapper function
def get_storm_seqs_predictions(listofseqs, alph, final_model_path, final_model_name, plot_path, plot_name, seq_len, sequence_type, model_type, n_seqs_to_optimize, target_y, num_of_optimization_rounds, class_of_interest, constraints):
    '''
        input_dir: string path of directory containing data and constraint files
        file_name: name of .csv file containing sequences and values
        constraints_file_name: name of .csv, .xls, or .xlsx containing template constraints
        output_file_dir: string path of directory for optimized sequences file
        final_model_path: string path of the final trained model
        n_seqs_to_optimize: int number of the first n sequences to optimize from the data file
        input_col_name: string name of the column containing sequences in the data file
        num_samples: the integer number of samples being optimized at a time
        target_y: the float target y value the sequences are optimized toward
        num_of_optimization_rounds: int number of rounds of storm optimization
        sequence_type: string name of sequence type
        final_weights_path: (optional) string name of path to model weights if saved separately from model
    '''

    if model_type != 'deepswarm':
        return(pd.DataFrame(), None)
    
    set_alph(alph, sequence_type)
    
    if n_seqs_to_optimize > len(listofseqs):
        n_seqs_to_optimize = len(listofseqs)
    # sample lists
    listofseqs = list(listofseqs)
    sample = random.sample(listofseqs, n_seqs_to_optimize)
    sample = [np.transpose(a) for a in sample]
    seqs = AutoMLBackend.reverse_onehot2seq(sample, alph, sequence_type, numeric = False)

    # read biological constraint template
    if constraints is not None: 
        base_seq = ['N'] * seq_len
        rev_constraints = []
        for con in constraints:
            if con[0] == 'exact_sequence':
                spec = con[1]
                start = con[2]
                end = con[3]
                base_seq[start:end+1] = spec
            elif con[0] == 'reverse_complement':
                rev_constraints.append(con)
        if sequence_type != 'glycan':
            bio_constraints = ''.join(base_seq)
        transform = get_revcomp_transform_opt(rev_constraints)
    else:
        bio_constraints = None
        transform = None

    # turn seqs back into one hot and get initial predicted values
    oh_data_input, numerical_data_input = AutoMLBackend.onehot_seqlist(seqs, sequence_type, alph, model_type = model_type)
    storm_pred_y_vals = AutoMLBackend.generic_predict(oh_data_input, numerical_data_input, model_type, final_model_path, final_model_name)
    storm_pred_y_vals = storm_pred_y_vals[:,int(class_of_interest)]
    y = np.array(storm_pred_y_vals).astype(np.float32)
    
    # setting up gradient ascent
    target = [[target_y], ] # keep in this format in case you want to adapt for multiple optimization targets

    # running gradient ascent
    optimized_pwms = [] # store the probabilities
    optimized_seqs = [] # store the converted sequences to be tested 
    predicted_targets = [] # store the original and predicted target values 

    # run n optimization rounds for each sequence- part of STORM algorithm
    num_at_a_time = 1
    for i in range(0, num_of_optimization_rounds):
        for idx, (seq, original_out) in enumerate(zip(seqs, y)): 
            optimized_pwm, _, predicted_out = run_gradient_ascent(seq, original_out, num_at_a_time, final_model_path+final_model_name, seq_len, alph, target, class_of_interest, bio_constraints, transform)
            optimized_pwms.append(np.reshape(optimized_pwm, [seq_len, len(alph)]))
            predicted_targets.append(predicted_out)
            if model_type == 'autokeras' or model_type == 'deepswarm':
                new_seq = AutoMLBackend.reverse_onehot2seq(optimized_pwm, alph, sequence_type, numeric = False)
            else:
                new_seq = AutoMLBackend.reverse_tpot2seq(optimized_pwm, alph, sequence_type)
            optimized_seqs.append(new_seq[0])
               
    # create final output table
    data_df = pd.DataFrame()
    data_df['old_seqs'] = list(itertools.chain.from_iterable(itertools.repeat(x, num_of_optimization_rounds) for x in seqs))
    data_df['old_predicted_y'] = list(itertools.chain.from_iterable(itertools.repeat(x, num_of_optimization_rounds) for x in storm_pred_y_vals))
    data_df['new_seqs'] = optimized_seqs
    data_df['STORM_predicted_y'] = predicted_targets
    data_df['optimized_pwm'] = optimized_pwms

    # use original model (not STORM model with PWM) to get fairer estimate of value 
    oh_data_input, numerical_data_input = AutoMLBackend.onehot_seqlist(optimized_seqs, sequence_type, alph, model_type = model_type)
    storm_pred_y_vals = AutoMLBackend.generic_predict(oh_data_input, numerical_data_input, model_type, final_model_path, final_model_name)
    storm_pred_y_vals = storm_pred_y_vals[:,int(class_of_interest)]
    y = np.array(storm_pred_y_vals).astype(np.float32)
    data_df['orig_model_predicted_y'] = np.reshape(y, [n_seqs_to_optimize*num_of_optimization_rounds,])

    #culling to have the best out of all optimization rounds
    # sometimes the STORM predicted value is much less than the actual predicted value due to the model re-engineering 
    # as part of STORM; the pre-trained model is hooked onto a PWM sampler at the end followed by the predictor model
    best_seqs = pd.DataFrame()
    y_col = data_df.columns.get_loc("orig_model_predicted_y") # choosing best sequence after constraints applied
    
    # cull so we have just the best out of each n rounds
    for i in range(0, n_seqs_to_optimize):
        start = i * num_of_optimization_rounds
        end = start + num_of_optimization_rounds
        best_seq_so_far = data_df.iloc[start,:]
        for j in range(start+1, end):
            curr_seq = data_df.iloc[j,:]
            lastone = abs(data_df.iloc[start, y_col] - target_y)
            currone = abs(data_df.iloc[j, y_col] - target_y)
            if (currone < lastone):
                best_seq_so_far = curr_seq
        best_seqs = pd.concat([best_seqs, best_seq_so_far], axis = 1)
    best_seqs = best_seqs.transpose()
    best_seqs.to_csv(plot_path + 'sequences_STORM_' + plot_name + '_optimized_all_info.csv', index = False)
    
    best_seqs = best_seqs[['new_seqs', 'orig_model_predicted_y']]
    best_seqs.columns = ['seqs', 'preds']
    best_seqs['preds'] = [float(x) for x in list(best_seqs['preds'])] # had to change for violinplot input
    return best_seqs, optimized_pwms

def integrated_design(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, plot_path, plot_name, sequence_type, model_type, design_params):
    # defaults
    try:
        k = design_params['k']
    except:
        k = 1
    try:
        substitution_type = design_params['substitution_type']
    except:
        substitution_type = 'random'
    try:
        target_y = design_params['target_y']
    except:
        target_y = 1
    try: 
        class_of_interest = design_params['class_of_interest']
    except:
        class_of_interest = 1
    try: 
        constraint_file_path = design_params['constraint_file_path']
    except:
        constraint_file_path = ''
    try: 
        de_novo_num_seqs_to_test = design_params['de_novo_num_seqs_to_test']
    except:
        de_novo_num_seqs_to_test = 100
    try:
        storm_num_seqs_to_test = design_params['storm_num_seqs_to_test']    
    except:
        storm_num_seqs_to_test = 5
    try:
        num_of_optimization_rounds = design_params['num_of_optimization_rounds']
    except:
        num_of_optimization_rounds = 3

    d = list(numerical)
    howmanyclasses = len(list(set(d)))
    seq_len = len(list(oh_data_input[0][0]))
    oh = np.array(oh_data_input)

    print("Checking generation methods...")
    k = check_generation_method(substitution_type, seq_len, alph, k)
    print(k)
    
    print("Reading in biological constraints...")
    constraints = read_bio_constraints(constraint_file_path, alph, sequence_type)
            
    # get top, worst, average seqs
    classes = []
    classes_orig = []
    storm_classes = []
    classlabels = []
    avg_preds, best_preds, worst_preds = None, None, None
    if howmanyclasses < 2:
        avg_preds, orig_preds_avg = get_denovo_seqs_predictions(oh, de_novo_num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'seq_logo_average_original', seq_len, model_type, k, substitution_type, 0, constraints)
        storm_avg_preds, _ = get_storm_seqs_predictions(oh, alph, final_model_path, final_model_name, plot_path, 'average', seq_len, sequence_type, model_type, storm_num_seqs_to_test, target_y, num_of_optimization_rounds, 0, constraints)
    
    elif howmanyclasses < 3:
        possvals= list(set(d))
        classlabels = possvals
        for poss in possvals:
            truthvals = [d1 == poss for d1 in d]
            classx = oh[truthvals]
            classx_preds, classx_preds_orig = get_denovo_seqs_predictions(classx, de_novo_num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'seq_logo_class_' + str(poss) + '_original', seq_len, model_type, k, substitution_type, poss, constraints)
            storm_classx_preds, _ = get_storm_seqs_predictions(classx, alph, final_model_path, final_model_name, plot_path, 'class_' + str(poss), seq_len, sequence_type, model_type, storm_num_seqs_to_test, target_y, num_of_optimization_rounds, poss, constraints)
            classes.append(classx_preds)
            classes_orig.append(classx_preds_orig)
            storm_classes.append(storm_classx_preds)
    else:
        if numericalbool:
            best = oh[d >= np.quantile(d, .9)]
            worst = oh[d < np.quantile(d, .1)]
            average = oh[[(x < np.quantile(d, .8) and x > np.quantile(d, 0.2)) for x in d]]

            best_preds, best_preds_orig = get_denovo_seqs_predictions(best, de_novo_num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'seq_logo_best_original', seq_len, model_type, k, substitution_type, class_of_interest, constraints)
            worst_preds, worst_preds_orig = get_denovo_seqs_predictions(worst, de_novo_num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'seq_logo_worst_original', seq_len, model_type, k, substitution_type, class_of_interest, constraints)
            avg_preds, orig_preds_avg = get_denovo_seqs_predictions(average, de_novo_num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'seq_logo_average_original', seq_len, model_type, k, substitution_type, class_of_interest, constraints)
            storm_best_preds, _ = get_storm_seqs_predictions(best, alph, final_model_path, final_model_name, plot_path, 'best', seq_len, sequence_type, model_type, storm_num_seqs_to_test, target_y, num_of_optimization_rounds, class_of_interest, constraints)
            storm_worst_preds, _ = get_storm_seqs_predictions(worst, alph, final_model_path, final_model_name, plot_path, 'worst', seq_len, sequence_type, model_type, storm_num_seqs_to_test, target_y, num_of_optimization_rounds, class_of_interest, constraints)
            storm_avg_preds, _ = get_storm_seqs_predictions(average, alph, final_model_path, final_model_name, plot_path, 'average', seq_len, sequence_type, model_type, storm_num_seqs_to_test, target_y, num_of_optimization_rounds, class_of_interest, constraints)

        else:
            possvals= list(set(d))
            classlabels = possvals
            for poss in possvals:
                truthvals = [d1 == poss for d1 in d]
                classx = oh[truthvals]
                classx_preds, classx_preds_orig = get_denovo_seqs_predictions(classx, de_novo_num_seqs_to_test, alph, sequence_type, final_model_path, final_model_name, plot_path, 'seq_logo_class_' + str(poss) + '_original', seq_len, model_type, k, substitution_type, class_of_interest, constraints)
                storm_classx_preds, _ = get_storm_seqs_predictions(classx, alph, final_model_path, final_model_name, plot_path, 'class_' + str(poss), seq_len, sequence_type, model_type, storm_num_seqs_to_test, target_y, num_of_optimization_rounds, class_of_interest, constraints)
                classes.append(classx_preds)
                classes_orig.append(classx_preds_orig)
                storm_classes.append(storm_classx_preds)
            
    # plot
    print("Plotting now...")
    x = list(range(0, seq_len))
    numbins = int(de_novo_num_seqs_to_test / 5)
    dfplot = []
    top10dfplot = []

    # plot random
    if avg_preds is not None:
        plt.hist(orig_preds_avg['preds'], bins = numbins, alpha = 0.7, label = 'Original-Random')
        orig_preds_avg['method'] = ['Original-Random'] * len(orig_preds_avg)
        dfplot.append(orig_preds_avg)
        orig_preds_avg = orig_preds_avg[orig_preds_avg['preds'] >= np.quantile(orig_preds_avg['preds'], 0.9)]
        top10dfplot.append(orig_preds_avg)

        plt.hist(avg_preds['preds'], bins = numbins, alpha = 0.7, label = 'Mutated-Random')
        avg_preds['method'] = ['Mutated-Random'] * len(avg_preds)
        dfplot.append(avg_preds)
        avg_preds = avg_preds[avg_preds['preds'] >= np.quantile(avg_preds['preds'], 0.9)]
        top10dfplot.append(avg_preds)

        if model_type == 'deepswarm':
            plt.hist(storm_avg_preds['preds'], bins = numbins, alpha = 0.7, label = 'STORM-Random')
            storm_avg_preds['method'] = ['STORM-Random'] * len(storm_avg_preds)
            dfplot.append(storm_avg_preds)
            storm_avg_preds = storm_avg_preds[storm_avg_preds['preds'] >= np.quantile(storm_avg_preds['preds'], 0.9)]
            top10dfplot.append(storm_avg_preds)

    if len(classes) > 0:
        index = 0
        for classx in classes:
            classx_orig = classes_orig[index]
            plt.hist(classx_orig['preds'], bins = numbins, alpha = 0.7, label = 'Original-Class ' + str(classlabels[index]))
            classx_orig['method'] = ['Original-Class' + str(classlabels[index])] * len(classx_orig)
            dfplot.append(classx_orig)
            classx_orig = classx_orig[classx_orig['preds'] >= np.quantile(classx_orig['preds'], 0.9)]
            top10dfplot.append(classx_orig)
            
            #oh_vectors.extend(oh)
            #labels.extend(['random'] * len(oh))
            
            plt.hist(classx['preds'], bins = numbins, alpha = 0.7, label = 'Mutated-Class ' + str(classlabels[index]))
            classx['method'] = ['Mutated-Class' + str(classlabels[index])] * len(classx)
            dfplot.append(classx)
            classx = classx[classx['preds'] >= np.quantile(classx['preds'], 0.9)]
            top10dfplot.append(classx)
            
            if model_type == 'deepswarm':
                storm_classx = storm_classes[index]
                plt.hist(storm_classx['preds'], bins = numbins, alpha = 0.7, label = 'STORM-Class ' + str(classlabels[index]))
                storm_classx['method'] = ['STORM-Class' + str(classlabels[index])] * len(storm_classx)
                dfplot.append(storm_classx)
                storm_classx = storm_classx[storm_classx['preds'] >= np.quantile(storm_classx['preds'], 0.9)]
                top10dfplot.append(storm_classx)
            
            index = index + 1
    #plot worst
    if worst_preds is not None:
        plt.hist(worst_preds_orig['preds'], bins = numbins, alpha = 0.7, label = 'Original-Bottom 10%')
        worst_preds_orig['method'] = ['Original-Bottom 10%'] * len(worst_preds_orig)
        dfplot.append(worst_preds_orig)
        worst_preds_orig = worst_preds_orig[worst_preds_orig['preds'] >= np.quantile(worst_preds_orig['preds'], 0.9)]
        top10dfplot.append(worst_preds_orig)
        
        plt.hist(worst_preds['preds'], bins = numbins, alpha = 0.7, label = 'Mutated-Bottom 10%')
        worst_preds['method'] = ['Mutated-Bottom 10%'] * len(worst_preds)
        dfplot.append(worst_preds)
        worst_preds = worst_preds[worst_preds['preds'] >= np.quantile(worst_preds['preds'], 0.9)]
        top10dfplot.append(worst_preds)
        
        if model_type == 'deepswarm':
            plt.hist(storm_worst_preds['preds'], bins = numbins, alpha = 0.7, label = 'STORM-Bottom 10%')
            storm_worst_preds['method'] = ['STORM-Bottom 10%'] * len(storm_worst_preds)
            dfplot.append(storm_worst_preds)
            storm_worst_preds = storm_worst_preds[storm_worst_preds['preds'] >= np.quantile(storm_worst_preds['preds'], 0.9)]
            top10dfplot.append(storm_worst_preds)

    # plot best
    if best_preds is not None:
        plt.hist(best_preds_orig['preds'], bins = numbins, alpha = 0.7, label = 'Original-Top 10%')
        best_preds_orig['method'] = ['Original-Top 10%'] * len(best_preds_orig)
        dfplot.append(best_preds_orig)
        best_preds_orig = best_preds_orig[best_preds_orig['preds'] >= np.quantile(best_preds_orig['preds'], 0.9)]
        top10dfplot.append(best_preds_orig)
        
        plt.hist(best_preds['preds'], bins = numbins, alpha = 0.7, label = 'Mutated-Top 10%')
        best_preds['method'] = ['Mutated-Top 10%'] * len(best_preds)
        dfplot.append(best_preds)
        best_preds = best_preds[best_preds['preds'] >= np.quantile(best_preds['preds'], 0.9)]
        top10dfplot.append(best_preds)

        if model_type == 'deepswarm':
            plt.hist(storm_best_preds['preds'], bins = numbins, alpha = 0.7, label = 'STORM-Top 10%')
            storm_best_preds['method'] = ['STORM-Top 10%'] * len(storm_best_preds)
            dfplot.append(storm_best_preds)
            storm_best_preds = storm_best_preds[storm_best_preds['preds'] >= np.quantile(storm_best_preds['preds'], 0.9)]
            top10dfplot.append(storm_best_preds)

    
    # do at the end for more efficiency
    dfplot = pd.concat(dfplot)
    top10dfplot = pd.concat(top10dfplot)
    dfplot
           
    fig, ax = plt.subplots(figsize = (6,6), dpi = 300)
    palette = ['cornflowerblue', 'navy', 'darkorange']
    sns.boxplot(data = dfplot, x = 'method', y = 'preds', palette = palette, saturation = 0.6, width = 1, linewidth=0.5, fliersize=1)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)   
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Predictions')
    ax.set_xlabel('Method')
    plt.tight_layout()
    plt.savefig(plot_path + 'plot_all_seqs_violinplot' + plot_name)
    plt.savefig(plot_path + 'plot_all_seqs_violinplot' + plot_name + '.svg')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (6,6), dpi = 300)
    sns.boxplot(data = top10dfplot, x = 'method', y = 'preds', palette = palette, saturation = 0.6, width = 1, linewidth=0.5, fliersize=1)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.ylabel('Predictions')
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Predictions')
    ax.set_xlabel('Method')
    plt.tight_layout()
    plt.savefig(plot_path + 'plot_top_ten_percent_seqs_violinplot' + plot_name)
    plt.savefig(plot_path + 'plot_top_ten_percent_seqs_violinplot' + plot_name + '.svg')
    plt.show()
    
    # one hot encode the top10df
    mat = compute_pairwise_distances(dfplot, sequence_type, plot_path, '_cosine_dist.png', alph, seq_len, model_type)
    dfplot.to_csv(plot_path + 'sequences_all_methods.csv', index = False)
    top10dfplot.to_csv(plot_path + 'sequences_top_ten_percent_all_methods.csv', index = False)