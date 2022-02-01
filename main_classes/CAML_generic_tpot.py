#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from CAML_generic_autokeras import AutoMLClassifier, AutoMLRegressor
# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'main_classes/')

from CAML_generic_automl_classes import *


# In[2]:


# ## Import Libraries

# # General system libraries
import os
import sys
import shutil
import math
import pickle
import itertools
import numpy as np
import pandas as pd
from time import time

# Multiprocessing
import multiprocessing

# pysster Lib
from pysster import utils
from pysster.Data import Data
from pysster.One_Hot_Encoder import One_Hot_Encoder
from pysster.Alphabet_Encoder import Alphabet_Encoder
#     Installation:
#     cd to src/pysster folder in terminal and then execute: 
#     python setup.py install --user

# Import sklearn libs
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

# Math & Visualization Libs
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import Image

# Tensorflow libs
import tensorflow as tf 

# Import Keras
from keras import optimizers, applications, regularizers
from keras import backend as K
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
from CAML_interpret_helpers import plot_mutagenesis, plot_rawseqlogos
from CAML_integrated_design_helpers import integrated_design


# AutoKeras libs
#    Requires:  pip install autokeras
#               pip install apex

# Import TPOT libs
from tpot import TPOTClassifier, TPOTRegressor

#import apex

# Visualize graph
#     Requires: conda install -c conda-forge python-graphviz
from graphviz import Digraph

# Warnings
import warnings
warnings.filterwarnings("ignore")

#Visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:

# functions that can be called by both autokeras regression and classifiers
# [Function] Convert to one-hot encoding  
# Note: tpot needs input as numerical (i.e. index of onehot encoding)
#       vocab space = 4 nucleotides (or more general - i.e. 20 amino acids)
def convert_tpot_input(df_data_input, df_data_output, pad_seqs, augment_data, sequence_type): 
    df_data_input, df_data_output, scrambled_df_data_input, alph = AutoMLBackend.clean_input(df_data_input, df_data_output, pad_seqs, augment_data, sequence_type)

    
    oh_data_input, numerical_data_input = AutoMLBackend.onehot_seqlist(df_data_input, sequence_type, alph, model_type = 'tpot')
    scrambled_oh_data_input, scrambled_numerical_data_input = AutoMLBackend.onehot_seqlist(scrambled_df_data_input, sequence_type, alph, model_type = 'tpot')
    print('Confirmed: Scrambled control generated.')

    return numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph

# [Function] Reformat data in a local format 
# Note: This is to be consistent w/ predictions while keeping data consistent across folds
def reformat_data(input_data, output_data): 
    # updating the format of the data so it's clean for tpot (sep for train and test)
    data_input_feature_names = [('n_' + str(item)) for item in list(range(len(input_data[0])))]
    data_tpot = pd.DataFrame(input_data, columns=data_input_feature_names)
    target_tpot = pd.DataFrame(output_data)
    target_tpot.rename(columns={0: 'target'}, inplace=True)
    #tpot_data = pd.concat([data_tpot, target_tpot], axis=1, join_axes=[data_tpot.index])
    tpot_data = pd.concat([data_tpot, target_tpot], axis=1)
    
    # below is the normal processing done by 
    features = tpot_data.drop('target', axis=1).values
    targets = tpot_data['target'].values
    return features, targets

# [Function] Reformat data in train/test set
# Note: This is to be consistent w/ predictions while keeping data consistent across folds
def reformat_data_traintest(train_data, train_target, test_data=None, test_target=None):
    # reformat SEPARATELY for train and test to keep split consistent 
    training_features, training_target = reformat_data(train_data, train_target)
    if test_data is None: return training_features, training_target#if want to train on all data, just process training
    else: testing_features, testing_target = reformat_data(test_data, test_target)
    
    return training_features, training_target, testing_features, testing_target 


# [Function] Remove the data reading step to keep train/test split consistent w/in folds
# Note: Help on file editing from: https://stackoverflow.com/questions/4710067/using-python-for-deleting-a-specific-line-in-a-file
def clean_file(model_path, model_path_tpot,): 
    # first read in generated tpot model file
    with open(model_path, "r") as f:
        lines = f.readlines()
    # now, remove lines b/w the "NOTE" and "average" comments in the file (consistent across all files)
    write_line = True
    #model_path_tpot='./main_classes/new_classification.py'
    #model_path_tpot = model_path
    with open(model_path_tpot, "w") as f:
        for line in lines:
            if "# NOTE" in line: write_line = False # start deleting
            elif "# Average" in line: write_line = True 
            if write_line: 
                f.write(line)
                
    return model_path_tpot

# convert the cleaned file into a function to be called
def convert_to_function(model_path, is_classification=True): 
    with open(model_path, "r") as f:
        lines = f.readlines()
    # now, remove lines b/w the "NOTE" and "average" comments in the file (consistent across all files)
    if is_classification: model_path_tpot = './main_classes/new_classification.py'
    else: model_path_tpot = './main_classes/new_regression.py'
    with open(model_path_tpot, "w") as f:
        f.write('def run_best_tpot(training_features, training_target, testing_features): \n')
        for line in lines:
            f.write('\t' + line)
        f.write('\n\treturn exported_pipeline, results')
    return model_path_tpot

class TPOTClassification(AutoMLClassifier): 
    
    def __init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, sequence_type, do_auto_bin=True, bin_threshold=None, verbosity=0, population_size=None,num_generations=None, input_col = 'seq', target_col = 'target', pad_seqs = 'max', augment_data = 'none', multiclass = False, dataset_robustness = False, run_interpretation = True, interpret_params = {}, run_design = True, design_params = {}):
        AutoMLClassifier.__init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, do_auto_bin, bin_threshold, input_col, target_col)

        self.verbosity = verbosity 
        self.population_size = population_size
        self.num_generations=num_generations
        self.best_model = None 
        self.model_path=None
        self.pad_seqs = pad_seqs
        self.augment_data = augment_data
        self.sequence_type = sequence_type
        self.multiclass = multiclass
        self.dataset_robustness = dataset_robustness
        self.run_interpretation = run_interpretation
        self.interpret_params = interpret_params
        self.run_design = run_design
        self.design_params = design_params
        
    def convert_input(self): 
        return convert_tpot_input(self.df_data_input, self.df_data_output, self.pad_seqs, self.augment_data, self.sequence_type)
    
    def find_best_architecture(self, X, y):

        # Train on all of the data (no need to have a split)
        # define max run-time for autokeras, in terms of minutes (also user specified)
        # model directory to save models
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        #print('num gens and pop size:' self.num_generations +self.population_size)
        # train tpot
        tpot = TPOTClassifier(generations = self.num_generations, 
                             population_size = self.population_size,
                             n_jobs = -1,
                             max_time_mins = self.max_runtime,
                             max_eval_time_mins = self.max_runtime,
                             #config_dict = 'TPOT MDR',
                             periodic_checkpoint_folder = self.model_folder,
                             verbosity = self.verbosity)
        tpot.fit(np.array(X), y)

        # save pipeline
        # Define path to save TPOT optimized predictive model (.py)
        model_filename = "optimized_tpot_pipeline_classification.py"
        model_path = self.model_folder + model_filename
        tpot.export(model_path)
         
        # Save model path
        self.model_path=model_path

        return tpot
        
    def train_architecture_kfold(self, X, y, transform_obj, seed,alph): 
        
        # run kfold cross-validation over the best model from autokeras
        # model_path is the path of the best pipeline from autokeras

        # set-up k-fold system 
        # default num folds = 10 (can be specified by user)
        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=seed)

        # keep track of metrics per fold
        cv_scores = []
        # save predictions
        predictions = [] # in order of folds
        true_targets = [] # in order of folds
        compiled_seqs = [] 

        fold_count = 1 # keep track of the fold (when saving models)
        for train, test in kfold.split(X, y):
            
             # edit data format for the exported tpot pipeline 
            # BUT, keep split consistent across folds
            training_features, training_target, testing_features, testing_target = reformat_data_traintest(np.array(X)[train], np.array(y)[train], np.array(X)[test], np.array(y)[test])

            # get best model
            # need to read-in from outputted file and instantiate the pipeline in current cell
            # first, clean file (to keep data split consistent)
            updated_model_path = self.model_folder + 'optimized_tpot_pipeline_classification_updated.py' # consistent across all to run
            model_path_updated = clean_file(self.model_path, updated_model_path)
            model_path_updated = convert_to_function(updated_model_path)
            from new_classification import run_best_tpot
            expored_pipeline, results = run_best_tpot(training_features,training_target,testing_features)
            os.remove(updated_model_path) # delete the temporary updated file (clear for next fold)
            os.remove(model_path_updated)

            true_targets.extend(testing_target)
            predictions.extend(results) # keep y_true, y_pred  
            #print(np.array(X)[test])
            compiled_seqs.extend(AutoMLBackend.reverse_tpot2seq(np.array(X)[test], alph, self.sequence_type))

            # save metrics
            # use function from luis per fold
            resulting_metrics, metric_names = self.classification_performance_eval(self.output_folder+'tpot_', testing_target, np.array(results), str(fold_count))#np.expand_dims(results,1), str(fold_count))
            cv_scores.append(resulting_metrics)
            fold_count += 1         
           
        return np.array(cv_scores), predictions, true_targets,compiled_seqs

    # [Function] Train deploy model using all data
    def fit_final_model(self, X, y): 
        # fit on all of the data 
        # NOTE: b/c TPOT, need to run w/ custom data formatting and need to reformat TPOT output file
        # similar to method used per fold 

        # edit data format for the exported tpot pipeline 
        training_features, training_target = reformat_data_traintest(X, y)

        # get best model
        # need to read-in from outputted file and instantiate the pipeline in current cell
        # first, clean file (to keep data split consistent)
        updated_model_path = self.model_folder + 'optimized_tpot_pipeline_classification_updated.py' # consistent across all to run
        model_path_updated = clean_file(self.model_path, updated_model_path)
        model_path_updated = convert_to_function(updated_model_path)
        from new_classification import run_best_tpot
        # ignore results for final fit (need to pass something for testing features, but all data, so just pass training features and ignore results)
        exported_pipeline, _ = run_best_tpot(training_features,training_target,training_features)
        # now, the variable exported pipeline is the fitted model (save weights)
        # save in custom way to avoid annoyances later (if possible...)
        # use pickle: https://stackabuse.com/scikit-learn-save-and-restore-models/
        pickle_model_file = self.output_folder + 'final_model_tpot_classification.pkl'
        with open(pickle_model_file, 'wb') as file:  
            pickle.dump(exported_pipeline, file)
        # can load as follows: with open(pickle_model_file, 'rb') as file:  pickle_model = pickle.load(file)   
        # can then call .score or .predict, etc. with the returned pickle_model
        self.best_model = exported_pipeline
        return exported_pipeline                                                            
                                                                
    def run_system(self): 

        start1 = time()
        print('Conducting architecture search now...')

        with suppress_stdout():

            # if user has specified an output folder that doesn't exist, create it 
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder)

            # transform input
            numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph = self.convert_input()
            
            self.df_data_output = df_data_output
            # transform output (target) into bins for
            transformed_output, transform_obj = self.transform_target(self.multiclass)
                
            # now, we have completed the pre-processing needed to feed our data into tpot
            # autokeras input: numerical_data_input
            # autokeras output: transformed_output
            X = numerical_data_input
            y = transformed_output # don't convert to categorical for autokeras
            
            # ensure replicability
            seed = 7
            np.random.seed(seed) 

            # run autokeras to find best model
            self.find_best_architecture(X, y)

            # run kfold cv over the resulting pipeline
            cv_scores, compiled_preds, compiled_true,compiled_seqs = self.train_architecture_kfold(X, y, transform_obj,seed,alph)

            # if failed to execute properly, just stop rest of execution
            if cv_scores is None: return None, None, None

            # now, get the average scores (find avg metric and std, to show variability) across folds 
            avg_metric_folds = np.mean(cv_scores, axis = 0) # avg over columns 
            std_metric_folds = np.std(cv_scores, axis = 0) # avg over columns 
            cv_scores = cv_scores.transpose()

            # now get the compiled metric and generate an overall plot 
            compiled_metrics, metric_names = self.classification_performance_eval(self.output_folder+'tpot_', np.array(compiled_true), np.array(compiled_preds), file_tag = 'compiled')#np.expand_dims(compiled_preds,1), file_tag = 'compiled')

        print('Metrics over folds: ')
        for i, metric in enumerate(metric_names):
            print('\tAverage ' + metric + ': ', avg_metric_folds[i])
            print('\tStd of ' + metric + ': ', std_metric_folds[i])

        # write results to a text file for the user to read
        results_file_path = self.write_results(metric_names,avg_metric_folds, std_metric_folds, compiled_metrics, cv_scores, scrambled=False)
        
        end1 = time()
        runtime_stat_time = 'Elapsed time for autoML search : ' + str(np.round(((end1 - start1) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')
        start2 = time()

        print('Testing scrambled control now...')

        with suppress_stdout():
            if not os.path.isdir(self.output_folder + 'scrambled/'):
                os.mkdir(self.output_folder + 'scrambled/')

            # test scrambled control on best architecture
            scr_X = scrambled_numerical_data_input
            scr_cv_scores, scr_compiled_preds, scr_compiled_true, scr_compiled_seqs = self.train_architecture_kfold(scr_X, y, transform_obj,seed,alph)
            # now, get the average scores (find avg metric and std, to show variability) across folds 
            scr_avg_metric_folds = np.mean(scr_cv_scores, axis = 0) # avg over columns 
            scr_std_metric_folds = np.std(scr_cv_scores, axis = 0) # avg over columns 
            scr_cv_scores = scr_cv_scores.transpose()

            # now get the compiled metric and generate an overall plot 
            scr_compiled_metrics, _ = self.classification_performance_eval(self.output_folder+'scrambled/',np.array(scr_compiled_true), np.array(scr_compiled_preds), file_tag = 'compiled')#np.expand_dims(compiled_preds,1), file_tag = 'compiled')

        print('Scrambled metrics over folds: ')
        for i, metric in enumerate(metric_names):
            print('\tAverage ' + metric + ': ', scr_avg_metric_folds[i])
            print('\tStd of ' + metric + ': ', scr_std_metric_folds[i])

        # write results to a text file for the user to read
        scr_results_file_path = self.write_results(metric_names,scr_avg_metric_folds, scr_std_metric_folds, scr_compiled_metrics, scr_cv_scores, scrambled=True)
        
        end2 = time()
        runtime_stat_time = 'Elapsed time for scrambled control : ' + str(np.round(((end2 - start2) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # dataset robustness test
        if self.dataset_robustness:
            start3 = time()
            dataset_size = len(X)
            if not os.path.isdir(self.output_folder + 'robustness/'):
                os.mkdir(self.output_folder + 'robustness/')

            while dataset_size > 1000:
                dataset_size = int(dataset_size / 2)
                print("Testing with dataset size of: " + str(dataset_size))
                
                with suppress_stdout():

                    cv_scores, compiled_preds, compiled_true,compiled_seqs = self.train_architecture_kfold(X[0:dataset_size], y[0:dataset_size], transform_obj,seed,alph)
                
                    # now, get the average scores (find avg metric and std, to show variability) across folds 
                    avg_metric_folds = np.mean(cv_scores, axis = 0) # avg over columns 
                    std_metric_folds = np.std(cv_scores, axis = 0) # avg over columns 
                    cv_scores = cv_scores.transpose()

                    # now get the compiled metric and generate an overall plot 
                    compiled_metrics, metric_names = self.classification_performance_eval(self.output_folder+'robustness/' + str(dataset_size) + '_', np.array(compiled_true), np.array(compiled_preds), file_tag = 'compiled')
                    # write results to a text file for the user to read
                    results_file_path = self.write_results(metric_names,avg_metric_folds, std_metric_folds, compiled_metrics, cv_scores, scrambled=False, subset = str(dataset_size))

                    scr_cv_scores, scr_compiled_preds, scr_compiled_true, scr_compiled_seqs = self.train_architecture_kfold(scr_X[0:dataset_size], y[0:dataset_size], transform_obj,seed,alph)
                    # now, get the average scores (find avg metric and std, to show variability) across folds 
                    scr_avg_metric_folds = np.mean(scr_cv_scores, axis = 0) # avg over columns 
                    scr_std_metric_folds = np.std(scr_cv_scores, axis = 0) # avg over columns 
                    scr_cv_scores = scr_cv_scores.transpose()

                    # now get the compiled metric and generate an overall plot 
                    scr_compiled_metrics, _ = self.classification_performance_eval(self.output_folder+'robustness/scrambled_' + str(dataset_size) + '_',np.array(scr_compiled_true), np.array(scr_compiled_preds), file_tag = 'compiled')
                    # write results to a text file for the user to read
                    scr_results_file_path = self.write_results(metric_names, scr_avg_metric_folds, scr_std_metric_folds, scr_compiled_metrics, scr_cv_scores, scrambled=True, subset = str(dataset_size))
                
            end3 = time()
            runtime_stat_time = 'Elapsed time for data ablation experiment : ' + str(np.round(((end3 - start3) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # get predictions
        
        results_df = pd.DataFrame(np.array([compiled_seqs,np.array(compiled_true).flatten(), np.array(compiled_preds).flatten()]).T, columns=['Seqs','True','Preds'])
        results_df.to_csv(self.output_folder+'compiled_results_tpot_classification.csv')

        print('Fitting final model now...')
        with suppress_stdout():
            # now train final model using all of the data and save for user to run predictions on 
            final_model = self.fit_final_model(X, y)

        final_model_path = self.output_folder
        final_model_name = 'final_model_tpot_classification.pkl'

        numerical = []
        numericalbool = True
        for x in list(df_data_output.values):
            try:
                x = float(x)
                numerical.append(x)
            except Exception as e:
                numericalbool = False
                numerical = list(df_data_output.values.flatten())
                break

        if self.run_interpretation:
            start4 = time()
            # make folder
            if not os.path.isdir(self.output_folder + 'interpretation/'):
                os.mkdir(self.output_folder + 'interpretation/')

            # in silico mutagenesis     
            print("Generating in silico mutagenesis plots...")
            with suppress_stdout():
                plot_mutagenesis(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'interpretation/', '_mutagenesis.png', self.sequence_type, model_type = 'tpot', interpret_params = self.interpret_params)
            
            end4 = time()
            runtime_stat_time = 'Elapsed time for interpretation : ' + str(np.round(((end4 - start4) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt') 
        if self.run_design:
            start5  = time()
            # make folder
            if not os.path.isdir(self.output_folder + 'design/'):
                os.mkdir(self.output_folder + 'design/')

            print("Generating designed sequences...")
            with suppress_stdout():
                integrated_design(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'design/', '_design.png', self.sequence_type, model_type = 'tpot', design_params = self.design_params)
            
            end5 = time()
            runtime_stat_time = 'Elapsed time for design : ' + str(np.round(((end5 - start5) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # metrics are saved in a file (as are plots)
        # return final model
        end = time()
        runtime_stat_time = 'Elapsed time for total : ' + str(np.round(((end - start1) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')
        return final_model, [compiled_metrics, avg_metric_folds, std_metric_folds], transform_obj


# In[ ]:


class TPOTRegression(AutoMLRegressor): 
    
    def __init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, sequence_type, do_transform=True, verbosity=0, population_size=None,num_generations=None, input_col = 'seq', target_col = 'target', pad_seqs = 'max', augment_data = 'none', dataset_robustness = False, run_interpretation = True, interpret_params = {}, run_design = True, design_params = {}):
        AutoMLRegressor.__init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, do_transform, input_col, target_col)

        self.verbosity = verbosity 
        self.population_size = population_size
        self.num_generations=num_generations
        self.best_model = None 
        self.model_path=None
        self.pad_seqs = pad_seqs
        self.augment_data = augment_data
        self.sequence_type = sequence_type
        self.dataset_robustness = dataset_robustness
        self.run_interpretation = run_interpretation
        self.interpret_params = interpret_params
        self.run_design = run_design
        self.design_params = design_params
        
    def convert_input(self): 
        return convert_tpot_input(self.df_data_input, self.df_data_output, self.pad_seqs, self.augment_data, self.sequence_type)
    
    def find_best_architecture(self, X, y):

        # Train on all of the data (no need to have a split)
        # define max run-time for autokeras, in terms of minutes (also user specified)
        # model directory to save models
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)

        tpot = TPOTRegressor(generations = self.num_generations, 
                             population_size = self.population_size,
                             n_jobs = -1,
                             max_time_mins = self.max_runtime,
                             max_eval_time_mins = 10,
                             #config_dict = 'TPOT MDR',
                             periodic_checkpoint_folder = self.model_folder,
                             verbosity = self.verbosity)
        tpot.fit(np.array(X), y)

        # save pipeline
        # Define path to save TPOT optimized predictive model (.py)
        model_filename = "optimized_tpot_pipeline_regression.py"
        model_path = self.model_folder + model_filename
        tpot.export(model_path)
        self.model_path=model_path

        return tpot
        
        
    def train_architecture_kfold(self, X, y, transform_obj, seed,alph): 
        
        # run kfold cross-validation over the best model from autokeras
        # model_path is the path of the best pipeline from autokeras

        # set-up k-fold system 
        # default num folds = 10 (can be specified by user)
        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=seed)

        # keep track of metrics per fold
        cv_scores = []
        # save predictions
        predictions = [] # in order of folds
        true_targets = [] # in order of folds
        compiled_seqs = [] 

        fold_count = 1 # keep track of the fold (when saving models)
        
        for train, test in kfold.split(X, y):
            print('Current fold:', fold_count)

             # edit data format for the exported tpot pipeline 
            # BUT, keep split consistent across folds
            training_features, training_target, testing_features, testing_target = reformat_data_traintest(np.array(X)[train], y[train], np.array(X)[test], y[test])

            # get best model
            # need to read-in from outputted file and instantiate the pipeline in current cell
            # first, clean file (to keep data split consistent)
            updated_model_path = self.model_folder + 'optimized_tpot_pipeline_regression_updated.py' # consistent across all to run
            model_path_updated = clean_file(self.model_path, updated_model_path)
            model_path_updated = convert_to_function(updated_model_path,is_classification=False)
            from new_regression import run_best_tpot
            expored_pipeline, results = run_best_tpot(training_features,training_target,testing_features)
            os.remove(updated_model_path) # delete the temporary updated file (clear for next fold)
            os.remove(model_path_updated)
            
            compiled_seqs.extend(AutoMLBackend.reverse_tpot2seq(np.array(X)[test], alph, self.sequence_type))

            # need to inverse transform (if user wanted output to be transformed originally)
            if self.do_transform:
                testing_target_invtransf = transform_obj.inverse_transform(y[test].reshape(-1,1))
                results_invtransf = transform_obj.inverse_transform(results.reshape(-1,1))
                # reverse uniform transformation operation   
                true_targets.extend(testing_target_invtransf)
                predictions.extend(results_invtransf) # keep y_true, y_pred   
            else: 
                true_targets.extend(y[test])
                predictions.extend(results) # keep y_true, y_pred   

            # save metrics
            # use function from luis per fold
            if self.do_transform:resulting_metrics = self.regression_performance_eval(self.output_folder+'tpot_', np.array(testing_target_invtransf), np.array(results_invtransf), str(fold_count))
            else: resulting_metrics = self.regression_performance_eval(self.output_folder+'tpot_', np.expand_dims(testing_target,1), np.expand_dims(results,1), str(fold_count))
            cv_scores.append(resulting_metrics)
            fold_count += 1

        return np.array(cv_scores), predictions, true_targets,compiled_seqs
        
        
    # [Function] Train deploy model using all data
    def fit_final_model(self, X, y): 
        # get best model
        # need to read-in from outputted file and instantiate the pipeline in current cell
        # first, clean file (to keep data split consistent)
        # edit data format for the exported tpot pipeline 
        training_features, training_target = reformat_data_traintest(X, y)

        # get best model
        # need to read-in from outputted file and instantiate the pipeline in current cell
        # first, clean file (to keep data split consistent)
        updated_model_path = self.model_folder + 'optimized_tpot_pipeline_regression_updated.py' # consistent across all to run
        model_path_updated = clean_file(self.model_path, updated_model_path)
        model_path_updated = convert_to_function(updated_model_path, is_classification=False)
        from new_regression import run_best_tpot
        # ignore results for final fit (need to pass something for testing features, but all data, so just pass training features and ignore results)
        exported_pipeline, _ = run_best_tpot(training_features,training_target,training_features)
        # now, the variable exported pipeline is the fitted model (save weights)
        # save in custom way to avoid annoyances later (if possible...)
        # use pickle: https://stackabuse.com/scikit-learn-save-and-restore-models/
        pickle_model_file = self.output_folder + 'final_model_tpot_regression.pkl'
        with open(pickle_model_file, 'wb') as file:  
            pickle.dump(exported_pipeline, file)
        # can load as follows: with open(pickle_model_file, 'rb') as file:  pickle_model = pickle.load(file)   
        # can then call .score or .predict, etc. with the returned pickle_model
        self.best_model = exported_pipeline
        return exported_pipeline                                                           
                                                                
    def run_system(self): 

        start1 = time()
        print('Conducting architecture search now...')

        with suppress_stdout():

            # if user has specified an output folder that doesn't exist, create it 
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder)

            # transform input
            numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph = self.convert_input()
            transformed_output, transform_obj = self.transform_target()

            # now, we have completed the pre-processing needed to feed our data into tpot
            # autokeras input: oh_data_input
            # autokeras output: transformed_output
            X = numerical_data_input
            y = transformed_output # don't convert to categorical for tpot
            
            # ensure replicability
            seed = 7
            np.random.seed(seed)
            
            # run autokeras to find best model
            self.find_best_architecture(X, y)

            # run kfold cv over the resulting pipeline
            cv_scores, compiled_preds, compiled_true,compiled_seqs = self.train_architecture_kfold(X, y, transform_obj,seed,alph)

            # if failed to execute properly, just stop rest of execution
            if cv_scores is None: return None, None

            # now, get the average scores (find avg r2 and std, to show variability) across folds 
            _, _, avg_r2_folds, _, _ = np.mean(cv_scores, axis = 0) # avg over columns 
            _, _, std_r2_folds, _, _ = np.std(cv_scores, axis = 0) # avg over columns 
            cv_scores = cv_scores.transpose()

            # now get the compiled r2 and generate an overall plot 
            # now get the compiled r2 and generate an overall plot 
            if self.do_transform: 
                _, _, compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'tpot_', np.array(compiled_true), np.array(compiled_preds), file_tag='compiled')
            else:
                _, _, compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'tpot_', np.expand_dims(compiled_true,1), np.expand_dims(compiled_preds,1), file_tag='compiled')

        print('Metrics over folds: \n\tAverage r2: ', avg_r2_folds)
        print('\tStd of r2: ', std_r2_folds)
        if compiled_r2 != avg_r2_folds: 
            #print('Compiled r2 does not match avg over folds. Check for error')
            print('\tOverall r2: ' + str(compiled_r2) + ", Average r2 over folds: " + str(avg_r2_folds))

        # write results to a text file for the user to read
        results_file_path = self.write_results(avg_r2_folds, std_r2_folds, compiled_r2, cv_scores, scrambled=False)

        end1 = time()
        runtime_stat_time = 'Elapsed time for autoML search : ' + str(np.round(((end1 - start1) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')
        start2 = time()

        print('Testing scrambled control now...')
        with suppress_stdout():
            if not os.path.isdir(self.output_folder + 'scrambled/'):
                os.mkdir(self.output_folder + 'scrambled/')

            # test scrambled control on best architecture
            scr_X = scrambled_numerical_data_input
            scr_cv_scores, scr_compiled_preds, scr_compiled_true, scr_compiled_seqs = self.train_architecture_kfold(scr_X, y, transform_obj,seed,alph)
            # now, get the average scores (find avg r2 and std, to show variability) across folds 
            _, _, scr_avg_r2_folds, _, _ = np.mean(scr_cv_scores, axis = 0) # avg over columns 
            _, _, scr_std_r2_folds, _, _ = np.std(scr_cv_scores, axis = 0) # avg over columns 
            scr_cv_scores = scr_cv_scores.transpose()

            # now get the compiled metric and generate an overall plot 
            if self.do_transform:
                _, _, scr_compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'scrambled/', np.array(scr_compiled_true), np.array(scr_compiled_preds), file_tag='compiled')
            else:
                _, _, scr_compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'scrambled/', np.expand_dims(scr_compiled_true,1), np.expand_dims(scr_compiled_preds,1), file_tag='compiled')

        print('Scrambled metrics over folds: ')
        print('Metrics over folds: \n\tAverage r2: ', scr_avg_r2_folds)
        print('\tStd of r2: ', scr_std_r2_folds)

        # write results to a text file for the user to read
        scr_results_file_path = self.write_results(scr_avg_r2_folds, scr_std_r2_folds, scr_compiled_r2, scr_cv_scores, scrambled=True)
                
        end2 = time()
        runtime_stat_time = 'Elapsed time for scrambled control : ' + str(np.round(((end2 - start2) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # dataset robustness test
        if self.dataset_robustness:
            start3 = time()
            dataset_size = len(X)
            if not os.path.isdir(self.output_folder + 'robustness/'):
                os.mkdir(self.output_folder + 'robustness/')
            while dataset_size > 1000:
                dataset_size = int(dataset_size / 2)
                print("Testing with dataset size of: " + str(dataset_size))
                
                with suppress_stdout():

                    # reshape for deepswarm
                    smallX = np.array(X)[0:dataset_size]

                    # run kfold cv over the resulting pipeline
                    cv_scores, compiled_preds, compiled_true,compiled_seqs = self.train_architecture_kfold(smallX, y[0:dataset_size], transform_obj,seed,alph)

                    # now, get the average scores (find avg r2 and std, to show variability) across folds 
                    _, _, avg_r2_folds, _, _ = np.mean(cv_scores, axis = 0) # avg over columns 
                    _, _, std_r2_folds, _, _ = np.std(cv_scores, axis = 0) # avg over columns
                    cv_scores = cv_scores.transpose()

                    # now get the compiled r2 and generate an overall plot 
                    if self.do_transform: 
                        _, _, compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'robustness/' + str(dataset_size) + '_', np.array(compiled_true), np.array(compiled_preds), file_tag='compiled')
                    else:
                        _, _, compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'robustness/' + str(dataset_size) + '_', np.expand_dims(compiled_true,1), np.expand_dims(compiled_preds,1), file_tag='compiled')

                    # write results to a text file for the user to read
                    results_file_path = self.write_results(avg_r2_folds, std_r2_folds, compiled_r2, cv_scores, scrambled = False, subset = str(dataset_size))
                    
                    # test scrambled control on best architecture
                    smallscrX = np.array(scr_X)[0:dataset_size]

                    scr_cv_scores, scr_compiled_preds, scr_compiled_true, scr_compiled_seqs = self.train_architecture_kfold(smallscrX, y[0:dataset_size], transform_obj,seed,alph)

                    # now, get the average scores (find avg r2 and std, to show variability) across folds 
                    _, _, scr_avg_r2_folds, _, _ = np.mean(scr_cv_scores, axis = 0) # avg over columns 
                    _, _, scr_std_r2_folds, _, _ = np.std(scr_cv_scores, axis = 0) # avg over columns 
                    scr_cv_scores = scr_cv_scores.transpose()

                    # now get the compiled metric and generate an overall plot 
                    if self.do_transform:
                        _, _, scr_compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'robustness/scrambled_' + str(dataset_size) + '_', np.array(scr_compiled_true), np.array(scr_compiled_preds), file_tag='compiled')
                    else:
                        _, _, scr_compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'robustness/scrambled_' + str(dataset_size) + '_', np.expand_dims(scr_compiled_true,1), np.expand_dims(scr_compiled_preds,1), file_tag='compiled')

                    # write results to a text file for the user to read
                    scr_results_file_path = self.write_results(scr_avg_r2_folds, scr_std_r2_folds, scr_compiled_r2, scr_cv_scores, scrambled=True, subset = str(dataset_size))
                
            end3 = time()
            runtime_stat_time = 'Elapsed time for data ablation experiment : ' + str(np.round(((end3 - start3) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # get predictions
        results_df = pd.DataFrame(np.array([compiled_seqs,np.array(compiled_true).flatten(), np.array(compiled_preds).flatten()]).T, columns=['Seqs','True','Preds'])
        results_df.to_csv(self.output_folder+'compiled_results_tpot_reg.csv')

        print('Fitting final model now...')
        with suppress_stdout():
            # now train final model using all of the data and save for user to run predictions on 
            final_model = self.fit_final_model(X, y)

        final_model_path = self.output_folder
        final_model_name = 'final_model_tpot_regression.pkl'

        numerical = []
        numericalbool = True
        for x in list(df_data_output.values):
            try:
                x = float(x)
                numerical.append(x)
            except Exception as e:
                numericalbool = False
                numerical = list(df_data_output.values.flatten())
                break
        
        if self.run_interpretation:
            start4 = time()
            # make folder
            if not os.path.isdir(self.output_folder + 'interpretation/'):
                os.mkdir(self.output_folder + 'interpretation/')

            # in silico mutagenesis     
            print("Generating in silico mutagenesis plots...")
            with suppress_stdout():
                plot_mutagenesis(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'interpretation/', '_mutagenesis.png', self.sequence_type, model_type = 'tpot', interpret_params = self.interpret_params)

            end4 = time()
            runtime_stat_time = 'Elapsed time for interpretation : ' + str(np.round(((end4 - start4) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')
        if self.run_design:
            start5 = time()
            # make folder
            if not os.path.isdir(self.output_folder + 'design/'):
                os.mkdir(self.output_folder + 'design/')

            print("Generating designed sequences...")
            # class_of_interest must be zero for regression
            with suppress_stdout():
                integrated_design(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'design/', '_design.png', self.sequence_type, model_type = 'tpot', design_params = self.design_params)
            
            end5 = time()
            runtime_stat_time = 'Elapsed time for design : ' + str(np.round(((end5 - start5) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # metrics are saved in a file (as are plots)
        # return final model
        end = time()
        runtime_stat_time = 'Elapsed time for total : ' + str(np.round(((end - start1) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')
        return final_model, [compiled_r2, avg_r2_folds, std_r2_folds], transform_obj


# In[ ]:




