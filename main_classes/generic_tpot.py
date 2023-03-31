#!/usr/bin/env python
# coding: utf-8

############## PART 1: IMPORT STATEMENTS ##############

# import generic functions
import sys
sys.path.insert(1, 'main_classes/')
from generic_automl_classes import *
from interpret_helpers import plot_ft_importance, plot_mutagenesis, plot_rawseqlogos
from integrated_design_helpers import integrated_design

# import system libraries
import os
import sys
import shutil
import math
import pickle
import itertools
import numpy as np
import pandas as pd
from time import time
import multiprocessing
from graphviz import Digraph # Requires: conda install -c conda-forge python-graphviz
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")

# import sklearn libs
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

# import Tensorflow libs
import tensorflow as tf 

# import Keras libs
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

# import TPOT libs
from tpot import TPOTClassifier, TPOTRegressor

############## PART 2: FUNCTIONS THAT CAN BE CALLED BY BOTH TPOT REGRESSORS AND CLASSIFIERS ##############

def reformat_data(input_data, output_data): 
    """reformat data in a local format, this is to be consistent with predictions while keeping data consistent across folds
    Parameters
    ----------
    input_data : numpy array of inputs 
    output_data : numpy array of target values
    
    Returns
    -------
    features : numpy array of reformatted inputs
    targets : numpy array of reformatted outputs
    """ 

    # updating the format of the data so it's clean for tpot (sep for train and test)
    data_input_feature_names = [('n_' + str(item)) for item in list(range(len(input_data[0])))]
    data_tpot = pd.DataFrame(input_data, columns=data_input_feature_names)
    target_tpot = pd.DataFrame(output_data)
    target_tpot.rename(columns={0: 'target'}, inplace=True)
    tpot_data = pd.concat([data_tpot, target_tpot], axis=1)
    
    # below is the normal processing done by 
    features = tpot_data.drop('target', axis=1).values
    targets = tpot_data['target'].values

    return features, targets

def reformat_data_traintest(train_data, train_target, test_data=None, test_target=None):
    """reformat data in train/test set, using above reformat_data helper function
    Parameters
    ----------
    train_data : numpy array of train set inputs 
    train_target : numpy array of train set target values
    test_data : numpy array of test set inputs 
    test_target : numpy array of test set target values
    
    Returns
    -------
    training_features : numpy array of reformatted train set inputs
    training_target : numpy array of reformatted train set outputs
    testing_features : numpy array of reformatted test set inputs
    testing_target : numpy array of reformatted test set outputs
    """ 

    training_features, training_target = reformat_data(train_data, train_target)
    if test_data is None: return training_features, training_target#if want to train on all data, just process training
    else: testing_features, testing_target = reformat_data(test_data, test_target)
    
    return training_features, training_target, testing_features, testing_target 

def clean_file(model_path, model_path_tpot): 
    """remove the data reading step to keep train/test split consistent w/in folds
    note: help on file editing from: https://stackoverflow.com/questions/4710067/using-python-for-deleting-a-specific-line-in-a-file

    Parameters
    ----------
    model_path : str with the path to the model folder
    model_path_tpot : str with the full path + file name of model
    
    Returns
    -------
    model_path_tpot : str with the full path + file name of model
    """ 

    # first read in generated tpot model file
    with open(model_path, "r") as f:
        lines = f.readlines()
    # now, remove lines b/w the "NOTE" and "average" comments in the file (consistent across all files)
    write_line = True

    with open(model_path_tpot, "w") as f:
        for line in lines:
            if "# NOTE" in line: write_line = False # start deleting
            elif "# Average" in line: write_line = True 
            if write_line: 
                f.write(line)
                
    return model_path_tpot

def convert_to_function(model_path, is_classification=True): 
    """convert the cleaned file into a function to be called

    Parameters
    ----------
    model_path : str with the path to the model folder
    is_classification : bool representing classification or regression
    
    Returns
    -------
    model_path_tpot : str with the full path + file name of model
    """ 

    with open(model_path, "r") as f:
        lines = f.readlines()
    # now, remove lines b/w the "NOTE" and "average" comments in the file (consistent across all files)
    if is_classification: model_path_tpot = 'settings/tpot_classification.py'
    else: model_path_tpot = 'settings/tpot_regression.py'
    with open(model_path_tpot, "w") as f:
        f.write('def run_best_tpot(training_features, training_target, testing_features): \n')
        for line in lines:
            f.write('\t' + line)
        f.write('\n\treturn exported_pipeline, results')
        
    return model_path_tpot

############## PART 3: TPOT CLASSIFIER ##############

class TPOTClassification(AutoMLClassifier): 
    
    def __init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, sequence_type, do_auto_bin, bin_threshold, verbosity, population_size,num_generations, input_col, target_col, pad_seqs, augment_data, multiclass, dataset_robustness, run_interpretation, interpret_params, run_design, design_params):
        """constructor for TPOTClassification, child of AutoMLClassifier"""
        
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
        """uses convert_generic_input function to convert input"""
        return convert_generic_input(self.df_data_input, self.df_data_output, self.pad_seqs, self.augment_data, self.sequence_type, model_type = 'tpot')
 
    def find_best_architecture(self, X, y):
        """find best model architecture via TPOT
        Parameters
        ----------
        X : numpy array of inputs
        y : numpy array of outputs    
        
        Returns
        -------
        tpot : TPOTClassifier object
        """ 

        # model directory to save models
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)

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

        # save pipeline and define path to save TPOT optimized predictive model (.py)
        model_filename = "optimized_tpot_pipeline_classification.py"
        model_path = self.model_folder + model_filename
        tpot.export(model_path)
         
        # Save model path
        self.model_path=model_path

        return tpot
        
    def train_architecture_kfold(self, X, y, transform_obj, seed, alph): 
        """run kfold cross-validation over the best model from tpot

        Parameters
        ----------
        X : numpy array of inputs
        y : numpy array of outputs    
        transform_obj : sklearn.preprocessing Scaler method
        seed : int corresponding to random seed for reproducibility
        alph : list representation of alphabet

        Returns
        -------
        cv_scores : numpy array corresponding to cross-validation metrics
        predictions : list of predicted targets
        true_targets : list of ground-truth targets
        compiled_seqs : list of sequences in order of targets
        """ 
 
        # set-up k-fold system 
        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=seed)

        # keep track of metrics per fold
        cv_scores = []
        # save predictions
        predictions = [] # in order of folds
        true_targets = [] # in order of folds
        compiled_seqs = [] 

        fold_count = 1 # keep track of the fold (when saving models)
        for train, test in kfold.split(X, y):
            
             # edit data format for the exported tpot pipeline but keep split consistent across folds
            training_features, training_target, testing_features, testing_target = reformat_data_traintest(np.array(X)[train], np.array(y)[train], np.array(X)[test], np.array(y)[test])

            # get best model
            # need to read-in from outputted file and instantiate the pipeline in current cell
            # first, clean file (to keep data split consistent)
            updated_model_path = self.model_folder + 'optimized_tpot_pipeline_classification_updated.py' # consistent across all to run
            model_path_updated = clean_file(self.model_path, updated_model_path)
            model_path_updated = convert_to_function(updated_model_path)
            sys.path.append('settings/')
            from tpot_classification import run_best_tpot
            expored_pipeline, results = run_best_tpot(training_features,training_target,testing_features)
            os.remove(updated_model_path) # delete the temporary updated file (clear for next fold)
            os.remove(model_path_updated)

            true_targets.extend(testing_target)
            predictions.extend(results) # keep y_true, y_pred  
            compiled_seqs.extend(AutoMLBackend.reverse_tpot2seq(np.array(X)[test], alph, self.sequence_type))

            # save metrics
            resulting_metrics, metric_names = self.classification_performance_eval(self.output_folder+'tpot_', testing_target, np.array(results), str(fold_count), display = self.verbosity)#np.expand_dims(results,1), str(fold_count))
            cv_scores.append(resulting_metrics)
            fold_count += 1         
           
        return np.array(cv_scores), predictions, true_targets, compiled_seqs

    def fit_final_model(self, X, y): 
        """train deployable model using all data

        Parameters
        ----------
        X : numpy array of inputs
        y : numpy array of outputs    

        Returns
        -------
        exported_pipeline : TPOTClassifier object
        """ 

        # fit on all of the data , but note b/c TPOT, need to run w/ custom data formatting and need to reformat TPOT output file
        # similar to method used per fold 
        # edit data format for the exported tpot pipeline 
        training_features, training_target = reformat_data_traintest(X, y)

        # get best model
        # need to read-in from outputted file and instantiate the pipeline in current cell
        # first, clean file (to keep data split consistent)
        updated_model_path = self.model_folder + 'optimized_tpot_pipeline_classification_updated.py' # consistent across all to run
        model_path_updated = clean_file(self.model_path, updated_model_path)
        model_path_updated = convert_to_function(updated_model_path)
        sys.path.append('settings/')
        from tpot_classification import run_best_tpot
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
        """put end-to-end training + deployment together

        Parameters
        ----------
        None

        Returns
        -------
        None
        """ 

        start1 = time()
        print('Conducting architecture search now...')

        with suppress_stdout(self.verbosity):

            # if user has specified an output folder that doesn't exist, create it 
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder)

            # transform input
            numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph = self.convert_input()
            
            self.df_data_output = df_data_output
            # transform output (target) into bins
            transformed_output, transform_obj = self.transform_target(self.multiclass)
                
            # now, we have completed the pre-processing needed to feed our data into tpot
            # tpot input: numerical_data_input
            # tpot output: transformed_output
            X = numerical_data_input
            y = transformed_output # don't convert to categorical for tpot
            
            # ensure replicability
            seed = 7
            np.random.seed(seed) 

            # run tpot to find best model
            self.find_best_architecture(X, y)

            # run kfold cv over the resulting pipeline
            compiled_seqs, compiled_true, compiled_preds = self.bin_train_architecture_helper(X, y, transform_obj,seed,alph, output_folder_mod='tpot_')

            # get predictions
            results_df = pd.DataFrame(np.array([compiled_seqs,np.array(compiled_true).flatten(), np.array(compiled_preds).flatten()]).T, columns=['Seqs','True','Preds'])
            results_df.to_csv(self.output_folder+'compiled_results_tpot_classification.csv')
            
            end1 = time()
            runtime_stat_time = 'Elapsed time for autoML search : ' + str(np.round(((end1 - start1) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')
            start2 = time()

        print('Testing scrambled control now...')
        with suppress_stdout(self.verbosity):

            if not os.path.isdir(self.output_folder + 'scrambled/'):
                os.mkdir(self.output_folder + 'scrambled/')

            # test scrambled control on best architecture
            scr_X = scrambled_numerical_data_input
            _, _, _ = self.bin_train_architecture_helper(scr_X, y, transform_obj,seed,alph,output_folder_mod='scrambled/', scrambled=True)

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
            
                with suppress_stdout(self.verbosity):

                    _, _, _ = self.bin_train_architecture_helper(X[0:dataset_size], y[0:dataset_size], transform_obj,seed,alph,output_folder_mod='robustness/'+ str(dataset_size) + '_', scrambled=False, subset = str(dataset_size))

                    _, _, _ = self.bin_train_architecture_helper(scr_X[0:dataset_size], y[0:dataset_size], transform_obj,seed,alph,output_folder_mod='robustness/scrambled_'+ str(dataset_size) + '_', scrambled=True, subset = str(dataset_size))

            end3 = time()
            runtime_stat_time = 'Elapsed time for data ablation experiment : ' + str(np.round(((end3 - start3) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        print('Fitting final model now...')
        with suppress_stdout(self.verbosity):
            # now train final model using all of the data and save for user to run predictions on 
            final_model = self.fit_final_model(X, y)

        final_model_path = self.output_folder
        final_model_name = 'final_model_tpot_classification.pkl'
        
        # convert target values to numerical if applicable - used for interpretation and design modules
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

            print("Plotting feature importances...")
            with suppress_stdout(self.verbosity):
                plot_ft_importance(oh_data_input, final_model_path, final_model_name, self.output_folder + 'interpretation/', '_feature_importances.png')

            # in silico mutagenesis     
            print("Generating in silico mutagenesis plots...")
            with suppress_stdout(self.verbosity):
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
            with suppress_stdout(self.verbosity):
                integrated_design(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'design/', '_design.png', self.sequence_type, model_type = 'tpot', design_params = self.design_params)
            
            end5 = time()
            runtime_stat_time = 'Elapsed time for design : ' + str(np.round(((end5 - start5) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # metrics are saved in a file (as are plots)
        end = time()
        runtime_stat_time = 'Elapsed time for total : ' + str(np.round(((end - start1) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

############## PART 4: TPOT REGRESSOR ##############

class TPOTRegression(AutoMLRegressor): 
    
    def __init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, sequence_type, do_transform, verbosity, population_size,num_generations, input_col, target_col, pad_seqs, augment_data, dataset_robustness, run_interpretation, interpret_params, run_design, design_params):
        """constructor for TPOTRegression, child of AutoMLRegressor"""

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
        """uses convert_generic_input function to convert input"""
        return convert_generic_input(self.df_data_input, self.df_data_output, self.pad_seqs, self.augment_data, self.sequence_type, model_type = 'tpot')
 
    def find_best_architecture(self, X, y):
        """find best model architecture via TPOT
        Parameters
        ----------
        X : numpy array of inputs
        y : numpy array of outputs    
        
        Returns
        -------
        tpot : TPOTClassifier object
        """ 

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
        
    def train_architecture_kfold(self, X, y, transform_obj, seed, alph): 
        """run kfold cross-validation over the best model from tpot

        Parameters
        ----------
        X : numpy array of inputs
        y : numpy array of outputs    
        transform_obj : sklearn.preprocessing Scaler method
        seed : int corresponding to random seed for reproducibility
        alph : list representation of alphabet

        Returns
        -------
        cv_scores : numpy array corresponding to cross-validation metrics
        predictions : list of predicted targets
        true_targets : list of ground-truth targets
        compiled_seqs : list of sequences in order of targets
        """ 
        
        # set-up k-fold system 
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

            # edit data format for the exported tpot pipeline but keep split consistent across folds
            training_features, training_target, testing_features, testing_target = reformat_data_traintest(np.array(X)[train], y[train], np.array(X)[test], y[test])

            # get best model
            # need to read-in from outputted file and instantiate the pipeline in current cell
            # first, clean file (to keep data split consistent)
            updated_model_path = self.model_folder + 'optimized_tpot_pipeline_regression_updated.py' # consistent across all to run
            model_path_updated = clean_file(self.model_path, updated_model_path)
            model_path_updated = convert_to_function(updated_model_path,is_classification=False)
            sys.path.append('settings/')
            from tpot_regression import run_best_tpot
            expored_pipeline, results = run_best_tpot(training_features,training_target,testing_features)
            os.remove(updated_model_path) # delete the temporary updated file (clear for next fold)
            os.remove(model_path_updated)
            
            compiled_seqs.extend(AutoMLBackend.reverse_tpot2seq(np.array(X)[test], alph, self.sequence_type))

            # need to inverse transform (if user wanted output to be transformed originally)
            testing_target_invtransf = transform_obj.inverse_transform(y[test].reshape(-1,1))
            results_invtransf = transform_obj.inverse_transform(results.reshape(-1,1))
            # reverse uniform transformation operation   
            true_targets.extend(testing_target_invtransf)
            predictions.extend(results_invtransf) # keep y_true, y_pred   

            # save metrics
            resulting_metrics = self.regression_performance_eval(self.output_folder+'tpot_', np.array(testing_target_invtransf), np.array(results_invtransf), str(fold_count), display = self.verbosity)
            cv_scores.append(resulting_metrics)
            fold_count += 1

        return np.array(cv_scores), predictions, true_targets,compiled_seqs       
        
    def fit_final_model(self, X, y): 
        """train deployable model using all data

        Parameters
        ----------
        X : numpy array of inputs
        y : numpy array of outputs    

        Returns
        -------
        exported_pipeline : TPOTClassifier object
        """ 
        
        # get best model
        # need to read-in from outputted file and instantiate the pipeline in current cell
        # first, clean file (to keep data split consistent)
        # edit data format for the exported tpot pipeline 
        training_features, training_target = reformat_data_traintest(X, y)
        updated_model_path = self.model_folder + 'optimized_tpot_pipeline_regression_updated.py' # consistent across all to run
        model_path_updated = clean_file(self.model_path, updated_model_path)
        model_path_updated = convert_to_function(updated_model_path, is_classification=False)
        sys.path.append('settings/')
        from tpot_regression import run_best_tpot
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
        """put end-to-end training + deployment together

        Parameters
        ----------
        None

        Returns
        -------
        None
        """ 

        start1 = time()
        print('Conducting architecture search now...')

        with suppress_stdout(self.verbosity):

            # if user has specified an output folder that doesn't exist, create it 
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder)

            # transform input
            numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph = self.convert_input()
            self.df_data_output = df_data_output
            
            # transform output (target)
            if self.do_transform:
                transformed_output, transform_obj = self.transform_target()
            else: 
                transformed_output = np.array(df_data_output) # don't do any modification to specified target! 
                transform_obj = preprocessing.FunctionTransformer(func=None) # when func is None, identity transform applied

            # now, we have completed the pre-processing needed to feed our data into tpot
            # tpot input: numerical_data_input
            # tpot output: transformed_output
            X = numerical_data_input
            y = transformed_output # don't convert to categorical for tpot
            
            # ensure replicability
            seed = 7
            np.random.seed(seed)
            
            # run tpot to find best model
            self.find_best_architecture(X, y)

            # run kfold cv over the resulting pipeline
            compiled_seqs, compiled_true, compiled_preds = self.reg_train_architecture_helper(X, y, transform_obj,seed,alph, output_folder_mod='tpot_')

            # get predictions
            results_df = pd.DataFrame(np.array([compiled_seqs,np.array(compiled_true).flatten(), np.array(compiled_preds).flatten()]).T, columns=['Seqs','True','Preds'])
            results_df.to_csv(self.output_folder+'compiled_results_tpot_reg.csv')
            
            end1 = time()
            runtime_stat_time = 'Elapsed time for autoML search : ' + str(np.round(((end1 - start1) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')
            start2 = time()

        print('Testing scrambled control now...')
        with suppress_stdout(self.verbosity):
            if not os.path.isdir(self.output_folder + 'scrambled/'):
                os.mkdir(self.output_folder + 'scrambled/')

            # test scrambled control on best architecture
            scr_X = scrambled_numerical_data_input
            _, _, _ = self.reg_train_architecture_helper(scr_X, y, transform_obj,seed,alph, output_folder_mod='scrambled/', scrambled = True)

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

                with suppress_stdout(self.verbosity):
                    # reshape for tpot
                    smallX = np.array(X)[0:dataset_size]

                    # run kfold cv over the resulting pipeline
                    _, _, _ = self.reg_train_architecture_helper(smallX, y[0:dataset_size], transform_obj,seed,alph, output_folder_mod='robustness/' + str(dataset_size) + '_', scrambled = False, subset = str(dataset_size))

                    # test scrambled control on best architecture
                    smallscrX = np.array(scr_X)[0:dataset_size]

                    # test scrambled control on best architecture
                    _, _, _ = self.reg_train_architecture_helper(smallscrX, y[0:dataset_size], transform_obj,seed,alph, output_folder_mod='robustness/scrambled_' + str(dataset_size) + '_', scrambled = True, subset = str(dataset_size))

            end3 = time()
            runtime_stat_time = 'Elapsed time for data ablation experiment : ' + str(np.round(((end3 - start3) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        print('Fitting final model now...')
        with suppress_stdout(self.verbosity):
            # now train final model using all of the data and save for user to run predictions on 
            final_model = self.fit_final_model(X, y)

        final_model_path = self.output_folder
        final_model_name = 'final_model_tpot_regression.pkl'

        # convert target values to numerical if applicable - used for interpretation and design modules
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

            print("Plotting feature importances...")
            with suppress_stdout(self.verbosity):
                plot_ft_importance(oh_data_input, final_model_path, final_model_name, self.output_folder + 'interpretation/', '_feature_importances.png')

            # in silico mutagenesis     
            print("Generating in silico mutagenesis plots...")
            with suppress_stdout(self.verbosity):
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
            with suppress_stdout(self.verbosity):
                integrated_design(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'design/', '_design.png', self.sequence_type, model_type = 'tpot', design_params = self.design_params)
            
            end5 = time()
            runtime_stat_time = 'Elapsed time for design : ' + str(np.round(((end5 - start5) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # metrics are saved in a file (as are plots)
        end = time()
        runtime_stat_time = 'Elapsed time for total : ' + str(np.round(((end - start1) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

############## END OF FILE ##############