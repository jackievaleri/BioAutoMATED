#!/usr/bin/env python
# coding: utf-8

############## PART 1: IMPORT STATEMENTS ##############

# import generic functions
import sys
sys.path.insert(1, 'main_classes/')
from BioSeqAutoML_generic_automl_classes import *
from BioSeqAutoML_interpret_helpers import plot_mutagenesis, plot_rawseqlogos
from BioSeqAutoML_integrated_design_helpers import integrated_design

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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import Image
from graphviz import Digraph # requires: conda install -c conda-forge python-graphviz
import warnings
warnings.filterwarnings("ignore")

# import sklearn libs
from sklearn import preprocessing
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
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# import AutoKeras libs
from autokeras.image.image_supervised import ImageClassifier, ImageRegressor # may change to 1D later...
from autokeras.utils import pickle_from_file

############## PART 2: FUNCTIONS THAT CAN BE CALLED BY BOTH AUTOKERAS REGRESSORS AND CLASSIFIERS ##############
 
def convert_autokeras_input(df_data_input, df_data_output, pad_seqs, augment_data, sequence_type): 
    """converts inputs and outputs for autokeras and cleans all data
    Parameters
    ----------
    df_data_input : pandas Series object with sequences 
    df_data_output : pandas Series object with target values
    pad_seqs : str indicating pad_seqs method, if any
    augment_data : str indicating data augmentation method, if any 
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    
    Returns
    -------
    numerical_data_input : a numpy array of sequences converted to numerical inputs 
    oh_data_input : list of sequences converted to one-hot encoded matrix inputs 
    df_data_output : pandas DataFrame with target values
    scrambled_numerical_data_input : a numpy array of scrambled sequences converted to numerical inputs 
    scrambled_oh_data_input : list of scrambled sequences converted to one-hot encoded matrix inputs 
    alph : list representation of alphabet
    """
    
    df_data_input, df_data_output, scrambled_df_data_input, alph = AutoMLBackend.clean_input(df_data_input, df_data_output, pad_seqs, augment_data, sequence_type)

    
    oh_data_input, numerical_data_input = AutoMLBackend.onehot_seqlist(df_data_input, sequence_type, alph, model_type = 'autokeras')
    scrambled_oh_data_input, scrambled_numerical_data_input = AutoMLBackend.onehot_seqlist(scrambled_df_data_input, sequence_type, alph, model_type = 'autokeras')
    print('Confirmed: Scrambled control generated.')

    return numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph

def generate_autokeras_graph(model_folder, output_folder, graph_file_name): 
    """generate a graph to visualize the best model found
    NOTE: model_folder must be same as where the original model was specified 
    pull visualization directly from their github, rather than running the main file automatically
    see https://github.com/keras-team/autokeras/blob/c84eb77e2765ef0bc3899f6a9dcb880e30328326/examples/visualizations/visualize.py
    
    Parameters
    ----------
    model_folder : str representing model file path
    output_folder : str representing output file path
    graph_file_name : str representing output file name
    
    Returns
    -------
    None
    """ 

    def to_pdf(graph, path):
        dot = Digraph()

        for index, node in enumerate(graph.node_list):
            dot.node(str(index), str(node.shape))

        for u in range(graph.n_nodes):
            for v, layer_id in graph.adj_list[u]:
                dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))
        
        dot.node_attr['shape']='box'
        dot.render(path)

    def visualize(path):
        cnn_module = pickle_from_file(os.path.join(path, 'module'))
        cnn_module.searcher.path = path
        for item in cnn_module.searcher.history:
            model_id = item['model_id']
            graph = cnn_module.searcher.load_model_by_id(model_id)
            to_pdf(graph, os.path.join(output_folder, graph_file_name))

    # will generate a .pdf in the same folder as the best model 
    visualize(model_folder)

def fit_final_model(model, final_model_path, X, y): 
    """trains deploy model using all data
    Parameters
    ----------
    final_model_path : str indicating the path where the final model should be exported to
    X : numpy array of inputs
    y : numpy array of outputs
    
    Returns
    -------
    model : autokeras supervised ImageClassifier or ImageRegressor
    """ 

    # NOTE: b/c autokeras, need to run w/ custom data formatting and need to reformat autokeras output file
    # similar to method used per fold 
    
    train_size =0.90
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, np.array(y).astype(float),train_size=train_size, test_size = 1-train_size)
    model.final_fit(np.array(X_train_new),np.array(y_train_new), np.array(X_test_new), np.array(y_test_new), retrain=True)
    
    # now, the variable exported pipeline is the fitted model (save weights)
    # save as an autokeras model -- if load as such, can predict w/ new data 
    model.export_autokeras_model(final_model_path)

############## PART 3: AUTOKERAS CLASSIFIER ##############

class AutoKerasClassification(AutoMLClassifier): 
    
    def __init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, sequence_type, do_auto_bin, bin_threshold, verbosity, input_col, target_col, pad_seqs, augment_data, multiclass, dataset_robustness, run_interpretation, interpret_params, run_design, design_params):
        """constructor for AutoKerasClassification, child of AutoMLClassifier"""

        AutoMLClassifier.__init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, do_auto_bin, bin_threshold, input_col, target_col)
        self.verbosity = verbosity 
        self.best_model = None 
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
        """uses convert_autokeras_input function to convert input"""

        return convert_autokeras_input(self.df_data_input, self.df_data_output, self.pad_seqs, self.augment_data, self.sequence_type)
    
    def generate_graph(self): 
        """uses generate_autokeras_graph function to generate graph"""

        generate_autokeras_graph(self.model_folder, self.output_folder, 'final_autokeras_classification_graph')
    
    def find_best_architecture(self, X, y):
        """find best model architecture
        must define max run-time for autokeras, in terms of minutes (also user specified)

        Parameters
        ----------
        X : numpy array of inputs
        y : numpy array of outputs    
        
        Returns
        -------
        model : autokeras supervised ImageClassifier
        """ 

        # model directory to save models
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)

        max_runtime_sec = 60 * self.max_runtime # convert max time to sec 

        # train autokeras
        model = ImageClassifier(verbose=self.verbosity, path=self.model_folder)
        model.fit(x= X, y=y,time_limit=max_runtime_sec, )

        # generate a .pdf of the graph (i.e. what's the structure of the best model found)
        self.generate_graph()

        self.best_model = model

        return model # return the best architecture
        
    def train_architecture_kfold(self, X, y, transform_obj, seed, alph): 
        """run kfold cross-validation over the best model from autokeras

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

            self.best_model.final_fit(np.array(X)[train],np.array(y)[train],np.array(X)[test],np.array(y)[test],retrain=True)
            # evaluate and get predictions
            results = self.best_model.predict(np.array(X)[test])
            testing_target=np.array(y)[test]

            # need to inverse transform (if user wanted output to be transformed originally)
            true_targets.extend(testing_target)
            predictions.extend(results) # keep y_true, y_pred  

            currarr = np.array(X)[test]
            currarr = [np.transpose(a) for a in currarr]
            compiled_seqs.extend(AutoMLBackend.reverse_onehot2seq(currarr, alph, self.sequence_type, numeric = False))

            # save metrics
            resulting_metrics, metric_names = self.classification_performance_eval(self.output_folder+'autokeras_', np.array(y)[test], np.array(results), str(fold_count), display = self.verbosity)
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
        model : autokeras supervised ImageClassifier
        """ 

        # get best model
        # need to read-in from outputted file and instantiate the pipeline in current cell
        final_model_path = self.model_folder + 'optimized_autokeras_pipeline_classification.h5' # consistent across all to run
        model = fit_final_model(self.best_model, final_model_path, X, y)
        # can load as follows: model = pickle_from_file(final_model_path)
        self.best_model = model     

        return model # final fitted model                                                                 
                                                                
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
            # transform output (target) into bins for 
            transformed_output, transform_obj = self.transform_target(self.multiclass)

            # now, we have completed the pre-processing needed to feed our data into autokeras
            # autokeras input: oh_data_input
            # autokeras output: transformed_output
            X = oh_data_input
            y = transformed_output # don't convert to categorical for autokeras
            
            # ensure replicability
            seed = 7
            np.random.seed(seed)
            
            # run autokeras to find best model
            self.find_best_architecture(X, y)

            # run kfold cv over the resulting pipeline
            cv_scores, compiled_preds, compiled_true,compiled_seqs = self.train_architecture_kfold(X, y, transform_obj,seed,alph)

            # if failed to execute properly, just stop rest of execution
            if cv_scores is None: return None, None

            # now, get the average scores (find avg metric and std, to show variability) across folds 
            avg_metric_folds = np.mean(cv_scores, axis = 0) # avg over columns 
            std_metric_folds = np.std(cv_scores, axis = 0) # avg over columns 
            cv_scores = cv_scores.transpose()

            # now get the compiled metric and generate an overall plot 
            compiled_metrics, metric_names = self.classification_performance_eval(self.output_folder+'autokeras_', np.array(compiled_true), np.array(compiled_preds), file_tag = 'compiled', display = self.verbosity)#np.expand_dims(compiled_preds,1), file_tag = 'compiled')

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
        with suppress_stdout(self.verbosity):

            if not os.path.isdir(self.output_folder + 'scrambled/'):
                os.mkdir(self.output_folder + 'scrambled/')

            # test scrambled control on best architecture
            scr_X = scrambled_oh_data_input
            scr_cv_scores, scr_compiled_preds, scr_compiled_true, scr_compiled_seqs = self.train_architecture_kfold(scr_X, y, transform_obj,seed,alph)
            # now, get the average scores (find avg metric and std, to show variability) across folds 
            scr_avg_metric_folds = np.mean(scr_cv_scores, axis = 0) # avg over columns 
            scr_std_metric_folds = np.std(scr_cv_scores, axis = 0) # avg over columns 
            scr_cv_scores = scr_cv_scores.transpose()

            # now get the compiled metric and generate an overall plot 
            scr_compiled_metrics, _ = self.classification_performance_eval(self.output_folder+'scrambled/',np.array(scr_compiled_true), np.array(scr_compiled_preds), file_tag = 'compiled', display = self.verbosity)#np.expand_dims(compiled_preds,1), file_tag = 'compiled')

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
        
                with suppress_stdout(self.verbosity):

                    cv_scores, compiled_preds, compiled_true,compiled_seqs = self.train_architecture_kfold(X[0:dataset_size], y[0:dataset_size], transform_obj,seed,alph)
                
                    # now, get the average scores (find avg metric and std, to show variability) across folds 
                    avg_metric_folds = np.mean(cv_scores, axis = 0) # avg over columns 
                    std_metric_folds = np.std(cv_scores, axis = 0) # avg over columns 
                    cv_scores = cv_scores.transpose()

                    # now get the compiled metric and generate an overall plot 
                    compiled_metrics, metric_names = self.classification_performance_eval(self.output_folder+'robustness/' + str(dataset_size) + '_', np.array(compiled_true), np.array(compiled_preds), file_tag = 'compiled', display = self.verbosity)
                    # write results to a text file for the user to read
                    results_file_path = self.write_results(metric_names,avg_metric_folds, std_metric_folds, compiled_metrics, cv_scores, scrambled=False, subset = str(dataset_size))

                    scr_cv_scores, scr_compiled_preds, scr_compiled_true, scr_compiled_seqs = self.train_architecture_kfold(scr_X[0:dataset_size], y[0:dataset_size], transform_obj,seed,alph)
                    # now, get the average scores (find avg metric and std, to show variability) across folds 
                    scr_avg_metric_folds = np.mean(scr_cv_scores, axis = 0) # avg over columns 
                    scr_std_metric_folds = np.std(scr_cv_scores, axis = 0) # avg over columns 
                    scr_cv_scores = scr_cv_scores.transpose()

                    # now get the compiled metric and generate an overall plot 
                    scr_compiled_metrics, _ = self.classification_performance_eval(self.output_folder+'robustness/scrambled_' + str(dataset_size) + '_',np.array(scr_compiled_true), np.array(scr_compiled_preds), file_tag = 'compiled', display = self.verbosity)
                    # write results to a text file for the user to read
                    scr_results_file_path = self.write_results(metric_names, scr_avg_metric_folds, scr_std_metric_folds, scr_compiled_metrics, scr_cv_scores, scrambled=True, subset = str(dataset_size))

            end3 = time()
            runtime_stat_time = 'Elapsed time for data ablation experiment : ' + str(np.round(((end3 - start3) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # get predictions
        results_df = pd.DataFrame(np.array([compiled_seqs,np.array(compiled_true).flatten(), compiled_preds]).T, columns=['Seqs','True','Preds'])
        results_df.to_csv(self.output_folder+'compiled_results_autokeras_classification.csv')

        print('Fitting final model now...')
        # now train final model using all of the data and save for user to run predictions on 
        with suppress_stdout(self.verbosity):
            final_model = self.fit_final_model(X, y)

        final_model_path = self.model_folder
        final_model_name = 'optimized_autokeras_pipeline_classification.h5'
    
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

            # in silico mutagenesis     
            print("Generating in silico mutagenesis plots...")
            with suppress_stdout(self.verbosity):
                plot_mutagenesis(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'interpretation/', '_mutagenesis.png', self.sequence_type, model_type = 'autokeras', interpret_params = self.interpret_params)

            end4 = time()
            runtime_stat_time = 'Elapsed time for interpretation : ' + str(np.round(((end4 - start4) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        if self.run_design:
            start5 = time()
            # make folder
            if not os.path.isdir(self.output_folder + 'design/'):
                os.mkdir(self.output_folder + 'design/')

            print("Generating designed sequences...")
            with suppress_stdout(self.verbosity):
                integrated_design(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'design/', '_design.png', self.sequence_type, model_type = 'autokeras', design_params = self.design_params)
            end5 = time()
            runtime_stat_time = 'Elapsed time for design : ' + str(np.round(((end5 - start5) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # metrics are saved in a file (as are plots)
        end = time()
        runtime_stat_time = 'Elapsed time for total : ' + str(np.round(((end - start1) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')
        print('Results are located at:', results_file_path)

############## PART 4: AUTOKERAS REGRESSOR ##############

class AutoKerasRegression(AutoMLRegressor): 
    
    def __init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, sequence_type, do_transform, verbosity, input_col, target_col, pad_seqs, augment_data, dataset_robustness, run_interpretation, interpret_params, run_design, design_params):
        """constructor for AutoKerasRegression, child of AutoMLRegressor"""

        AutoMLRegressor.__init__(self, data_path, model_folder, output_folder, max_runtime, num_folds, do_transform, input_col, target_col)
        self.verbosity = verbosity 
        self.best_model = None 
        self.pad_seqs = pad_seqs
        self.augment_data = augment_data
        self.sequence_type = sequence_type
        self.dataset_robustness = dataset_robustness
        self.run_interpretation = run_interpretation
        self.interpret_params = interpret_params
        self.run_design = run_design
        self.design_params = design_params
        
    def convert_input(self): 
        """uses convert_autokeras_input function to convert input"""

        return convert_autokeras_input(self.df_data_input, self.df_data_output, self.pad_seqs, self.augment_data, self.sequence_type)
    
    def generate_graph(self): 
        """uses generate_autokeras_graph function to generate graph"""

        generate_autokeras_graph(self.model_folder, self.output_folder, 'final_autokeras_regression_graph')

    def find_best_architecture(self, X, y):
        """find best model architecture
        must define max run-time for autokeras, in terms of minutes (also user specified)

        Parameters
        ----------
        X : numpy array of inputs
        y : numpy array of outputs    
        
        Returns
        -------
        model : autokeras supervised ImageRegressor
        """ 

        # define max run-time for autokeras, in terms of minutes (also user specified)
        # model directory to save models
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)

        max_runtime_sec = 60 * self.max_runtime # convert max time to sec 

        # train autokeras
        model = ImageRegressor(verbose=self.verbosity, augment=False, path=self.model_folder,)
        model.fit(x= X, y=y,time_limit=max_runtime_sec, )

        # generate a .pdf of the graph (i.e. what's the structure of the best model found)
        self.generate_graph()

        self.best_model = model

        return model # return the best architecture
        
    def train_architecture_kfold(self, X, y, transform_obj, seed,alph): 
        """run kfold cross-validation over the best model from deepswarm

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

            # re-train (re-initialize weights) of best architecture from autokeras
            self.best_model.final_fit(np.array(X)[train],np.array(y)[train],np.array(X)[test],np.array(y)[test],retrain=True)

            # evaluate and get predictions
            results = self.best_model.predict(np.array(X)[test])
            currarr = np.array(X)[test]
            currarr = [np.transpose(a) for a in currarr]
            compiled_seqs.extend(AutoMLBackend.reverse_onehot2seq(currarr, alph, self.sequence_type, numeric = False))

            # need to inverse transform (if user wanted output to be transformed originally)
            testing_target_invtransf = transform_obj.inverse_transform(y[test].reshape(-1,1))
            results_invtransf = transform_obj.inverse_transform(results.reshape(-1,1))
            # reverse uniform transformation operation   
            true_targets.extend(testing_target_invtransf)
            predictions.extend(results_invtransf) # keep y_true, y_pred   
          
            # save metrics
            resulting_metrics = self.regression_performance_eval(self.output_folder+'autokeras_', np.array(testing_target_invtransf), np.array(results_invtransf), str(fold_count), display = self.verbosity)
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
        model : autokeras supervised ImageRegressor
        """ 

        # need to read-in from outputted file and instantiate the pipeline in current cell
        # first, clean file (to keep data split consistent)
        final_model_path = self.model_folder + 'optimized_autokeras_pipeline_regression.h5' # consistent across all to run
        model = fit_final_model(self.best_model, final_model_path, X, y)
        self.best_model = model        

        return model # final fitted model                                                                 
                                                                
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

            # now, we have completed the pre-processing needed to feed our data into autokeras
            # autokeras input: oh_data_input
            # autokeras output: transformed_output
            X = oh_data_input
            y = transformed_output # don't convert to categorical for autokeras
            
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
            _, _, compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'autokeras_', np.array(compiled_true), np.array(compiled_preds), file_tag='compiled', display = self.verbosity)

            print('Metrics over folds: \n\tAverage r2: ', avg_r2_folds)
            print('\tStd of r2: ', std_r2_folds)
            if compiled_r2 != avg_r2_folds: 
                # this does not indicate a problem with cross validation!
                print('\tOverall r2: ' + str(compiled_r2) + ", Average r2 over folds: " + str(avg_r2_folds))

            # write results to a text file for the user to read
            results_file_path = self.write_results(avg_r2_folds, std_r2_folds, compiled_r2, cv_scores, scrambled = False)
            
            end1 = time()
            runtime_stat_time = 'Elapsed time for autoML search : ' + str(np.round(((end1 - start1) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')
            start2 = time()

        print('Testing scrambled control now...')
        with suppress_stdout(self.verbosity):
            if not os.path.isdir(self.output_folder + 'scrambled/'):
                os.mkdir(self.output_folder + 'scrambled/')

            # test scrambled control on best architecture
            scr_X = scrambled_oh_data_input
            scr_cv_scores, scr_compiled_preds, scr_compiled_true, scr_compiled_seqs = self.train_architecture_kfold(scr_X, y, transform_obj,seed,alph)

            # now, get the average scores (find avg r2 and std, to show variability) across folds 
            _, _, scr_avg_r2_folds, _, _ = np.mean(scr_cv_scores, axis = 0) # avg over columns 
            _, _, scr_std_r2_folds, _, _ = np.std(scr_cv_scores, axis = 0) # avg over columns 
            scr_cv_scores = scr_cv_scores.transpose()

            # now get the compiled metric and generate an overall plot 
            _, _, scr_compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'scrambled/', np.array(scr_compiled_true), np.array(scr_compiled_preds), file_tag='compiled', display = self.verbosity)

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

                with suppress_stdout(self.verbosity):
                    # run kfold cv over the resulting pipeline
                    cv_scores, compiled_preds, compiled_true,compiled_seqs = self.train_architecture_kfold(X[0:dataset_size], y[0:dataset_size], transform_obj,seed,alph)

                    # now, get the average scores (find avg r2 and std, to show variability) across folds 
                    _, _, avg_r2_folds, _, _ = np.mean(cv_scores, axis = 0) # avg over columns 
                    _, _, std_r2_folds, _, _ = np.std(cv_scores, axis = 0) # avg over columns
                    cv_scores = cv_scores.transpose()

                    # now get the compiled r2 and generate an overall plot 
                    _, _, compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'robustness/' + str(dataset_size) + '_', np.array(compiled_true), np.array(compiled_preds), file_tag='compiled', display = self.verbosity)

                    # write results to a text file for the user to read
                    results_file_path = self.write_results(avg_r2_folds, std_r2_folds, compiled_r2, cv_scores, scrambled = False, subset = str(dataset_size))
                    
                    # test scrambled control on best architecture
                    scr_X = scrambled_oh_data_input
                    scr_cv_scores, scr_compiled_preds, scr_compiled_true, scr_compiled_seqs = self.train_architecture_kfold(scr_X[0:dataset_size], y[0:dataset_size], transform_obj,seed,alph)

                    # now, get the average scores (find avg r2 and std, to show variability) across folds 
                    _, _, scr_avg_r2_folds, _, _ = np.mean(scr_cv_scores, axis = 0) # avg over columns 
                    _, _, scr_std_r2_folds, _, _ = np.std(scr_cv_scores, axis = 0) # avg over columns 
                    scr_cv_scores = scr_cv_scores.transpose()

                    # now get the compiled metric and generate an overall plot 
                    _, _, scr_compiled_r2, _, _ = self.regression_performance_eval(self.output_folder +'robustness/scrambled_' + str(dataset_size) + '_', np.array(scr_compiled_true), np.array(scr_compiled_preds), file_tag='compiled', display = self.verbosity)

                    # write results to a text file for the user to read
                    scr_results_file_path = self.write_results(scr_avg_r2_folds, scr_std_r2_folds, scr_compiled_r2, scr_cv_scores, scrambled=True, subset = str(dataset_size))

            end3 = time()
            runtime_stat_time = 'Elapsed time for data ablation experiment : ' + str(np.round(((end3 - start3) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # get predictions
        results_df = pd.DataFrame(np.array([compiled_seqs,np.array(compiled_true).flatten(), np.array(compiled_preds).flatten()]).T, columns=['Seqs','True','Preds'])
        results_df.to_csv(self.output_folder+'compiled_results_autokeras_reg.csv')
        
        result_data = pd.DataFrame(zip(compiled_preds,compiled_true))

        print('Fitting final model now...')
        # now train final model using all of the data and save for user to run predictions on 
        with suppress_stdout(self.verbosity):
            final_model = self.fit_final_model(X, y)

        final_model_path = self.model_folder
        final_model_name = 'optimized_autokeras_pipeline_regression.h5'
  
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

            # in silico mutagenesis     
            print("Generating in silico mutagenesis plots...")           
            with suppress_stdout(self.verbosity):
                plot_mutagenesis(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'interpretation/', '_mutagenesis.png', self.sequence_type, model_type = 'autokeras', interpret_params = self.interpret_params)
        
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
                integrated_design(numerical_data_input, oh_data_input, alph, numerical, numericalbool, final_model_path, final_model_name, self.output_folder + 'design/', '_design.png', self.sequence_type, model_type = 'autokeras', design_params = self.design_params)
            
            end5 = time()
            runtime_stat_time = 'Elapsed time for design : ' + str(np.round(((end5 - start5) / 60), 2))  + ' minutes'
            AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

        # metrics are saved in a file (as are plots)
        # return final model
        end = time()
        runtime_stat_time = 'Elapsed time for total : ' + str(np.round(((end - start1) / 60), 2))  + ' minutes'
        AutoMLBackend.print_stats([runtime_stat_time], self.output_folder+ 'runtime_statistics.txt')

############## END OF FILE ##############