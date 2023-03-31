import scipy.stats as sp
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
import math

from generic_automl_classes import AutoMLBackend, convert_generic_input, read_in_data_file


# Data Processing Transfer Learning Helper Functions

def transform_classification_target(df_data_output, multiclass, do_auto_bin = True, bin_threshold = None): 
    """transform target values if desired
    Parameters
    ----------
    df_data_output : pandas Series object with target values
    multiclass : bool representing if multiclass classification or binary classification
    do_auto_bin : bool representing if target values should be automatically binned
    bin_threshold : float representing threshold for positive and negative classes
    Returns
    -------
    transformed_output : numpy array of transformed target values
    data_transformer : sklearn.preprocessing Scaler method
    """

    if multiclass:
        col = df_data_output.iloc[:,0]
        transformed_output = col.astype("category").cat.codes
        transformed_output = list(transformed_output)
        data_transformer = None
    else:
        if do_auto_bin: 
            # first, transform to the uniform distribution 
            #data_transformer = preprocessing.QuantileTransformer(random_state=0)
            data_transformer = preprocessing.RobustScaler()
            transformed_output = data_transformer.fit_transform(df_data_output.values.reshape(-1, 1))
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
                binned_output = (df_data_output > bin_threshold).astype(int)
            else: 
                # default bin_threshold = the median of the data (handle class skewing)
                bin_threshold = np.median(df_data_output.values)
                binned_output = (df_data_output > bin_threshold).astype(int)
            # pull out the values from the data frame vector 
            binned_output = binned_output.values
            # transform to categorical (need to reshape to match desired format)
            min_max_scaler = preprocessing.MinMaxScaler()
            transformed_output = to_categorical(min_max_scaler.fit_transform(binned_output.reshape(-1, 1)))
        
    return transformed_output,  data_transformer

def transform_regression_target(df_data_output): 
    """transform output vector to desired distribution
    Parameters
    ----------
    df_data_output : pandas Series object with target values

    Returns
    -------
    transformed_output : numpy array of transformed target values
    data_transformer : sklearn.preprocessing Scaler method
    """

    data_transformer = preprocessing.RobustScaler()
    transformed_output = data_transformer.fit_transform(df_data_output.values.reshape(-1, 1))
        
    return transformed_output,  data_transformer


# DeepSwarm Transfer Learning Helper Functions

def fit_final_deepswarm_model(model, task, num_epochs, X, y): 
    """trains deploy model using all data
    Parameters
    ----------
    model : tensorflow keras engine training model
    task : str, one of 'binary_classification', 'multiclass_classification', 'regression'
    num_epochs : int of number of epochs to train the final model
    X : numpy array of inputs
    y : numpy array of outputs
    
    Returns
    -------
    model : tensorflow keras engine training model
    """ 

    # NOTE: b/c deepswarm, need to run w/ custom data formatting and need to reformat deepswarm output file
    # similar to method used per fold 
    
    training_features = np.array(X)[:]
    training_target = y[:]
    
    if 'classification' in task:
        # Recreate the exact same model purely from the file
        compile_deepswarm_classification_model(model, 'multiclass' in task)
    else:
        compile_deepswarm_regression_model(model)
    
    # Train topology for N epochs in deepswarm
    validation_spit_init = 0.1
    callbacks_list = [EarlyStopping(monitor='val_loss', patience=math.ceil(num_epochs*0.1), verbose = False)]     # Callback to be used for checkpoint generation and early stopping
    model_history = model.fit(training_features, training_target, validation_split=validation_spit_init, epochs=num_epochs, callbacks=callbacks_list)
        
    return model 


def compile_deepswarm_classification_model(model, multiclass):
    """specifies optimizer parameters for binary classification and multiclass classification
    Parameters
    ----------
    model : tensorflow keras engine training model
    multiclass : bool representing if multiclass classification or binary classification
    
    Returns
    -------
    None
    """ 

    if multiclass:
        optimizer_parameters = {
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy', 'categorical_crossentropy'],
        }
    else:
        optimizer_parameters = {
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy', 'binary_crossentropy'],
        } 

    model.compile(**optimizer_parameters)

def compile_deepswarm_regression_model(model):
    """specifies optimizer parameters for regression
    Parameters
    ----------
    model : tensorflow keras engine training model

    Returns
    -------
    None
    """ 

    optimizer_parameters = {
        'optimizer': 'adam',
        'loss': 'mean_squared_error',
        'metrics': ['mean_squared_error'],
    }
    model.compile(**optimizer_parameters)

def compute_statistics(y_pred, y_true, cutoff_true, cutoff_pred):
    """helper function to calculate statistics on performance for model deployment and validation

    Parameters
    ----------
    y_pred : list of predicted values
    y_true : pandas DataFrame with true labels
    cutoff_true : value at which to binarize the true values as positive/negative; matters for auROC and MCC
    cutoff_pred : value at which to binarize the predicted values as positive/negative; matters for MCC
        
    Returns
    -------
    None 
    """
    
    y_true_bin = [1 if t > cutoff_true else 0 for t in y_true]
    y_pred_bin = [1 if t > cutoff_pred else 0 for t in y_pred]
    print('Number of labeled binary positives with cutoff of ' + str(np.round(cutoff_true,5)) + ': ' + str(sum(y_true_bin)))
    print('Number of total test set: ' + str(len(y_true_bin)))
    print('\nComputing statistics now...')
    slope, intercept, r_val, p_val, std_error = sp.linregress(y_true, y_pred)
    print('R2: ', np.round(r_val ** 2, 5), ' with a p-val of ', np.round(p_val, 5))
    pear = sp.pearsonr(y_true, y_pred)
    print('Pearson R: ', np.round(pear[0], 5) , ' with a p-val of ', np.round(pear[1], 5))
    spear = sp.spearmanr(y_true, y_pred)
    print('Spearman R: ', np.round(spear[0], 5) , ' with a p-val of ', np.round(spear[1], 5))
    print('auROC: ', np.round(sklearn.metrics.roc_auc_score(y_true_bin, y_pred), 5))
    print('MCC: ', np.round(sklearn.metrics.matthews_corrcoef(y_true_bin, y_pred_bin), 5))
    
def read_in_format_data_and_pred(task, data_folder, data_file, input_col, target_col, pad_seqs, augment_data, sequence_type, model_type, model_folder, output_folder, class_of_interest, cutoff_true = None, cutoff_pred = None, stats = True):
    """helper function to calculate statistics on performance for model deployment and validation

    Parameters
    ----------
    task : str, one of 'binary_classification', 'multiclass_classification', 'regression'
    data_folder : str representing folder where data is stored
    data_file : str representing file name where data is stored
    input_col : str representing input column name where sequences can be located
    target_col : str representing target column name where target values can be located
    pad_seqs : str indicating pad_seqs method, either 'max', 'min', 'average'
    augment_data : str, either 'none', 'complement', 'reverse_complement', or 'both_complements'
    sequence_type : str, either 'nucleic_acid', 'peptide', or 'glycan'
    model_type : str representing which AutoML search technique models should be interpreted, one of 'deepswarm', 'autokeras', 'tpot'
    model_folder : str representing folder where models are to be stored
    output_folder : str representing folder where output is to be stored
    class_of_interest : int corresponding to which class should be considered for predictions (for regression, must be 0)
    cutoff_true : value at which to binarize the true values as positive/negative; matters for auROC and MCC
    cutoff_pred : value at which to binarize the predicted values as positive/negative; matters for MCC
    stats : boolean, says whether to compute statistics or not  
    
    Returns
    -------
    y_pred : list of predicted values
    y_true : pandas DataFrame with true labels 

    """
    # allows user to validate model with data not in the original training set
    # so apply typical cleaning pipeline
    df_data_input, df_data_output, _ = read_in_data_file(data_folder + data_file, input_col, target_col)
    
    # format data inputs appropriately for autoML platform
    numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph = convert_generic_input(df_data_input, df_data_output, pad_seqs, augment_data, sequence_type, model_type = model_type)
    
    # get correct paths for models and outputs
    output_path_suffix = model_type + '/' + task + '/'

    if model_type == 'deepswarm':
        final_model_path = output_folder + output_path_suffix
        final_model_name = 'deepswarm_deploy_model.h5'
    if model_type == 'autokeras':   
        final_model_path = model_folder + output_path_suffix
        if 'classification' in task:
            final_model_name = 'optimized_autokeras_pipeline_classification.h5'
        else:
            final_model_name = 'optimized_autokeras_pipeline_regression.h5'
    if model_type == 'tpot':
        final_model_path = output_folder + output_path_suffix
        if 'classification' in task:
            final_model_name = 'final_model_tpot_classification.pkl'
        else:
            final_model_name = 'final_model_tpot_regression.pkl'
  
    preds = AutoMLBackend.generic_predict(oh_data_input, numerical_data_input, model_type, final_model_path, final_model_name)
    preddf = pd.DataFrame(preds)
    y_pred = preddf.iloc[:,class_of_interest]
    y_true = list(df_data_output.iloc[:,0])
    if stats:
        compute_statistics(y_pred, y_true, cutoff_true, cutoff_pred)
    return(y_pred, y_true)