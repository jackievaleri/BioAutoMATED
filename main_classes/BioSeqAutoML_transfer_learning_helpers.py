from sklearn import preprocessing
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import math

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