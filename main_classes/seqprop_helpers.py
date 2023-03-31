#!/usr/bin/env python
# coding: utf-8

############## PART 1: IMPORT STATEMENTS ##############

# import system libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import itertools

# import keras libraries
import keras as keras
from keras.models import load_model
from keras.regularizers import l2
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute
from keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
import keras.losses

# import other libraries
from pysster.One_Hot_Encoder import One_Hot_Encoder
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.layers import Layer, InputSpec
from tensorflow.python.framework import ops

# import core libraries
from generic_automl_classes import AutoMLBackend

############## PART 2: RANDOM HELPER FUNCTIONS ##############

def set_alph(alphabet, sequence_type_curr):
    """helper function to define global alphabet
    Parameters
    ----------
    alphabet : list representation of alphabet
    sequence_type_curr : str, either 'nucleic_acid', 'peptide', or 'glycan'

    Returns
    -------
    None
    """

    global alph
    alph = alphabet
    global sequence_type
    sequence_type = sequence_type_curr

############## PART 3: SEQPROP GENERATOR FUNCTIONS ##############

#### Note, the following code will not have the same documentation as all other BioAutoMATED functions.
#### This is because these functions are pulled directly from the seqprop library by https://github.com/johli/seqprop
#### and documented in Bogard et al. Cell 2019. However, this library was only applicable for 4-nucleotide alphabets
#### Hence, the seqprop library has been extended to be alphabet- and sequence type-agnostic.
#### All changes are marked by the following comment : "#JV change for var alph" so that it is easy to find the changes.

class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#Stochastic Binarized Neuron helper functions (Tensorflow)
#ST Estimator code adopted from https://r2rt.com/beyond-binary-ternary-and-one-hot-neurons.html
#See Github https://github.com/spitis/

def st_sampled_softmax(logits):
    with ops.name_scope("STSampledSoftmax") as namescope :
        nt_probs = tf.nn.softmax(logits)
        onehot_dim = logits.get_shape().as_list()[1]
        sampled_onehot = tf.one_hot(tf.squeeze(tf.multinomial(logits, 1), 1), onehot_dim, 1.0, 0.0)
        with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
            return tf.ceil(sampled_onehot * nt_probs)

def st_hardmax_softmax(logits):
    with ops.name_scope("STHardmaxSoftmax") as namescope :
        nt_probs = tf.nn.softmax(logits)
        onehot_dim = logits.get_shape().as_list()[1]
        sampled_onehot = tf.one_hot(tf.argmax(nt_probs, 1), onehot_dim, 1.0, 0.0)
        with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
            return tf.ceil(sampled_onehot * nt_probs)

@ops.RegisterGradient("STMul") #JV change to integrate
def st_mul(op, grad):
    return [grad, grad]

def st_sampled(logits):
    with ops.name_scope("STSampled") as namescope :
        #nt_probs = tf.nn.softmax(logits)
        onehot_dim = logits.get_shape().as_list()[1]
        sampled_onehot = tf.one_hot(tf.squeeze(tf.multinomial(logits, 1), 1), onehot_dim, 1.0, 0.0)
        with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul', 'Softmax' : 'Identity'}):
            return tf.ceil(sampled_onehot * tf.nn.softmax(logits))

#PWM Masking and Sampling helper functions

def mask_pwm(inputs) :
    pwm, onehot_template, onehot_mask = inputs

    return pwm * onehot_mask + onehot_template

def sample_pwm_only(pwm_logits) :
    n_sequences = pwm_logits.get_shape().as_list()[0]
    seq_length = pwm_logits.get_shape().as_list()[1]
    
    flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, len(alph))) #JV change for var alph
    sampled_pwm = st_sampled_softmax(flat_pwm)

    return K.reshape(sampled_pwm, (n_sequences, seq_length, len(alph), 1)) #JV change for var alph

def sample_pwm_simple(pwm_logits) :
    n_sequences = pwm_logits.get_shape().as_list()[0]
    seq_length = pwm_logits.get_shape().as_list()[1]
    
    flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, len(alph))) #JV change for var alph
    sampled_pwm = st_sampled(flat_pwm)

    return K.reshape(sampled_pwm, (n_sequences, seq_length, len(alph), 1)) #JV change for var alph

def sample_pwm(pwm_logits) :
    n_sequences = pwm_logits.get_shape().as_list()[0]
    seq_length = pwm_logits.get_shape().as_list()[1]
    
    flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, len(alph))) #JV change for var alph
    sampled_pwm = K.switch(K.learning_phase(), st_sampled_softmax(flat_pwm), st_hardmax_softmax(flat_pwm))

    return K.reshape(sampled_pwm, (n_sequences, seq_length, len(alph), 1)) #JV change for var alph

def max_pwm(pwm_logits) :
    n_sequences = pwm_logits.get_shape().as_list()[0]
    seq_length = pwm_logits.get_shape().as_list()[1]
    
    flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, len(alph))) #JV change for var alph
    sampled_pwm = st_hardmax_softmax(flat_pwm)

    return K.reshape(sampled_pwm, (n_sequences, seq_length, len(alph), 1)) #JV change for var alph

#Gumbel-Softmax (The Concrete Distribution) for annealed nucleotide sampling

def gumbel_softmax(logits, temperature=0.1) :
    gumbel_dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
    batch_dim = logits.get_shape().as_list()[0]
    onehot_dim = logits.get_shape().as_list()[1]
    return gumbel_dist.sample()

def sample_gumbel(pwm_logits) :
    n_sequences = K.shape(pwm_logits)[0]
    seq_length = K.shape(pwm_logits)[1]
    
    flat_pwm = K.reshape(pwm_logits, (n_sequences * seq_length, len(alph))) #JV change for var alph
    sampled_pwm = gumbel_softmax(flat_pwm, temperature=0.1)

    return K.reshape(sampled_pwm, (n_sequences, seq_length, len(alph), 1)) #JV change for var alph
 
#SeqProp helper functions

def initialize_sequence_templates(generator, sequence_templates) :


    #JV change for var alph
    #encoder = iso.OneHotEncoder(seq_length=len(sequence_templates[0]))
    #onehot_templates = np.concatenate([encoder(sequence_template).reshape((1, len(sequence_template), len(alph), 1)) for sequence_template in sequence_templates], axis=0) 

    onehot_templates,_ = AutoMLBackend.onehot_seqlist(sequence_templates, sequence_type, alph, model_type = 'deepswarm')
    onehot_templates = [np.transpose(onehot_template) for onehot_template in onehot_templates]
    onehot_templates = np.concatenate([onehot_template.reshape((1, len(sequence_templates[0]), len(alph), 1)) for onehot_template in onehot_templates], axis=0) 
    
    for i in range(len(sequence_templates)) :
        sequence_template = sequence_templates[i]

        for j in range(len(sequence_template)) :
            if sequence_type == 'nucleic_acid':
                if sequence_template[j] != 'N' : #JV change for var alph
                    if sequence_template[j] != 'Q' : #JV change for var alph
                        nt_ix = np.argmax(onehot_templates[i, j, :, 0])
                        onehot_templates[i, j, :, :] = -1 * len(alph) #JV change for var alph
                        onehot_templates[i, j, nt_ix, :] = 10
                    else :
                        onehot_templates[i, j, :, :] = -1
            if sequence_type != 'nucleic_acid': #JV change for var alph
                if sequence_template[j] != 'X' : #JV change for var alph
                    if sequence_template[j] != 'Q' : #JV change for var alph 
                        nt_ix = np.argmax(onehot_templates[i, j, :, 0])
                        onehot_templates[i, j, :, :] = -1 * len(alph) #JV change for var alph
                        onehot_templates[i, j, nt_ix, :] = 10
                    else :
                        onehot_templates[i, j, :, :] = -1

    onehot_masks = np.zeros((len(sequence_templates), len(sequence_templates[0]), len(alph), 1)) #JV change for var alph
    for i in range(len(sequence_templates)) :
        sequence_template = sequence_templates[i]
        for j in range(len(sequence_template)) :
            if sequence_type == 'nucleic_acid': #JV change for var alph
                if sequence_template[j] == 'N' :
                    onehot_masks[i, j, :, :] = 1.0
            else: #JV change for var alph
                if sequence_template[j] == 'X' :
                    onehot_masks[i, j, :, :] = 1.0


    generator.get_layer('template_dense').set_weights([onehot_templates.reshape(1, -1)])
    generator.get_layer('template_dense').trainable = False

    generator.get_layer('mask_dense').set_weights([onehot_masks.reshape(1, -1)])
    generator.get_layer('mask_dense').trainable = False

def initialize_sequences(generator, init_sequences, p_init) :

    #JV change for var alph
    #encoder = iso.OneHotEncoder(seq_length=len(init_sequences[0])) 
    #onehot_sequences = np.concatenate([encoder(init_sequence).reshape((1, len(init_sequence), len(alph), 1)) for init_sequence in init_sequences], axis=0) #JV change for var alph
    onehot_sequences,_ = AutoMLBackend.onehot_seqlist(init_sequences, sequence_type, alph, model_type = 'deepswarm')
    onehot_sequences = [np.transpose(onehot_sequence) for onehot_sequence in onehot_sequences]
    onehot_sequences = np.concatenate([onehot_sequence.reshape((1, len(init_sequences[0]), len(alph), 1)) for onehot_sequence in onehot_sequences], axis=0) 

    onehot_sequences = np.array(onehot_sequences)
    onehot_logits = generator.get_layer('policy_pwm').get_weights()[0].reshape((len(init_sequences), len(init_sequences[0]), len(alph), 1)) #JV change for var alph

    on_logit = np.log(p_init / (1. - p_init))

    p_off = (1. - p_init) / float(len(alph)-1) #JV change for var alph
    off_logit = np.log(p_off / (1. - p_off))

    for i in range(len(init_sequences)) :
        init_sequence = init_sequences[i]
        
        for j in range(len(init_sequence)) :
            nt_ix = -1
            #JV change for var alph
            try:
                nt_ix = alph.index(init_sequence[j])
            except:
                nt_ix = -1
            #if init_sequence[j] == 'A' :
            #    nt_ix = 0
            #elif init_sequence[j] == 'C' :
            #    nt_ix = 1
            #elif init_sequence[j] == 'G' :
            #    nt_ix = 2
            #elif init_sequence[j] == 'T' :
            #    nt_ix = 3

            onehot_logits[i, j, :, :] = off_logit
            if nt_ix != -1 :
                onehot_logits[i, j, nt_ix, :] = on_logit

    generator.get_layer('policy_pwm').set_weights([onehot_logits.reshape(1, -1)])

#SeqProp Generator Model definitions

#Generator that samples a single one-hot sequence per trainable PWM
def build_generator(seq_length, n_sequences=1, n_samples=None, sequence_templates=None, init_sequences=None, p_init=0.5, batch_normalize_pwm=False, pwm_transform_func=None, validation_sample_mode='max', master_generator=None) :

    use_samples = True
    if n_samples is None :
        use_samples = False
        n_samples = 1

    #Seed input for all dense/embedding layers
    ones_input = Input(tensor=K.ones((1, 1)), name='seed_input')

    #Initialize a Lambda layer to reshape flat matrices into PWM tensors
    reshape_layer = Lambda(lambda x: K.reshape(x, (n_sequences, seq_length, len(alph), 1)), name='onehot_reshape') #JV change for var alph
    
    #Initialize Template, Masking and Trainable PWMs
    onehot_template_dense = Dense(n_sequences * seq_length * len(alph), use_bias=False, kernel_initializer='zeros', name='template_dense') #JV change for var alph
    onehot_mask_dense = Dense(n_sequences * seq_length * len(alph), use_bias=False, kernel_initializer='ones', name='mask_dense') #JV change for var alph
    dense_seq_layer = Dense(n_sequences * seq_length * len(alph), use_bias=False, kernel_initializer='glorot_uniform', name='policy_pwm') #JV change for var alph

    if master_generator is not None :
        dense_seq_layer = master_generator.get_layer('policy_pwm')
    
    #Initialize Templating and Masking Lambda layer
    masking_layer = Lambda(mask_pwm, output_shape = (seq_length, len(alph), 1), name='masking_layer') #JV change for var alph
    
    #Get Template, Mask and Trainable PWM logits
    onehot_template = reshape_layer(onehot_template_dense(ones_input))
    onehot_mask = reshape_layer(onehot_mask_dense(ones_input))
    onehot_logits = reshape_layer(dense_seq_layer(ones_input))

    #Batch Normalize PWM Logits
    if batch_normalize_pwm :
        pwm_norm_layer = InstanceNormalization(axis=-2, name='policy_batch_norm')
        if master_generator is not None :
            pwm_norm_layer = master_generator.get_layer('policy_batch_norm')
        onehot_logits = pwm_norm_layer(onehot_logits)
    
    #Add Template and Multiply Mask
    pwm_logits = masking_layer([onehot_logits, onehot_template, onehot_mask])
    
    #Get PWM from logits
    pwm = Softmax(axis=-2, name='pwm')(pwm_logits)

    #Optionally tile each PWM to sample from
    if use_samples :
        pwm_logits = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]))(pwm_logits)
    
    #Sample proper One-hot coded sequences from PWMs
    if validation_sample_mode == 'max' :
        sampled_pwm = Lambda(sample_pwm, name='pwm_sampler')(pwm_logits)
    elif validation_sample_mode == 'gumbel' :
        sampled_pwm = Lambda(sample_gumbel, name='pwm_sampler')(pwm_logits)
    elif validation_sample_mode == 'simple_sample' :
        sampled_pwm = Lambda(sample_pwm_simple, name='pwm_sampler')(pwm_logits)
    else :
        sampled_pwm = Lambda(sample_pwm_only, name='pwm_sampler')(pwm_logits)
    
    #PWM & Sampled One-hot custom transform function
    if pwm_transform_func is not None :
        pwm = Lambda(lambda pwm_seq: pwm_transform_func(pwm_seq))(pwm)
        sampled_pwm = Lambda(lambda pwm_seq: pwm_transform_func(pwm_seq))(sampled_pwm)
    
    #Optionally create sample axis
    if use_samples :
        sampled_pwm = Lambda(lambda x: K.reshape(x, (n_samples, n_sequences, seq_length, len(alph), 1)))(sampled_pwm) #JV change for var alph

    generator_model = Model(
        inputs=[
            ones_input      #Dummy Seed Input
        ],
        outputs=[
            pwm_logits,     #Logits of the Templated and Masked PWMs
            pwm,            #Templated and Masked PWMs
            sampled_pwm     #Sampled One-hot sequences (n_samples per trainable PWM)
        ]
    )

    if sequence_templates is not None :
        initialize_sequence_templates(generator_model, sequence_templates)

    if init_sequences is not None :
        initialize_sequences(generator_model, init_sequences, p_init)

    #Lock all generator layers except policy layers
    for generator_layer in generator_model.layers :
        generator_layer.trainable = False
        
        if 'policy' in generator_layer.name :
            generator_layer.trainable = True

    return 'seqprop_generator', generator_model

#(Re-)Initialize PWM weights
def reset_generator(generator_model) :
    session = K.get_session()
    for generator_layer in generator_model.layers :
        if 'policy' in generator_layer.name :
            for v in generator_layer.__dict__:
                v_arg = getattr(generator_layer, v)
                if hasattr(v_arg,'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
                    print('reinitializing layer {}.{}'.format(generator_layer.name, v))

############## PART 4: SEQPROP OPTIMIZE FUNCTIONS ##############

#### Note, the following code will not have the same documentation as all other BioAutoMATED functions.
#### This is because these functions are pulled directly from the seqprop library by https://github.com/johli/seqprop
#### and documented in Bogard et al. Cell 2019. However, this library was only applicable for 4-nucleotide alphabets
#### Hence, the seqprop library has been extended to be alphabet- and sequence type-agnostic.
#### All changes are marked by the following comment : "#JV change for var alph" so that it is easy to find the changes.

#SeqProp loss helper functions

def get_target_entropy_mse(pwm_start=0, pwm_end=100, target_bits=2.0) :
    
    def target_entropy_mse(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        return K.mean((conservation - target_bits)**2, axis=-1)
    
    return target_entropy_mse

def get_target_entropy_mae(pwm_start=0, pwm_end=100, target_bits=2.0) :
    
    def target_entropy_mae(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        return K.mean(K.abs(conservation - target_bits), axis=-1)
    
    return target_entropy_mae

def get_target_entropy_sme(pwm_start=0, pwm_end=100, target_bits=2.0) :
    
    def target_entropy_sme(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        return (K.mean(conservation, axis=-1) - target_bits)**2
    
    return target_entropy_sme

def get_target_entropy_ame(pwm_start=0, pwm_end=100, target_bits=2.0) :
    
    def target_entropy_ame(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        return K.abs(K.mean(conservation, axis=-1) - target_bits)
    
    return target_entropy_ame

def get_margin_entropy(pwm_start=0, pwm_end=100, min_bits=2.0) :
    
    def margin_entropy(pwm) :
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -K.log(K.clip(pwm_section, K.epsilon(), 1. - K.epsilon())) / K.log(2.0)
        entropy = K.sum(entropy, axis=(2, 3))
        conservation = 2.0 - entropy

        mean_conservation = K.mean(conservation, axis=-1)

        margin_entropy_cost = K.switch(mean_conservation < K.constant(min_bits, shape=(1,)), min_bits - mean_conservation, K.zeros_like(mean_conservation))
    
        return margin_entropy_cost

    
    return margin_entropy

############## PART 5: OTHER SEQPROP HELPER FUNCTIONS - CUSTOM FOR BioAutoMATED ##############

#### Note, the following code will not have the same documentation as all other BioAutoMATED functions.
#### This is because these functions are pulled directly from the seqprop library by https://github.com/johli/seqprop
#### and documented in Bogard et al. Cell 2019. However, this library was only applicable for 4-nucleotide alphabets
#### Hence, the seqprop library has been extended to be alphabet- and sequence type-agnostic.
#### All changes below were adapted from the functions described in https://github.com/876lkj/seqprop

def loss_func_mod(predictor_outputs, target, class_of_interest, load_predictor_function):

    pwm_logits, pwm, sampled_pwm, predicted_out = predictor_outputs
    seq_input = Lambda(lambda X: K.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1)))(sampled_pwm)
    predictor_inputs, predictor_outputs, post_compile_function = load_predictor_function(seq_input)
    
    #Create target constant -- want predicted value for modified input to be close to target input 
    target_out = K.tile(K.constant(target), (K.shape(sampled_pwm)[0], 1)) # make list of lists of targets
    # get loss from predicted =/= actual
    target_cost = (target_out - predicted_out)**2 # predicted_out is still list of lists

    # needed to add cast because of weird dtype issue https://stackoverflow.com/questions/56357004/typeerror-value-passed-to-parameter-indices-has-datatype-float32-not-in-list
    target_cost = tf.gather_nd(target_cost,[[0,K.cast(class_of_interest, dtype = 'int32')]])
    # only at this point do we grab the actual class for the loss function - didn't want to mess with

    loss = target_cost
    
    return loss

# helper functions adapted from: https://github.com/876lkj/seqprop 
# need to re-create EXACT SAME layers as final trained model
# fix weights of layers so only input layer is modified
def load_saved_predictor(model_path, seq_len, alph):
    with CustomObjectScope({'GlorotUniform': glorot_uniform(), 'BatchNormalizationV1': BatchNormalization()}): # , 'BatchNormalizationV1': BatchNormalization()
        try:
            saved_model = load_model(model_path)
        except:
            saved_model = tf.keras.models.load_model(model_path)   
        saved_model.load_weights(model_path)

    def _initialize_predictor_weights(predictor_model, saved_model=saved_model):
        
        for layer in saved_model.layers:
            if layer.get_weights():
                predictor_model.get_layer(layer.name).set_weights(layer.get_weights())
                predictor_model.get_layer(layer.name).trainable = False

    def _load_predictor_func(sequence_input, saved_model=saved_model) :

        # input space parameters 
        seq_length = seq_len
        num_letters = len(alph) # num nt 
        # expanded version b/c seqprop built for 2d 
        seq_input_shape = (seq_len, num_letters, 1) # modified
        
        reshaped_input = Reshape(target_shape=(seq_len, num_letters, 1),name='reshaped_input')(sequence_input)
        if 'deepswarm' not in model_path: # reshapes if extra dimension in deepswarm input is not needed
            reshaped_input = Reshape(target_shape=(seq_len, num_letters),name='reshaped_input')(sequence_input)

        print(reshaped_input)
        print(type(reshaped_input))

        prior_layer = reshaped_input
        for layer in saved_model.layers:
            print(layer)
            H = layer(prior_layer)
            prior_layer = H
        out_y = prior_layer 
        
        predictor_inputs = []
        predictor_outputs = [out_y]
        return predictor_inputs, predictor_outputs, _initialize_predictor_weights

    return _load_predictor_func

# adapted from: https://github.com/876lkj/seqprop 
def build_loss_model(predictor_model, loss_func, target, class_of_interest, load_predictor_function): # adapted from seqprop package to take target as input
    loss_out = Lambda(lambda out: loss_func(out, target, class_of_interest, load_predictor_function), output_shape = (1,))(predictor_model.outputs)
    loss_model = Model(predictor_model.inputs, loss_out)
    return 'loss_model', loss_model

# adapted from: https://github.com/876lkj/seqprop 
# adapted from seqprop package to take chnage shape of seq_input for DS models
def build_predictor(generator_model, load_predictor_function, final_model_path, n_sequences=1, n_samples=None, eval_mode='pwm') : 

    use_samples = True
    if n_samples is None:
        use_samples = False
        n_samples = 1

    #Get PWM outputs from Generator Model
    pwm = generator_model.outputs[1]
    sampled_pwm = generator_model.outputs[2]

    seq_input = None
    if eval_mode == 'pwm' :
        seq_input = pwm
    elif eval_mode == 'sample' :
        seq_input = sampled_pwm
        if use_samples : # use_samples is False 
            seq_input = Lambda(lambda x: K.reshape(x, (K.shape(x)[0] * K.shape(x)[1], K.shape(x)[2], K.shape(x)[3], K.shape(x)[4])))(seq_input)
    
    X = seq_input
    if 'deepswarm' in final_model_path: # adding dimension if DeepSwarm model
        seq_input = Lambda(lambda X: K.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1)))(seq_input)
    
    predictor_inputs, predictor_outputs, post_compile_function = load_predictor_function(seq_input)
    
    
    #Optionally create sample axis
    if use_samples : # use_samples is False 
        predictor_outputs = [
            #Lambda(lambda x: K.reshape(x, (n_samples, n_sequences, K.shape(x)[-1])))(predictor_output)
            Lambda(lambda x: K.reshape(x, K.concatenate([K.constant([n_samples, n_sequences], dtype=tf.int32), K.shape(x)[1:]], axis=0)))(predictor_output)
        for predictor_output in predictor_outputs
        ]
    
    predictor_model = Model(
        inputs = generator_model.inputs + predictor_inputs,
        outputs = generator_model.outputs + predictor_outputs # issue is in the predictor_outputs, this line is adding two lists
    )

    post_compile_function(predictor_model)

    #Lock all layers except policy layers
    for predictor_layer in predictor_model.layers :
        predictor_layer.trainable = False
        
        if 'policy' in predictor_layer.name :
            predictor_layer.trainable = True

    return 'seqprop_predictor_aparent_large', predictor_model

############## END OF FILE ##############