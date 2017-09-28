# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
from keras.layers import Dense,Input,merge, Convolution2D, LeakyReLU, MaxPooling2D, Dropout, Flatten,Layer
from keras.models import Model, Sequential
from keras.regularizers import l2
import keras.backend as K
import numpy as np

def create_smodel(N_mod, img_rows, img_cols, index=0):
    index = str(index)
    smodel = Sequential(name='Seq_' + index)
    # 1x33x33 -> 24x14x14
    smodel.add(Convolution2D(24, 5, 5,
                             input_shape=(N_mod, img_rows, img_cols), name='conv1_' + index,W_regularizer='l2'))  # 1x33x33 -> 24x28x28
    smodel.add(LeakyReLU(name='leakyrelu1_' + index))
    smodel.add(MaxPooling2D(pool_size=(2, 2), name='maxpool1_' + index))  # 24x28x28 -> 24x14x14
    #smodel.add(Dropout(0.25, name='drop1_' + index))

    # 24x14x14 -> 33x6x6
    smodel.add(Convolution2D(33, 3, 3, name='conv2_' + index,W_regularizer='l2'))  # 24x14x14 -> 33x12x12
    smodel.add(LeakyReLU(name='leakyrelu2_' + index))
    smodel.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2_' + index))  # 33x12x12 -> 33x6x6
    #smodel.add(Dropout(0.25, name='drop2_' + index))

    # 33x6x6 -> 48x4x4
    smodel.add(Convolution2D(48, 3, 3, name='conv3_' + index,W_regularizer='l2'))
    smodel.add(LeakyReLU(name='leakyrelu3_' + index))
    #smodel.add(Dropout(0.25, name='drop3_' + index))

    smodel.add(Flatten(name='flat1_' + index))
    smodel.add(Dense(16, name='dense1_' + index))
    smodel.add(LeakyReLU(name='leakyrelu4_' + index))
    #smodel.add(Dropout(0.25, name='drop4_' + index))
    return smodel


def one_predictor_model(N_mod=1, img_rows=33, img_cols=33, index=0):
    predictor = create_smodel(N_mod, img_rows, img_cols, index)
    predictor.add(Dense(16,name='dense2_{}'.format(index), w_regularizer="l2"))
    predictor.add(LeakyReLU(name='perception_{}'.format(index)))
    predictor.add(Dense(1, activation='sigmoid', name='out{}'.format(index), w_regularizer="l2"))
    return predictor

def gating_model(N_exp,N_mod, img_rows, img_cols):
    gate = create_smodel(N_exp*N_mod, img_rows, img_cols, index='gate')
    gate.add(Dense(16, name='dense_gate',W_regularizer='l2'))
    gate.add(Dense(N_exp,activation='softmax',name='out_gate',W_regularizer='l2'))
    return gate

def gating_model_use_parameters(N_exp):
    input = Input(shape=(N_exp*16,))
    dense1 = Dense(16, name='dense1_gate', W_regularizer="l2",input_shape=(48,))(input)
    dense2 = Dense(16, name='dense2_gate', W_regularizer="l2")(dense1)
    dense3 = Dense(N_exp, activation='softmax', name='out_gate', W_regularizer="l2")(dense2)

    model = Model(input=input,output=dense3)
    return model

def two_predictors_combined_model():
    
    init_bias = np.full(shape=(1,),fill_value=-1)
    init_weights = np.ones((2,1))
    first_predictor = one_predictor_model(index=0)
    second_predictor = one_predictor_model(index=1)
    first_predictor_data = Input(shape=(1,33,33))
    second_predictor_data = Input(shape=(1,33,33))
    decide1= first_predictor(first_predictor_data)
    decide2= second_predictor(second_predictor_data)
    merged = merge([decide1,decide2],mode='concat',concat_axis=1)
    out = Dense(1,activation='sigmoid',weights=[init_weights,init_bias])(merged)
    model = Model(input=[first_predictor_data,second_predictor_data],output=out)
    return model


def two_parameters_combined_model():
    model1 = create_smodel(1,33,33,0)
    model2 = create_smodel(1,33,33,1)
    first_predictor_data = Input(shape=(1,33,33))
    second_predictor_data = Input(shape=(1,33,33))
    param1 = model1(first_predictor_data)
    param2 = model2(second_predictor_data)
    merged = merge(inputs=[param1,param2],mode='concat',concat_axis=1)
    out = Dense(1,activation='sigmoid')(merged)
    model = Model(input=[first_predictor_data,second_predictor_data],output=out)    
    return model

    
def average_two_models_prediction():    
    first_predictor = one_predictor_model(index=0)
    second_predictor = one_predictor_model(index=1)
    first_predictor_data = Input(shape=(1,33,33))
    second_predictor_data = Input(shape=(1,33,33))
    decide1= first_predictor(first_predictor_data)
    decide2= second_predictor(second_predictor_data)
    merged = merge([decide1,decide2],mode='ave',concat_axis=1)
    model = Model(input=[first_predictor_data,second_predictor_data],output=merged)
    return model

def average_n_models_prediction(N_mod = 1, img_rows = 33, img_cols = 33,n=2):
    predictors = []
    decisions = []
    data = []
    for i in range(n):
        predictors.append(one_predictor_model(index=i))
        data.append(Input(shape=(N_mod, img_rows, img_cols),name='input{}'.format(i)))
        decisions.append(predictors[i](data[i]))
    merged = merge(decisions,mode='ave',concat_axis=1)
    model = Model(input=data,output=merged)
    return model


def n_predictors_combined_model(N_mod = 1, img_rows = 33, img_cols = 33,n=2):
    init_bias = np.full(shape=(1,), fill_value=-1)
    init_weights = np.ones((n, 1))
    predictors = []
    decisions = []
    data = []
    for i in range(n):
        predictors.append(one_predictor_model(N_mod, img_rows, img_cols,index=i))
        data.append(Input(shape=(N_mod, img_rows, img_cols),name='input{}'.format(i)))
        decisions.append(predictors[i](data[i]))
    merged = merge(decisions, mode='concat', concat_axis=1)
    out = Dense(1, activation='sigmoid', weights=[init_weights, init_bias],W_regularizer=l2(0.01))(merged)
    model = Model(input=data, output=out)
    return model

def n_parameters_combined_model(N_mod = 1, img_rows = 33, img_cols = 33,n=2):
    predictors = []
    params = []
    data = []
    for i in range(n):
        predictors.append(create_smodel(N_mod, img_rows, img_cols,index=i))
        data.append(Input(shape=(N_mod, img_rows, img_cols), name='input{}'.format(i)))
        params.append(predictors[i](data[i]))
    merged = merge(inputs=params, mode='concat', concat_axis=1)
    out = Dense(1, activation='sigmoid',W_regularizer=l2(0.01))(merged)
    model = Model(input=data, output=out)
    return model


def n_experts_combined_model(N_mod = 4, img_rows = 33, img_cols = 33,n=3):
    predictors = []
    decisions = []
    data = []

    for i in range(n):
        predictors.append(one_predictor_model(N_mod, img_rows, img_cols,index=i))
        data.append(Input(shape=(N_mod, img_rows, img_cols),name='input{}'.format(i)))
        decisions.append(predictors[i](data[i]))

    gate = gating_model(N_exp=n,N_mod = 4, img_rows = 33, img_cols = 33)
    merged_input = merge(inputs=data,mode='concat',concat_axis=1)
    merged_decisions = merge(inputs=decisions,mode='concat',concat_axis=1)

    coefficients = gate(merged_input)
    weighted_prediction = merge(inputs=[coefficients,merged_decisions],mode='dot',concat_axis=1)


    model = Model(input=data, output=weighted_prediction)
    return model

def n_experts_combined_model_gate_parameters(N_mod=4, img_rows=33, img_cols=33, n=3):
    predictors = []
    decisions = []
    denses = []
    denses2 = []
    perceptions =[]
    data = []

    for i in range(n):
        predictors.append(create_smodel(N_mod, img_rows, img_cols, index=i))
        data.append(Input(shape=(N_mod, img_rows, img_cols), name='input{}'.format(i)))
        denses.append(predictors[i](data[i]))
        denses2.append(Dense(16, name='dense2_{}'.format(i), W_regularizer="l2")(denses[i]))
        perceptions.append(LeakyReLU(name='perception_{}'.format(i))(denses2[i]))
        decisions.append(Dense(1, activation='sigmoid', name='out{}'.format(i), W_regularizer="l2")(perceptions[i]))

    merged_decisions = merge(inputs=decisions,concat_axis=1,mode='concat')

    gate = gating_model_use_parameters(N_exp=n)

    gate_input = merge(inputs=perceptions,concat_axis=1,mode='concat')
    coefficients = gate(gate_input)
    weighted_prediction = merge([coefficients, merged_decisions],mode='dot',concat_axis=1)

    model = Model(input=data, output=weighted_prediction)
    return model
