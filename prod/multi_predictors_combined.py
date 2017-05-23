# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
from keras.layers import Dense,Input,merge, Convolution2D, LeakyReLU, MaxPooling2D, Dropout, Flatten
from keras.models import Model, Sequential
import numpy as np


def create_smodel(N_mod, img_rows, img_cols, index=0):
    index = str(index)
    smodel = Sequential(name='Seq_' + index)
    # 1x32x32 -> 24x14x14
    smodel.add(Convolution2D(24, 5, 5,
                             input_shape=(N_mod, img_rows, img_cols), name='conv1_' + index,W_regularizer='l2',b_regularizer='l2'))  # 1x32x32 -> 24x28x28
    smodel.add(LeakyReLU(name='leakyrelu1_' + index))
    smodel.add(MaxPooling2D(pool_size=(2, 2), name='maxpool1_' + index))  # 24x28x28 -> 24x14x14
    #smodel.add(Dropout(0.25, name='drop1_' + index))

    # 24x14x14 -> 32x6x6
    smodel.add(Convolution2D(32, 3, 3, name='conv2_' + index,W_regularizer='l2',b_regularizer='l2'))  # 24x14x14 -> 32x12x12
    smodel.add(LeakyReLU(name='leakyrelu2_' + index))
    smodel.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2_' + index))  # 32x12x12 -> 32x6x6
    #smodel.add(Dropout(0.25, name='drop2_' + index))

    # 32x6x6 -> 48x4x4
    smodel.add(Convolution2D(48, 3, 3, name='conv3_' + index,W_regularizer='l2',b_regularizer='l2'))
    smodel.add(LeakyReLU(name='leakyrelu3_' + index))
    #smodel.add(Dropout(0.25, name='drop3_' + index))

    smodel.add(Flatten(name='flat1_' + index))
    smodel.add(Dense(16, name='dense1_' + index))
    smodel.add(LeakyReLU(name='leakyrelu4_' + index))
    #smodel.add(Dropout(0.25, name='drop4_' + index))
    return smodel



def one_predictor_model(N_mod = 1, img_rows = 32, img_cols = 32,index=0):
    predictor = create_smodel(N_mod,img_rows,img_cols,index)
    predictor.add(Dense(1,activation='sigmoid',name='out{}'.format(index),W_regularizer='l2',b_regularizer='l2'))
    return predictor


def two_predictors_combined_model():
    
    init_bias = np.full(shape=(1,),fill_value=-1)
    init_weights = np.ones((2,1))
    first_predictor = one_predictor_model(index=0)
    second_predictor = one_predictor_model(index=1)
    first_predictor_data = Input(shape=(1,32,32))
    second_predictor_data = Input(shape=(1,32,32))
    decide1= first_predictor(first_predictor_data)
    decide2= second_predictor(second_predictor_data)
    merged = merge([decide1,decide2],mode='concat',concat_axis=1)
    out = Dense(1,activation='sigmoid',weights=[init_weights,init_bias])(merged)
    model = Model(input=[first_predictor_data,second_predictor_data],output=out)
    return model


def two_parameters_combined_model():
    model1 = create_smodel(1,32,32,0)
    model2 = create_smodel(1,32,32,1)
    first_predictor_data = Input(shape=(1,32,32))
    second_predictor_data = Input(shape=(1,32,32))
    param1 = model1(first_predictor_data)
    param2 = model2(second_predictor_data)
    merged = merge(inputs=[param1,param2],mode='concat',concat_axis=1)
    out = Dense(1,activation='sigmoid')(merged)
    model = Model(input=[first_predictor_data,second_predictor_data],output=out)    
    return model

    
def average_two_models_prediction():    
    first_predictor = one_predictor_model(index=0)
    second_predictor = one_predictor_model(index=1)
    first_predictor_data = Input(shape=(1,32,32))
    second_predictor_data = Input(shape=(1,32,32))
    decide1= first_predictor(first_predictor_data)
    decide2= second_predictor(second_predictor_data)
    merged = merge([decide1,decide2],mode='ave',concat_axis=1)
    model = Model(input=[first_predictor_data,second_predictor_data],output=merged)
    return model

def average_n_models_prediction(N_mod = 1, img_rows = 32, img_cols = 32,n=2):
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


def n_predictors_combined_model(N_mod = 1, img_rows = 32, img_cols = 32,n=2):
    init_bias = np.full(shape=(1,), fill_value=-1)
    init_weights = np.ones((n, 1))
    predictors = []
    decisions = []
    data = []
    for i in range(n):
        predictors.append(one_predictor_model(index=i))
        data.append(Input(shape=(N_mod, img_rows, img_cols),name='input{}'.format(i)))
        decisions.append(predictors[i](data[i]))
    merged = merge(decisions, mode='concat', concat_axis=1)
    out = Dense(1, activation='sigmoid', weights=[init_weights, init_bias],W_regularizer='l2',b_regularizer='l2')(merged)
    model = Model(input=data, output=out)
    return model

def n_parameters_combined_model(N_mod = 1, img_rows = 32, img_cols = 32,n=2):
    predictors = []
    params = []
    data = []
    for i in range(n):
        predictors.append(create_smodel(N_mod, img_rows, img_cols,index=i))
        data.append(Input(shape=(N_mod, img_rows, img_cols), name='input{}'.format(i)))
        params.append(predictors[i](data[i]))
    merged = merge(inputs=params, mode='concat', concat_axis=1)
    out = Dense(1, activation='sigmoid')(merged)
    model = Model(input=data, output=out)
    return model


