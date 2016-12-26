# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
from model_atom_base import atom_model_conv
from keras.layers import merge,Dense
from keras.models import Model

def two_predictors_combained_model():
    
    first_predictor = atom_model_conv(sequence_num=1)
    second_predictor = atom_model_conv(sequence_num=2)
    x = merge([first_predictor, second_predictor], mode='concat')
    out = Dense(1,activation='sigmoid')(x)
    
    model = Model(output=out)
    return model


mymodel = two_predictors_combained_model()
from keras.utils.visualize_util import plot
plot(mymodel, to_file='model_combained-yaniv.png',show_layer_names=True,show_shapes=True)
    
