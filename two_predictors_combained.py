# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
from model_atom_base import atom_model_conv,create_smodel
from keras.layers import Merge,Dense,Input,merge
from keras.models import Model,Sequential


def one_predictor_model():
    predictor = create_smodel(1,32,32)
    predictor.add(Dense(1,activation='sigmoid'))
    return predictor

def two_predictors_combained_model():
    
    #atom_input1 = Input(shape=(1,32,32),name='input_num1')
    #atom_input2 = Input(shape=(1,32,32),name='input_num2')
#    first_predictor = atom_model_conv(sequence_num=1)
#    second_predictor = atom_model_conv(sequence_num=2)
#    x1  = first_predictor(atom_input1)
#    x2  = second_predictor(atom_input2)
#    x = merge([x1, x2], mode='concat')
#    out = Dense(1,activation='sigmoid')(x)
    first_predictor = one_predictor_model()
    second_predictor = one_predictor_model()
    first_predictor_data = Input(shape=(1,32,32))
    second_predictor_data = Input(shape=(1,32,32))
    decide1= first_predictor(first_predictor_data)
    decide2= second_predictor(second_predictor_data)
    merged = merge([decide1,decide2],mode='concat',concat_axis=1)
    out = Dense(1,activation='sigmoid')(merged)
    model = Model(input=[first_predictor_data,second_predictor_data],output=out)
    return model

flair_axial_model = one_predictor_model()
flair_coronal_model = one_predictor_model()
my_combained_model = two_predictors_combained_model()#two_predictors_combained_model()


from keras.utils.visualize_util import plot
plot(mymodel, to_file='model_combained-yaniv.png',show_layer_names=True,show_shapes=True)
    
