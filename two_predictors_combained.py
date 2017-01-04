# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
from model_atom_base import create_smodel
from keras.layers import Merge,Dense,Input,merge
from keras.models import Model,Sequential
from keras.callbacks import EarlyStopping,LambdaCallback
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




def one_predictor_model(index=0):
    predictor = create_smodel(1,32,32,index)
    predictor.add(Dense(1,activation='sigmoid'))
    return predictor

def two_predictors_combained_model():
    
    first_predictor = one_predictor_model(index=0)
    second_predictor = one_predictor_model(index=1)
    first_predictor_data = Input(shape=(1,32,32))
    second_predictor_data = Input(shape=(1,32,32))
    decide1= first_predictor(first_predictor_data)
    decide2= second_predictor(second_predictor_data)
    merged = merge([decide1,decide2],mode='concat',concat_axis=1)
    out = Dense(1,activation='sigmoid')(merged)
    model = Model(input=[first_predictor_data,second_predictor_data],output=out)
    return model
    
def two_parameters_combained_model():
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

if __name__ == "__main__":          
    weight_path = r'/home/yaniv/src/data/weights_predictor'    
    with open('patches_axial_01.lst', 'rb') as fp1 ,open('patches_coronal_01.lst', 'rb') as fp2,open('labels_01.lst', 'rb') as fp3 :
        axial_samples_train = np.array(pickle.load(fp1))
        coronal_samples_train = np.array(pickle.load(fp2))
        labels_samples_train =  np.array(pickle.load(fp3))
    
    with open('patches_axial_02.lst', 'rb') as fp1 ,open('patches_coronal_02.lst', 'rb') as fp2,open('labels_02.lst', 'rb') as fp3 :
        axial_samples_test = np.array(pickle.load(fp1))
        coronal_samples_test = np.array(pickle.load(fp2))
        labels_samples_test =  np.array(pickle.load(fp3))

    axial_samples_train=np.expand_dims(axial_samples_train,1)
    coronal_samples_train=np.expand_dims(coronal_samples_train,1)
    labels_samples_train=np.expand_dims(labels_samples_train,1)
    
    axial_samples_test=np.expand_dims(axial_samples_test,1)
    coronal_samples_test=np.expand_dims(coronal_samples_test,1)
    labels_samples_test=np.expand_dims(labels_samples_test,1)
    
    #permute data    
    permute = np.random.permutation(len(axial_samples_train))
    axial_samples_train = axial_samples_train[permute]
    coronal_samples_train = coronal_samples_train[permute]
    labels_samples_train = labels_samples_train[permute]
    #divide train-test
#    train_samples = [axial_samples[permute[ : int(len(permute) * .75)]],coronal_samples[permute[ : int(len(permute) * .75)]] ]    
#    train_labels = [labels_samples[permute[ : int(len(permute) * .75)]],labels_samples[permute[ : int(len(permute) * .75)]]]    
#    test_samples = [axial_samples[permute[ int(len(permute) * .75):]],coronal_samples[permute[ int(len(permute) * .75):]]]     
#    test_labels = [labels_samples[permute[ int(len(permute) * .75):]],labels_samples[permute[ int(len(permute) * .75):]]]  
    train_samples = [axial_samples_train,coronal_samples_train]    
    train_labels = [labels_samples_train,labels_samples_train]    
    test_samples = [axial_samples_test,coronal_samples_test]     
    test_labels = [labels_samples_test,labels_samples_test]    


    predictors = []
    results = []
    for i in range(2):
        predictors.append(one_predictor_model(index=i))
        predictors[i].compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        predictors[i].fit(train_samples[i], train_labels[i],batch_size=300,shuffle=True)
        results.append(predictors[i].evaluate(test_samples[i],test_labels[i]))
        predictors[i].save_weights(weight_path+'%d.h5'%(i))
    avg = (results[0][1]+results[1][1])/2

    # Plot the loss after every epoch.
    plot_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch), logs['loss']))
    stop_by_loss_callback = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')

    result_combained = []
    combained_model_predictors = two_predictors_combained_model()
    layer_dict = dict([(layer.name, layer) for layer in combained_model_predictors.layers])
    layer_dict["Seq_0"].load_weights(weight_path+'0.h5',by_name=True)
    layer_dict["Seq_1"].load_weights(weight_path+'1.h5',by_name=True)
    combained_model_predictors.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    combained_model_predictors.fit(train_samples,train_labels[0],nb_epoch=100,batch_size=300,callbacks=[ stop_by_loss_callback])
    result_combained.append(combained_model_predictors.evaluate(test_samples,test_labels[0]))
    
    combained_model_parameters = two_parameters_combained_model()
    layer_dict = dict([(layer.name, layer) for layer in combained_model_parameters.layers])
    layer_dict["Seq_0"].load_weights(weight_path+'0.h5',by_name=True)
    layer_dict["Seq_1"].load_weights(weight_path+'1.h5',by_name=True)
    combained_model_parameters.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    combained_model_parameters.fit(train_samples,train_labels[0],nb_epoch=100,batch_size=300,callbacks=[ stop_by_loss_callback])
    result_combained.append(combained_model_parameters.evaluate(test_samples,test_labels[0]))
