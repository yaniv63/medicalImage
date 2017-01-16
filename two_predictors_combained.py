# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
from model_atom_base import create_smodel
from keras.layers import Merge,Dense,Input,merge
from keras.models import Model,Sequential
from keras.callbacks import EarlyStopping
import numpy as np
import pickle
from sklearn.model_selection import train_test_split



def one_predictor_model(index=0):
    predictor = create_smodel(1,32,32,index)
    predictor.add(Dense(1,activation='sigmoid'))
    return predictor

def two_predictors_combained_model():
    
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
    #init_weights = np.ones()
    
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

#    train_samples = [axial_samples_train,coronal_samples_train]    
#    train_labels = [labels_samples_train,labels_samples_train]    
#    test_samples = [axial_samples_test,coronal_samples_test]     
#    test_labels = [labels_samples_test,labels_samples_test]    
    train_samples = [axial_samples_train,axial_samples_train]    
    train_labels = [labels_samples_train,labels_samples_train]    
    test_samples = [axial_samples_test,axial_samples_test]     
    test_labels = [labels_samples_test,labels_samples_test]  

    stop_train_callback1 = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
    stop_train_callback2 = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=3, verbose=0, mode='auto')

    predictors = []
    results = []
    predictions= []
    for i in range(2):
        predictors.append(one_predictor_model(index=i))
        predictors[i].compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        predictors[i].fit(train_samples[i], train_labels[i],batch_size=300,nb_epoch=10,validation_split=0.2,callbacks=[stop_train_callback1,stop_train_callback2],shuffle=True)
        results.append(predictors[i].evaluate(test_samples[i],test_labels[i]))
        predictions.append(predictors[i].predict(test_samples[i]))
        predictors[i].save_weights(weight_path+'%d.h5'%(i))    
    combained_model = [average_two_models_prediction(),two_parameters_combained_model(),two_predictors_combained_model()]
    for i in range(3):    
        layer_dict = dict([(layer.name, layer) for layer in combained_model[i].layers])
        layer_dict["Seq_0"].load_weights(weight_path+'0.h5',by_name=True)
        layer_dict["Seq_1"].load_weights(weight_path+'1.h5',by_name=True)
        combained_model[i].compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        combained_model[i].fit(train_samples,train_labels[0],nb_epoch=100,batch_size=300,validation_split=0.2,callbacks=[stop_train_callback1,stop_train_callback2])

    combained_model_results = [r.evaluate(test_samples,test_labels[0]) for r in combained_model ]    
    #avg predicor
    avg_predict = ((predictions[0] + predictions[1])/2).round()
    avg_success = np.equal(avg_predict,test_labels[0])
    avg_precentage = avg_success.tolist().count([True])/float(len(avg_predict))
    
    confusion_mat=[]
    for i in range(3):
        from sklearn.metrics import confusion_matrix
        predict = combained_model[i].predict(test_samples).round()
        confusion_mat.append(confusion_matrix(test_labels[0],predict))
    
    
        
