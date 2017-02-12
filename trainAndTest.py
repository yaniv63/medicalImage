# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
from imghdr import what

from keras.callbacks import EarlyStopping
import pickle
import numpy as np

from two_predictors_combained import one_predictor_model, average_two_models_prediction, two_parameters_combained_model, \
    two_predictors_combained_model


def generate_train(patchType, personList, batchSize=128):
    while True:
        for index in personList:
            for index2 in range(1, 5):
                with open(patches + "patches_"+patchType+ "_train_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels_train_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                    samples_train = np.array(pickle.load(fp1))
                    labels_train = np.array(pickle.load(fp2))

                samples_train = np.expand_dims(samples_train, 1)
                labels_train = np.expand_dims(labels_train, 1)
                k = samples_train.shape[0]/batchSize

                # divide batches
                for i in range(k):
                    yield (samples_train[i*batchSize:(i+1)*batchSize],labels_train[i*batchSize:(i+1)*batchSize])

def generate_train_combained(patchType1,patchType2, personList, batchSize=128):
    while True:
        for index in personList:
            for index2 in range(1, 5):
                with open(patches + "patches_" + patchType1 + "_train_0{}_0{}.lst".format(index, index2),'rb') as fp1, \
                 open(patches + "patches_" + patchType2 + "_train_0{}_0{}.lst".format(index, index2), 'rb') as fp2, \
                        open(patches + "labels_train_0{}_0{}.lst".format(index, index2), 'rb') as fp3:
                    samples1_train = np.array(pickle.load(fp1))
                    samples2_train = np.array(pickle.load(fp2))
                    labels_train = np.array(pickle.load(fp3))

                samples1_train = np.expand_dims(samples1_train, 1)
                samples2_train = np.expand_dims(samples2_train, 1)
                labels_train = np.expand_dims(labels_train, 1)
                k = samples1_train.shape[0] / batchSize

                # divide batches
                for i in range(k):
                    yield ([samples1_train[i * batchSize:(i + 1) * batchSize],
                           samples2_train[i * batchSize:(i + 1) * batchSize]],
                           labels_train[i * batchSize:(i + 1) * batchSize])

def aggregate_test(personList,patchType):
    samples_test = []
    labels_test = []
    for index in personList:
        for index2 in range(1, 5):
            with open(patches + "patches_"+patchType+ "_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                samples_test += pickle.load(fp1)
                labels_test += pickle.load(fp2)

    samples_test = np.array(samples_test)
    labels_test = np.array(labels_test)

    samples_test = np.expand_dims(samples_test, 1)
    labels_test = np.expand_dims(labels_test, 1)

    return (samples_test,labels_test)

def aggregate_val(personList,patchType):
    samples_val = []
    labels_val = []
    for index in personList:
        for index2 in range(1, 5):
            with open(patches + "patches_"+patchType+ "_val_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels__val_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                samples_val += pickle.load(fp1)
                labels_val += pickle.load(fp2)

    samples_val = np.array(samples_val)
    labels_val = np.array(labels_val)

    samples_val = np.expand_dims(samples_val, 1)
    labels_val = np.expand_dims(labels_val, 1)

    return (samples_val,labels_val)

weight_path = r'./trained_weights/'
patches = r'./patches/'

######## create models

stop_train_callback1 = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
stop_train_callback2 = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=3, verbose=0, mode='auto')
mycallbacks = [stop_train_callback1,stop_train_callback2]
predictors = []

for i in range(2):
    predictors.append(one_predictor_model(index=i))
    predictors[i].compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    combained_model = [average_two_models_prediction(), two_parameters_combained_model(),
               two_predictors_combained_model()]
for i in range(3):
    combained_model[i].compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

PersonTrainList = [1,2,3,4]
val_axial_set,val_axial_labels = aggregate_val("axial",PersonTrainList)
val_coronal_set,val_coronal_labels = aggregate_val("coronal",PersonTrainList)
val_sets = [(val_axial_set,val_axial_labels),(val_coronal_set,val_coronal_labels)]
combained_val = ([val_axial_set,val_coronal_set],val_coronal_labels)
######## train individual predictors
axial_generator = generate_train("axial",PersonTrainList)
coronal_generator = generate_train("coronal",PersonTrainList)
train_generator = [axial_generator,coronal_generator]
for i in range(2):
    predictors[i].fit_generator(train_generator[i], samples_per_epoch=10000, nb_epoch=10, callbacks=mycallbacks,nb_worker=4,validation_data=val_sets[i])
    predictors[i].save_weights(weight_path + '%d.h5' % (i))
######## test individual predictors
test_axial_samples,test_axial_labels = aggregate_test([5],"axial")
test_coronal_samples,test_coronal_labels = aggregate_test([5],"coronal")

test_samples = [test_axial_samples, test_coronal_samples]
test_labels = [test_axial_labels, test_coronal_labels]

results = []
predictions = []

for i in range(2):
    results.append(predictors[i].evaluate(test_samples[i], test_labels[i]))
    predictions.append(predictors[i].predict(test_samples[i]))

######## train predictors combinations
for i in range(3):
    layer_dict = dict([(layer.name, layer) for layer in combained_model[i].layers])
    layer_dict["Seq_0"].load_weights(weight_path + '0.h5', by_name=True)
    layer_dict["Seq_1"].load_weights(weight_path + '1.h5', by_name=True)
    combained_model[i].fit_generator(generate_train_combained("axial","coronal",[1,2,3,4]), samples_per_epoch=10000, nb_epoch=30, callbacks=mycallbacks,nb_worker=4,validation_data=combained_val)

######## test predictors combinations

combained_model_results = [r.evaluate(test_samples, test_labels[0]) for r in combained_model]
# avg predicor
avg_predict = ((predictions[0] + predictions[1]) / 2).round()
avg_success = np.equal(avg_predict, test_labels[0])
avg_precentage = avg_success.tolist().count([True]) / float(len(avg_predict))

confusion_mat = []
dice = []
for i in range(3):
    from sklearn.metrics import confusion_matrix
    predict = combained_model[i].predict(test_samples).round()
    confusion_mat.append(confusion_matrix(test_labels[0], predict))
    print("dice {} is " + str(float(2) * confusion_mat[i][1][1] / (
    2 * confusion_mat[i][1][1] + confusion_mat[i][1][0] + confusion_mat[i][0][1])).format(i))


