# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:32:39 2016

@author: yaniv
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
import numpy as np
np.random.seed(178)
from os import path, makedirs
from datetime import datetime
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
import pickle
from sklearn.metrics import confusion_matrix

from two_predictors_combined import one_predictor_model, average_two_models_prediction, two_parameters_combined_model, \
    two_predictors_combined_model
from logging_tools import get_logger

weight_path = r'./trained_weights/'
patches = r'./patches/'
runs_dir = r'./runs/'


def generate_train(patchType, personList, batchSize=256):
    while True:
        for index in personList:
            for index2 in range(1, 5):
                with open(patches + "patches_"+patchType+ "_train_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels_train_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                    samples_train = np.array(pickle.load(fp1))
                    labels_train = np.array(pickle.load(fp2))

                samples_train = np.expand_dims(samples_train, 1)
                labels_train = np.expand_dims(labels_train, 1)

		permute = np.random.permutation(len(samples_train))
        	samples_train = np.array(samples_train)[permute]
                labels_train = np.array(labels_train)[permute]

                k = samples_train.shape[0] / batchSize

                # divide batches
                for i in range(k):
                    yield (
                    samples_train[i * batchSize:(i + 1) * batchSize], labels_train[i * batchSize:(i + 1) * batchSize])


def generate_train_combined(patchType1, patchType2, personList, batchSize=256):
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
                    inputs = [samples1_train[i * batchSize:(i + 1) * batchSize],
                              samples2_train[i * batchSize:(i + 1) * batchSize]]
                    labels = labels_train[i * batchSize:(i + 1) * batchSize]
                    t = (inputs, labels)
                    yield (inputs, labels)


def aggregate_test(personList, patchType):
    i = 1
    for index in personList:
        for index2 in range(1, 5):
            with open(patches + "patches_"+patchType+ "_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                if i == 1:
                    samples_test = pickle.load(fp1)
                    labels_test = pickle.load(fp2)
                    i = 2
                else:
                    samples_test = np.append(samples_test, pickle.load(fp1), axis=0)
                    labels_test = np.append(labels_test, pickle.load(fp2), axis=0)

    samples_test = np.expand_dims(samples_test, 1)
    labels_test = np.expand_dims(labels_test, 1)

    return (samples_test, labels_test)


def aggregate_val(personList, patchType):
    i = 1
    for index in personList:
        for index2 in range(1, 5):
            with open(patches + "patches_"+patchType+ "_val_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels__val_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                if i == 1:
                    samples_val = pickle.load(fp1)
                    labels_val = pickle.load(fp2)
                    i = 2
                else:
                    t1 = pickle.load(fp1)
                    t2 = pickle.load(fp2)
                    samples_val = np.append(samples_val, t1, axis=0)
                    labels_val = np.append(labels_val, t2, axis=0)

    samples_val = np.array(samples_val)
    labels_val = np.array(labels_val)

    samples_val = np.expand_dims(samples_val, 1)
    labels_val = np.expand_dims(labels_val, 1)

    return (samples_val, labels_val)


def calc_confusion_mat(model,samples,labels,identifier=None):
    predict = model.predict(samples).round()
    confusion_mat = confusion_matrix(labels,predict)
    logger.debug("confusion_mat {} is {} ".format(identifier, str(confusion_mat)))
    return confusion_mat


def calc_dice(confusion_mat,identifier):
    dice = float(2) * confusion_mat[1][1] / (
        2 * confusion_mat[1][1] + confusion_mat[1][0] + confusion_mat[0][1])
    logger.info("model {} dice {} is ".format(identifier,dice))

# create run folder
time = datetime.now().strftime('%d_%m_%Y_%H_%M')
run_dir = runs_dir+time + '/'
if not path.exists(run_dir):
    makedirs(run_dir)
# create logger
logger = get_logger(run_dir)

# ######## create callbacks

logger.info("creating callbacks")
stop_train_callback1 = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
stop_train_callback2 = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, verbose=1, mode='auto')
print_logs = LambdaCallback(on_epoch_end=lambda epoch, logs:
logger.debug("epoch {} loss {:.5f} acc {:.5f} fmeasure {:.5f} val_loss {:.5f} val_acc {:.5f} val_fmeasure{:.5f} ".
             format(epoch, logs['loss'], logs['acc'],logs['fmeasure'], logs['val_loss'], logs['val_acc'],logs['val_fmeasure'])))

mycallbacks = [print_logs,stop_train_callback1, stop_train_callback2]
predictors = []
logger.info("creating models")
for i in range(1):
    predictors.append(one_predictor_model(index=i))
    predictors[i].compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy','fmeasure'])

PersonTrainList = [1,2,3,4]
val_axial_set,val_axial_labels = aggregate_val(PersonTrainList,"axial")
val_coronal_set,val_coronal_labels = aggregate_val(PersonTrainList,"coronal")
val_sets = [(val_axial_set,val_axial_labels),(val_coronal_set,val_coronal_labels)]
combined_val = ([val_axial_set,val_coronal_set],val_coronal_labels)
######## train individual predictors
logger.info("training individual models")
axial_generator = generate_train("axial", PersonTrainList)
coronal_generator = generate_train("coronal", PersonTrainList)
train_generator = [axial_generator, coronal_generator]

for i in range(1):
    logger.debug("training individual model {}".format(i))
    history = predictors[i].fit_generator(train_generator[i], samples_per_epoch=340000, nb_epoch=50, callbacks=mycallbacks,
                                 nb_worker=4, validation_data=val_sets[i])
    predictors[i].save_weights(run_dir +'%d.h5' % (i))
# ######## test individual predictors
logger.info("testing individual models")

test_axial_samples, test_axial_labels = aggregate_test([5], "axial")
test_coronal_samples, test_coronal_labels = aggregate_test([5], "coronal")

test_samples = [test_axial_samples, test_coronal_samples]
test_labels = [test_axial_labels, test_coronal_labels]

results = []
predictions = []

for i in range(1):
    results.append(predictors[i].evaluate(test_samples[i], test_labels[i]))
    predictions.append(predictors[i].predict(test_samples[i]))
    logger.info("predictor {} loss {} acc {}".format(i, results[i][0], results[i][1]))
    confusion_mat = calc_confusion_mat(predictors[i], val_sets[i][0], val_sets[i][1], "individual val {}".format(i))
    calc_dice(confusion_mat, "individual val {}".format(i))
    confusion_mat = calc_confusion_mat(predictors[i], test_samples[i], test_labels[i], "individual test {}".format(i))
    calc_dice(confusion_mat, "individual test {}".format(i))

metrics = ['acc','val_acc','loss','val_loss','fmeasure','val_fmeasure']
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pylab as plt
for i in [0,2,4]:
    plt.clf()
    plt.plot(history.history[metrics[i]])
    plt.plot(history.history[metrics[i+1]])
    plt.title('model ' + metrics[i])
    plt.ylabel(metrics[i])
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(run_dir+'model_' + metrics[i]+'.png')
