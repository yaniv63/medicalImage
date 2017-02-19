# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""

from keras.callbacks import EarlyStopping, LambdaCallback
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

from two_predictors_combined import one_predictor_model, average_two_models_prediction, two_parameters_combined_model, \
    two_predictors_combined_model
from logging_tools import get_logger

weight_path = r'./trained_weights/'
patches = r'./patches/'

logger = get_logger()


def generate_train(patchType, personList, batchSize=5000):
    while True:
        for index in personList:
            for index2 in range(1, 5):
                with open(patches + "patches_"+patchType+ "_train_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels_train_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                    samples_train = np.array(pickle.load(fp1))
                    labels_train = np.array(pickle.load(fp2))

                samples_train = np.expand_dims(samples_train, 1)
                labels_train = np.expand_dims(labels_train, 1)
                k = samples_train.shape[0] / batchSize

                # divide batches
                for i in range(k):
                    yield (
                    samples_train[i * batchSize:(i + 1) * batchSize], labels_train[i * batchSize:(i + 1) * batchSize])


def generate_train_combined(patchType1, patchType2, personList, batchSize=5000):
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


# ######## create callbacks
logger.info("creating callbacks")
stop_train_callback1 = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
stop_train_callback2 = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=0, mode='auto')
print_logs = LambdaCallback(on_epoch_end=lambda epoch, logs:
logger.debug("epoch {} loss {:.5f} acc {:.5f} val_los {:.5f} val_acc {:.5f}".
             format(epoch, logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc'])))

mycallbacks = [print_logs,stop_train_callback1, stop_train_callback2]
predictors = []
logger.info("creating models")
for i in range(2):
    predictors.append(one_predictor_model(index=i))
    predictors[i].compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy','fmeasure'])

combined_model = [average_two_models_prediction(), two_parameters_combined_model(),
                      two_predictors_combined_model()]
for i in range(3):
    combined_model[i].compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy','fmeasure'])

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

for i in range(2):
    logger.debug("training individual model {}".format(i))
    predictors[i].fit_generator(train_generator[i], samples_per_epoch=500000, nb_epoch=8, callbacks=mycallbacks,
                                nb_worker=4, validation_data=val_sets[i])
    predictors[i].save_weights(weight_path + '%d.h5' % (i))
######## test individual predictors
logger.info("testing individual models")

test_axial_samples, test_axial_labels = aggregate_test([5], "axial")
test_coronal_samples, test_coronal_labels = aggregate_test([5], "coronal")

test_samples = [test_axial_samples, test_coronal_samples]
test_labels = [test_axial_labels, test_coronal_labels]

results = []
predictions = []

for i in range(2):
    results.append(predictors[i].evaluate(test_samples[i], test_labels[i]))
    predictions.append(predictors[i].predict(test_samples[i]))
    logger.info("predictor {} loss {} acc {}".format(i, results[i][0], results[i][1]))
    confusion_mat = calc_confusion_mat(predictors[i], val_sets[i][0], val_sets[i][1], "individual val {}".format(i))
    calc_dice(confusion_mat, "individual val {}".format(i))
    confusion_mat = calc_confusion_mat(predictors[i], test_samples[i], test_labels[i], "individual test {}".format(i))
    calc_dice(confusion_mat, "individual test {}".format(i))
######## train predictors combinations
logger.info("training combined models")
gen = generate_train_combined("axial", "coronal", PersonTrainList)

for i in range(3):
    logger.debug("training combined model {}".format(i))
    layer_dict = dict([(layer.name, layer) for layer in combined_model[i].layers])
    layer_dict["Seq_0"].load_weights(weight_path + '0.h5', by_name=True)
    layer_dict["Seq_1"].load_weights(weight_path + '1.h5', by_name=True)
    combined_model[i].fit_generator(gen, samples_per_epoch=500000, nb_epoch=8, callbacks=mycallbacks,
                                    validation_data=combined_val)
    combined_model[i].save_weights(weight_path + 'combined_%d.h5' % (i))
######## test predictors combinations
logger.info("test combined models")

combined_model_results = [r.evaluate(test_samples, test_labels[0]) for r in combined_model]
for i in range(3):
    logger.info(
        "combined_model {} loss {} acc {}".format(i, combined_model_results[i][0], combined_model_results[i][1]))
#avg predicor
avg_predict = ((predictions[0] + predictions[1]) / 2).round()
avg_success = np.equal(avg_predict, test_labels[0])
avg_precentage = avg_success.tolist().count([True]) / float(len(avg_predict))
logger.info("avg_precentage  (acc)  {}".format(avg_precentage))

for i in range(3):
    confusion_mat = calc_confusion_mat(combined_model[i], combined_val[0],  combined_val[1], "combined val {}".format(i))
    calc_dice(confusion_mat, "combined val {}".format(i))
    confusion_mat = calc_confusion_mat(combined_model[i],test_samples,test_labels[0],"combined test {}".format(i))
    calc_dice(confusion_mat,"combined test {}".format(i))
logger.info("finish train and test models")



