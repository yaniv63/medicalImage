# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:32:39 2016

@author: yaniv
"""

# -*- coding: utf-8 -*-
from sklearn import pipeline

"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
import numpy as np
from os import path, makedirs
from datetime import datetime
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


from two_predictors_combined import one_predictor_model
from logging_tools import get_logger

weight_path = r'./trained_weights/'
patches = r'./patches/'
runs_dir = r'./runs/'

def extract_axial(vol,xc, yc, zc, sz, w):
    try:
        x = np.arange(xc - w, xc + w , 1)
        y = np.arange(yc - w, yc + w , 1)
        indexes = np.ix_(y, x)
        patch = vol[zc][indexes]
        return  patch
    except IndexError as e:
        return 0

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

def create_callbacks(name,fold):
    save_weights = ModelCheckpoint(filepath=run_dir + 'model_{}_fold_{}.h5'.format(name, fold), monitor='val_acc',
                                   save_best_only=True,
                                   save_weights_only=True)
    stop_train_callback1 = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
    stop_train_callback2 = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, verbose=1, mode='auto')
    print_logs = LambdaCallback(on_epoch_end=lambda epoch, logs:
    logger.debug("epoch {} loss {:.5f} acc {:.5f} fmeasure {:.5f} val_loss {:.5f} val_acc {:.5f} val_fmeasure{:.5f} ".
                 format(epoch, logs['loss'], logs['acc'], logs['fmeasure'], logs['val_loss'], logs['val_acc'],
                        logs['val_fmeasure'])))
    mycallbacks = [print_logs, stop_train_callback1, stop_train_callback2,save_weights]
    return mycallbacks

def train(model,PersonTrainList,PersonValList,patch_type,fold_num,name):
    logger.debug("training model {} fold {}".format(name,fold_num))
    logger.debug("creating callbacks")
    callbacks = create_callbacks(name,fold_num)
    logger.debug("creating validation set")
    val_set = aggregate_val(PersonValList,patch_type)
    logger.info("training individual model")
    train_generator = generate_train(patch_type, PersonTrainList)
    history = model.fit_generator(train_generator, samples_per_epoch=1000, nb_epoch=5, callbacks=callbacks,
                                      validation_data=val_set)
    confusion_mat = calc_confusion_mat(model, val_set[0], val_set[1], "individual val {}".format(fold_num))
    calc_dice(confusion_mat, "individual val {}".format(fold_num))
    return history

def test(model,patch_type,testList):
    # ######## test individual predictors
    logger.info("testing individual models")
    test_samples, test_labels = aggregate_test(testList, patch_type)
    results = model.evaluate(test_samples, test_labels)
    #predictions = model.predict(test_samples)
    logger.info("predictor loss {} acc {}".format(results[0], results[1]))
    confusion_mat = calc_confusion_mat(model, test_samples, test_labels, "individual test ")
    calc_dice(confusion_mat, "individual test ")

def plot_training(logs):
    metrics = ['acc', 'val_acc', 'loss', 'val_loss', 'fmeasure', 'val_fmeasure']
    linestyles = ['-', '--', '-.', ':']
    for j,history in enumerate(logs):
        for i in [0,2,4]:
            params = {'figure_name': metrics[i], 'y':history.history[metrics[i]],'title':'model ' + metrics[i],
                      'ylabel':metrics[i],'xlabel':'epoch',"line_att":dict(linestyle=linestyles[j])}
            generic_plot(params)
            params = {'figure_name': metrics[i], 'y':history.history[metrics[i+1]],"line_att":dict(linestyle=linestyles[j])}
            generic_plot(params)
    for i in [0, 2, 4]:
        params = {'figure_name': metrics[i], 'legend': ['train', 'validation']*len(logs),
                  'save_file': run_dir + 'model_' + metrics[i] + '.png'}
        generic_plot(params)


def generic_plot(kwargs):
    import matplotlib
    matplotlib.use('Agg')
    # import pylab as plt
    import matplotlib.pyplot as plt
    if kwargs.has_key("figure_name"):
        f1 = plt.figure(kwargs["figure_name"])
    if kwargs.has_key("title"):
        plt.title(kwargs["title"])
    if kwargs.has_key("ylabel"):
        plt.ylabel(kwargs["ylabel"])
    if kwargs.has_key("xlabel"):
        plt.xlabel(kwargs["xlabel"])
    if kwargs.has_key("line_att"):
        line_attribute = kwargs["line_att"]
    else:
        line_attribute = ''
    if kwargs.has_key("x"):
        plt.plot(kwargs["x"],kwargs["y"],**line_attribute)
    elif  kwargs.has_key("y"):
        plt.plot(kwargs["y"],**line_attribute)
    if kwargs.has_key("legend"):
        plt.legend(kwargs["legend"], loc='upper left')
    if kwargs.has_key("save_file"):
        plt.savefig(kwargs["save_file"])

def probability_plot(model, vol):
    prob_plot = np.zeros(vol.shape)
    import itertools
    from scipy.interpolate import RegularGridInterpolator
    from prepro_pipeline import  sz, w
    import matplotlib
    import matplotlib.pylab as plt
    x = np.linspace(0, vol.shape[2] - 1, vol.shape[2], dtype='int')
    y = np.linspace(0, vol.shape[1] - 1, vol.shape[1], dtype='int')
    z = np.linspace(0, vol.shape[0] - 1, vol.shape[0], dtype='int')
    interp3 = RegularGridInterpolator((z, y, x), vol)
    voxel_list = itertools.product(y,x)
    patches_list = []
    i=100
    logger.info("patches for model")
    for j, k in voxel_list:
        axial_p = extract_axial(vol, k, j, i, sz, w)
        if type(axial_p) == np.ndarray:
            patches_list.append((i,j,k,axial_p))

    patches = [v[3] for v in patches_list]
    patches = np.expand_dims(patches, 1)
    logger.info("predict model")

    predictions = model.predict(patches)
    for index,(i, j, k,_) in enumerate(patches_list):
        prob_plot[i, j, k] = predictions[index]*255
    plt.imshow(prob_plot[100, :, :], cmap=matplotlib.cm.gray)
    plt.savefig(run_dir +'slice_prob' + '.png')

# create run folder
time = datetime.now().strftime('%d_%m_%Y_%H_%M')
run_dir = runs_dir+time + '/'
if not path.exists(run_dir):
    makedirs(run_dir)
# create logger
logger = get_logger(run_dir)

# ######## train model

logger.info("creating model")
predictor = one_predictor_model()
predictor.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy','fmeasure'])

# person_indices =np.array([1,2,3,4])
# kf = KFold(n_splits=4)
# runs = []
# for i,(train_index, val_index) in enumerate(kf.split(person_indices)):
#     print("TRAIN:", person_indices[train_index], "TEST:", person_indices[val_index])
#     history = train(predictor,person_indices[train_index],person_indices[val_index], "axial", i, name=0)
#     runs.append(history)
# plot_training(runs)

# test model

Src_Path = r"./train/"
Data_Path = r"data/"
FLAIR_filename = Src_Path+Data_Path+"Person01_Time01_FLAIR.npy"
vol = np.load(FLAIR_filename)
probability_plot(predictor,vol)


