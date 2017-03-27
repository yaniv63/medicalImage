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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os import path, makedirs
from datetime import datetime
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from collections import defaultdict


from two_predictors_combined import one_predictor_model
from logging_tools import get_logger

weight_path = r'./trained_weights/'
patches = r'./patches/'
runs_dir = r'./runs/'
Src_Path = r"./train/"
Data_Path = r"data/"

def extract_axial(vol,xc, yc, zc, w):
    try:
        x = np.arange(xc - w, xc + w , 1)
        y = np.arange(yc - w, yc + w , 1)
        indexes = np.ix_(y, x)
        patch = vol[zc][indexes]
        return  patch
    except IndexError as e:
        return 0

def load_patches_list(person_list):
    import pickle
    with open(patches + "positive_list_person_{}.lst".format(str(person_list)), 'rb') as fp1, \
            open(patches + "negative_list_person_{}.lst".format(str(person_list)), 'rb') as fp2:
            positive_list_np = np.array(pickle.load(fp1))
            negative_list_np = np.array(pickle.load(fp2))
    return positive_list_np,negative_list_np

#
def load_images(person_list):
    image_list =defaultdict(dict)
    for person in person_list:
        for time in range(1,5):
            image_list[person][time] = np.load(Src_Path+Data_Path+"Person0{}_Time0{}_FLAIR.npy".format(person,time))
    return image_list

def load_data(person_list):
    pos_list,neg_list = load_patches_list(person_list)
    images = load_images(person_list)
    return images,pos_list,neg_list

def generator(positive_list,negative_list,data,batch_size=256,patch_width = 16):
    batch_pos = batch_size/2
    batch_num = len(positive_list)/batch_pos
    while True:
        #modify list to divide by batch_size
        positive_list_np = np.random.permutation(positive_list)
        positive_list_np = positive_list_np[:batch_num*batch_pos]
        negative_list_np = np.random.permutation(negative_list)
        for batch in range(batch_num):
            positive_batch = positive_list_np[batch*batch_pos:(batch+1)*batch_pos]
            positive_batch_patches = [[extract_axial(data[person][time],k,j,i,patch_width),1] for person,time,i,j,k in positive_batch]
            negative_batch = negative_list_np[batch * batch_pos:(batch + 1) * batch_pos]
            negative_batch_patches = [[extract_axial(data[person][time], k, j, i,patch_width),0] for person, time, i, j, k in
                                      negative_batch]
            final_batch = np.random.permutation(positive_batch_patches + negative_batch_patches)
            samples =  [patches for patches,_ in final_batch]
            samples = np.expand_dims(samples, 1)

            labels = [labels for _,labels in final_batch]
            yield (samples,labels)

def calc_epoch_size(patch_list,batch_size):
    batch_pos = batch_size / 2
    batch_num = len(patch_list) / batch_pos
    return batch_num * batch_pos*2

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

def train(model,PersonTrainList,PersonValList,patch_type,fold_num,name,batch_size=256):
    logger.debug("training model {} fold {}".format(name,fold_num))
    logger.debug("creating callbacks")
    callbacks = create_callbacks(name,fold_num)

    logger.debug("creating train & val generators")

    train_images,pos_train_list,neg_train_list = load_data(PersonTrainList)
    train_generator = generator(pos_train_list, neg_train_list, train_images)

    val_images,pos_val_list,neg_val_list = load_data(PersonValList)
    val_generator = generator(pos_val_list, neg_val_list, val_images)

    logger.info("training individual model")
    epoch_size = calc_epoch_size(pos_train_list,batch_size)
    val_size = calc_epoch_size(pos_val_list,batch_size)
    history = model.fit_generator(train_generator, samples_per_epoch=epoch_size, nb_epoch=5, callbacks=callbacks,
                                      validation_data=val_generator,nb_val_samples=val_size)
    #confusion_mat = calc_confusion_mat(model, val_set[0], val_set[1], "individual val {}".format(fold_num))
    #calc_dice(confusion_mat, "individual val {}".format(fold_num))
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
    #import pylab as plt
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
    import itertools
    from scipy.interpolate import RegularGridInterpolator

    prob_plot = np.zeros(vol.shape)
    x = np.linspace(0, vol.shape[2] - 1, vol.shape[2], dtype='int')
    y = np.linspace(0, vol.shape[1] - 1, vol.shape[1], dtype='int')
    z = np.linspace(0, vol.shape[0] - 1, vol.shape[0], dtype='int')
    interp3 = RegularGridInterpolator((z, y, x), vol)
    voxel_list = itertools.product(y,x)
    patches_list = []
    i=100
    logger.info("patches for model")
    for j, k in voxel_list:
        axial_p = extract_axial(vol, k, j, i,16)
        if type(axial_p) == np.ndarray:
            patches_list.append((i,j,k,axial_p))

    patches = [v[3] for v in patches_list]

    patches = np.expand_dims(patches, 1)
    logger.info("predict model")

    predictions = model.predict(patches)
    for index,(i, j, k,_) in enumerate(patches_list):
        prob_plot[i, j, k] = predictions[index]*255
    plt.clf()
    plt.imshow(prob_plot[i, :, :], cmap=matplotlib.cm.gray)
    plt.savefig(run_dir +'slice_prob' + '.png')

# create run folder
time = datetime.now().strftime('%d_%m_%Y_%H_%M')
run_dir = runs_dir+time + '/'
if not path.exists(run_dir):
    makedirs(run_dir)
# create logger
logger = get_logger(run_dir)

# ######## train model


person_indices =np.array([1,2,3,4])
kf = KFold(n_splits=4)
runs = []
predictors = []
for i,(train_index, val_index) in enumerate(kf.split(person_indices)):
    print("TRAIN:", person_indices[train_index], "TEST:", person_indices[val_index])
    predictors.append(one_predictor_model())
    predictors[i].compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    history = train(predictors[i],person_indices[train_index],person_indices[val_index], "axial", i, name=i)
    runs.append(history)
plot_training(runs)

# test model

FLAIR_filename = Src_Path+Data_Path+"Person05_Time01_FLAIR.npy"
vol = np.load(FLAIR_filename)
probability_plot(predictors[0],vol)


