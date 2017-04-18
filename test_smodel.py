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
from keras.optimizers import SGD
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

def generator(positive_list,negative_list,data,batch_size=256,patch_width = 16,only_once=False):
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
        if only_once:
            break

def calc_epoch_size(patch_list,batch_size):
    batch_pos = batch_size / 2
    batch_num = len(patch_list) / batch_pos
    return batch_num * batch_pos*2

def aggregate_genrated_samples(pos_list,neg_list,data):
    samples = []
    labels = []
    for batch_samples,batch_labels in generator(pos_list,neg_list,data,only_once=True):
        samples.extend(batch_samples)
        labels.extend(batch_labels)

    return (np.array(samples), np.array(labels))




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
    val_set = aggregate_genrated_samples(pos_val_list, neg_val_list, val_images)

    logger.info("training individual model")
    epoch_size = calc_epoch_size(pos_train_list,batch_size)
    history = model.fit_generator(train_generator, samples_per_epoch=epoch_size, nb_epoch=50, callbacks=callbacks,
                                      validation_data=val_set)
    confusion_mat = calc_confusion_mat(model, val_set[0], val_set[1], "individual val {}".format(fold_num))
    calc_dice(confusion_mat, "individual val {}".format(fold_num))
    return history

def test(model,patch_type,testList,name):
    # ######## test individual predictors
    logger.info("testing individual models {}".format(name))
    test_images, pos_test_list, neg_test_list = load_data(testList)
    #test_generator = generator(pos_test_list, neg_test_list, test_images,only_once=True)
    test_samples,test_labels = aggregate_genrated_samples(pos_test_list, neg_test_list, test_images)
    results = model.evaluate(test_samples,test_labels)
    #predictions = model.predict(test_samples)
    logger.info("predictor loss {} acc {}".format(results[0], results[1]))
    confusion_mat = calc_confusion_mat(model, test_samples, test_labels, "individual test ")
    calc_dice(confusion_mat, "individual test ")

def probability_plot(model, vol,fold,threshold=0.5,slice = 95):
    import itertools

    prob_plot = np.zeros(vol.shape)
    final_decision = np.zeros(vol.shape)

    x = np.linspace(0, vol.shape[2] - 1, vol.shape[2], dtype='int')
    y = np.linspace(0, vol.shape[1] - 1, vol.shape[1], dtype='int')
    z = np.linspace(0, vol.shape[0] - 1, vol.shape[0], dtype='int')
    voxel_list = itertools.product(y,x)
    patches_list = []
    logger.info("patches for model")
    for j, k in voxel_list:
        axial_p = extract_axial(vol, k, j, slice,16)
        if type(axial_p) == np.ndarray:
            patches_list.append((slice,j,k,axial_p))

    patches = [v[3] for v in patches_list]

    patches = np.expand_dims(patches, 1)
    logger.info("predict model")

    predictions = model.predict(patches)
    for index,(slice, j, k,_) in enumerate(patches_list):
        if predictions[index] > threshold:
            final_decision[slice, j, k] = 255
        prob_plot[slice, j, k] = predictions[index] * 255

    params = {'figure_name': "slice_prob_{}".format(fold),"image" : prob_plot[slice, :, :],
              "image_att": dict(cmap=matplotlib.cm.gray),'save_file': run_dir +'slice_prob_{}'.format(fold) + '.png'}
    generic_plot(params)
    params["image"] = final_decision[slice, :, :]
    params["save_file"] = run_dir +'slice_decision_{}'.format(fold) + '.png'
    generic_plot(params)

def predict_image(model, vol,threshold=0.5):
    import itertools

    prob_plot = np.zeros(vol.shape,dtype='uint8')
    segmentation = np.zeros(vol.shape,dtype='uint8')

    x = np.linspace(0, vol.shape[2] - 1, vol.shape[2], dtype='int')
    y = np.linspace(0, vol.shape[1] - 1, vol.shape[1], dtype='int')
    z = np.linspace(0, vol.shape[0] - 1, vol.shape[0], dtype='int')
    voxel_list = itertools.product(y,x)
    patches_list = []
    logger.info("patches for model")
    for i in z:
        for j, k in voxel_list:
            axial_p = extract_axial(vol, k, j, i,16)
            if type(axial_p) == np.ndarray:
                patches_list.append((i,j,k,axial_p))

        patches = [v[3] for v in patches_list]

        patches = np.expand_dims(patches, 1)
        logger.info("predict model")

        predictions = model.predict(patches)
        for index,(i, j, k,_) in enumerate(patches_list):
            if predictions[index] > threshold:
                segmentation[i, j, k] = 1
            prob_plot[i, j, k] = predictions[index] * 255

    return segmentation

    # params = {'figure_name': "slice_prob_{}".format(fold),"image" : prob_plot[slice, :, :],
    #           "image_att": dict(cmap=matplotlib.cm.gray),'save_file': run_dir +'slice_prob_{}'.format(fold) + '.png'}
    # generic_plot(params)
    # params["image"] = final_decision[slice, :, :]
    # params["save_file"] = run_dir +'slice_decision_{}'.format(fold) + '.png'
    # generic_plot(params)




def generic_plot(kwargs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 10
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
    if kwargs.has_key("image_att"):
        image_attribute = kwargs["image_att"]
    else:
        image_attribute = {}
    if kwargs.has_key("x"):
        plt.plot(kwargs["x"],kwargs["y"],**line_attribute)
    elif  kwargs.has_key("y"):
        plt.plot(kwargs["y"],**line_attribute)
    elif kwargs.has_key("image"):
        plt.imshow(kwargs["image"],**image_attribute)
    if kwargs.has_key("legend"):
        plt.legend(kwargs["legend"], loc=0)
    if kwargs.has_key("save_file"):
        plt.savefig(kwargs["save_file"],dpi=100)

def plot_training(logs):
    metrics = ['acc', 'val_acc', 'loss', 'val_loss', 'fmeasure', 'val_fmeasure']
    linestyles = ['-', '--']
    colors = ['b','y','r','g']
    for j,history in enumerate(logs):
        for i in [0,2,4]:
            params = {'figure_name': metrics[i], 'y':history[metrics[i]],'title':'model ' + metrics[i],
                      'ylabel':metrics[i],'xlabel':'epoch',"line_att":dict(linestyle=linestyles[0],color=colors[j])}
            generic_plot(params)
            params = {'figure_name': metrics[i], 'y':history[metrics[i+1]],"line_att":dict(linestyle=linestyles[1],color=colors[j])}
            generic_plot(params)
    for i in [0, 2, 4]:
        params = {'figure_name': metrics[i], 'legend': ['train', 'validation']*len(logs),
                  'save_file': run_dir + 'model_' + metrics[i] + '.png'}
        generic_plot(params)

# create run folder
time = datetime.now().strftime('%d_%m_%Y_%H_%M')
run_dir = '/media/sf_shared/src/medicalImaging/tmp/tm2/play/'
if not path.exists(run_dir):
    makedirs(run_dir)
# create logger
logger = get_logger(run_dir)

# ######## train model
logger.info("use adadelta")
person_indices =np.array([1,2,3,4])
kf = KFold(n_splits=4)
runs = []
predictors = []

for i,(train_index, val_index) in enumerate(kf.split(person_indices)):
    logger.info("Train: {} Val {} ".format( person_indices[train_index],person_indices[val_index]) )
    predictors.append(one_predictor_model())
    sgd = SGD(nesterov=True)
    predictors[i].compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    predictors[i].load_weights(run_dir + 'model_{}_fold_{}.h5'.format(i,i))
    #history = train(predictors[i],person_indices[train_index],person_indices[val_index], "axial", i, name=i)
    #runs.append(history.history)

# with open(run_dir + 'cross_valid_stats.lst', 'wb') as fp:
#         pickle.dump(runs, fp)
# plot_training(runs)


with open(run_dir + 'cross_valid_stats.lst', 'rb') as fp:
       runs = pickle.load(fp)

FLAIR_filename = Src_Path+Data_Path+"Person05_Time01_FLAIR.npy"
vol = np.load(FLAIR_filename)
# test model
#for i in range(4):
    #test(predictors[i],"axial",[5],i)
    #probability_plot(predictors[i],vol,i,slice=80,threshold=0.8)
seg = predict_image(predictors[0],vol)
with open(run_dir + 'segmantation.npy', 'wb') as fp:
    np.save(fp,seg)

