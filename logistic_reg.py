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
#np.random.seed(42)
import nibabel as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from os import path, makedirs
from datetime import datetime

import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier

from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score ,roc_auc_score ,recall_score,precision_score
import scipy.ndimage.morphology as mrph



from logging_tools import get_logger

weight_path = r'./trained_weights/'
patches = r'./patches/'
runs_dir = r'./runs/'
Labels_Path = r"seg/"
Src_Path = r"./train/"
Data_Path = r"data/"
WM_path = r"WM/"

def binary_disk(r):
    arr = np.ones((2*r+1,2*r+1,2*r+1))
    arr[r,r,r] = 0

    dt = mrph.distance_transform_bf(arr,metric='euclidean')
    disk = dt<=r
    disk = disk.astype('float32')

    return disk


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
        # positive_list_np = positive_list[:batch_num*batch_pos]
        # negative_list_np = negative_list
        for batch in range(batch_num):
            positive_batch = positive_list_np[batch*batch_pos:(batch+1)*batch_pos]
            positive_batch_patches = [[extract_axial(data[person][time],k,j,i,patch_width),1] for person,time,i,j,k in positive_batch]
            negative_batch = negative_list_np[batch * batch_pos:(batch + 1) * batch_pos]
            negative_batch_patches = [[extract_axial(data[person][time], k, j, i,patch_width),0] for person, time, i, j, k in
                                      negative_batch]
            final_batch =  np.random.permutation(positive_batch_patches + negative_batch_patches) #positive_batch_patches + negative_batch_patches
            samples =  [patches for patches,_ in final_batch]
            samples = np.expand_dims(samples, 1)

            labels = [labels for _,labels in final_batch]
            yield (samples,labels)
        if only_once:
            break

def aggregate_genrated_samples(pos_list,neg_list,data):
    samples = []
    labels = []
    for batch_samples,batch_labels in generator(pos_list,neg_list,data,only_once=True):
        samples.extend(batch_samples)
        labels.extend(batch_labels)

    return (np.array(samples), np.array(labels))

def calc_epoch_size(patch_list,batch_size):
    batch_pos = batch_size / 2
    batch_num = len(patch_list) / batch_pos
    return batch_num * batch_pos*2 ,batch_num




def calc_confusion_mat(model,samples,labels,identifier=None):
    predict = model.predict(samples).round()
    confusion_mat = confusion_matrix(labels,predict)
    logger.debug("confusion_mat {} is {} ".format(identifier, str(confusion_mat)))
    return confusion_mat


def calc_dice(confusion_mat,identifier):
    dice = float(2) * confusion_mat[1][1] / (
        2 * confusion_mat[1][1] + confusion_mat[1][0] + confusion_mat[0][1])
    logger.info("model {} dice {} is ".format(identifier,dice))



def train(PersonTrainList,PersonValList,patch_type,fold_num,name,batch_size=256):
    logger.debug("training model {} fold {}".format(name,fold_num))
    logger.debug("creating callbacks")

    logger.debug("creating train & val generators")

    train_images,pos_train_list,neg_train_list = load_data(PersonTrainList)
    # pos_train_list = np.random.permutation(pos_train_list)
    # neg_train_list = np.random.permutation(neg_train_list)

    train_generator = generator(pos_train_list, neg_train_list, train_images)

    val_images,pos_val_list,neg_val_list = load_data(PersonValList)
    val_set = aggregate_genrated_samples(pos_val_list, neg_val_list, val_images)
    val_set_x = reshape_input(val_set[0])
    logger.info("training individual model")
    epoch_size,batch_num = calc_epoch_size(pos_train_list,batch_size)

    model = SGDClassifier(loss='log')
    epoch_num = 30
    for i in range(epoch_num):
        for j in range(batch_num):
            x,y = train_generator.next()
            x=reshape_input(x)
            model.partial_fit(x,y,classes= np.unique(y))
        predicted_val = model.predict(val_set_x)
        args = val_set[1],predicted_val
        logger.info("val f1 {}, auc {} , recall {} , precision {}  ".format(
            f1_score(*args),roc_auc_score(*args),recall_score(*args),precision_score(*args)))
    return model
    # confusion_mat = calc_confusion_mat(model, val_set[0], val_set[1], "individual val {}".format(fold_num))
    # calc_dice(confusion_mat, "individual val {}".format(fold_num))
    # return history

def reshape_input(x):
    return x.squeeze(axis=1).reshape(x.shape[0],-1)

def post_process(seg, thresh):
    from scipy import ndimage
    connected_comp = ndimage.generate_binary_structure(3, 2) * 1
    label_weight = 30
    connected_comp[1, 1, 1] = label_weight
    res = ndimage.convolve(seg, connected_comp, mode='constant', cval=0.)
    return (res > (thresh + label_weight)) * 1

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


def predict_image(model, vol,mask, threshold=0.5):
    import itertools

    prob_plot = np.zeros(vol.shape, dtype='float16')
    segmentation = np.zeros(vol.shape, dtype='uint8')

    x = np.linspace(0, vol.shape[2] - 1, vol.shape[2], dtype='int')
    y = np.linspace(0, vol.shape[1] - 1, vol.shape[1], dtype='int')
    z = np.linspace(0, vol.shape[0] - 1, vol.shape[0], dtype='int')
    logger.info("patches for model")
    for i in z:
        patches_list = []
        voxel_list = itertools.product(y, x)
        for j, k in voxel_list:
            if mask[i][j][k] :
                axial_p = extract_axial(vol, k, j, i, 16)
                if type(axial_p) == np.ndarray:
                    patches_list.append((i, j, k, axial_p))
        if len(patches_list) > 0:
            patches = [v[3] for v in patches_list]

            patches = np.expand_dims(patches, 1)
            patches= reshape_input(patches)
            predictions = model.predict(patches)
            for index, (i, j, k, _) in enumerate(patches_list):
                if predictions[index] > threshold:
                    segmentation[i, j, k] = 1
                prob_plot[i, j, k] = predictions[index]

    return segmentation ,prob_plot

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
run_dir = runs_dir+time + '/'
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
    logger.info("Train: {} Val {} ".format(person_indices[train_index] ,person_indices[val_index]) )
    #predictors[i].load_weights(run_dir + 'model_{}_fold_{}.h5'.format(i,i))
    model = train(person_indices[train_index] ,person_indices[val_index], "axial", i, name=i)
    break



# with open(run_dir + 'cross_valid_stats.lst', 'rb') as fp:
#        runs = pickle.load(fp)


#test segmantation
person=5
mri_time=1

FLAIR_filename = Src_Path+Data_Path+"Person0{}_Time0{}_FLAIR.npy".format(person,mri_time)
vol = np.load(FLAIR_filename)

FLAIR_labels_1 = Src_Path + Labels_Path + "training0{}_0{}_mask1.nii".format(person,mri_time)
labels = nb.load(FLAIR_labels_1).get_data()
labels = labels.T
labels = np.rot90(labels, 2, axes=(1, 2))
test_labels = labels.flatten().tolist()



#create mask
wm_mask = np.load(Src_Path + WM_path + "Person0{}_Time0{}.npy".format(person, mri_time))
sel = binary_disk(2)
WM_dilated = mrph.filters.maximum_filter(wm_mask, footprint=sel)

# apply thresholds
FLAIR_th = 0.91
WM_prior_th = 0.5
FLAIR_mask = vol > FLAIR_th
WM_mask = WM_dilated > WM_prior_th

# final mask: logical AND
candidate_mask = np.logical_and(FLAIR_mask, WM_mask)


# test model
for i in range(1):
    #test(predictors[i],"axial",[5],i)
    # probability_plot(predictors[i],vol,i,slice=80,threshold=0.8)
    segmantation,prob_map = predict_image(model,vol,candidate_mask)
    segmantation = post_process(segmantation,9)
    with open(run_dir + 'segmantation{}.npy'.format(i), 'wb') as fp, open(run_dir + 'prob_plot{}.npy'.format(i), 'wb') as fp1:
        np.save(fp, segmantation)
        np.save(fp1, prob_map)
    test_seg = segmantation.flatten().tolist()
    logger.info("predictor {} f1 is {} , auc is {} recall is {} precision is {} ".format
                    (i,f1_score(test_labels, test_seg),roc_auc_score(test_labels,test_seg) ,
                     recall_score(test_labels,test_seg),precision_score(test_labels,test_seg)))


