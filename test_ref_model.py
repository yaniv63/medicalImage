# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:32:39 2016

@author: yaniv
"""

# -*- coding: utf-8 -*-
import Queue

from sklearn import pipeline

"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading

from os import path, makedirs
from datetime import datetime
import nibabel as nb

import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from collections import defaultdict
from itertools import product
from scipy.interpolate import RegularGridInterpolator
import scipy.ndimage.morphology as mrph

from MIMTP_Detection_demo import create_full_model
from logging_tools import get_logger
from prepro_pipeline import binary_disk, FLAIR_th, WM_prior_th

weight_path = r'./trained_weights/'
patches = r'./patches/'
Labels_Path = r"seg/"
runs_dir = r'./runs/'
Src_Path = r"./train/"
Data_Path = r"data/"
WM_path = r"WM/"


def multi_dimensions(n, type=None):
    if n <= 0:
        if type is not None:
            return type()
        return None
    return defaultdict(lambda: multi_dimensions(n - 1, type))


def can_extract_patch(shape, xc, yc, zc, w):
    return (xc - w + 0.5) >= 0 and (xc + w + 0.5) <= (shape[2] - 1) and (yc - w + 0.5) >= 0 and\
           (yc + w + 0.5) <= (shape[1] - 1) and (zc - w + 0.5) >= 0 and (zc + w + 0.5) <= (shape[0] - 1)


def extract_axial(interp3, xc, yc, zc, sz, w):
    x = np.arange(xc - w + 0.5, xc + w + 0.5, 1)
    y = np.arange(yc + w + 0.5, yc - w + 0.5, -1)

    # axial patch voxels
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape((xx.shape[0] * xx.shape[1], 1))
    yy = yy.reshape((yy.shape[0] * yy.shape[1], 1))
    zz = zc * np.ones(xx.shape)
    pts = np.concatenate((zz, yy, xx), axis=1)

    # interpolate
    p_axial = interp3(pts)
    p_axial = p_axial.reshape((sz, sz))
    p_axial = np.flipud(p_axial)
    return p_axial


# In[30]:

def extract_coronal(interp3, xc, yc, zc, sz, w):
    x = np.arange(xc - w + 0.5, xc + w + 0.5, 1)
    z = np.arange(zc - w + 0.5, zc + w + 0.5, 1)

    # coronal patch voxels
    xx, zz = np.meshgrid(x, z)
    xx = xx.reshape((xx.shape[0] * xx.shape[1], 1))
    zz = zz.reshape((zz.shape[0] * zz.shape[1], 1))
    yy = yc * np.ones(xx.shape)
    pts = np.concatenate((zz, yy, xx), axis=1)

    # interpolate
    p_coronal = interp3(pts)
    p_coronal = p_coronal.reshape((sz, sz))

    return p_coronal


# In[31]:

def extract_sagittal(interp3, xc, yc, zc, sz, w):
    y = np.arange(yc + w + 0.5, yc - w + 0.5, -1)
    z = np.arange(zc + w + 0.5, zc - w + 0.5, -1)

    # sagittal patch voxels
    zz, yy = np.meshgrid(z, y)
    yy = yy.reshape((yy.shape[0] * yy.shape[1], 1))
    zz = zz.reshape((zz.shape[0] * zz.shape[1], 1))
    xx = xc * np.ones(yy.shape)
    pts = np.concatenate((zz, yy, xx), axis=1)

    # interpolate
    p_sagittal = interp3(pts)
    p_sagittal = p_sagittal.reshape((sz, sz))
    p_sagittal = np.fliplr(p_sagittal)
    p_sagittal = np.rot90(p_sagittal, 3)
    return p_sagittal


def extract_patch(contrast, vol, person, time, view, voxel, w, z, y, x):
    volume = vol[person][time][contrast]
    interp3 = RegularGridInterpolator((z, y, x), volume, method='nearest')
    extract_func = globals()['extract_' + view]
    return extract_func(interp3, voxel[0], voxel[1], voxel[2], w * 2, w)


def load_wm_masks(person_list, time_list):
    wm_list = multi_dimensions(2)
    indexes = product(person_list, time_list)
    for person, time in indexes:
        wm_list[person][time] = np.load(Src_Path + WM_path + "Person0{}_Time0{}.npy".format(person, time))
    return wm_list


def create_image_masks(wm_list, image_list, person_list, time_list):
    sel = binary_disk(2)
    masks_list = multi_dimensions(2)
    indexes = product(person_list, time_list)
    for person, time in indexes:
        WM_dilated = mrph.filters.maximum_filter(wm_list[person][time], footprint=sel)

        # apply thresholds
        FLAIR_mask = image_list[person][time]['FLAIR'] > FLAIR_th
        WM_mask = WM_dilated > WM_prior_th

        # final mask: logical AND
        masks_list[person][time] = np.logical_and(FLAIR_mask, WM_mask)
    return masks_list


def is_masks_positive(mask_list, person, time_list, index):
    mask_list = map(lambda time: mask_list[person][time][index[0], index[1], index[2]], time_list)
    return mask_list.count(True) > 0 # == len(mask_list)


def load_images(person_list, time_list, type_list):
    image_list = multi_dimensions(3)
    indexes = product(person_list, time_list, type_list)
    for person, time, type in indexes:
        image_list[person][time][type] = np.load(
            Src_Path + Data_Path + "Person0{}_Time0{}_{}.npy".format(person, time, type))
    return image_list


def post_process(seg, thresh):
    from scipy import ndimage
    connected_comp = ndimage.generate_binary_structure(3, 2) * 1
    label_weight = 30
    connected_comp[1, 1, 1] = label_weight
    res = ndimage.convolve(seg, connected_comp, mode='constant', cval=0.)
    return (res > (thresh + label_weight)) * 1


# def predict_image(model, vol, masks, person, time_list, view_list, MRI_list, threshold=0.5, w=16):
#     import itertools
#     vol_shape = vol[person][time_list[0]][MRI_list[0]].shape
#     prob_plot = np.zeros(vol_shape, dtype='float16')
#     segmentation = np.zeros(vol_shape, dtype='uint8')
#
#     x = np.linspace(0, vol_shape[2] - 1, vol_shape[2], dtype='int')
#     y = np.linspace(0, vol_shape[1] - 1, vol_shape[1], dtype='int')
#     z = np.linspace(0, vol_shape[0] - 1, vol_shape[0], dtype='int')
#     logger.info("patches for model")
#     for i in z:
#         index_list = []
#         patch_dict = defaultdict(list)
#         voxel_list = itertools.product(y, x)
#         for j, k in voxel_list:
#             voxel_patches = itertools.product(time_list, view_list)
#             if can_extract_patch(vol_shape, k, j, i, w) and is_masks_positive(masks, person, time_list, (i, j, k)):
#                 index_list.append((i, j, k))
#                 for time, view in voxel_patches:
#                     additionalArgument = vol, person, time, view, (k, j, i), w, z, y, x
#                     l = map(lambda p: extract_patch(p, *additionalArgument), MRI_list)
#                     patch_dict[str(time) + view].append(np.array(l))
#         if len(index_list) > 0:
#             logger.info("predict model z is {}".format(i))
#             predictions = model.predict({'s0_curr': np.array(patch_dict[str(time_list[1]) + 'axial']),
#                                          's0_prev': np.array(patch_dict[str(time_list[0]) + 'axial']),
#                                          's1_curr': np.array(patch_dict[str(time_list[1]) + 'coronal']),
#                                          's1_prev': np.array(patch_dict[str(time_list[0]) + 'coronal']),
#                                          's2_curr': np.array(patch_dict[str(time_list[1]) + 'sagittal']),
#                                          's2_prev': np.array(patch_dict[str(time_list[0]) + 'sagittal'])})
#             out_pred = predictions['output'][:, 1]
#             for index, (i, j, k,) in enumerate(index_list):
#                 if out_pred[index] > threshold:
#                     segmentation[i, j, k] = 1
#                 prob_plot[i, j, k] = out_pred[index]
#
#     return segmentation, prob_plot


def predict_image( vol, masks, person, time_list, view_list, MRI_list,vol_shape, w=16):
    x = np.linspace(0, vol_shape[2] - 1, vol_shape[2], dtype='int')
    y = np.linspace(0, vol_shape[1] - 1, vol_shape[1], dtype='int')
    z = np.linspace(0, vol_shape[0] - 1, vol_shape[0], dtype='int')
    logger.info("patches for model")
    for i in z:
        index_list = []
        patch_dict = defaultdict(list)
        voxel_list = product(y, x)
        for j, k in voxel_list:
            voxel_patches = product(time_list, view_list)
            if can_extract_patch(vol_shape, k, j, i, w) and is_masks_positive(masks, person, time_list, (i, j, k)):
                index_list.append((i, j, k))
                for time, view in voxel_patches:
                    additionalArgument = vol, person, time, view, (k, j, i), w, z, y, x
                    l = map(lambda p: extract_patch(p, *additionalArgument), MRI_list)
                    patch_dict[str(time) + view].append(np.array(l))
        if len(index_list) > 0:
            patch_q.put((index_list, patch_dict))
            logger.info("put layer {}".format(i))

def model_pred(model,time_list,vol_shape):
    max = vol_shape[0]
    while True:
        indexes,patches =patch_q.get()
        curr_layer = indexes[0][0]
        predictions = model.predict({'s0_curr': np.array(patches[str(time_list[1]) + 'axial']),
                                     's0_prev': np.array(patches[str(time_list[0]) + 'axial']),
                                     's1_curr': np.array(patches[str(time_list[1]) + 'coronal']),
                                     's1_prev': np.array(patches[str(time_list[0]) + 'coronal']),
                                     's2_curr': np.array(patches[str(time_list[1]) + 'sagittal']),
                                     's2_prev': np.array(patches[str(time_list[0]) + 'sagittal'])})
        out_pred = predictions['output'][:, 1]
        prediction_q.put((indexes,out_pred))
        logger.info("predicted layer {} ".format(curr_layer))
        if curr_layer >=max:
            break

def get_segmantation(vol_shape,queue,threshold=0.5):
    prob_plot = np.zeros(vol_shape, dtype='float16')
    segmentation = np.zeros(vol_shape, dtype='uint8')
    while True:
        indexes,pred = prediction_q.get()
        curr_layer = indexes[0][0]
        for index, (i, j, k,) in enumerate(indexes):
            if pred[index] > threshold:
                segmentation[i, j, k] = 1
            prob_plot[i, j, k] = pred[index]
        if curr_layer >= max:
            break
        logger.info("segmanted layer {}".format(curr_layer))
    queue.put((segmentation,prob_plot))






# create run folder
time = datetime.now().strftime('%d_%m_%Y_%H_%M')
run_dir = './runs/' + time + '/'
if not path.exists(run_dir):
    makedirs(run_dir)
# create logger
logger = get_logger(run_dir)

person_list = [2]
time_list = [2,3]
MR_modalities = ['FLAIR', 'T2', 'MPRAGE', 'PD']
view_list = ['axial', 'coronal', 'sagittal']
# load images
logger.info("load images")
images = load_images(person_list, time_list, MR_modalities)
vol_shape = images[person_list[0]][time_list[0]][MR_modalities[0]].shape
wm_masks = load_wm_masks(person_list, time_list)
masks = create_image_masks(wm_masks, images, person_list, time_list)

# load model
logger.info("create model")
model = create_full_model()
model_weights = r"./ref/MIMTP_model_weights.h5"
logger.info("model load weights")
model.load_weights(model_weights)
model.compile(optimizer='adadelta', loss={'output': 'categorical_crossentropy'})

logger.info("predict images")
BUF_SIZE = 15
patch_q = Queue.Queue(BUF_SIZE)
prediction_q =  Queue.Queue(BUF_SIZE)
seg_q =  Queue.Queue(1)
patch_thread = threading.Thread(target=predict_image,args=(images, masks, person_list[0], time_list, view_list, MR_modalities,vol_shape))
model_thread = threading.Thread(target=model_pred,args=(model,time_list,vol_shape))
seg_thread = threading.Thread(target=get_segmantation,args=(vol_shape,seg_q))
thread_list = [patch_thread,model_thread,seg_thread]
for i in thread_list:
    i.start()
for i in thread_list:
    i.join()
segmantation, prob_map = seg_q.get()

#segmantation, prob_map = predict_image(model, images, masks, 1, time_list, view_list, MR_modalities)
with open(run_dir + 'segmantation.npy', 'wb') as fp, open(run_dir + 'prob_plot.npy', 'wb') as fp1:
    np.save(fp, segmantation)
    np.save(fp1, prob_map)

FLAIR_labels_1 = Src_Path + Labels_Path + "training0{}_0{}_mask1.nii".format(person_list[0],time_list[1])
labels = nb.load(FLAIR_labels_1).get_data()
labels = labels.T
labels = np.rot90(labels, 2, axes=(1, 2))

test_labels = labels.flatten().tolist()
test_seg = segmantation.flatten().tolist()
from sklearn.metrics import accuracy_score, f1_score

logger.info("f1 is {} , accuracy is {} ".format(f1_score(test_labels, test_seg), accuracy_score(test_labels, test_seg)))
# xc = 100
# yc = 158
# zc =82
# w=16
# FLAIR_filename = Src_Path+Data_Path+"Person05_Time02_FLAIR.npy"
# vol = np.load(FLAIR_filename)
# plt.imshow(vol[:, :, xc], cmap=matplotlib.cm.gray)
# plt.figure()
# a = extract_sagittal(vol,xc,yc,zc,w)
# plt.imshow(a, cmap=matplotlib.cm.gray)
# plt.show()

# x = np.linspace(0, 180, 181, dtype='int')
# y = np.linspace(0, 216, 217, dtype='int')
# z = np.linspace(0, 180, 181, dtype='int')
# count=0
# for i,j,k in product(z,y,x):
#     if labels[i,j,k] ==segmantation[i,j,k]:
#         count +=1
# print float(count)/181*181*217
