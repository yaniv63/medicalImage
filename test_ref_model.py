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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os import path, makedirs
from datetime import datetime
import nibabel as nb

import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from collections import defaultdict
from itertools import product

from MIMTP_Detection_demo import create_full_model
from logging_tools import get_logger
weight_path = r'./trained_weights/'
patches = r'./patches/'
Labels_Path = r"seg/"
runs_dir = r'./runs/'
Src_Path = r"./train/"
Data_Path = r"data/"


def multi_dimensions(n,type=None):
    if n<=0:
        if type is not None:
            return type()
        return None
    return defaultdict(lambda:multi_dimensions(n-1,type))

def can_extract_patch(shape,xc, yc, zc, w):
    return (xc-w) >= 0 and (xc+w)<=shape[2] and (yc-w) >= 0 and (yc+w)<=shape[1] and (zc-w) >= 0 and (zc+w)<=shape[0]


def extract_axial(vol,xc, yc, zc, w):
    x = np.arange(xc - w, xc + w , 1)
    y = np.arange(yc - w, yc + w , 1)
    indexes = np.ix_(y,x) #choose patch indices left for rows,right for columns
    patch = vol[zc][indexes]
    return  patch

def extract_coronal(vol, xc, yc, zc, w):
    x = np.arange(xc - w, xc + w , 1)
    z = np.arange(zc - w, zc + w , 1)
    indexes = np.ix_(z, x)
    patch = vol[:,yc,:][indexes]
    return  patch


def extract_sagittal(vol, xc, yc, zc, w):
    y = np.arange(yc - w, yc + w , 1)
    z = np.arange(zc - w, zc + w , 1)
    indexes = np.ix_(z, y)
    patch = vol[:,:,xc][indexes]
    return  patch

def extract_patch(contrast,vol,person,time,view,voxel,w):
    volume = vol[person][time][contrast]
    extract_func = globals()['extract_' + view]
    return extract_func(volume, voxel[0],voxel[1],voxel[2], w)

def load_images(person_list,time_list,type_list):
    image_list =multi_dimensions(3)
    indexes = product(person_list,time_list,type_list)
    for person,time,type in indexes:
            image_list[person][time][type] = np.load(Src_Path+Data_Path+"Person0{}_Time0{}_{}.npy".format(person,time,type))
    return image_list

def predict_image(model, vol,person,time_list,view_list,MRI_list,threshold=0.5,w=16):
    import itertools
    vol_shape = vol[person][time_list[0]][MRI_list[0]].shape
    prob_plot = np.zeros(vol_shape,dtype='uint8')
    segmentation = np.zeros(vol_shape,dtype='uint8')

    x = np.linspace(0, vol_shape[2] - 1, vol_shape[2], dtype='int')
    y = np.linspace(0, vol_shape[1] - 1, vol_shape[1], dtype='int')
    z = np.linspace(0, vol_shape[0] - 1, vol_shape[0], dtype='int')
    logger.info("patches for model")
    for i in z:
        index_list = []
        patch_dict = defaultdict(list)
        voxel_list = itertools.product(y, x)
        for j, k in voxel_list:
            voxel_patches = itertools.product(time_list, view_list)
            if can_extract_patch(vol_shape,k,j,i,w):
                index_list.append((i,j,k))
                for time,view in voxel_patches:
                    additionalArgument = vol,person,time,view,(k,j,i),w
                    l = map(lambda p: extract_patch(p, *additionalArgument), MRI_list)
                    patch_dict[str(time)+view].append(np.array(l))
        if len(index_list)>0:
            logger.info("predict model")
            predictions = model.predict({'s0_curr':np.array(patch_dict[str(time_list[1])+'axial']), 's0_prev':np.array(patch_dict[str(time_list[0])+'axial']),
                                         's1_curr':np.array(patch_dict[str(time_list[1])+'coronal']), 's1_prev':np.array(patch_dict[str(time_list[0])+'coronal']),
                                         's2_curr':np.array(patch_dict[str(time_list[1])+'sagittal']), 's2_prev':np.array(patch_dict[str(time_list[0])+'sagittal'])})
            out_pred = predictions['output'][:,1]
            for index,(i, j, k,) in enumerate(index_list):
                if out_pred[index] > threshold:
                    segmentation[i, j, k] = 1
                prob_plot[i, j, k] = out_pred[index] * 255

    return segmentation,prob_plot




# create run folder
time = datetime.now().strftime('%d_%m_%Y_%H_%M')
run_dir = './' +time + '/'
if not path.exists(run_dir):
    makedirs(run_dir)
# create logger
logger = get_logger(run_dir)

person_list = [1]
time_list = [1,2]
MR_modalities = ['FLAIR', 'T2', 'MPRAGE', 'PD']
view_list  = ['axial','coronal','sagittal']
#load images
logger.info("load images")
images = load_images(person_list,time_list,MR_modalities)
#load model
logger.info("create model")
model = create_full_model()
model_weights = r"/media/sf_shared/src/medicalImaging/ref/MIMTP_model_weights.h5"
logger.info("model load weights")
model.load_weights(model_weights)
model.compile(optimizer='adadelta', loss={'output':'categorical_crossentropy'})

logger.info("predict images")
segmantation, prob_map = predict_image(model,images,1,time_list,view_list,MR_modalities)
with open(run_dir + 'segmantation.npy', 'wb') as fp,open(run_dir + 'prob_plot.npy', 'wb') as fp1:
    np.save(fp,segmantation)
    np.save(fp,prob_map)


FLAIR_labels_1 = Src_Path+Labels_Path+"training01_02_mask1.nii"
labels = nb.load(FLAIR_labels_1).get_data()
labels = labels.T
labels = np.rot90(labels, 2, axes=(1, 2))

test_labels = labels.flatten().tolist()
test_seg = segmantation.flatten().tolist()
from sklearn.metrics import accuracy_score,f1_score
logger.info("f1 is {} , accuracy is {} ".format(f1_score(test_labels,test_seg),accuracy_score(test_labels,test_seg)))
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
