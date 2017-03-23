# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:35:06 2016

@author: yaniv
"""
import numpy as np
import random
import pylab
import scipy.ndimage.morphology as mrph
import scipy.ndimage as ndimage
import scipy.io
import itertools
import nibabel as nb
import scipy.misc
from collections import defaultdict

from logging_tools import  get_logger

FLAIR_th = 0.91
WM_prior_th = 0.5
valSize = 0.2
negative_threshold = 0.4

Src_Path = r"./train/"
Data_Path = r"data/"
WM_Path = r"WM/"
Labels_Path = r"seg/"
Output_Path=r"patches/"

# patch size
sz = 32
w = sz/2

# patch center
xc = 74
yc = 145
zc = 100

def binary_disk(r):
    arr = np.ones((2*r+1,2*r+1,2*r+1))
    arr[r,r,r] = 0

    dt = mrph.distance_transform_bf(arr,metric='euclidean')
    disk = dt<=r
    disk = disk.astype('float32')

    return disk


# In[11]:
def apply_masks(FLAIR_vol_filename,WM_filename):
    FLAIR_vol = np.load(FLAIR_vol_filename)
    WM = np.load(WM_filename)
    
    # dilate WM mask
    sel = binary_disk(2)
    WM_dilated = mrph.filters.maximum_filter(WM, footprint=sel)
    
    # apply thresholds
    FLAIR_mask = FLAIR_vol > FLAIR_th
    WM_mask = WM_dilated > WM_prior_th
    
    # final mask: logical AND
    candidate_mask = np.logical_and(FLAIR_mask, WM_mask)
    return candidate_mask

import matplotlib
import pylab

z = 70

#%% patching

from scipy.interpolate import RegularGridInterpolator

def extract_axial(interp3, xc, yc, zc, sz, w):
    x = np.arange(xc-w+0.5, xc+w+0.5, 1)
    y = np.arange(yc+w+0.5, yc-w+0.5, -1)

    # axial patch voxels
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape((xx.shape[0]*xx.shape[1],1))
    yy = yy.reshape((yy.shape[0]*yy.shape[1],1))
    zz = zc*np.ones(xx.shape)
    pts = np.concatenate((zz,yy,xx),axis=1)

    # interpolate
    try:
        p_axial = interp3(pts)
        p_axial = p_axial.reshape((sz,sz))
        return p_axial
    except ValueError as e:
        return 0


# In[3]:

def extract_coronal(interp3, xc, yc, zc, sz, w):
    x = np.arange(xc-w+0.5, xc+w+0.5, 1)
    z = np.arange(zc-w+0.5, zc+w+0.5, 1)

    # coronal patch voxels
    xx, zz = np.meshgrid(x, z)
    xx = xx.reshape((xx.shape[0]*xx.shape[1],1))
    zz = zz.reshape((zz.shape[0]*zz.shape[1],1))
    yy = yc*np.ones(xx.shape)
    pts = np.concatenate((zz,yy,xx),axis=1)

    # interpolate
    try:    
        p_coronal = interp3(pts)
        p_coronal = p_coronal.reshape((sz,sz))
        return p_coronal
    except ValueError as e:
        return 0

def split_train_validation(data, labels, _valSize):
    total_samples = len(data)
    val_slice = int((1 - _valSize) * total_samples)
    val_data, val_labels = data[val_slice:], labels[val_slice:]
    train_data, train_labels = data[:val_slice], labels[:val_slice]
    return train_data, train_labels, val_data, val_labels


def sample_negative_samples(axial_arr, coronal_arr, labels):
    output_axial = []
    output_coronal = []
    output_labels = []
    for i in range(len(axial_arr)):
        if labels[i] == 1 or random.random() < negative_threshold:
            output_axial.append(axial_arr[i])
            output_coronal.append(coronal_arr[i])
            output_labels.append(labels[i])

    return output_axial, output_coronal, output_labels

logger = get_logger()


# In[6]:

# load volume
def create_positive_list(person_list):
    positive_list = []
    for index in person_list:
        for index2 in range(1,5):
            logger.info("person {} time {} creating patches".format(index,index2))

            #Person = "person0%d"%(index)
            FLAIR_filename = Src_Path+Data_Path+"Person0{}_Time0{}_FLAIR.npy".format(index,index2)
            WM_filename = Src_Path+WM_Path+"Person0{}_Time0{}.npy".format(index,index2)
            FLAIR_labels_1 = Src_Path+Labels_Path+"training0{}_0{}_mask1.nii".format(index,index2)
            vol = np.load(FLAIR_filename)
            labels = nb.load(FLAIR_labels_1).get_data()
            labels = labels.T
            labels = np.rot90(labels, 2, axes=(1, 2))

            # In[7]:

            # initialize interpolator
            x = np.linspace(0, vol.shape[2]-1,vol.shape[2],dtype='int')
            y = np.linspace(0, vol.shape[1]-1,vol.shape[1],dtype='int')
            z = np.linspace(0, vol.shape[0]-1,vol.shape[0],dtype='int')
            #%%
            candidate_mask = apply_masks(FLAIR_filename,WM_filename)

            # In[12]:
            voxel_list = itertools.product(z,y,x)
            for i,j,k in voxel_list:
                if candidate_mask[i][j][k] and labels[i][j][k] == 1:
                    positive_list.append((index,index2,i,j,k))


    import pickle
    with open(Output_Path+"positive_list_person_{}.lst".format(str(person_list)), 'wb') as fp1:
                pickle.dump(positive_list, fp1)
                logger.info("person {} finished patches and saved".format(index))



person_list = [1,2,3]
from sklearn.model_selection import KFold
kf = KFold(4, n_folds=4)
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)


def load_positive_list(person_list):
    import pickle
    with open(Output_Path + "positive_list_person_{}.lst".format(str(person_list)), 'rb') as fp1:
            positive_list_np = np.array(pickle.load(fp1))
    return positive_list_np

#
def load_data(person_list):
    image_list =defaultdict(dict)
    for person in person_list:
        for time in range(1,5):
            image_list[person][time] = np.load(Src_Path+Data_Path+"Person0{}_Time0{}_FLAIR.npy".format(person,time))
    return image_list

def list_generator(positive_list,batch_size=256):
    batch_num = len(positive_list) / batch_size
    while True:
        # modify list to divide by batch_size
        positive_list_np = np.random.permutation(positive_list)
        positive_list_np = positive_list_np[:batch_num * batch_size]
        for i in range(batch_num):
            positive_batch = positive_list_np[i * batch_size:(i + 1) * batch_size]


def generator(positive_list,data,batch_size=256):
    batch_num = len(positive_list)/batch_size
    while True:
        #modify list to divide by batch_size
        positive_list_np = np.random.permutation(positive_list)
        positive_list_np = positive_list_np[:batch_num*batch_size]
        for i in range(batch_num):
            positive_batch = positive_list_np[i*batch_size:(i+1)*batch_size]


p = load_data(person_list)
a = load_positive_list(person_list)
print 3
