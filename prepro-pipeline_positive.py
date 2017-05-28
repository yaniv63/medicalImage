# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:35:06 2016

@author: yaniv
"""
import itertools
import random
from collections import defaultdict
import logging

import nibabel as nb
import numpy as np
import scipy.ndimage.morphology as mrph

logger = logging.getLogger('root')

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


z = 70

#%% patching

def extract_axial(vol,xc, yc, zc, w):
    try:
        x = np.arange(xc - w, xc + w , 1)
        y = np.arange(yc - w, yc + w , 1)
        indexes = np.ix_(y, x)
        patch = vol[zc][indexes]
        return  patch
    except IndexError as e:
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



# In[6]:

# load volume
def create_patches_list(person_list,time_list):
    positive_list = []
    neg_candidate_list = []
    for index in person_list:
        for index2 in time_list:
            logger.debug("person {} time {} creating patches".format(index,index2))

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
                elif candidate_mask[i][j][k]:
                    neg_candidate_list.append((index,index2,i,j,k))



    import pickle
    with open(Output_Path+"positive_list_person_{}.lst".format(str(person_list)), 'wb') as fp1,\
        open(Output_Path + "negative_list_person_{}.lst".format(str(person_list)), 'wb') as fp2:
                pickle.dump(positive_list, fp1)
                pickle.dump(neg_candidate_list, fp2)
                logger.debug("person {} finished patches and saved".format(index))

def create_ROI_list_person_time(person,time):
    positive_list = []
    neg_candidate_list = []

    logger.debug("person {} time {} creating patches".format(person,time))
    FLAIR_filename = Src_Path+Data_Path+"Person0{}_Time0{}_FLAIR.npy".format(person,time)
    WM_filename = Src_Path+WM_Path+"Person0{}_Time0{}.npy".format(person,time)
    FLAIR_labels_1 = Src_Path+Labels_Path+"training0{}_0{}_mask1.nii".format(person,time)
    vol = np.load(FLAIR_filename)
    labels = nb.load(FLAIR_labels_1).get_data()
    labels = labels.T
    labels = np.rot90(labels, 2, axes=(1, 2))

    x = np.linspace(0, vol.shape[2]-1,vol.shape[2],dtype='int')
    y = np.linspace(0, vol.shape[1]-1,vol.shape[1],dtype='int')
    z = np.linspace(0, vol.shape[0]-1,vol.shape[0],dtype='int')
    candidate_mask = apply_masks(FLAIR_filename,WM_filename)

    voxel_list = itertools.product(z,y,x)
    for i,j,k in voxel_list:
        if candidate_mask[i][j][k] and labels[i][j][k] == 1:
            positive_list.append((person,time,i,j,k))
        elif candidate_mask[i][j][k]:
            neg_candidate_list.append((person,time,i,j,k))

    import pickle
    with open(Output_Path+"positive_person{}_time{}.lst".format(person,time), 'wb') as fp1,\
            open(Output_Path + "negative_person{}_time{}.lst".format(person,time), 'wb') as fp2:
                pickle.dump(positive_list, fp1)
                pickle.dump(neg_candidate_list, fp2)
                logger.debug("person{}_time{} finished patches and saved".format(person,time))





if __name__ == "__main__":
   for person in range(1,6) :
       for time in range(1,5):
           create_ROI_list_person_time(person,time)