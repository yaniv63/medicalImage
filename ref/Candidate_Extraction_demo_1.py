
# coding: utf-8

# In[1]:
import itertools
import numpy as np
import pylab
import scipy.ndimage.morphology as mrph
import scipy.ndimage as ndimage
import scipy.io

import scipy.misc


# In[8]:

# thresholds
FLAIR_th = 0.91
WM_prior_th = 0.5


# In[9]:

FLAIR_vol_filename = r"../train/data/Person03_Time01_FLAIR.npy"
WM_filename = r"../train/WM/Person03_Time01.npy"
labels_file = r"../train/seg/training03_01_mask1.nii"
labels2_file = r"../train/seg/training01_01_mask2.nii"

# In[10]:

def binary_disk(r):
    arr = np.ones((2*r+1,2*r+1,2*r+1))
    arr[r,r,r] = 0

    dt = mrph.distance_transform_bf(arr,metric='euclidean')
    disk = dt<=r
    disk = disk.astype('float32')

    return disk


# In[11]:

FLAIR_vol = np.load(FLAIR_vol_filename)
FLAIR_vol = np.rot90(FLAIR_vol,2,axes=(1,2))
WM = np.load(WM_filename)
WM = np.rot90(WM,2,axes=(1,2))


# dilate WM mask
sel = binary_disk(2)
WM_dilated = mrph.filters.maximum_filter(WM, footprint=sel)

# apply thresholds
FLAIR_mask = FLAIR_vol > FLAIR_th
WM_mask = WM_dilated > WM_prior_th

# final mask: logical AND
candidate_mask = np.logical_and(FLAIR_mask, WM_mask)


# In[15]:

# display resulted mask
#get_ipython().magic(u'matplotlib qt')

import matplotlib
import matplotlib.pyplot as pylab
z = 100
pylab.ion()
pylab.figure()
pylab.imshow(FLAIR_vol[z,:,:], cmap=matplotlib.cm.gray)
pylab.figure()
pylab.imshow(candidate_mask[z,:,:], cmap=matplotlib.cm.gray)
import nibabel as nib
labels = nib.load(labels_file).get_data()
a =  labels.T
e = np.rot90(a,2,axes=(1,2))
pylab.figure()
pylab.imshow(e[z,:,:], cmap=matplotlib.cm.gray)
pylab.show()
pylab.waitforbuttonpress()
pylab.close('all')


# In[ ]:

#
# from scipy.interpolate import RegularGridInterpolator

#
#
# x = np.linspace(0, FLAIR_vol.shape[2] - 1, FLAIR_vol.shape[2])
# y = np.linspace(0, FLAIR_vol.shape[1] - 1, FLAIR_vol.shape[1])
# z = np.linspace(0, FLAIR_vol.shape[0] - 1, FLAIR_vol.shape[0])
# interp3 = RegularGridInterpolator((z, y, x), FLAIR_vol)
# # %%
#
# # In[12]:
# patches = {}
#
# voxel_list = itertools.product(x, y, z)
# zero_count = 0
#
# patches_labels_0 = 0
# patches_labels_1 = 0
# labels_1_and_candidate = 0
# original_labels_0 = 0
#
# for i, j, k in voxel_list:
#     if candidate_mask[i][j][k] == True:
#             patches_labels_1 += 1
#     else:
#             patches_labels_0 += 1
#
#     if candidate_mask[i][j][k] == True and labels[i][j][k] == True:
#         labels_1_and_candidate += 1
#
# print("person  original label 1 count {}".format(patches_labels_1))
# print("person  original label 0 count {}".format(patches_labels_0))
# print("person   label and candidate 1 count {}".format(labels_1_and_candidate))
