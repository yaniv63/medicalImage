
# coding: utf-8

# In[1]:

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

FLAIR_vol_filename = r"/home/yaniv/src/medicalImaging/person1_time1/Person01_Time01_FLAIR.npy"
WM_filename = r"/home/yaniv/src/medicalImaging/person1_time1/Person01_Time01.npy"


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
WM = np.load(WM_filename)

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
import pylab

z = 70
pylab.figure()
pylab.imshow(FLAIR_vol[z,:,:], cmap=matplotlib.cm.gray)
pylab.figure()
pylab.imshow(WM_dilated[z,:,:], cmap=matplotlib.cm.gray)
pylab.figure()
pylab.imshow(candidate_mask[z,:,:], cmap=matplotlib.cm.gray)


# In[ ]:



