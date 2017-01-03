# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:35:06 2016

@author: yaniv
"""
import numpy as np
import pylab
import scipy.ndimage.morphology as mrph
import scipy.ndimage as ndimage
import scipy.io
import itertools
import nibabel as nb
import scipy.misc

FLAIR_th = 0.91
WM_prior_th = 0.5

Src_Path = r"/media/sf_ubuntuFolder/src/medicalImaging/data/"



# In[10]:

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


# patch size
sz = 32
w = sz/2

# patch center
xc = 74
yc = 145
zc = 100


# In[6]:

# load volume
for i in range(2):
    index = i+1
    Person = "person0%d"%(index)
    FLAIR_filename = Src_Path+Person+"/"+Person+"_Time01_FLAIR.npy"
    WM_filename = Src_Path+Person+"/"+Person+"_Time01.npy"    
    FLAIR_labels_1 = Src_Path+Person+"/"+"training0%d_01_mask1.nii"%(index)
    vol = np.load(FLAIR_filename)
    labels = nb.load(FLAIR_labels_1).get_data()
    
    # In[7]:
    
    # initialize interpolator
    x = np.linspace(0, vol.shape[2]-1,vol.shape[2])
    y = np.linspace(0, vol.shape[1]-1,vol.shape[1])
    z = np.linspace(0, vol.shape[0]-1,vol.shape[0])
    interp3 = RegularGridInterpolator((z,y,x), vol)
#%%
    candidate_mask = apply_masks(FLAIR_filename,WM_filename)    

    # In[12]:
    patches= {}
    patches_axial = []
    patches_coronal = []
    patches_labels = []
    voxel_list = itertools.product(x,y,z)
    zero_count = 0
    for i,j,k in voxel_list:
        if candidate_mask[i][j][k] == True:
            axial_p = extract_axial(interp3, i, j, k, sz, w)
            coronal_p = extract_coronal(interp3, i, j, k, sz, w)
            if type(axial_p) == np.ndarray and type(coronal_p) == np.ndarray:        
                if labels[i][j][k] == 0 and zero_count < 1500:            
                    zero_count= zero_count+1                
                    patches_axial.append(axial_p)
                    patches_coronal.append(coronal_p)
                    patches_labels.append(labels[i][j][k])
                elif  labels[i][j][k] == 1:
                    patches_axial.append(axial_p)
                    patches_coronal.append(coronal_p)
                    patches_labels.append(labels[i][j][k])
        if len(patches_axial) > 3000:
            break
    
    axial_p = extract_axial(interp3, 73, 101, 104, sz, w)
    coronal_p = extract_coronal(interp3, 73, 101, 104, sz, w)
    patches_axial.append(axial_p)
    patches_coronal.append(coronal_p)
    patches_labels.append(labels[i][j][k])
    
    import pickle
    
    with open('patches_axial_0%d.lst'%(index), 'wb') as fp1 ,open('patches_coronal_0%d.lst'%(index), 'wb') as fp2,open('labels_0%d.lst'%(index), 'wb') as fp3 :
        pickle.dump(patches_axial, fp1)
        pickle.dump(patches_coronal, fp2)
        pickle.dump(patches_labels, fp3)


#%%
## extract patches
p_axial = extract_axial(interp3, 80, 101, 104, sz, w)
p_coronal = extract_coronal(interp3, 80, 101, 104, sz, w)
#
#
# axial
pylab.figure()
pylab.imshow(p_axial, cmap=matplotlib.cm.gray, interpolation='nearest')

# coronal
pylab.figure()
pylab.imshow(p_coronal, cmap=matplotlib.cm.gray, interpolation='nearest')