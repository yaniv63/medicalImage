
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib qt')

import matplotlib
import pylab
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# In[2]:

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
    p_axial = interp3(pts)
    p_axial = p_axial.reshape((sz,sz))
    
    return p_axial


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
    p_coronal = interp3(pts)
    p_coronal = p_coronal.reshape((sz,sz))
    
    return p_coronal


# In[4]:

def extract_sagittal(interp3, xc, yc, zc, sz, w):
    y = np.arange(yc+w+0.5, yc-w+0.5, -1)
    z = np.arange(zc+w+0.5, zc-w+0.5, -1)
    
    # sagittal patch voxels
    zz, yy = np.meshgrid(z, y)
    yy = yy.reshape((yy.shape[0]*yy.shape[1],1))
    zz = zz.reshape((zz.shape[0]*zz.shape[1],1))
    xx = xc*np.ones(yy.shape)
    pts = np.concatenate((zz,yy,xx),axis=1)

    # interpolate
    p_sagittal = interp3(pts)
    p_sagittal = p_sagittal.reshape((sz,sz))
    
    return p_sagittal


# In[16]:

# patch size
sz = 32
w = sz/2

# patch center
xc = 74
yc = 145
zc = 100


# In[6]:

# load volume
FLAIR_filename = r"/home/yaniv/src/medicalImaging/person1_time1/Person01_Time01_FLAIR.npy"
vol = np.load(FLAIR_filename)
vol.shape


# In[7]:

# initialize interpolator
x = np.linspace(0, vol.shape[2]-1,vol.shape[2])
y = np.linspace(0, vol.shape[1]-1,vol.shape[1])
z = np.linspace(0, vol.shape[0]-1,vol.shape[0])
interp3 = RegularGridInterpolator((z,y,x), vol)


# In[12]:

# extract patches
p_axial = extract_axial(interp3, xc, yc, zc, sz, w)
p_coronal = extract_coronal(interp3, xc, yc, zc, sz, w)
p_sagittal = extract_sagittal(interp3, xc, yc, zc, sz, w)


# In[14]:

# display full slices

# axial
pylab.figure()
pylab.imshow(vol[zc,:,:], cmap=matplotlib.cm.gray, interpolation='nearest')

# coronal
pylab.figure()
pylab.imshow(vol[:,yc,:], cmap=matplotlib.cm.gray, interpolation='nearest')

# sagittal
pylab.figure()
pylab.imshow(vol[:,:,xc], cmap=matplotlib.cm.gray, interpolation='nearest')


# In[13]:

# display patches

# axial
pylab.figure()
pylab.imshow(p_axial, cmap=matplotlib.cm.gray, interpolation='nearest')

# coronal
pylab.figure()
pylab.imshow(p_coronal, cmap=matplotlib.cm.gray, interpolation='nearest')

# sagittal
pylab.figure()
pylab.imshow(p_sagittal, cmap=matplotlib.cm.gray, interpolation='nearest')


# In[15]:

#pylab.close('all')


# In[ ]:



