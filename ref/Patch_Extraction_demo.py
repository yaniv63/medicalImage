
# coding: utf-8

# In[2]:

#get_ipython().magic(u'matplotlib qt')

import matplotlib
import pylab
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# In[29]:

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


# In[30]:

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


# In[31]:

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


# In[57]:

# patch size
sz = 32
w = sz/2

# patch center
xc = 51 #74
yc = 185 #145
zc = 140


# In[58]:

# load volume
FLAIR_filename = r"/media/sf_shared/src/medicalImaging/train/data/Person03_Time01_FLAIR.npy"
vol = np.load(FLAIR_filename)
print vol.shape


# In[59]:

# initialize interpolator
x = np.linspace(0, vol.shape[2]-1,vol.shape[2])
y = np.linspace(0, vol.shape[1]-1,vol.shape[1])
z = np.linspace(0, vol.shape[0]-1,vol.shape[0])
interp3 = RegularGridInterpolator((z,y,x), vol,method='nearest')


# In[60]:

# extract patches
p_axial = extract_axial(interp3, xc, yc, zc, sz, w)
p_coronal = extract_coronal(interp3, xc, yc, zc, sz, w)
p_sagittal = extract_sagittal(interp3, xc, yc, zc, sz, w)
p_sagittal = np.fliplr(p_sagittal)
p_sagittal = np.rot90(p_sagittal,3)
p_axial = np.flipud(p_axial)


# In[63]:

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


# In[64]:

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
pylab.show()
pylab.waitforbuttonpress()
pylab.close('all')


# In[ ]:



