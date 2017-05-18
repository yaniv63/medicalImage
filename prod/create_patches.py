import numpy as np


def can_extract_patch(shape, zc, yc, xc, w):
    return (xc-w) >= 0 and (xc+w)<=(shape[2]-1) and (yc-w) >= 0 and (yc+w)<=(shape[1]-1)\
           and (zc-w) >= 0 and (zc+w)<=(shape[0]-1)


def extract_axial(vol,zc, yc, xc, w):
    x = np.arange(xc - w, xc + w , 1)
    y = np.arange(yc - w, yc + w , 1)
    indexes = np.ix_(y,x) #choose patch indices left for rows,right for columns
    patch = vol[zc][indexes]
    return  patch


def extract_coronal(vol, zc, yc, xc, w):
    x = np.arange(xc - w, xc + w , 1)
    z = np.arange(zc - w, zc + w , 1)
    indexes = np.ix_(z, x)
    patch = vol[:,yc,:][indexes]
    return  patch

def extract_sagittal(vol, zc, yc, xc, w):
    y = np.arange(yc - w, yc + w , 1)
    z = np.arange(zc - w, zc + w , 1)
    indexes = np.ix_(z, y)
    patch = vol[:,:,xc][indexes]
    return  patch

def extract_patch(volume, view, voxel, w):
    extract_func = globals()['extract_' + view]
    return extract_func(volume, voxel[0], voxel[1], voxel[2], w)