import itertools
from datetime import datetime
from os import path, makedirs

import nibabel as nb
import numpy as np
import scipy.ndimage.morphology as mrph
from scipy.interpolate import RegularGridInterpolator

from prod.logging_tools import  get_logger

FLAIR_th = 0.91
WM_prior_th = 0.5
valSize = 0.2

Src_Path = r"../train/"
Data_Path = r"data/"
WM_Path = r"WM/"
Labels_Path = r"seg/"
Output_Path = r"patches/"

# patch size
sz = 32
w = sz / 2



def binary_disk(r):
    arr = np.ones((2 * r + 1, 2 * r + 1, 2 * r + 1))
    arr[r, r, r] = 0

    dt = mrph.distance_transform_bf(arr, metric='euclidean')
    disk = dt <= r
    disk = disk.astype('float32')

    return disk


# In[11]:
def apply_masks(FLAIR_vol_filename, WM_filename):
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
    return FLAIR_mask

# create run folder
time = datetime.now().strftime('%d_%m_%Y_%H_%M')
run_dir = '../runs/' + time + '/'
if not path.exists(run_dir):
    makedirs(run_dir)
# create logger
logger = get_logger(run_dir)


for index in range(3, 4):
    for index2 in range(5, 6):
        labeled0_and_not_candidate = 0
        labeled1_and_not_candidate = 0
        labeled0_and_candidate = 0
        labeled1_and_candidate = 0
        labeled1 = 0
        labeled0 = 0

        # Person = "person0%d"%(index)
        FLAIR_filename = Src_Path + Data_Path + "Person0{}_Time0{}_FLAIR.npy".format(index, index2)
        WM_filename = Src_Path + WM_Path + "Person0{}_Time0{}.npy".format(index, index2)
        FLAIR_labels_1 = Src_Path + Labels_Path + "training0{}_0{}_mask1.nii".format(index, index2)
        vol = np.load(FLAIR_filename)
        labels = nb.load(FLAIR_labels_1).get_data()
        labels = labels.T
        labels = np.rot90(labels, 2, axes=(1, 2))

        # In[7]:

        # initialize interpolator
        x = np.linspace(0, vol.shape[2] - 1, vol.shape[2],dtype='int')
        y = np.linspace(0, vol.shape[1] - 1, vol.shape[1],dtype='int')
        z = np.linspace(0, vol.shape[0] - 1, vol.shape[0],dtype='int')
        interp3 = RegularGridInterpolator((z, y, x), vol)
        # %%
        candidate_mask = apply_masks(FLAIR_filename, WM_filename)

        # In[12]:
        patches = {}

        voxel_list = itertools.product(z, y, x)
        zero_count = 0


        for i, j, k in voxel_list:
            if labels[i][j][k] == 1:
                labeled1 += 1
                if candidate_mask[i][j][k] == True:
                    labeled1_and_candidate += 1
                else:
                    labeled1_and_not_candidate += 1
            else:
                labeled0 += 1
                if candidate_mask[i][j][k] == True:
                    labeled0_and_candidate += 1
                else:
                    labeled0_and_not_candidate += 1

        logger.info("person {} time {} labeled0 count {}".format(index,index2,labeled0))
        logger.info("person {} time {} labeled1 count {}".format(index,index2,labeled1))
        logger.info("person {} time {} labeled0_and_not_candidate count {}".format(index,index2,labeled0_and_not_candidate))
        logger.info("person {} time {} labeled1_and_not_candidate count {}".format(index,index2,labeled1_and_not_candidate))
        logger.info("person {} time {} labeled0_and_candidate count {}".format(index,index2,labeled0_and_candidate))
        logger.info("person {} time {} labeled1_and_candidate count {}".format(index,index2,labeled1_and_candidate))



