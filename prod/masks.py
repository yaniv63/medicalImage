
import numpy as np
from itertools import product
import scipy.ndimage.morphology as mrph

from paths import  Src_Path,WM_path
from data_containers import multi_dimensions

FLAIR_th = 0.91
WM_prior_th = 0.5

def binary_disk(r):
    arr = np.ones((2*r+1,2*r+1,2*r+1))
    arr[r,r,r] = 0

    dt = mrph.distance_transform_bf(arr,metric='euclidean')
    disk = dt<=r
    disk = disk.astype('float32')

    return disk

def load_wm_masks(person_list, time_list):
    wm_list = multi_dimensions(2)
    indexes = product(person_list, time_list)
    for person, time in indexes:
        wm_list[person][time] = np.load(Src_Path + WM_path + "Person0{}_Time0{}.npy".format(person, time))
    return wm_list


def create_image_masks(wm_list, image_list, person_list, time_list):
    sel = binary_disk(2)
    masks_list = multi_dimensions(2)
    indexes = product(person_list, time_list)
    for person, time in indexes:
        WM_dilated = mrph.filters.maximum_filter(wm_list[person][time], footprint=sel)

        # apply thresholds
        FLAIR_mask = image_list[person][time]['FLAIR'] > FLAIR_th
        WM_mask = WM_dilated > WM_prior_th

        # final mask: logical AND
        masks_list[person][time] = np.logical_and(FLAIR_mask, WM_mask)
    return masks_list


def is_masks_positive(mask_list, person, time_list, index):
    mask_list = map(lambda time: mask_list[person][time][index[0], index[1], index[2]], time_list)
    return mask_list.count(True) > 0 # == len(mask_list)