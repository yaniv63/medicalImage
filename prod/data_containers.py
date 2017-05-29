import numpy as np
import nibabel as nb
import pickle
from collections import defaultdict
from itertools import product
from paths import Src_Path,Data_Path,patches,Labels_Path

def multi_dimensions(n, type=None):
    if n <= 0:
        if type is not None:
            return type()
        return None
    return defaultdict(lambda: multi_dimensions(n - 1, type))


def load_images(person_list, contrast_type):
    image_list =defaultdict(dict)
    for (person,time) in person_list:
        image_list[person][time] = load_image(person, time, contrast_type)
    return image_list

def load_data(person_list,contrast_type):
    pos_list,neg_list = create_ROI_list(person_list)
    images = load_images(person_list,contrast_type)
    return images,pos_list,neg_list

def load_all_data(person_list, contrast_list):
    pos_list,neg_list = create_ROI_list(person_list)
    images = load_all_images(person_list, contrast_list)
    return images,pos_list,neg_list

def load_all_images(person_list, contrast_list):
    image_list = multi_dimensions(3)
    indexes = product(person_list, contrast_list)
    for (person, time), contrast in indexes:
        image_list[person][time][contrast] = load_image(person, time, contrast)
    return image_list

def load_image(person, time, contrast):
    return np.load(
            Src_Path + Data_Path + "Person0{}_Time0{}_{}.npy".format(person, time, contrast))

def load_contrasts(person, time,contrast_list):
    contrasts = {}
    for contrast in contrast_list:
        contrasts[contrast] = load_image(person,time,contrast)
    return contrasts

def load_lables(person,time,doc_num):
    path = Src_Path + Labels_Path + "training0{}_0{}_mask{}.nii".format(person,time,doc_num)
    labels = nb.load(path).get_data()
    labels = labels.T
    labels = np.rot90(labels, 2, axes=(1, 2))
    return labels


def load_patch(person,time):
    with open(patches + "positive_person{}_time{}.lst".format(person,time), 'rb') as fp1, \
            open(patches + "negative_person{}_time{}.lst".format(person,time), 'rb') as fp2:
            positive_list_np = pickle.load(fp1)
            negative_list_np = pickle.load(fp2)
    return positive_list_np,negative_list_np


def create_ROI_list(input_list):
    positive_list = []
    negative_list = []
    for person, time in input_list:
        positive,negative = load_patch(person,time)
        positive_list.extend(positive)
        negative_list.extend(negative)
    return np.array(positive_list),np.array(negative_list)

