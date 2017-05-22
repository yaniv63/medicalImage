import numpy as np
from collections import defaultdict
from itertools import product
from paths import Src_Path,Data_Path,patches

def multi_dimensions(n, type=None):
    if n <= 0:
        if type is not None:
            return type()
        return None
    return defaultdict(lambda: multi_dimensions(n - 1, type))

def load_patches_list(person_list):
    import pickle
    with open(patches + "positive_list_person_{}.lst".format(str(person_list)), 'rb') as fp1, \
            open(patches + "negative_list_person_{}.lst".format(str(person_list)), 'rb') as fp2:
            positive_list_np = np.array(pickle.load(fp1))
            negative_list_np = np.array(pickle.load(fp2))
    return positive_list_np,negative_list_np


def load_images(person_list, contrast_type):
    image_list =defaultdict(dict)
    for person in person_list:
        for time in range(1,5):
            image_list[person][time] = load_image(person, time, contrast_type)
    return image_list

def load_data(person_list,contrast_type):
    pos_list,neg_list = load_patches_list(person_list)
    images = load_images(person_list,contrast_type)
    return images,pos_list,neg_list

def load_all_data(person_list, time_list, contrast_list):
    pos_list,neg_list = load_patches_list(person_list)
    images = load_all_images(person_list, time_list, contrast_list)
    return images,pos_list,neg_list

def load_all_images(person_list, time_list, contrast_list):
    image_list = multi_dimensions(3)
    indexes = product(person_list, time_list, contrast_list)
    for person, time, contrast in indexes:
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

# def load_patches_list_tr(person_list):
#     import pickle
#     with open(patches + "tempP", 'rb') as fp1, \
#             open(patches + "tempN", 'rb') as fp2:
#             positive_list_np = np.array(pickle.load(fp1))
#             negative_list_np = np.array(pickle.load(fp2))
#     return positive_list_np,negative_list_np
#
# def load_patches_list_val(person_list):
#     import pickle
#     with open(patches + "tempP_v", 'rb') as fp1, \
#             open(patches + "tempN_v", 'rb') as fp2:
#             positive_list_np = np.array(pickle.load(fp1))
#             negative_list_np = np.array(pickle.load(fp2))
#     return positive_list_np,negative_list_np
# def load_data_v(person_list,contrast_type):
#     pos_list,neg_list = load_patches_list_val(person_list)
#     images = load_images(person_list,contrast_type)
#     return images,pos_list,neg_list