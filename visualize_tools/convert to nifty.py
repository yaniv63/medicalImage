import os
import nibabel as nb
import numpy as np
from prod.data_containers import load_lables


def np_2_nifty(vol, affine_path, save_path):
    affine = np.load(affine_path)
    c = np.flip(vol, 2)
    c = np.flip(c, 1)
    d = np.swapaxes(c, 0, 2).astype('float')
    e = nb.Nifti1Image(d, affine)
    nb.save(e, save_path)

init_path = '/media/sf_shared/results/1_2/'
affine_path = init_path+'affine.npy'
data_path = '/media/sf_shared/src/medicalImaging/'
vol_path = data_path + 'train/data/Person01_Time02_FLAIR.npy'
vol = np.load(vol_path)
np_2_nifty(vol, affine_path, init_path+'my_1_2_FLAIR.nii')
labels = load_lables(1,2,1)
np_2_nifty(labels, affine_path, init_path+'my_labels_1_2.nii')
