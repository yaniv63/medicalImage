import numpy as np
import nibabel as nb
import matplotlib
import matplotlib.pylab as plt


Src_Path = r"../train/"
Data_Path = r"data/"
Labels_Path = r"seg/"
Output_Path=r"patches/"
index = 1
index2  = 1
FLAIR_filename = Src_Path+Data_Path+"Person0{}_Time0{}_FLAIR.npy".format(index,index2)
FLAIR_labels_1 = Src_Path+Labels_Path+"training0{}_0{}_mask1.nii".format(index,index2)
vol = np.load(FLAIR_filename)
labels = nb.load(FLAIR_labels_1).get_data()
labels = labels.T
labels = np.rot90(labels, 2, axes=(1, 2))

plt.imshow(vol[100, :, :], cmap=matplotlib.cm.gray)
plt.figure()
plt.imshow(labels[100, :, :], cmap=matplotlib.cm.gray)
plt.show()