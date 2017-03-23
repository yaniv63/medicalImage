import numpy as np

sample_patches_filename = 'ref/sample_patches.npz'
Npz = np.load(sample_patches_filename)
pos_curr = Npz['IPosCurr']
pos_prev = Npz['IPosPrev']
neg_curr = Npz['INegCurr']
neg_prev = Npz['INegPrev']

print pos_curr.shape

import matplotlib
import matplotlib.pylab as plt
import pickle

patches = r'./patches/'

# plt.figure()
# plt.imshow(pos_curr[2,0,:,:,0],cmap=matplotlib.cm.gray)
# plt.figure()
# plt.imshow(pos_curr[3,0,:,:,0],cmap=matplotlib.cm.gray)
# plt.figure()
# plt.imshow(neg_curr[10,0,:,:,0],cmap=matplotlib.cm.gray)
# plt.figure()
# plt.imshow(neg_curr[1,0,:,:,0],cmap=matplotlib.cm.gray)
# plt.show()

def generate_train(patchType, personList, batchSize=256):
    while True:
        for index in personList:
            for index2 in range(1, 5):
                with open(patches + "patches_"+patchType+ "_train_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels_train_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                    samples_train = np.array(pickle.load(fp1))
                    labels_train = np.array(pickle.load(fp2))

                samples_train = np.expand_dims(samples_train, 1)
                labels_train = np.expand_dims(labels_train, 1)
                k = samples_train.shape[0] / batchSize

                # divide batches
                for i in range(k):
                    yield (
                    samples_train[i * batchSize:(i + 1) * batchSize], labels_train[i * batchSize:(i + 1) * batchSize])

PersonTrainList = [2]
a = generate_train(patchType="axial",personList=PersonTrainList)

plt.figure()
plt.imshow(train[3,0,:,:],cmap=matplotlib.cm.gray)
plt.figure()
plt.imshow(train[226,0,:,:],cmap=matplotlib.cm.gray)
plt.figure()
plt.imshow(train[237,0,:,:],cmap=matplotlib.cm.gray)
plt.figure()
plt.imshow(train[253,0,:,:],cmap=matplotlib.cm.gray)
plt.figure()
plt.imshow(train[199,0,:,:],cmap=matplotlib.cm.gray)
plt.figure()
plt.imshow(train[205,0,:,:],cmap=matplotlib.cm.gray)
plt.figure()
plt.imshow(train[32,0,:,:],cmap=matplotlib.cm.gray)
plt.figure()
plt.imshow(train[8,0,:,:],cmap=matplotlib.cm.gray)


plt.show()