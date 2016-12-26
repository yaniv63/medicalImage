# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:28:28 2016

@author: yaniv
"""


# coding: utf-8

# In[ ]:

# import os
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt

# init keras
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator


# In[10]:

# parameters
img_rows, img_cols = 32, 32
img_channels = 3
nb_classes = 2
MR_modalities = ['FLAIR', 'T2', 'MPRAGE', 'PD']
N_mod = len(MR_modalities)

# data
model_weights = r"/home/yaniv/src/medicalImaging/ref/MIMTP_model_weights.h5"
sample_patches_filename = r"/home/yaniv/src/medicalImaging/ref/sample_patches.npz"


# In[11]:

def create_smodel():
    smodel = Sequential()

    # 1x32x32 -> 24x14x14
    smodel.add(Convolution2D(24, 5, 5,
                             input_shape=(N_mod, img_rows, img_cols))) # 1x32x32 -> 24x28x28
    smodel.add(LeakyReLU())
    smodel.add(MaxPooling2D(pool_size=(2, 2)))                     # 24x28x28 -> 24x14x14
    smodel.add(Dropout(0.25))

    # 24x14x14 -> 32x6x6
    smodel.add(Convolution2D(32, 3, 3)) # 24x14x14 -> 32x12x12
    smodel.add(LeakyReLU())
    smodel.add(MaxPooling2D(pool_size=(2, 2)))                     # 32x12x12 -> 32x6x6
    smodel.add(Dropout(0.25))

    # 32x6x6 -> 48x4x4
    smodel.add(Convolution2D(48, 3, 3))
    smodel.add(LeakyReLU())
    smodel.add(Dropout(0.25))
    
    smodel.add(Flatten())
    smodel.add(Dense(16))
    smodel.add(LeakyReLU())
    smodel.add(Dropout(0.25))
    smodel.add(Dense(1,activation='sigmoid'))    
    
    return smodel

def create_2_predicters_model():
    graph = Graph()
    
    # basic model for a single slice image
    first_predict_model = create_smodel()
    second_predict_model = create_smodel()
    
    # add to graph
    graph.add_input(name='first_predict_input', input_shape=(N_mod,32,32))
    graph.add_input(name='second_predict_input', input_shape=(N_mod,32,32))
    graph.add_node(first_predict_model, name='first_predict_model', input='first_predict_input')
    graph.add_node(second_predict_model, name='second_predict_model', input='second_predict_input')
    graph.add_node(layer=Dense(1,activation='sigmoid'), name='final_predict', inputs=['first_predict_model', 'second_predict_model'], 
                   merge_mode='concat', concat_axis=1)
    graph.add_output(name='output', input='final_predict')
    
    return graph


# In[14]:
small_model = create_smodel()
model =  create_2_predicters_model()

from keras.utils.visualize_util import plot
plot(model, to_file='model-yaniv.png')


# compile net
model.compile(optimizer='adadelta', loss={'output':'categorical_crossentropy'})

# In[ ]:

# load sample patches
Npz = np.load(sample_patches_filename)
pos_curr = Npz['IPosCurr']
neg_curr = Npz['INegCurr']
plt.imshow(neg_curr[1,0,:,:,2],cmap='gray')
flair_samples = np.concatenate((pos_curr[:,0,:,:,1],neg_curr[:,0,:,:,1]))
flair_labels = np.concatenate((np.ones(flair_samples.shape[0]/2),np.zeros(flair_samples.shape[0]/2)))
T2_samples = np.concatenate((pos_curr[:,1,:,:,1],neg_curr[:,1,:,:,1]))
T2_labels = np.concatenate((np.ones(flair_samples.shape[0]/2),np.zeros(flair_samples.shape[0]/2)))
print('samples x channels x width x height x views = ')
print(pos_curr.shape)


model.fit([flair_samples,T2_samples][flair_labels,T2_labels],nb_epoch=5,batch_size=5)




# In[ ]:

# predict positive patches
#Sp = model.predict({'first_predict_input':pos_curr[:,1,:,:,1]+neg_curr[:,1,:,:,1],'second_predict_input':pos_curr[:,2,:,:,1]+neg_curr[:,2,:,:,1]},{'output': })
#Sp = Sp['output'][:,1]
#
## display scores
#print(Sp)


