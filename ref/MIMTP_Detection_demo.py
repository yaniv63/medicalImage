
# coding: utf-8

# In[ ]:

# import os
import numpy as np
from keras.utils import np_utils


# In[9]:

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
    
    return smodel


# In[12]:

def create_slice_model():
    slice_model = Sequential()
    
    slice_model.add(Convolution2D(48, 1, 1, input_shape=(96,4,4)))
    slice_model.add(LeakyReLU())
    slice_model.add(Dropout(0.25))
    
    # output is a flat vector
    slice_model.add(Flatten())
    slice_model.add(Dense(16))
    slice_model.add(LeakyReLU())
    slice_model.add(Dropout(0.25))
    
    return slice_model


# In[13]:

def create_full_model():
    graph = Graph()
    slice_model_node_list = []
    
    for ch in range(img_channels):
        # basic model for a single slice image
        s_curr_model = create_smodel()
        s_prev_model = create_smodel()
        
        # combining current and previous image
        s_model = create_slice_model()
        
        # add to graph
        graph.add_input(name='s%d_curr'%(ch), input_shape=(N_mod,32,32))
        graph.add_input(name='s%d_prev'%(ch), input_shape=(N_mod,32,32))
        graph.add_node(s_curr_model, name='s%d_curr_model_node'%(ch), input='s%d_curr'%(ch))
        graph.add_node(s_prev_model, name='s%d_prev_model_node'%(ch), input='s%d_prev'%(ch))
        graph.add_node(layer=s_model, name='s%d'%(ch), inputs=['s%d_curr_model_node'%(ch), 's%d_prev_model_node'%(ch)], 
                       merge_mode='concat', concat_axis=1)
        
        slice_model_node_list.append('s%d'%(ch))
    

    # merge slices
    graph.add_node(layer=Dense(16), name='slices_Dense', inputs=slice_model_node_list)
    graph.add_node(layer=Dropout(0.25), name='slices_Dense_Dropout', input='slices_Dense')
    graph.add_node(layer=Dense(nb_classes, activation='softmax'), name='slices_out', input='slices_Dense_Dropout')
    graph.add_output(name='output', input='slices_out')
    
    return graph


# In[14]:

# init net structure
model = create_full_model()
# load net weights
model.load_weights(model_weights)

# compile net
model.compile(optimizer='adadelta', loss={'output':'categorical_crossentropy'})


# In[ ]:

# load sample patches
Npz = np.load(sample_patches_filename)
pos_curr = Npz['IPosCurr']
pos_prev = Npz['IPosPrev']
neg_curr = Npz['INegCurr']
neg_prev = Npz['INegPrev']

print('samples x channels x width x height x views = ')
print(pos_curr.shape)


# In[ ]:

# predict positive patches
Sp = model.predict({'s0_curr':pos_curr[:,:,:,:,0], 's0_prev':pos_prev[:,:,:,:,0], 
                   's1_curr':pos_curr[:,:,:,:,1], 's1_prev':pos_prev[:,:,:,:,1],
                   's2_curr':pos_curr[:,:,:,:,2], 's2_prev':pos_prev[:,:,:,:,2]})

Sp = Sp['output'][:,1]

# display scores
print(Sp)


# In[ ]:

# predict negative patches
Sn = model.predict({'s0_curr':neg_curr[:,:,:,:,0], 's0_prev':neg_prev[:,:,:,:,0], 
                   's1_curr':neg_curr[:,:,:,:,1], 's1_prev':neg_prev[:,:,:,:,1],
                   's2_curr':neg_curr[:,:,:,:,2], 's2_prev':neg_prev[:,:,:,:,2]})

Sn = Sn['output'][:,1]

# display scores
print(Sn)


# In[ ]:



