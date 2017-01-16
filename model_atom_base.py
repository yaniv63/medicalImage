# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:47:23 2016

@author: yaniv
"""

from keras.layers import Input, Dense,Flatten,Convolution2D,MaxPooling2D,Dropout,LeakyReLU
from keras.models import Model,Sequential
import numpy as np

#def atom_model_flat(sequence_num=1):
#    
#    atom_input = Input(shape=(32,32,),name='input_num%d'%(sequence_num))
#    flat = Flatten()(atom_input)
#    dense_output =Dense(10,activation='relu')(flat)
#    prediction =Dense(1,activation='sigmoid')(dense_output)
#    model = Model(input=atom_input,output=prediction)
#    return model
#
#def atom_model_conv(sequence_num=1):
#    #atom_input = Input(shape=(1,32,32),name='input_num%d'%(sequence_num))
#    conv1 = Convolution2D(24, 5, 5)#(atom_input) # 1x32x32 -> 24x28x28
#    leakyrelu1 = LeakyReLU()(conv1)    
#    maxpool1= MaxPooling2D(pool_size=(2, 2))(leakyrelu1)                     # 24x28x28 -> 24x14x14
#    dropout1 = Dropout(0.25)(maxpool1)    # 24x14x14 -> 32x6x6    
#
#    conv2 = Convolution2D(32, 3, 3)(dropout1) # 24x14x14 -> 32x12x12
#    leakyrelu2 = LeakyReLU()(conv2)        
#    maxpool2= MaxPooling2D(pool_size=(2, 2))(leakyrelu2)                     # 24x28x28 -> 24x14x14
#    dropout2 = Dropout(0.25)(maxpool2)    
#
#    # 32x6x6 -> 48x4x4
#    conv3 = Convolution2D(48, 3, 3)(dropout2)
#    leakyrelu3 = LeakyReLU()(conv3)            
#    dropout3 = Dropout(0.25)(leakyrelu3)    
#    
#    flat = Flatten()(dropout3)
#    dense = Dense(16)(flat)
#    leakyrelu4 = LeakyReLU()(dense)        
#    dropout_f = Dropout(0.25)(leakyrelu4)
#    decision = Dense(1,activation='sigmoid')(dropout_f)   
#    
#    model = Model(output=decision)
#    #model = Model(input=atom_input,output=decision)
#    return model
    
def create_smodel(N_mod, img_rows, img_cols,index=0):
    index = str(index)    
    smodel = Sequential(name='Seq_'+index)
    # 1x32x32 -> 24x14x14
    smodel.add(Convolution2D(24, 5, 5,
                             input_shape=(N_mod, img_rows, img_cols),name='conv1_'+index)) # 1x32x32 -> 24x28x28
    smodel.add(LeakyReLU(name='leakyrelu1_'+index))
    smodel.add(MaxPooling2D(pool_size=(2, 2),name='maxpool1_'+index))                     # 24x28x28 -> 24x14x14
    smodel.add(Dropout(0.25,name='drop1_'+index))

    # 24x14x14 -> 32x6x6
    smodel.add(Convolution2D(32, 3, 3,name='conv2_'+index)) # 24x14x14 -> 32x12x12
    smodel.add(LeakyReLU(name='leakyrelu2_'+index))
    smodel.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2_'+index))                     # 32x12x12 -> 32x6x6
    smodel.add(Dropout(0.25,name='drop2_'+index))

    # 32x6x6 -> 48x4x4
    smodel.add(Convolution2D(48, 3, 3,name='conv3_'+index))
    smodel.add(LeakyReLU(name='leakyrelu3_'+index))
    smodel.add(Dropout(0.25,name='drop3_'+index))
    
    smodel.add(Flatten(name='flat1_'+index))
    smodel.add(Dense(16,name='dense1_'+index))
    smodel.add(LeakyReLU(name='leakyrelu4_'+index))
    smodel.add(Dropout(0.25,name='drop4_'+index))    
    return smodel


if __name__ == "__main__":          
    # load sample patches
    sample_patches_filename = r"/media/sf_ubuntuFolder/src/medicalImaging/ref/sample_patches.npz"
    Npz = np.load(sample_patches_filename)
    pos_curr = Npz['IPosCurr']
    neg_curr = Npz['INegCurr']
    flair_samples = np.concatenate((pos_curr[:,0,:,:,1],neg_curr[:,0,:,:,1])).reshape(100,1,32,32)
    flair_labels = np.concatenate((np.ones(flair_samples.shape[0]/2),np.zeros(flair_samples.shape[0]/2)))              
    #mymodel=atom_model_flat()
    mymodel = atom_model_conv()
    mymodel.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    from keras.utils.visualize_util import plot
    plot(mymodel, to_file='model_atom-yaniv.png',show_layer_names=True,show_shapes=True)
    
    mymodel.fit(flair_samples, flair_labels,batch_size=10,shuffle=True)
    mymodel.save_weights(r'/media/sf_ubuntuFolder/src/medicalImaging/weights.h5')
    result = mymodel.evaluate(flair_samples,flair_labels) 
    a = mymodel.get_weights()
    
    