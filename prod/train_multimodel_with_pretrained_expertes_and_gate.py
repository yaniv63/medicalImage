# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:32:39 2016

@author: yaniv
"""
# create logger

from paths import *
from prod.logging_tools import get_logger

run_dir = get_run_dir()
logger = get_logger(run_dir)

from keras.optimizers import Adadelta,SGD
import pickle
from  itertools import product
from sklearn.model_selection import KFold
import sys
import numpy as np
np.seed

from prod.multi_predictors_combined import one_predictor_model,n_predictors_combined_model,n_parameters_combined_model,n_experts_combined_model,n_experts_combined_model_gate_parameters
from train_tools import create_callbacks,generator,combined_generator,aggregate_genrated_samples\
    , calc_epoch_size,combined_aggregate_genrated_samples
from data_containers import load_data,load_all_data
from metrics import calc_confusion_mat,calc_dice
from plotting_tools import *
from train_proccesses import TrainGenerator
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix



def train_combined(model,PersonTrainList,PersonValList,contrast_list,view_list,name,batch_size=16):

    callbacks = create_callbacks(name, fold=0)
    logger.debug("creating train & val generators")
    train_images,positive_list, negative_list = load_all_data(PersonTrainList,contrast_list)
    train_generator = TrainGenerator(train_images,positive_list, negative_list,contrast_list,view_list,batch_size,w=16)
    val_images, pos_val_list, neg_val_list = load_all_data(PersonValList,contrast_list)
    val_set = combined_aggregate_genrated_samples(val_images,pos_val_list, neg_val_list,contrast_list,view_list,batch_size,w=16,aug_args=None)

    logger.info("predict val")
    pred = model.predict(val_set[0])
    prediction = [0 if p[0] <0.5 else 1 for p in pred]
    true_labels = val_set[1]
    logger.info("classification \n \n {}".format(classification_report(true_labels, prediction)))
    logger.info("accuracy \n \n {}".format(accuracy_score(true_labels, prediction)))
    logger.info("confusion matrix  \n \n {}".format(confusion_matrix(true_labels, prediction)))


    logger.info("training combined model")
    epoch_size = calc_epoch_size(positive_list, batch_size)
    val_size = calc_epoch_size(pos_val_list, batch_size)
    gen = train_generator.get_generator()
    history = model.fit_generator(gen, samples_per_epoch=epoch_size, nb_epoch=200, callbacks=callbacks,
                                  validation_data=val_set,nb_val_samples=val_size)
    gen.close()
    train_generator.close()

    return history

def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))

# Install exception handler
sys.excepthook = my_handler

# ######## train model
logger.debug("start script")
MR_modalities = ['FLAIR', 'T2', 'MPRAGE', 'PD']
view_list = ['axial','coronal', 'sagittal']




train_data = np.array([[(2,x) for x in range(1,5)],[(3,x) for x in range(1,6)],[(4,x) for x in range(1,5)]] )
val_d = [(1,x) for x in range(1,2)]
kf = KFold(n_splits=5)

#X_train = data[train_index]
#val_d = X_train[-1]
#train_data =X_train[:-1].tolist()
train_d = [item for sublist in train_data for item in sublist]
test_person = 5# data[test_index][0][0][0]
logger.info("TRAIN: {} VAL: {} , TEST: {}".format(train_d,val_d,test_person))


name="test_{}".format(test_person)
logger.info("training model {}".format(name))
runs = []
w_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/'
w_path_gate = '/media/sf_shared/src/medicalImaging/results/'

moe = n_experts_combined_model_gate_parameters(n=3, N_mod=4, img_rows=33, img_cols=33)
moe.get_layer('Seq_0').load_weights(w_path + 'model_test_1_axial_fold_0.h5', by_name=True)
moe.load_weights(w_path + 'model_test_1_axial_fold_0.h5', by_name=True)

moe.get_layer('Seq_1').load_weights(w_path + 'model_test_1_coronal_fold_0.h5', by_name=True)
moe.load_weights(w_path + 'model_test_1_coronal_fold_0.h5', by_name=True)

moe.get_layer('Seq_2').load_weights(w_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)
moe.load_weights(w_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)

moe.load_weights(w_path_gate + 'gate_batching_hard.h5',by_name=True)

layer_dict = dict([(layer.name, layer) for layer in moe.layers])

for name,layer in layer_dict.items():
    if 'gate' not in name:
        layer.trainable = False
optimizer = SGD(lr=0.01, nesterov=True)
moe.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
history = train_combined(moe, train_d, val_d, MR_modalities, view_list,
                         name=name)
runs.append(history.history)

with open(run_dir + 'cross_valid_stats_{}.lst'.format(name), 'wb') as fp:
        pickle.dump(runs, fp)
plot_training(runs,name = name)

