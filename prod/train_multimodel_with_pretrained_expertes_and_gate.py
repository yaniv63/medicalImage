# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:32:39 2016

@author: yaniv
"""
# create logger

from paths import *
from logging_tools import get_logger

run_dir = get_run_dir()
logger = get_logger(run_dir)

from keras.optimizers import Adadelta,SGD
import pickle
from  itertools import product
from sklearn.model_selection import KFold
import sys


from multi_predictors_combined import one_predictor_model,n_predictors_combined_model,n_parameters_combined_model,n_experts_combined_model,n_experts_combined_model_gate_parameters
from train_tools import create_callbacks,generator,combined_generator,aggregate_genrated_samples\
    , calc_epoch_size,combined_aggregate_genrated_samples
from data_containers import load_data,load_all_data
from metrics import calc_confusion_mat,calc_dice
from plotting_tools import *
from train_proccesses import TrainGenerator


def train_combined(model,PersonTrainList,PersonValList,contrast_list,view_list,name,batch_size=256):

    callbacks = create_callbacks(name, fold=0)
    logger.debug("creating train & val generators")
    train_images,positive_list, negative_list = load_all_data(PersonTrainList,contrast_list)
    train_generator = TrainGenerator(train_images,positive_list, negative_list,contrast_list,view_list,batch_size,w=16)
    val_images, pos_val_list, neg_val_list = load_all_data(PersonValList,contrast_list)
    #val_generator = combined_generator(pos_val_list, neg_val_list, val_images,contrast_list,view_list)
    val_set = combined_aggregate_genrated_samples(val_images,pos_val_list, neg_val_list,contrast_list,view_list,batch_size,w=16,aug_args=None)
    logger.info("training combined model")
    epoch_size = calc_epoch_size(positive_list, batch_size)
    val_size = calc_epoch_size(pos_val_list, batch_size)
    gen = train_generator.get_generator()
    history = model.fit_generator(gen, samples_per_epoch=epoch_size, nb_epoch=1, callbacks=callbacks,
                                  validation_data=val_set,nb_val_samples=val_size)
    gen.close()
    train_generator.close()
    # confusion_mat = calc_confusion_mat(model, val_set[0], val_set[1], "individual val {}".format(0))
    # calc_dice(confusion_mat, "individual val {}".format(0))
    return history

def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))

# Install exception handler
sys.excepthook = my_handler

# ######## train model
logger.debug("start script")
MR_modalities = ['FLAIR', 'T2', 'MPRAGE', 'PD']
view_list = ['axial','coronal', 'sagittal']

station = 'server'
data = np.array([[(1,x) for x in range(1,5)],[(2,x) for x in range(1,5)],[(3,x) for x in range(1,6)],[(4,x) for x in range(1,5)],
        [(5,x) for x in range(1,5)]])
kf = KFold(n_splits=5)

for train_index, test_index in kf.split(data):
    X_train = data[train_index]
    val_d = X_train[-1]
    train_data =X_train[:-1].tolist()
    train_d = [item for sublist in train_data for item in sublist]
    test_person = data[test_index][0][0][0]
    if test_person != 1:
        continue
    logger.info("TRAIN: {} VAL: {} , TEST: {}".format(train_d,val_d,test_person))

    name="test_{}".format(test_person)
    logger.info("training model {} . multimodel with pretrained expertes and gate. use parameters gate".format(name))
    runs = []
    if station=='desktop':
	w_path =  '/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/'
        w_path_gate = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run9-pretrain gate parameters/'
    else:
	w_path = weight_path + 'moe/'

    moe = n_experts_combined_model_gate_parameters(n=3, N_mod=4, img_rows=33, img_cols=33)
    moe.get_layer('Seq_0').load_weights(w_path + 'model_test_1_axial_fold_0.h5', by_name=True)
    moe.load_weights(w_path + 'model_test_1_axial_fold_0.h5', by_name=True)

    moe.get_layer('Seq_1').load_weights(w_path + 'model_test_1_coronal_fold_0.h5', by_name=True)
    moe.load_weights(w_path + 'model_test_1_coronal_fold_0.h5', by_name=True)

    moe.get_layer('Seq_2').load_weights(w_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)
    moe.load_weights(w_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)

    moe.load_weights(w_path + 'gate_parameters_test1.h5',by_name=True)
    #moe.load_weights(w_path + 'w_test1.h5')

    optimizer = SGD(lr=0.01, nesterov=True)
    moe.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    history = train_combined(moe, train_d, val_d, MR_modalities, view_list,
                             name=name)
    runs.append(history.history)

    with open(run_dir + 'cross_valid_stats_{}.lst'.format(name), 'wb') as fp:
            pickle.dump(runs, fp)
    plot_training(runs,name = name)

