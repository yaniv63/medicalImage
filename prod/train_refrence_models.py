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
import os.path

from multi_predictors_combined import one_predictor_model,average_n_models_prediction,n_parameters_combined_model
from train_tools import create_callbacks,generator,combined_generator,aggregate_genrated_samples\
    , calc_epoch_size,combined_aggregate_genrated_samples,create_callbacks_refrences
from data_containers import load_data,load_all_data
from plotting_tools import *
from train_proccesses import TrainGenerator

comp = "server"
val_loc = "./patches/" if comp=="server" else "../runs/MOE runs/run17-multilabel-cross-validation/"

def train_combined(model,PersonTrainList,PersonValList,contrast_list,view_list,name,batch_size=256):#256

    callbacks = create_callbacks_refrences(name, fold=0)
    logger.debug("creating train & val generators")
    train_images,positive_list, negative_list = load_all_data(PersonTrainList,contrast_list)
    train_generator = TrainGenerator(train_images,positive_list, negative_list,contrast_list,view_list,batch_size,w=16,num_labels=1)
    val_images, pos_val_list, neg_val_list = load_all_data(PersonValList,contrast_list)
    if os.path.isfile(val_loc + 'val_set_{}'.format(test_person)):
        logger.info("use existing val set")
        with open(val_loc+'val_set_{}'.format(test_person), 'rb') as f:
            val_set = pickle.load(f)
        if model_ref == "single":
            val_samples = val_set[0][view_list.index(angle)]
        else:
            val_samples = val_set[0]
        val_l = val_set[1][0]
        val_set = (val_samples, val_l)
    else:
        val_set = combined_aggregate_genrated_samples(val_images,pos_val_list, neg_val_list,contrast_list,view_list,batch_size,w=16,aug_args=None,num_labels=1)

    logger.info("training combined model")
    epoch_size = calc_epoch_size(positive_list, batch_size)
    val_size = calc_epoch_size(pos_val_list, batch_size)
    gen = train_generator.get_generator()
    e = gen.next()
    history = model.fit_generator(gen, samples_per_epoch=epoch_size, nb_epoch=200, callbacks=callbacks,
                                  validation_data=val_set,nb_val_samples=val_size)
    gen.close()
    train_generator.close()
    return history

def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))

# Install exception handler
#sys.excepthook = my_handler

# ######## train model


logger.debug("start script")
MR_modalities = ['FLAIR', 'T2', 'MPRAGE', 'PD']
view_list = ['axial','coronal', 'sagittal']

model_ref = "single"
view_angle = "c"


if model_ref =="single":
    angle = {"a": 'axial', "c": 'coronal', "s": 'sagittal'}.get(view_angle)
    view_list = [angle]
    predictor = one_predictor_model(N_mod=4)
elif model_ref == "av":
    predictor = average_n_models_prediction(N_mod=4,n=3)
elif model_ref == "conct":
    predictor = n_parameters_combined_model(N_mod=4,n=3)



data = np.array([[(1,x) for x in range(1,5)],[(2,x) for x in range(1,5)],[(3,x) for x in range(1,6)],[(4,x) for x in range(1,5)],
        [(5,x) for x in range(1,5)]])
kf = KFold(n_splits=5)

for train_index, test_index in kf.split(data):
    X_train = data[train_index]
    val_d = X_train[-1]
    train_data =X_train[:-1].tolist()
    train_d = [item for sublist in train_data for item in sublist]
    test_person = data[test_index][0][0][0]
    if test_person != 5:
        continue
    logger.info("TRAIN: {} VAL: {} , TEST: {}".format(train_d,val_d,test_person))

    name="{}_test_{}".format(model_ref,test_person)
    if model_ref=="single":
        name = name+"_{}".format(angle)
    logger.info("training model {}".format(name))
    runs = []
    # predictor = average_n_models_prediction(N_mod = 4, img_rows = 33, img_cols = 33,n=3)
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    predictor.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'fmeasure'])
    history = train_combined(predictor, train_d, val_d, MR_modalities,view_list,
                             name=name)
    runs.append(history.history)

    with open(run_dir + 'cross_valid_stats_{}.lst'.format(name), 'wb') as fp:
            pickle.dump(runs, fp)
    #plot_training(runs,name = name)

