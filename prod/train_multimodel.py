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


from multi_predictors_combined import one_predictor_model,n_experts_combined_model_gate_parameters
from train_tools import create_callbacks,generator,combined_generator,aggregate_genrated_samples\
    , calc_epoch_size,combined_aggregate_genrated_samples
from data_containers import load_data,load_all_data
from metrics import calc_confusion_mat,calc_dice
from plotting_tools import *
from train_proccesses import TrainGenerator

def train(model,PersonTrainList,PersonValList,view_type,contrast_type,fold_num,name,batch_size=256):
    logger.debug("creating callbacks")
    callbacks = create_callbacks(name,fold_num)

    logger.debug("creating train & val generators")

    train_images,pos_train_list,neg_train_list = load_data(PersonTrainList,contrast_type)
    train_generator = generator(pos_train_list, neg_train_list, train_images,view_type)
    val_images,pos_val_list,neg_val_list = load_data(PersonValList,contrast_type)
    val_set = aggregate_genrated_samples(pos_val_list, neg_val_list, val_images,view_type)

    logger.info("training individual model")
    epoch_size = calc_epoch_size(pos_train_list,batch_size)
    history = model.fit_generator(train_generator, samples_per_epoch=epoch_size, nb_epoch=300, callbacks=callbacks,
                                      validation_data=val_set)
    confusion_mat = calc_confusion_mat(model, val_set[0], val_set[1], "individual val {}".format(fold_num))
    calc_dice(confusion_mat, "individual val {}".format(fold_num))
    return history

def train_combined(model,PersonTrainList,PersonValList,contrast_list,view_list,name,batch_size=256):#256

    callbacks = create_callbacks(name, fold=0)
    logger.debug("creating train & val generators")
    train_images,positive_list, negative_list = load_all_data(PersonTrainList,contrast_list)
    train_generator = TrainGenerator(train_images,positive_list, negative_list,contrast_list,view_list,batch_size,w=16,num_labels=4)
    val_images, pos_val_list, neg_val_list = load_all_data(PersonValList,contrast_list)
    #val_generator = combined_generator(pos_val_list, neg_val_list, val_images,contrast_list,view_list)
    val_set = combined_aggregate_genrated_samples(val_images,pos_val_list, neg_val_list,contrast_list,view_list,batch_size,w=16,aug_args=None,num_labels=4)
    logger.info("training combined model")
    epoch_size = calc_epoch_size(positive_list, batch_size)
    val_size = calc_epoch_size(pos_val_list, batch_size)
    gen = train_generator.get_generator()
    e = gen.next()
    history = model.fit_generator(gen, samples_per_epoch=epoch_size, nb_epoch=200, callbacks=callbacks,
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

station = 'server'
if station == 'desktop':
    experts_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/'
    w_path_gate = '/media/sf_shared/src/medicalImaging/results/'
else:
    experts_path = weight_path + '/moe/'
    w_path_gate = weight_path + '/moe/'


logger.debug("start script")
MR_modalities = ['FLAIR', 'T2', 'MPRAGE', 'PD']
view_list = ['axial','coronal', 'sagittal']


data = np.array([[(1,x) for x in range(1,5)],[(2,x) for x in range(1,5)],[(3,x) for x in range(1,6)],[(4,x) for x in range(1,5)],
        [(5,x) for x in range(1,5)]])
kf = KFold(n_splits=5)

for train_index, test_index in kf.split(data):
    X_train = data[train_index]
    val_d = X_train[-1]
    train_data =X_train[:-1].tolist()
    train_d = [item for sublist in train_data for item in sublist]
    test_person = data[test_index][0][0][0]
    if test_person != 2:
        continue
    logger.info("TRAIN: {} VAL: {} , TEST: {}".format(train_d,val_d,test_person))

    name="test_{}".format(test_person)
    logger.info("training model {}".format(name))
    runs = []
    predictor = n_experts_combined_model_gate_parameters()
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    predictor.compile(optimizer=optimizer,
                  loss={'main_output': 'binary_crossentropy',
                        'out0': 'binary_crossentropy',
                        'out1': 'binary_crossentropy',
                        'out2': 'binary_crossentropy',
                        },
                  loss_weights=[1., 0.2, 0.2, 0.2],
                  metrics=['accuracy', 'fmeasure'])
    history = train_combined(predictor, train_d, val_d, MR_modalities,view_list,
                             name=name)
    runs.append(history.history)

    with open(run_dir + 'cross_valid_stats_{}.lst'.format(name), 'wb') as fp:
            pickle.dump(runs, fp)
    #plot_training(runs,name = name)


#
# # and trained it via:
# model.fit({'main_input': headline_data, 'aux_input': additional_data},
#           {'main_output': labels, 'aux_output': labels},
#           nb_epoch=50, batch_size=32)
