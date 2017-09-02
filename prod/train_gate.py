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

from keras.optimizers import SGD
import pickle
from sklearn.model_selection import KFold
import sys


from multi_predictors_combined import gating_model
from train_tools import create_callbacks, calc_epoch_size,combined_aggregate_genrated_samples_multiclass
from data_containers import load_all_data,load_all_images,separate_classes_indexes
from plotting_tools import *
from train_proccesses_gate import TrainGeneratorMultiClass,TrainGeneratorMultiClassAggregator
from train_tools import saperate_set_major_minor



def train_combined(model,images,train_list,contrast_list,view_list,name,batch_size=16):

    callbacks = create_callbacks(name, fold=0)
    logger.debug("creating train & val generators")

    train, val = saperate_set_major_minor(train_list)
    indexes_per_class_tr = separate_classes_indexes(train, 3)
    indexes_per_class_val = separate_classes_indexes(val, 3)

    train_generator = TrainGeneratorMultiClass(images, indexes_per_class_tr, contrast_list, view_list, batch_size, w=16)
    # val_images, pos_val_list, neg_val_list = load_all_data(PersonValList,contrast_list)
    val_set = combined_aggregate_genrated_samples_multiclass(images,indexes_per_class_val,contrast_list,view_list,batch_size,w=16,aug_args=None)
    logger.info("training combined model")
    smallest_set = min([len(set) for set in indexes_per_class_tr])
    class_num = len(indexes_per_class_tr)
    epoch_size = class_num * smallest_set
    val_size = len(val)
    gen = train_generator.get_generator()
    history = model.fit_generator(gen, samples_per_epoch=10624, nb_epoch=5,nb_worker=1,validation_data=val_set,nb_val_samples=val_size, callbacks=callbacks)
    gen.close()
    train_generator.close()
    return history


# def train_combined(model,PersonTrainList,PersonValList,contrast_list,view_list,name,batch_size=256):
#
#     callbacks = create_callbacks(name, fold=0)
#     logger.debug("creating train & val generators")
#     train_images,positive_list, negative_list = load_all_data(PersonTrainList,contrast_list)
#     train_generator = TrainGenerator(train_images,positive_list, negative_list,contrast_list,view_list,batch_size,w=16)
#     val_images, pos_val_list, neg_val_list = load_all_data(PersonValList,contrast_list)
#     val_set = combined_aggregate_genrated_samples(val_images,pos_val_list, neg_val_list,contrast_list,view_list,batch_size,w=16,aug_args=None)
#     logger.info("training combined model")
#     epoch_size = calc_epoch_size(positive_list, batch_size)
#     val_size = calc_epoch_size(pos_val_list, batch_size)
#     gen = train_generator.get_generator()
#     history = model.fit_generator(gen, samples_per_epoch=epoch_size, nb_epoch=200, callbacks=callbacks,
#                                   validation_data=val_set,nb_val_samples=val_size)
#     gen.close()
#     train_generator.close()
#     return history

def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))

# Install exception handler
sys.excepthook = my_handler

# ######## train model
logger.debug("start script")
MR_modalities = ['FLAIR', 'T2', 'MPRAGE', 'PD']
view_list = ['axial','coronal', 'sagittal']

np.random.seed(42)

data = np.array([[(1,x) for x in range(1,5)],[(2,x) for x in range(1,5)],[(3,x) for x in range(1,6)],[(4,x) for x in range(1,5)],
        [(5,x) for x in range(1,5)]])
kf = KFold(n_splits=5)

for view in view_list:
    for train_index, test_index in kf.split(data):
        X_train = data[train_index]
        val_d = X_train[-1]
        train_data =X_train[:-1].tolist()
        train_d = [item for sublist in train_data for item in sublist]
        test_person = data[test_index][0][0][0]
        if test_person != 1:
            continue
        logger.info("TRAIN: {} VAL: {} , TEST: {}".format(train_d,val_d,test_person))

        name="test_{}_{}".format(test_person,view)
        logger.info("training model {}".format(name))
        runs = []
        predictor = gating_model(N_exp=3, N_mod=4, img_rows=33, img_cols=33)
        optimizer = SGD(lr=0.01, nesterov=True)
        predictor.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'fmeasure'])

        PersonTrainList = [(1, 2)]
        index_path = '/media/sf_shared/src/medicalImaging/stats/test1_gate_indexes.npy'

        train_images = load_all_images(PersonTrainList, MR_modalities)
        classes_indexes = np.load(index_path)
        train, test = saperate_set_major_minor(classes_indexes)


        history = train_combined(predictor,train_images, train, MR_modalities,view_list,
                                 name=name)
        runs.append(history.history)

        with open(run_dir + 'cross_valid_stats_{}.lst'.format(name), 'wb') as fp:
                pickle.dump(runs, fp)
        plot_training(runs,name = name)

