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

from keras.optimizers import SGD
import pickle
from sklearn.model_selection import KFold
import sys


from multi_predictors_combined import gating_model
from train_tools import create_callbacks, calc_epoch_size,combined_aggregate_genrated_samples_multiclass
from data_containers import load_all_data,load_all_images,separate_classes_indexes,load_index_list
from plotting_tools import *
#from train_proccesses_gate2 import generator_gate
from train_proccesses_gate import TrainGeneratorMultiClass
station = 'dist'


def train_combined(model,train_list,val_list,contrast_list,view_list,name,batch_size=16):

    callbacks = create_callbacks(name, fold=0)
    logger.debug("creating train & val generators")
    train_indexes = load_index_list("gate_indexes_person",train_list)
    train_images = load_all_images(train_list,contrast_list)
    val_indexes = load_index_list("gate_indexes_person",val_list)
    val_images = load_all_images(val_list,contrast_list)

    indexes_per_class_tr = separate_classes_indexes(train_indexes, 3)
    indexes_per_class_val = separate_classes_indexes(val_indexes, 3)

    #train_generator = generator_gate(indexes_per_class_tr,train_images,  contrast_list, view_list, batch_size, w=16,predictor_num=3)
    train_generator = TrainGeneratorMultiClass(train_images,indexes_per_class_tr,contrast_list, view_list, batch_size, w=16,workers_num=3)
    gen =train_generator.get_generator()
    val_set = combined_aggregate_genrated_samples_multiclass(val_images,indexes_per_class_val,contrast_list,view_list,batch_size,w=16,aug_args=None)
    logger.info("training combined model")
    smallest_set = min([len(set) for set in indexes_per_class_tr])
    class_num = len(indexes_per_class_tr)
    epoch_size = class_num * smallest_set - ((class_num * smallest_set) % batch_size)
    val_size = len(val_indexes)
    history = model.fit_generator(gen, samples_per_epoch=epoch_size, nb_epoch=200,nb_worker=1,validation_data=val_set,nb_val_samples=val_size, callbacks=callbacks)

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

np.random.seed(42)

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

    name="test_1"
    logger.info("training model {}".format(name))
    runs = []
    predictor = gating_model(N_exp=3, N_mod=4, img_rows=33, img_cols=33)
    optimizer = SGD(lr=0.01, nesterov=True)
    predictor.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'fmeasure'])

    history = train_combined(predictor,train_d,val_d, MR_modalities,view_list,
                             name=name)
    runs.append(history.history)

    with open(run_dir + 'cross_valid_stats_{}.lst'.format(name), 'wb') as fp:
            pickle.dump(runs, fp)
    plot_training(runs,name = name)

