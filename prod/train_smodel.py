# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:32:39 2016

@author: yaniv
"""
# create logger
from logging_tools import get_logger
from paths import *
run_dir = get_run_dir()
logger = get_logger(run_dir)

from keras.optimizers import Adadelta
import pickle
from  itertools import product
from sklearn.model_selection import KFold

from two_predictors_combined import one_predictor_model,n_predictors_combined_model
from train_tools import create_callbacks,generator,aggregate_genrated_samples,calc_epoch_size
from data_containers import load_data,load_data_v
from metrics import calc_confusion_mat,calc_dice
from plotting_tools import *


def train(model,PersonTrainList,PersonValList,view_type,contrast_type,fold_num,name,batch_size=256):
    logger.debug("training model {} fold {}".format(name,fold_num))
    logger.debug("creating callbacks")
    callbacks = create_callbacks(name,fold_num)

    logger.debug("creating train & val generators")

    train_images,pos_train_list,neg_train_list = load_data(PersonTrainList,contrast_type)
    train_generator = generator(pos_train_list, neg_train_list, train_images,view_type)
    logger.info("after tr")
    val_images,pos_val_list,neg_val_list = load_data(PersonValList,contrast_type)
    val_set = aggregate_genrated_samples(pos_val_list, neg_val_list, val_images,view_type)
    logger.info("after val")

    logger.info("training individual model")
    epoch_size = calc_epoch_size(pos_train_list,batch_size)
    history = model.fit_generator(train_generator, samples_per_epoch=epoch_size, nb_epoch=1, callbacks=callbacks,
                                      validation_data=val_set)
    confusion_mat = calc_confusion_mat(model, val_set[0], val_set[1], "individual val {}".format(fold_num))
    calc_dice(confusion_mat, "individual val {}".format(fold_num))
    return history



# ######## train model
MR_modalities = ['FLAIR', 'T2', 'MPRAGE', 'PD']
view_list = ['axial', 'coronal', 'sagittal']
image_types = product(MR_modalities,view_list)
for contrast_type,view_type in image_types:
    logger.info("training {} {} model".format(contrast_type,view_type))
    person_indices =np.array([1,2,3,4])
    kf = KFold(n_splits=4)
    runs = []
    predictors = []

    for i,(train_index, val_index) in enumerate(kf.split(person_indices)):
        logger.info("Train: {} Val {} ".format( person_indices[train_index],person_indices[val_index]) )
        predictors.append(one_predictor_model())
        opt = Adadelta(lr=0.05)
        predictors[i].compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
        history = train(predictors[i],person_indices[train_index],person_indices[val_index],view_type,contrast_type, i, name="{}_{}".format(contrast_type,view_type))
        runs.append(history.history)
        break

    with open(run_dir + 'cross_valid_stats{}_{}.lst'.format(view_type,contrast_type), 'wb') as fp:
            pickle.dump(runs, fp)
    plot_training(runs,view_type,contrast_type)

combined_model = n_predictors_combined_model(n=len(MR_modalities)*len(view_list))
layer_dict = dict([(layer.name, layer) for layer in combined_model.layers])
for i,contrast,view in enumerate(product(MR_modalities,view_list)):
    layer_dict["Seq_{}".format(i)].load_weights(run_dir + 'model_{}_{}.h5'.format(contrast,view), by_name=True)


