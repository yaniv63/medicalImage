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

from keras.optimizers import Adadelta
import pickle
from  itertools import product

from prod.multi_predictors_combined import one_predictor_model,n_predictors_combined_model
from train_tools import create_callbacks,generator,combined_generator,aggregate_genrated_samples\
    , calc_epoch_size
from data_containers import load_data,load_all_data
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

def train_combined(model,PersonTrainList,PersonValList,contrast_list,view_list,name,batch_size=256):

    callbacks = create_callbacks(name, fold=0)
    logger.debug("creating train & val generators")
    train_images,positive_list, negative_list = load_all_data(PersonTrainList,range(1,5),contrast_list)
    train_generator = combined_generator(positive_list, negative_list, train_images,contrast_list,view_list)
    logger.info("after tr")
    val_images, pos_val_list, neg_val_list = load_all_data(PersonValList,range(1,5),contrast_list)
    #val_set = combined_aggregate_genrated_samples(pos_val_list, neg_val_list, val_images, contrast_list,view_list)
    val_generator = combined_generator(pos_val_list, neg_val_list, val_images,contrast_list,view_list)
    logger.info("after val")
    logger.info("training individual model")
    epoch_size = calc_epoch_size(positive_list, batch_size)
    val_size = calc_epoch_size(pos_val_list, batch_size)
    history = model.fit_generator(train_generator, samples_per_epoch=epoch_size, nb_epoch=1, callbacks=callbacks,
                                  validation_data=val_generator,nb_val_samples=val_size)
    # confusion_mat = calc_confusion_mat(model, val_set[0], val_set[1], "individual val {}".format(0))
    # calc_dice(confusion_mat, "individual val {}".format(0))
    return history


# ######## train model
logger.debug("start script")
MR_modalities = ['FLAIR']#, 'T2']#, 'MPRAGE', 'PD']
view_list = ['axial', 'coronal']#, 'sagittal']
image_types = product(MR_modalities,view_list)
optimizer = Adadelta(lr=0.05)

for contrast_type,view_type in image_types:
    # logger.info("training {} {} model".format(contrast_type,view_type))
    person_indices =np.array([1,2,3,4])
    # kf = KFold(n_splits=4)
    runs = []
    predictors = []
    train_index = [0,1,2];val_index = [3]

    #for i,(train_index, val_index) in enumerate(kf.split(person_indices)):
    logger.info("Train: {} Val {} ".format( person_indices[train_index],person_indices[val_index]) )
    predictors.append(one_predictor_model())
    predictors[0].compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    history = train(predictors[0],person_indices[train_index],person_indices[val_index],view_type,contrast_type, 0, name="{}_{}".format(contrast_type,view_type))
    runs.append(history.history)


    with open(run_dir + 'cross_valid_stats{}_{}.lst'.format(view_type,contrast_type), 'wb') as fp:
            pickle.dump(runs, fp)
    plot_training(runs,view_type,contrast_type)


combined_model = n_predictors_combined_model(n=len(MR_modalities)*len(view_list))
combined_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
layer_dict = dict([(layer.name, layer) for layer in combined_model.layers])

for i,(contrast,view) in enumerate(product(MR_modalities,view_list)):
    layer_dict["Seq_{}".format(i)].load_weights(run_dir + 'model_{}_{}_{}.h5'.format(contrast,view,0), by_name=True)

history = train_combined(combined_model, [1, 2, 4], [3], MR_modalities, view_list, "combined")
with open(run_dir + 'cross_valid_stats_multimodel.lst', 'wb') as fp:
    pickle.dump(history, fp)
