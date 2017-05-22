from paths import *
from logging_tools import get_logger

run_dir = get_run_dir()
logger = get_logger(run_dir)

from keras.optimizers import Adadelta
from itertools import  product

from multi_predictors_combined import n_predictors_combined_model
from masks import get_combined_mask,load_wm_mask
from data_containers import load_all_images

MR_modalities = ['FLAIR']#, 'T2']#, 'MPRAGE', 'PD']
view_list = ['axial', 'coronal', 'sagittal']
optimizer = Adadelta(lr=0.05)
test_person = 5
test_time = 1

#load model
combined_model = n_predictors_combined_model(n=len(MR_modalities)*len(view_list))
combined_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
layer_dict = dict([(layer.name, layer) for layer in combined_model.layers])

for i,(contrast,view) in enumerate(product(MR_modalities,view_list)):
    layer_dict["Seq_{}".format(i)].load_weights(run_dir + 'model_{}_{}_{}.h5'.format(contrast,view,0), by_name=True)

#load test
test_images = load_all_images([test_person],[test_time],MR_modalities)
wm_mask = load_wm_mask(test_person,test_time)
mask = get_combined_mask(wm_mask,test_images[test_person][test_time]['FLAIR'])
