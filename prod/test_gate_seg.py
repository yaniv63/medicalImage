import numpy as np
from collections import defaultdict
from keras.models import Model
from multi_predictors_combined import gating_model,n_experts_combined_model
from create_patches import can_extract_patch,extract_patch
from masks import get_combined_mask,load_wm_mask
from data_containers import load_contrasts,load_lables
from sklearn.metrics import classification_report
from paths import *
from logging_tools import get_logger
run_dir = get_run_dir()
logger = get_logger(run_dir)

test_indexes = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run6-train gate/test_data_test_1.npy'
weight_path ='/media/sf_shared/src/medicalImaging/runs/MOE runs/run6-train gate/gate_1_2.h5'
experts_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/'
mri_contrasts = ['FLAIR', 'T2', 'MPRAGE', 'PD']
views =['axial', 'coronal', 'sagittal']
person  =1
time =2

def patch_index_list(test_index_list, images, mask, contrasts, views, vol_shape, w=16):
    logger.info("start create patch process ")
    index_list = []
    patch_dict = defaultdict(list)
    samples = []
    for index in test_index_list:
        person,time,i,j,k = index
        if mask[i, j, k] and can_extract_patch(vol_shape, i, j, k, w):
            index_list.append((i, j, k))
            for view in views:
                patch_list = []
                for contrast in contrasts:
                    patch = extract_patch(images[contrast], view, (i, j, k), w)
                    patch_list.append(patch)
                patch_dict[view].append(patch_list)
    for v in views:
        sample = np.array(patch_dict[v])
        samples.append(sample)
    samples = np.concatenate(samples, axis=1)
    return index_list,samples

def model_pred(model, patches,has_stats):
    logger.info("start predict process")
    logger.info("loading model")
    predictions = model.predict(patches)
    if has_stats:
        predictions = zip(*predictions)
    return predictions

gate = gating_model(N_exp=3,N_mod = 4, img_rows=33, img_cols=33)
gate.load_weights(weight_path)
test_indexes_list = np.load(test_indexes)
test_indexes = [index for index,_ in test_indexes_list]
test_labels = [label for _, label in test_indexes_list]

test_images = load_contrasts(person, time, mri_contrasts)
wm_mask = load_wm_mask(person, time)
mask = get_combined_mask(wm_mask,test_images['FLAIR'])
vol_shape = test_images[mri_contrasts[0]].shape

index_list,patches = patch_index_list(test_indexes,test_images,mask,mri_contrasts,views,vol_shape)
predictions = model_pred(gate,patches,has_stats=False)
pred_hard = np.argmax(predictions,axis = 1)
logger.info("classification \n \n {}".format(classification_report(test_labels,pred_hard)))

moe = n_experts_combined_model(n=3,N_mod = 4, img_rows=33, img_cols=33)
moe.get_layer('Seq_0').load_weights(experts_path+'model_test_1_axial_fold_0.h5')
moe.get_layer('Seq_1').load_weights(experts_path+'model_test_1_coronal_fold_0.h5')
moe.get_layer('Seq_2').load_weights(experts_path+'model_test_1_sagittal_fold_0.h5')
#moe.get_layer('Seq_gate').load_weights(weight_path)
moe.load_weights(experts_path + 'combined_weights_1.h5')
moe_stats = Model(input=moe.input,
          output=[moe.output,moe.get_layer("Seq_gate").get_output_at(1), moe.get_layer("Seq_0").get_output_at(1),
                  moe.get_layer("Seq_1").get_output_at(1), moe.get_layer("Seq_2").get_output_at(1)])
patches = np.split(patches,3,axis=1)
predictions = model_pred(moe_stats,patches,has_stats=True)
labels = load_lables(person, time, doc_num=1)
gold_truth = []
for index in test_indexes:
    p,t,i,j,k = index
    label = labels[(i,j,k)]
    gold_truth.append(label)

pred_hard = [0 if pred[0] <0.5 else 1 for pred in predictions]
logger.info("classification \n \n {}".format(classification_report(gold_truth,pred_hard)))





