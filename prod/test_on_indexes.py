import numpy as np
from keras.models import Model
from collections import defaultdict
from multi_predictors_combined import n_experts_combined_model_gate_parameters,n_parameters_combined_model,one_predictor_model
from create_patches import can_extract_patch,extract_patch
from masks import get_combined_mask,load_wm_mask
from data_containers import load_contrasts,load_lables,separate_classes_indexes,load_index_list,load_all_images,create_ROI_list
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
from paths import *
from logging_tools import get_logger
run_dir = get_run_dir()
logger = get_logger(run_dir)

mri_contrasts = ['FLAIR', 'T2', 'MPRAGE', 'PD']
views =['axial', 'coronal', 'sagittal']
# person  =1
# time =2


weight_path ='/media/sf_shared/src/medicalImaging/runs/MOE runs/run13- multilabel moe/lr_0.0001/model_test_1_fold_0.h5'



def patch_index_list(test_index_list, images, contrasts, views, vol_shape, w=16):
    logger.info("start create patch process ")
    index_list = []
    patch_dict = defaultdict(list)
    samples = []
    for index in test_index_list:
        person,time,i,j,k = index
        if can_extract_patch(vol_shape, i, j, k, w):
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
    #samples = np.concatenate(samples, axis=1)
    return index_list,samples

def model_pred(model, patches,has_stats):
    logger.info("start predict process")
    logger.info("loading model")
    predictions = model.predict(patches)
    if has_stats:
        predictions = zip(*predictions)
    return predictions


def get_model():
    # model = n_experts_combined_model_gate_parameters(n=3,N_mod=4)
    # model.load_weights(weight_path)
    # out_model = Model(input=model.input, output=model.outputs[0])
    weight_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/'
    model = one_predictor_model(N_mod=4,index=0)
    model.load_weights(weight_path + 'model_test_1_axial_fold_0.h5')
    return model

model = get_model()
test_list = [(1,x) for x in range(1,5)]
for person,time in test_list:
    logger.info("test person {} time {}".format(person,time))

    pos_list, neg_list = create_ROI_list([(person,time)])
    test_indexes = pos_list.tolist() + neg_list.tolist()
    test_images = load_contrasts(person, time, mri_contrasts)
    wm_mask = load_wm_mask(person, time)
    mask = get_combined_mask(wm_mask,test_images['FLAIR'])
    vol_shape = test_images[mri_contrasts[0]].shape

    index_list,patches = patch_index_list(test_indexes,test_images,mri_contrasts,[views[0]],vol_shape)
    predictions = model_pred(model,patches,has_stats=False)
    pred_hard = [0 if pred[0] <0.5 else 1 for pred in predictions]

    labels = load_lables(person, time, doc_num=1)
    gold_truth = []
    for index in test_indexes:
        p,t,i,j,k = index
        if can_extract_patch(vol_shape,i,j,k,16):
            label = labels[(i,j,k)]
            gold_truth.append(label)

    logger.info("classification \n \n {}".format(classification_report(gold_truth,pred_hard)))
    logger.info("accuracy \n \n {}".format(accuracy_score(gold_truth,pred_hard)))
    logger.info("confusion matrix  \n \n {}".format(confusion_matrix(gold_truth,pred_hard)))
    logger.info("fmeasure  \n \n {}".format(f1_score(gold_truth,pred_hard)))