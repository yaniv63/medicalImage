import numpy as np
from collections import defaultdict
from keras.models import Model
from multi_predictors_combined import gating_model,n_experts_combined_model,n_experts_combined_model_gate_parameters
from create_patches import can_extract_patch,extract_patch
from masks import get_combined_mask,load_wm_mask
from data_containers import load_contrasts,load_lables,separate_classes_indexes,load_index_list,load_all_images
from train_tools import combined_aggregate_genrated_samples_multiclass
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from paths import *
from logging_tools import get_logger
run_dir = get_run_dir()
logger = get_logger(run_dir)

test_indexes = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run6-train gate/test_data_test_1.npy'
weight_path ='/media/sf_shared/src/medicalImaging/runs/MOE runs/run6-train gate/gate_1_2.h5'
experts_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/'
w_path_gate = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run9-pretrain gate parameters/'
mri_contrasts = ['FLAIR', 'T2', 'MPRAGE', 'PD']
views =['axial', 'coronal', 'sagittal']
person  =1
time =2
check_on_gate = False

class_label_method = 'expert_labels_soft'

if class_label_method =='expert_labels_hard':
    indexes_path = 'gate vectors - hard decision/'
else:
    indexes_path = 'gate vectors - soft decision - exponent=8/'


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
    samples = np.concatenate(samples, axis=1)
    return index_list,samples

def model_pred(model, patches,has_stats):
    logger.info("start predict process")
    logger.info("loading model")
    predictions = model.predict(patches)
    if has_stats:
        predictions = zip(*predictions)
    return predictions



person_time_list = [[(2, x) for x in range(1, 5)], [(3, x) for x in range(1, 6)], [(4, x) for x in range(1, 5)],
                    [(5,x) for x in range(1,5)], [(1,x) for x in range(1,5)]]
for i,dlist in enumerate(person_time_list):
    train_d = [item for sublist in dlist for item in sublist]
    person = dlist[0][0]
    indexes_list = load_index_list(indexes_path + "gate_indexes_{}".format(class_label_method)+"_person", dlist)
    indexes_per_class = separate_classes_indexes(indexes_list, 2)
    images_set = load_all_images(dlist, mri_contrasts)

    patches,test_labels,experts = combined_aggregate_genrated_samples_multiclass(images_set, indexes_per_class, mri_contrasts, views,
                                                                                 batch_size=16, w=16, aug_args=None)


    moe = n_experts_combined_model_gate_parameters(n=3,N_mod = 4, img_rows=33, img_cols=33)
    moe.get_layer('Seq_0').load_weights(experts_path+'model_test_1_axial_fold_0.h5',by_name=True)
    moe.load_weights(experts_path+'model_test_1_axial_fold_0.h5',by_name=True)

    moe.get_layer('Seq_1').load_weights(experts_path+'model_test_1_coronal_fold_0.h5',by_name=True)
    moe.load_weights(experts_path+'model_test_1_coronal_fold_0.h5',by_name=True)

    moe.get_layer('Seq_2').load_weights(experts_path+'model_test_1_sagittal_fold_0.h5',by_name=True)
    moe.load_weights(experts_path+'model_test_1_sagittal_fold_0.h5',by_name=True)

    #moe.load_weights(w_path_gate + 'gate_parameters_test1.h5',by_name=True)
    #moe.get_layer('Seq_gate').load_weights(weight_path)
    #moe.load_weights(experts_path + 'combined_weights_1.h5')
    moe_stats = Model(input=moe.input,
              output=[moe.output,moe.get_layer("out_gate").output,
                      # moe.get_layer("Seq_0").get_output_at(1),
                      # moe.get_layer("Seq_1").get_output_at(1),
                      # moe.get_layer("Seq_2").get_output_at(1),
                      moe.get_layer("perception_0").output,
                      moe.get_layer("perception_1").output,
                      moe.get_layer("perception_2").output
                      ])
    patches = np.split(patches,3,axis=1)
    predictions = model_pred(moe_stats,patches,has_stats=True)


    train_set = []
    for stat,label,expert in zip(predictions,test_labels,experts):
        sample = ((stat[2],stat[3],stat[4]),label,expert)
        train_set.append(sample)

    with open('gate_parameters_samples_test1_set_{}.npy'.format(person), 'wb') as fp:
        np.save(fp,np.array(train_set))

if check_on_gate == True:
    gate = gating_model(N_exp=3,N_mod = 4, img_rows=33, img_cols=33)
    gate.load_weights(weight_path)
    test_indexes_list = np.load(test_indexes)
    test_indexes = [index for index,_ in test_indexes_list]
    test_labels = [label for _, label in test_indexes_list]
    test_images = load_contrasts(person, time, mri_contrasts)
    wm_mask = load_wm_mask(person, time)
    mask = get_combined_mask(wm_mask,test_images['FLAIR'])
    vol_shape = test_images[mri_contrasts[0]].shape

    index_list,patches = patch_index_list(test_indexes,test_images,mri_contrasts,views,vol_shape)
    predictions = model_pred(gate,patches,has_stats=False)
    pred_hard = np.argmax(predictions,axis = 1)
    test_labels_num = np.argmax(test_labels,axis = 1)
    logger.info("classification \n \n {}".format(classification_report(test_labels_num,pred_hard)))
    logger.info("accuracy \n \n {}".format(accuracy_score(test_labels_num,pred_hard)))
    logger.info("confusion matrix  \n \n {}".format(confusion_matrix(test_labels_num,pred_hard)))

    # labels = load_lables(person, time, doc_num=1)
    # gold_truth = []
    # for index in test_indexes:
    #     p,t,i,j,k = index
    #     label = labels[(i,j,k)]
    #     gold_truth.append(label)
    #
    # pred_hard = [0 if pred[0] <0.5 else 1 for pred in predictions]
    # logger.info("classification \n \n {}".format(classification_report(gold_truth,pred_hard)))