import numpy as np
import logging
from itertools import product
from collections import defaultdict

from create_patches import can_extract_patch,extract_patch

logger = logging.getLogger('root')
experts_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/'
w_path_gate = '/media/sf_shared/src/medicalImaging/results/'
parameters_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run4-parameters equal to moe/'

def post_process(seg, thresh):
    from scipy import ndimage
    connected_comp = ndimage.generate_binary_structure(3, 2) * 1
    label_weight = 30
    connected_comp[1, 1, 1] = label_weight
    res = ndimage.convolve(seg, connected_comp, mode='constant', cval=0.)
    return (res > (thresh + label_weight)) * 1

def patch_image(images, mask,contrasts, views,  vol_shape, output_q, w=16):
    x = np.linspace(0, vol_shape[2] - 1, vol_shape[2], dtype='int')
    y = np.linspace(0, vol_shape[1] - 1, vol_shape[1], dtype='int')
    z = np.linspace(0, vol_shape[0] - 1, vol_shape[0], dtype='int')
    logger.info("start create patch process ")
    logger.info("patches for model")
    for i in z:
        index_list = []
        samples = []
        patch_dict = defaultdict(list)
        voxel_list = product(y, x)
        for j, k in voxel_list:
            if mask[i,j,k] and can_extract_patch(vol_shape, i, j, k, w):
                index_list.append((i, j, k))
                for view in views:
                    patch_list = []
                    for contrast in contrasts:
                        patch = extract_patch(images[contrast], view, (i, j, k), w)
                        patch_list.append(patch)
                    patch_dict[view].append(patch_list)
                # for contrast, view in product(contrasts, views):
                #     patch = extract_patch(images[contrast], view, (i,j,k), w)
                #     patch_dict[contrast+'_'+view].append(patch)
        if len(index_list) > 0:
            for v in views:
                sample = np.array(patch_dict[v])
                samples.append(sample)
            output_q.put((index_list, samples))
    output_q.put((None,None))
    output_q.close()
    logger.info("finish create patch process ")

def patch_index_list(test_index_list, images, mask, contrasts, views, vol_shape, output_q, w=16):
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
    output_q.put((index_list, samples))
    output_q.put((None, None))
    output_q.close()
    logger.info("finish create patch process ")

def model_pred(weight_dir, input_q, output_q, args, unimodel=False):
    logger.info("start predict process")
    logger.info("loading model")
    use_stats_model = args['use_stats_model']
    if unimodel:
        model = load_unimodel(weight_dir,args)
    else:
        model = load_model(weight_dir,args)
    if use_stats_model:
        stats_model = test_coefficients_model(weight_dir,args)
    logger.info("start predict with model")
    while True:
        indexes,patches =input_q.get()
        if indexes == None:
            input_q.task_done()
            break
        curr_layer = indexes[0][0]
        predictions = model.predict(patches)
        if not use_stats_model:
            output_q.put((indexes,predictions))
        else:
            stats = stats_model.predict(patches)
            stats = zip(*stats)#zip(stats[0],stats[1],stats[2],stats[3])#,stats[4],stats[5],stats[6])
            output_q.put((indexes,predictions,stats))
        input_q.task_done()
    output_q.put((indexes, patches))
    output_q.close()
    logger.info("finish predict process")


def get_segmentation(vol_shape, input_q, output_queue, args,threshold=0.5):
    logger.info("start segmentation process")
    prob_plot = np.zeros(vol_shape, dtype='float16')
    segmentation = np.zeros(vol_shape, dtype='uint8')
    stats_list = []
    use_stats = args['use_stats_model']
    while True:
        batch= input_q.get()
        if batch[0] == None:
            input_q.task_done()
            break
        if use_stats:
            indexes, pred,stats = batch
            stats_list.extend(zip(indexes,stats))
        else:
            indexes, pred = batch

        curr_layer = indexes[0][0]
        for index, (i, j, k,) in enumerate(indexes):
            if pred[index] > threshold:
                segmentation[i, j, k] = 1
            prob_plot[i, j, k] = pred[index]

        input_q.task_done()
    output_queue.put((segmentation,prob_plot,stats_list))
    output_queue.close()
    logger.info("finish segmentation process")


def load_unimodel(weight_dir,args):
    from multi_predictors_combined import one_predictor_model
    from keras.optimizers import SGD
    optimizer = SGD(lr=0.01, nesterov=True)

    model = one_predictor_model(N_mod=4, img_rows=33, img_cols=33,index=2)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    #model.load_weights('/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/model_test_1_coronal_fold_0.h5')
    model.load_weights(weight_dir + 'model_{}_fold_{}.h5'.format(args['name'],args['fold']),by_name=True)
    model.load_weights(weight_dir + 'sagittal_expert.h5',by_name=True)

    return model


def load_model(weight_dir,args):
    from multi_predictors_combined import n_predictors_combined_model,n_parameters_combined_model,n_experts_combined_model,n_experts_combined_model_gate_parameters,n_experts_combined_model_gate_parameters_test
    from keras.optimizers import SGD

    optimizer = SGD(lr=0.01,nesterov=True)
    #combined_model = n_predictors_combined_model(n=args['n'])
    #combined_model = n_parameters_combined_model(n=args['n'], N_mod=4, img_rows=33, img_cols=33)
    #combined_model = n_experts_combined_model(n=args['n'])
    #combined_model = n_experts_combined_model_gate_parameters(n=args['n'], N_mod=4, img_rows=33, img_cols=33)

    #combined_model = n_experts_combined_model_gate_parameters(n=args['n'],N_mod=4)
    combined_model = n_experts_combined_model_gate_parameters(n=args['n'],N_mod=4)
    #combined_model.load_weights(weight_dir + 'coronal_expert.h5', by_name=True)

    # combined_model.get_layer('Seq_0').load_weights(experts_path + 'model_test_1_axial_fold_0.h5', by_name=True)
    # #combined_model.load_weights(experts_path + 'model_test_1_axial_fold_0.h5', by_name=True)
    #
    # combined_model.get_layer('Seq_1').load_weights(experts_path + 'model_test_1_coronal_fold_0.h5', by_name=True)
    # #combined_model.load_weights(experts_path + 'model_test_1_coronal_fold_0.h5', by_name=True)
    #
    # combined_model.get_layer('Seq_2').load_weights(experts_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)
    #combined_model.load_weights(experts_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)
    layer_dict = dict([(layer.name, layer) for layer in combined_model.layers])
    for i in range(3):
        expert_layers_dict = dict([(layer.name, layer) for layer in layer_dict['Seq_{}'.format(i)].layers])
        for layer in expert_layers_dict.values():
            if 'dense2' not in layer.name:
                layer.trainable = False
    # for layer in layer_dict.values():
    #     if 'gate' in layer.name:
    #         layer.trainable = False
    optimizer = SGD(lr=0.01, nesterov=True)
    combined_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    #combined_model.load_weights(w_path_gate + 'gate_batching_hard.h5', by_name=True)
    combined_model.load_weights(weight_dir + 'model_test_1_fold_0.h5',by_name=True)

   # combined_model.load_weights(parameters_path +'combined_weights_{}.h5'.format(args['test_person']))

    combined_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])


    # combined_model.get_layer('Seq_0').load_weights(weight_dir + 'model_test_1_axial_fold_0.h5', by_name=True)
    # combined_model.get_layer('Seq_1').load_weights(weight_dir + 'model_test_1_coronal_fold_0.h5', by_name=True)
    # combined_model.get_layer('Seq_2').load_weights(weight_dir + 'model_test_1_sagittal_fold_0.h5', by_name=True)
    return combined_model


def test_coefficients_model(w_path,args):
    from multi_predictors_combined import n_experts_combined_model_gate_parameters,n_experts_combined_model,n_experts_combined_model_gate_parameters_test
    from keras.models import Model
    #model = n_experts_combined_model_gate_parameters(n=3, N_mod=4, img_rows=33, img_cols=33)  # create the original model
    #model = n_experts_combined_model(n=3, N_mod=4, img_rows=33, img_cols=33)
    #model = n_experts_combined_model_gate_parameters()
    model=n_experts_combined_model_gate_parameters()
    #model.load_weights(w_path+ 'model_test_1_fold_0.h5'.format(args['test_person']),by_name=True)
    # model.load_weights(weight_dir + 'model_{}_fold_{}.h5'.format(args['name'], args['fold']), by_name=True)
    # model.load_weights(weight_dir + 'axial_expert.h5', by_name=True)

    # model.get_layer('Seq_0').load_weights(experts_path + 'model_test_1_axial_fold_0.h5', by_name=True)
    # #model.load_weights(experts_path + 'model_test_1_axial_fold_0.h5', by_name=True)
    #
    # model.get_layer('Seq_1').load_weights(experts_path + 'model_test_1_coronal_fold_0.h5', by_name=True)
    # #model.load_weights(experts_path + 'model_test_1_coronal_fold_0.h5', by_name=True)
    #
    # model.get_layer('Seq_2').load_weights(experts_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)
    # #model.load_weights(experts_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)
    #
    # model.load_weights(w_path_gate + 'gate_batching_hard.h5', by_name=True)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    for i in range(3):
        expert_layers_dict = dict([(layer.name, layer) for layer in layer_dict['Seq_{}'.format(i)].layers])
        for layer in expert_layers_dict.values():
            if 'dense' not in layer.name:
                layer.trainable = False
    for layer in layer_dict.values():
        if 'gate' in layer.name:
            layer.trainable = False
    model.load_weights(w_path+ 'model_test_1_fold_0.h5'.format(args['test_person']),by_name=True)


    layer_name1 = 'out_gate'
    intermediate_layer_model = Model(input=model.input,
                                     output=[model.output,
                                             model.get_layer(layer_name1).get_output_at(0),
                                             model.get_layer('out0').output,
                                             model.get_layer('out1').output,
                                             model.get_layer('out2').output,
                                             model.get_layer("perception_0").output,
                                             model.get_layer("perception_1").output,
                                             model.get_layer("perception_2").output])
    # intermediate_layer_model = Model(input=model.input,
    #       output=[model.get_layer("Seq_gate").get_output_at(1), model.get_layer("Seq_0").get_output_at(1),
    #               model.get_layer("Seq_1").get_output_at(1), model.get_layer("Seq_2").get_output_at(1)])

    return intermediate_layer_model



