import numpy as np
import logging
from itertools import product
from collections import defaultdict

from create_patches import can_extract_patch,extract_patch

logger = logging.getLogger('root')

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
        if len(index_list) > 0:
            for v in views:
                sample = np.array(patch_dict[v])
                samples.append(sample)
            output_q.put((index_list, samples))
    output_q.put((None,None))
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
    logger.info("start predict with model")
    while True:
        indexes,patches =input_q.get()
        if indexes == None:
            input_q.task_done()
            break
        curr_layer = indexes[0][0]
        predictions = model.predict(patches)
        stats = zip(*predictions)
        output_q.put((indexes,stats))
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
        indexes, stats = batch
        st = [stat[1:] for stat in stats]
        stats_list.extend(zip(indexes,st))
        preds = [stat[0] for stat in stats]
        indexes, pred = batch
        for index, (i, j, k,) in enumerate(indexes):
            if preds[index] > threshold:
                segmentation[i, j, k] = 1
            prob_plot[i, j, k] = preds[index]
        input_q.task_done()
    output_queue.put((segmentation,prob_plot,stats_list))
    output_queue.close()
    logger.info("finish segmentation process")


def load_unimodel(weight_dir,args):
    from multi_predictors_combined import one_predictor_model
    from keras.optimizers import SGD
    optimizer = SGD(lr=0.01, nesterov=True)

    model = one_predictor_model(N_mod=4, img_rows=33, img_cols=33)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    model.load_weights(weight_dir + 'model_{}_fold_{}.h5'.format(args['name'],args['fold']))
    return model


def load_model(weight_dir,args):
    from multi_predictors_combined import n_predictors_combined_model,n_parameters_combined_model,n_experts_combined_model,n_experts_combined_model_gate_parameters
    from keras.optimizers import SGD
    from keras.models import Model

    optimizer = SGD(lr=0.01,nesterov=True)
    combined_model = n_experts_combined_model(n=args['n'])
    # combined_model.get_layer('Seq_0').load_weights(weight_dir + 'model_test_1_axial_fold_0.h5')
    # combined_model.get_layer('Seq_1').load_weights(weight_dir + 'model_test_1_coronal_fold_0.h5')
    # combined_model.get_layer('Seq_2').load_weights(weight_dir + 'model_test_1_sagittal_fold_0.h5')
    # combined_model.get_layer('Seq_gate').load_weights(weight_dir + 'gate_1_2.h5')
    combined_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    combined_model.load_weights(weight_dir + 'combined_weights_1.h5')
    combined_model_stats = Model(input=combined_model.input,
                      output=[combined_model.output, combined_model.get_layer("Seq_gate").get_output_at(1),
                              combined_model.get_layer("Seq_0").get_output_at(1),
                              combined_model.get_layer("Seq_1").get_output_at(1), combined_model.get_layer("Seq_2").get_output_at(1)])
    combined_model_stats.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    return combined_model_stats


def test_coefficients_model(w_path,args):
    from multi_predictors_combined import n_experts_combined_model_gate_parameters,n_experts_combined_model
    from keras.models import Model
    #model = n_experts_combined_model_gate_parameters(n=3, N_mod=4, img_rows=33, img_cols=33)  # create the original model
    model = n_experts_combined_model(n=3, N_mod=4, img_rows=33, img_cols=33)
    model.load_weights(w_path+ 'combined_weights_{}.h5'.format(args['test_person']),by_name=True)
    layer_name1 = 'out_gate'
    # intermediate_layer_model = Model(input=model.input,
    #                                  output=[model.get_layer(layer_name1).get_output_at(0),
    #                                          model.get_layer('out0').output,
    #                                          model.get_layer('out1').output,
    #                                          model.get_layer('out2').output,
    #                                          model.get_layer("perception_0").output,
    #                                          model.get_layer("perception_1").output,
    #                                          model.get_layer("perception_2").output])
    intermediate_layer_model = Model(input=model.input,
          output=[model.get_layer("Seq_gate").get_output_at(1), model.get_layer("Seq_0").get_output_at(1),
                  model.get_layer("Seq_1").get_output_at(1), model.get_layer("Seq_2").get_output_at(1)])

    return intermediate_layer_model



