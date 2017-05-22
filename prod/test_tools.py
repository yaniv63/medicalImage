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
                for contrast, view in product(contrasts, views):
                    patch = extract_patch(images[contrast], view, (i,j,k), w)
                    patch_dict[contrast+'_'+view].append(patch)
        if len(index_list) > 0:
            for c, v in product(contrasts, views):
                sample = np.array(patch_dict[c+'_'+v])
                samples.append(np.expand_dims(sample,1))
            output_q.put((index_list, samples))
            logger.info("put layer {}".format(i))
    logger.info("finish create patch process ")
    output_q.put((-1,-1))


def model_pred(weight_dir,n_predictors,input_q,output_q):
    logger.info("start predict process")
    logger.info("loading model")
    model = load_model(weight_dir,n_predictors)
    logger.info("start predict with model")
    while True:
        indexes,patches =input_q.get()
        curr_layer = indexes[0][0]
        if curr_layer == -1:
            break
        predictions = model.predict(patches)
        output_q.put((indexes,predictions))
        logger.info("predicted layer {} ".format(curr_layer))
    logger.info("finish predict process")
    output_q.put((-1,-1))


def get_segmentation(vol_shape, input_q, output_queue, threshold=0.5):
    logger.info("start segmentation process")
    prob_plot = np.zeros(vol_shape, dtype='float16')
    segmentation = np.zeros(vol_shape, dtype='uint8')
    while True:
        indexes,pred = input_q.get()
        curr_layer = indexes[0][0]
        if curr_layer == -1:
            break
        for index, (i, j, k,) in enumerate(indexes):
            if pred[index] > threshold:
                segmentation[i, j, k] = 1
            prob_plot[i, j, k] = pred[index]
        logger.info("segmented layer {}".format(curr_layer))
    logger.info("finish segmentation process")
    output_queue.put((segmentation,prob_plot))

def load_model(weight_dir,n_predictors):
    from multi_predictors_combined import n_predictors_combined_model
    from keras.optimizers import Adadelta

    optimizer = Adadelta(lr=0.05)
    combined_model = n_predictors_combined_model(n=n_predictors)
    combined_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    combined_model.load_weights(weight_dir + 'combined_weights.h5')
    return combined_model
