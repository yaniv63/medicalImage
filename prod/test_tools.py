import numpy as np
import logging
import itertools

from create_patches import extract_axial

logger = logging.getLogger('root')

def post_process(seg, thresh):
    from scipy import ndimage
    connected_comp = ndimage.generate_binary_structure(3, 2) * 1
    label_weight = 30
    connected_comp[1, 1, 1] = label_weight
    res = ndimage.convolve(seg, connected_comp, mode='constant', cval=0.)
    return (res > (thresh + label_weight)) * 1




def predict_image(model, vol,mask, threshold=0.5):

    prob_plot = np.zeros(vol.shape, dtype='float16')
    segmentation = np.zeros(vol.shape, dtype='uint8')

    x = np.linspace(0, vol.shape[2] - 1, vol.shape[2], dtype='int')
    y = np.linspace(0, vol.shape[1] - 1, vol.shape[1], dtype='int')
    z = np.linspace(0, vol.shape[0] - 1, vol.shape[0], dtype='int')
    logger.info("patches for model")
    for i in z:
        patches_list = []
        voxel_list = itertools.product(y, x)
        for j, k in voxel_list:
            if mask[i][j][k] :
                axial_p = extract_axial(vol, k, j, i, 16)
                if type(axial_p) == np.ndarray:
                    patches_list.append((i, j, k, axial_p))
        if len(patches_list) > 0:
            patches = [v[3] for v in patches_list]

            patches = np.expand_dims(patches, 1)

            predictions = model.predict(patches)
            for index, (i, j, k, _) in enumerate(patches_list):
                if predictions[index] > threshold:
                    segmentation[i, j, k] = 1
                prob_plot[i, j, k] = predictions[index]

    return segmentation ,prob_plot