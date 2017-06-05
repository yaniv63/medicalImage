import numpy as np
import itertools
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
import logging

from create_patches import extract_axial
from paths import get_run_dir

run_dir = get_run_dir()
logger = logging.getLogger('root')


def generic_plot(kwargs):

    if kwargs.has_key("figure_name"):
        f1 = plt.figure(kwargs["figure_name"])
    if kwargs.has_key("title"):
        plt.title(kwargs["title"])
    if kwargs.has_key("ylabel"):
        plt.ylabel(kwargs["ylabel"])
    if kwargs.has_key("xlabel"):
        plt.xlabel(kwargs["xlabel"])
    if kwargs.has_key("line_att"):
        line_attribute = kwargs["line_att"]
    else:
        line_attribute = ''
    if kwargs.has_key("image_att"):
        image_attribute = kwargs["image_att"]
    else:
        image_attribute = {}
    if kwargs.has_key("x"):
        plt.plot(kwargs["x"],kwargs["y"],**line_attribute)
    elif  kwargs.has_key("y"):
        plt.plot(kwargs["y"],**line_attribute)
    elif kwargs.has_key("image"):
        plt.imshow(kwargs["image"],**image_attribute)
    if kwargs.has_key("legend"):
        plt.legend(kwargs["legend"], loc=0)
    if kwargs.has_key("save_file"):
        plt.savefig(kwargs["save_file"],dpi=100)

def plot_training(logs,name):
    metrics = ['acc', 'val_acc', 'loss', 'val_loss', 'fmeasure', 'val_fmeasure']
    linestyles = ['-', '--']
    colors = ['b','y','r','g']
    for j,history in enumerate(logs):
        for i in [0,2,4]:
            params = {'figure_name': metrics[i]+name, 'y':history[metrics[i]],'title':'model_{} '.format(name) + metrics[i],
                      'ylabel':metrics[i],'xlabel':'epoch',"line_att":dict(linestyle=linestyles[0],color=colors[j])}
            generic_plot(params)
            params = {'figure_name': metrics[i]+name, 'y':history[metrics[i+1]],"line_att":dict(linestyle=linestyles[1],color=colors[j])}
            generic_plot(params)
    for i in [0, 2, 4]:
        params = {'figure_name': metrics[i]+name, 'legend': ['train', 'validation']*len(logs),
                  'save_file': run_dir + 'model_{}'.format(name) + metrics[i] + '.png'}
        generic_plot(params)

def probability_plot(model, vol,fold,threshold=0.5,slice = 95):

    prob_plot = np.zeros(vol.shape)
    final_decision = np.zeros(vol.shape)

    x = np.linspace(0, vol.shape[2] - 1, vol.shape[2], dtype='int')
    y = np.linspace(0, vol.shape[1] - 1, vol.shape[1], dtype='int')
    z = np.linspace(0, vol.shape[0] - 1, vol.shape[0], dtype='int')
    voxel_list = itertools.product(y,x)
    patches_list = []
    logger.info("patches for model")
    for j, k in voxel_list:
        axial_p = extract_axial(vol, k, j, slice,16)
        if type(axial_p) == np.ndarray:
            patches_list.append((slice,j,k,axial_p))

    patches = [v[3] for v in patches_list]

    patches = np.expand_dims(patches, 1)
    logger.info("predict model")

    predictions = model.predict(patches)
    for index,(slice, j, k,_) in enumerate(patches_list):
        if predictions[index] > threshold:
            final_decision[slice, j, k] = 255
        prob_plot[slice, j, k] = predictions[index] * 255

    params = {'figure_name': "slice_prob_{}".format(fold),"image" : prob_plot[slice, :, :],
              "image_att": dict(cmap=matplotlib.cm.gray),'save_file': run_dir +'slice_prob_{}'.format(fold) + '.png'}
    generic_plot(params)
    params["image"] = final_decision[slice, :, :]
    params["save_file"] = run_dir +'slice_decision_{}'.format(fold) + '.png'
    generic_plot(params)