import numpy as np
import nibabel as nb
from itertools import product
from sklearn.metrics import f1_score,classification_report,confusion_matrix

from paths import *
from logging_tools import get_logger

run_dir = get_run_dir()
logger = get_logger(run_dir)

init_path = './runs/test4/'
prob_plot = []
person = 1

mri_contrasts = ['FLAIR', 'T2', 'MPRAGE', 'PD']
views =['axial', 'coronal', 'sagittal']
test_data =[(4,1),(4,2),(4,3),(4,4)]#[(4,1),(4,2),(4,3),(4,4)]#[(3,2),(3,3),(3,4),(3,5)]#[(4,1),(4,2),(4,3),(4,4)]#[(3,2),(3,3),(3,4),(3,5)]#[(1,2),(1,3),(1,4)]#[(2,2),(2,3),(2,4)] #[(1,2),(1,3),(1,4)]#,(2,2),(2,3),(2,4),(3,2),(3,3),(3,4),(3,5),(4,1),(4,2),(4,3),(4,4),(5,2),(5,3),(5,4)]
unimodel = [False,True]


def load_lables(person,time,doc_num):
    path = Src_Path + Labels_Path + "training0{}_0{}_mask{}.nii".format(person,time,doc_num)
    labels = nb.load(path).get_data()
    labels = labels.T
    labels = np.rot90(labels, 2, axes=(1, 2))
    return labels
for person, time in  test_data:
    prob_plots = []
    logger.info("check avarage person {}  time {}".format(person,time))
    for contrast,view in product(mri_contrasts,views):
        prob_plots.append(np.load(init_path+'prob_plot_{}_{}_{}_{}.npy'.format(person,time,contrast,view)))
    av = np.array(prob_plots).mean(axis=0)
    pred = (av > 0.5)*1
    labels = load_lables(person,time,1)
    logger.info("{}".format(confusion_matrix(labels.flatten().tolist(), pred.flatten().tolist())))
    logger.info("\n" + classification_report(labels.flatten().tolist(), pred.flatten().tolist()))
    logger.info("f1 score is {}".format(f1_score(labels.flatten().tolist(), pred.flatten().tolist())))

