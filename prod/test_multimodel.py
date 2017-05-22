from multiprocessing import  Queue,Process
import numpy as np

from paths import *
from logging_tools import get_logger

run_dir = get_run_dir()
logger = get_logger(run_dir)

from masks import get_combined_mask,load_wm_mask
from data_containers import load_contrasts
from test_tools import model_pred,patch_image,get_segmentation

contrasts = ['FLAIR']#, 'T2', 'MPRAGE', 'PD']
views = ['axial', 'coronal', 'sagittal']
test_person = 5
test_time = 1


#load test
test_images = load_contrasts(test_person, test_time, contrasts)
wm_mask = load_wm_mask(test_person,test_time)
mask = get_combined_mask(wm_mask,test_images['FLAIR'])
vol_shape = test_images[contrasts[0]].shape

#predict test
logger.info("predict images")
BUF_SIZE = 50
patch_q = Queue(BUF_SIZE)
prediction_q =  Queue(BUF_SIZE)
seg_q =  Queue(1)
patch_p = Process(target=patch_image, args=(test_images, mask,contrasts, views, vol_shape, patch_q))
model_p = Process(target=model_pred,args=(weight_path,len(contrasts)*len(views),patch_q,prediction_q))
seg_p = Process(target=get_segmentation,args=(vol_shape, prediction_q, seg_q))
process_list = [patch_p, model_p, seg_p]
for i in process_list:
    i.start()
for i in process_list:
    i.join()
segmentation, prob_map = seg_q.get()

with open(run_dir + 'segmantation.npy', 'wb') as fp, open(run_dir + 'prob_plot.npy', 'wb') as fp1:
    np.save(fp, segmentation)
    np.save(fp1, prob_map)

