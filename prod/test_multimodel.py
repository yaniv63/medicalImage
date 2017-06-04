from multiprocessing import  JoinableQueue,Process
import numpy as np
from sklearn.metrics import classification_report,f1_score

from paths import *
from logging_tools import get_logger

run_dir = get_run_dir()
logger = get_logger(run_dir)

from masks import get_combined_mask,load_wm_mask
from data_containers import load_contrasts,load_lables
from test_tools import model_pred,patch_image,get_segmentation

contrasts = ['FLAIR']#, 'T2', 'MPRAGE', 'PD']
views =['sagittal']#, 'coronal', 'sagittal']
test_person = 1
test_time = 4
test_unimodel =True

weight_path = '/media/sf_shared/src/medicalImaging/tmp/tm2/train_adadelta/multimodel/run9-3pred,train_on_5,200_epoch_300_epoch_combined/'
#load test
test_images = load_contrasts(test_person, test_time, contrasts)
wm_mask = load_wm_mask(test_person,test_time)
mask = get_combined_mask(wm_mask,test_images['FLAIR'])
vol_shape = test_images[contrasts[0]].shape

#predict test
logger.info("predict images")
BUF_SIZE = 50
patch_q = JoinableQueue(BUF_SIZE)
prediction_q =  JoinableQueue(BUF_SIZE)
seg_q =  JoinableQueue(1)
patch_p = Process(target=patch_image, args=(test_images, mask,contrasts, views, vol_shape, patch_q),name='patcher')
args = {}
if test_unimodel:
    args['contrast'] = contrasts[0]
    args['view'] = views[0]
    args['fold'] = 0
    model_p = Process(target=model_pred,args=(weight_path,patch_q,prediction_q,args,True),name='predictor')
else:
    args['n']=len(contrasts)*len(views)
    model_p = Process(target=model_pred,args=(weight_path,patch_q,prediction_q,args),name='predictor')
seg_p = Process(target=get_segmentation,args=(vol_shape, prediction_q, seg_q),name='segmentor')
process_list = [patch_p, model_p, seg_p]
for i in process_list:
    i.daemon = True
    i.start()
segmentation, prob_map = seg_q.get()
for i in process_list:
    i.join()
logger.info("finished prediction")

with open(run_dir + 'segmantation.npy', 'wb') as fp, open(run_dir + 'prob_plot.npy', 'wb') as fp1:
    np.save(fp, segmentation)
    np.save(fp1, prob_map)

# compare
doctor_num = 1
labels = load_lables(test_person,test_time,doctor_num)
logger.info("\n"+classification_report(labels.flatten().tolist(),segmentation.flatten().tolist()))
logger.info("f1 score is {}".format(f1_score(labels.flatten().tolist(),segmentation.flatten().tolist())))