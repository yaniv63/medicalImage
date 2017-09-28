from multiprocessing import  JoinableQueue,Process
import numpy as np
from sklearn.metrics import classification_report,f1_score,confusion_matrix
from itertools import product


from paths import *
from logging_tools import get_logger

run_dir = get_run_dir()
logger = get_logger(run_dir)

from masks import get_combined_mask,load_wm_mask
from data_containers import load_contrasts,load_lables
from test_tools import model_pred,patch_image,get_segmentation
from plotting_tools import watch_predictions


def test_model(weight_path,person,time,is_unimodel,contrasts,views,view=None,use_stats=False):
    #load test
    test_images = load_contrasts(person, time, mri_contrasts)
    wm_mask = load_wm_mask(person, time)
    mask = get_combined_mask(wm_mask,test_images['FLAIR'])
    vol_shape = test_images[contrasts[0]].shape

    #predict test
    logger.info("predict images")
    BUF_SIZE = 50
    patch_q = JoinableQueue(BUF_SIZE)
    prediction_q =  JoinableQueue(BUF_SIZE)
    seg_q =  JoinableQueue(1)
    args = {}
    args['use_stats_model'] = use_stats
    patch_p = Process(target=patch_image, args=(test_images, mask, contrasts, views, vol_shape, patch_q),
                      name='patcher')
    if is_unimodel:
        args['name'] ='test_'+str(person)+'_' + view
        args['fold'] = 0
        model_p = Process(target=model_pred,args=(weight_path,patch_q,prediction_q,args,True),name='predictor')
    else:
        args['n']=len(views)
        args['test_person'] = person
        model_p = Process(target=model_pred,args=(weight_path,patch_q,prediction_q,args),name='predictor')


    seg_p = Process(target=get_segmentation,args=(vol_shape, prediction_q, seg_q,args),name='segmentor')
    process_list = [patch_p, model_p, seg_p]
    for i in process_list:
        i.daemon = True
        i.start()
    segmentation, prob_map, stats = seg_q.get()
    if use_stats:
        with open(run_dir + 'stats_{}.npy'.format(str(person)+'_'+str(time)), 'wb') as fp:
            np.save(fp, stats)
    for i in process_list:
        i.join()
    logger.info("finished prediction")
    model_name = str(person)+'_'+str(time)+'_'+view if is_unimodel else str(person)+'_'+str(time) + '_multimodel'
    with open(run_dir + 'segmantation_{}.npy'.format(model_name), 'wb') as fp, open(run_dir + 'prob_plot_{}.npy'.format(model_name), 'wb') as fp1:
        np.save(fp, segmentation)
        np.save(fp1, prob_map)

    # compare
    doctor_num = 1
    labels = load_lables(person, time, doctor_num)
    logger.info("{}".format(confusion_matrix(labels.flatten().tolist(),segmentation.flatten().tolist())))
    logger.info("\n"+classification_report(labels.flatten().tolist(),segmentation.flatten().tolist()))
    logger.info("f1 score is {}".format(f1_score(labels.flatten().tolist(),segmentation.flatten().tolist())))

    #plot predictiorn patches
    # if is_unimodel:
    #     watch_predictions(test_images[contrasts[0]],labels,segmentation,views[0],w=16)
test_data = {1:[(1,1),(1,2),(1,3),(1,4)],2:[(2,1),(2,2),(2,3),(2,4)],3:[(3,1),(3,2),(3,3),(3,4),(3,5)],4:[(4,1),(4,2),(4,3),(4,4)],5:[(5,1),(5,2),(5,3),(5,4)]}
test_person=1
test = test_data[test_person]
mri_contrasts = ['FLAIR', 'T2', 'MPRAGE', 'PD']
views =['axial', 'coronal', 'sagittal']
unimodel = [False,True]
weight_path ='/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/'#'/media/sf_shared/src/medicalImaging/runs/MOE runs/run3-return to inputs to gate/'# '/home/yaniv/Desktop/'

logger.info("checking unimodel coronal")
for person, time in  test:
    logger.info("person {} time {}".format(person,time))
    test_model(weight_path, person, time, True, mri_contrasts, [views[2]],view = views[2], use_stats=False)
    #test_model(weight_path, person, time, False, mri_contrasts,views, use_stats=False)
    # test_model(weight_path,person,time,False,mri_contrasts,views,use_stats=True)
    # for contrast,view in product(mri_contrasts,views):
    #     logger.info("checking individual model on person {} time {} contrast {} view {}".format(person,time,contrast,view))
    #     test_model(weight_path, person, time, True,[contrast],[view])
