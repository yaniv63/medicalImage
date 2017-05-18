import numpy as np
from keras.callbacks import  LambdaCallback, ModelCheckpoint
import logging

from paths import get_run_dir
from create_patches import extract_patch

logger = logging.getLogger('root')
run_dir =get_run_dir()


def generator(positive_list,negative_list,data,view,batch_size=256,patch_width = 16,only_once=False):
    half_batch,batch_num = calc_batch_params(positive_list,batch_size)
    while True:
        #modify list to divide by batch_size
        positive_list_np = np.random.permutation(positive_list)
        positive_list_np = positive_list_np[:batch_num*half_batch]
        negative_list_np = np.random.permutation(negative_list)
        for batch in range(batch_num):
            positive_batch = positive_list_np[batch*half_batch:(batch+1)*half_batch]
            positive_batch_patches = [[extract_patch(data[person][time],view,(i,j,k),patch_width),1] for person,time,i,j,k in positive_batch]
            negative_batch = negative_list_np[batch * half_batch:(batch + 1) * half_batch]
            negative_batch_patches = [[extract_patch(data[person][time],view,(i,j,k),patch_width),0] for person, time, i, j, k in
                                      negative_batch]
            final_batch = np.random.permutation(positive_batch_patches + negative_batch_patches)
            samples =  [patches for patches,_ in final_batch]
            samples = np.expand_dims(samples, 1)

            labels = [labels for _,labels in final_batch]
            yield (samples,labels)
        if only_once:
            break

def combined_generator(positive_list,negative_list,data,batch_size=256,patch_width = 16,only_once=False):
    half_batch,batch_num = calc_batch_params(positive_list,batch_size)
    while True:


def calc_batch_params(patch_list,batch_size):
    half_batch = batch_size / 2
    batch_num = len(patch_list) / half_batch
    return half_batch,batch_num


def calc_epoch_size(patch_list,batch_size):
    half_batch,batch_num = calc_batch_params(patch_list,batch_size)
    return batch_num * half_batch*2

def aggregate_genrated_samples(pos_list,neg_list,data,view):
    samples = []
    labels = []
    for batch_samples,batch_labels in generator(pos_list,neg_list,data,view,only_once=True):
        samples.extend(batch_samples)
        labels.extend(batch_labels)

    return (np.array(samples), np.array(labels))


def create_callbacks(name,fold):
    save_weights = ModelCheckpoint(filepath=run_dir + 'model_{}_fold_{}.h5'.format(name, fold), monitor='val_fmeasure',
                                   save_best_only=True,
                                   save_weights_only=True)
    print_logs = LambdaCallback(on_epoch_end=lambda epoch, logs:
    logger.debug("epoch {} loss {:.5f} acc {:.5f} fmeasure {:.5f} val_loss {:.5f} val_acc {:.5f} val_fmeasure{:.5f} ".
                 format(epoch, logs['loss'], logs['acc'], logs['fmeasure'], logs['val_loss'], logs['val_acc'],
                        logs['val_fmeasure'])))
    mycallbacks = [print_logs,save_weights]
    return mycallbacks