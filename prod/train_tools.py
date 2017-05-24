import numpy as np
from keras.callbacks import  LambdaCallback, ModelCheckpoint,ReduceLROnPlateau
import logging
from random import shuffle
from itertools import product
from collections import defaultdict


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

def combined_generator(positive_list,negative_list,data,contrasts,views,batch_size=256,w = 16,only_once=False):
    half_batch,batch_num = calc_batch_params(positive_list,batch_size)
    while True:
        positive_list_np = np.random.permutation(positive_list).tolist()
        positive_list_np = positive_list_np[:batch_num * half_batch]
        negative_list_np = np.random.permutation(negative_list).tolist()
        for batch in range(batch_num):
            samples = []
            positive_batch = [(x,1) for x in positive_list_np[batch*half_batch:(batch+1)*half_batch]]
            negative_batch = [(x,0) for x in negative_list_np[batch * half_batch:(batch + 1) * half_batch]]
            mix_batch = positive_batch+negative_batch
            mix_batch = np.random.permutation(mix_batch).tolist()
            patch_dict = defaultdict(list)
            labels = []
            for (person,time,i,j,k),label in mix_batch:
                labels.append(label)
                for contrast in contrasts:
                    volume = data[person][time][contrast]
                    for view in views:
                        patch_dict[contrast+'_'+view].append(extract_patch(volume,view,(i,j,k),w))
            for x, y in product(contrasts, views):
                sample = np.array(patch_dict[x+'_'+y])
                samples.append(np.expand_dims(sample,1))
            yield (samples,labels)

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


def combined_aggregate_genrated_samples(pos_list,neg_list,data,contrast_list,view_list):
    samples = []
    labels = []
    for batch_samples,batch_labels in combined_generator(pos_list,neg_list,data,contrast_list,view_list,only_once=True):
        if len(samples) == 0:
            samples = batch_samples
        else:
            samples = [np.concatenate((samples[i], batch_samples[i])) for i in range(len(batch_samples))]
        labels.extend(batch_labels)
    return samples,labels

def create_callbacks(name,fold):
    save_weights = ModelCheckpoint(filepath=run_dir + 'model_{}_fold_{}.h5'.format(name, fold), monitor='val_fmeasure',
                                   save_best_only=True,
                                   save_weights_only=True)
    print_logs = LambdaCallback(on_epoch_end=lambda epoch, logs:
    logger.debug("epoch {} loss {:.5f} acc {:.5f} fmeasure {:.5f} val_loss {:.5f} val_acc {:.5f} val_fmeasure{:.5f} ".
                 format(epoch, logs['loss'], logs['acc'], logs['fmeasure'], logs['val_loss'], logs['val_acc'],
                        logs['val_fmeasure'])))
    lr_reduce = ReduceLROnPlateau(verbose=1)
    mycallbacks = [print_logs,save_weights,lr_reduce]
    return mycallbacks