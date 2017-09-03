import numpy as np
from itertools import chain
from keras.utils.np_utils import to_categorical
from multiprocessing import JoinableQueue, Event, Process
from Queue import Empty
from augmentations2 import AugmentationWorker2
from paths import *
import logging

run_dir = get_run_dir()
logger =logging.getLogger('root')

def batch_generator(batch_queue,event,concat=True):
    logger.debug("start batch generator")
    while not event.is_set():
        batch, labels = batch_queue.get()
        if batch == -1:
            batch_queue.task_done()
            break
        labels = to_categorical(labels)
        if concat:
            batch = np.concatenate(batch,axis=1)
        yield (batch,labels)
    batch_queue.close()
    logger.info("finish batch generator")

def random_batch_indexes(class_indexes_set,batch_size):
    classes_num = len(class_indexes_set)
    size = batch_size / classes_num
    mod_size = batch_size % classes_num
    index_lists =[ [] for _ in range(len(class_indexes_set))]
    batch = []
    labels = []
    for i,class_index in enumerate(class_indexes_set):
        index_lists[i] =  np.random.permutation(class_index).tolist()
    for index in range(size):
        for j in range(classes_num):
            batch.append(index_lists[j][index])
            labels.append(j)
    for i in range(mod_size):  # complete batch size
        batch.append(index_lists[i][size])
        labels.append(i)
    return batch,labels


def batch_all_indexes(class_indexes_set,classes_indexes_queues):
    logger.debug("start batch_all_indexes")

    for i,index_list in enumerate(class_indexes_set):
        for index in index_list:
            classes_indexes_queues[i].put(index)
    for q in classes_indexes_queues:
        q.close()
    logger.info("finish batch_all_indexes")

def collect_all_patches(patches_queues,output_queue,size):
    patches_list = []
    labels_list = []
    while len(patches_list) <size :
        try:
             for i,q in enumerate(patches_queues):
                if q.qsize() > 0:
                    patches = q.get(True,5)
                    patches = np.concatenate(patches,axis=0)
                    patches_list.append(patches)
                    labels_list.append(i)
        except Empty as e:
            logger.info("was empty")
    #labels_list = to_categorical(labels_list)
    #patches_list = np.array(patches_list)
    output_queue.put((patches_list,labels_list))
    output_queue.close()
    logger.info("finish collect_all_patches")




def collect_batch(batch_list,predictor_num,patcher):
    patches = []
    for index in batch_list:
       patch = patcher.worker_augmentation(index)
       patches.append(patch)
    batch = [[] for i in range(predictor_num)]
    for sample in patches:
       for num,patch in enumerate(sample):
            batch[num].append(patch)
    #batch = [np.array(x) for x in batch]
    batch = np.concatenate(batch, axis=1)
    return  batch


class TrainGeneratorMultiClass(object):

    def __initialize_proccesses(self):
        max_size = 10000
        batch_q = JoinableQueue(max_size)
        index_queues = [JoinableQueue(max_size) for _ in range(self.classes_num)]
        patches_queues = [JoinableQueue(max_size) for _ in range(self.classes_num)]
        self.queues = [batch_q, index_queues,patches_queues]

        # create proccesses
        if self.random_batch:
            index_p = Process(target=random_batch_indexes,
                              args=(self.classes_index_lists, index_queues, self.event),
                              name='random_batch_indexes')
        else:
            index_p = Process(target=batch_all_indexes,
                              args=(self.classes_index_lists, index_queues),
                              name='batch_all_indexes')
        collector_p = Process(target=collect_batch,
                              args=(patches_queues,
                                    batch_q, self.batch_size,
                                    self.event,len(self.views)),
                              name='collect_batch')
        self.proccesses.append(index_p)
        self.proccesses.append(collector_p)
        for i in range(self.workers_num):
            for j in range(self.classes_num):
                worker = AugmentationWorker(index_queues[j], patches_queues[j], self.data, self.contrasts, self.views, self.w, self.event,
                                                 self.aug_args)
                worker_p = Process(target=worker.start_calc, name='class_{}_worker{}'.format(j,i))
                self.workers.append(worker)
                self.proccesses.append(worker_p)

    def __init__(self, data, classes_index_lists, contrasts, views, batch_size,w,workers_num=1, aug_args=None,random_batches=True,concat_batch=True):
        self.data = data
        self.classes_index_lists = classes_index_lists
        self.classes_num = len(classes_index_lists)
        self.contrasts = contrasts
        self.views = views
        self.proccesses = []
        self.queues = []
        self.workers = []
        self.batch_size = batch_size
        self.w = w
        self.event = Event()
        self.workers_num = workers_num
        self.random_batch = random_batches
        self.concat_batch = concat_batch

        if aug_args is None:
            self.aug_args = {'flip': True, 'flip_p': 0.5, 'rescale': True, 'rescale_lowbound': 0.8,
                             'rescale_highbound': 1.2, 'rotate': True, 'rot_angle': 5}
        self.__initialize_proccesses()

    def get_generator(self):
        batch_p = batch_generator(self.queues[0], self.event,self.concat_batch)
        for i in self.proccesses:
            i.daemon = True
            i.start()
        return batch_p

    def close(self):
        self.event.set()
        for i in self.workers:
            i.finish()
        for i in self.proccesses:
            i.terminate()
        logger.info("finished")


class TrainGeneratorMultiClassAggregator(object):

    def __initialize_proccesses(self):
        max_size = 10000
        size = 1434
        set_q = JoinableQueue(max_size)
        index_queues = [JoinableQueue(max_size) for _ in range(self.classes_num)]
        patches_queues = [JoinableQueue(max_size) for _ in range(self.classes_num)]
        self.queues = [set_q, index_queues, patches_queues]

        # create proccesses
        index_p = Process(target=batch_all_indexes,
                              args=(self.classes_index_lists, index_queues),
                              name='batch_all_indexes')

        collector_p = Process(target=collect_all_patches,
                              args=(patches_queues,set_q,size),
                              name='collect_all_patches')
        self.proccesses.append(index_p)
        self.proccesses.append(collector_p)
        for i in range(self.workers_num):
            for j in range(self.classes_num):
                worker = AugmentationWorker(index_queues[j], patches_queues[j], self.data, self.contrasts, self.views,
                                            self.w, self.event,
                                            self.aug_args)
                worker_p = Process(target=worker.start_calc, name='class_{}_worker{}'.format(j, i))
                self.workers.append(worker)
                self.proccesses.append(worker_p)

    def __init__(self, data, classes_index_lists, contrasts, views, batch_size,w,workers_num=1, aug_args=None,random_batches=True,concat_batch=True):
        self.data = data
        self.classes_index_lists = classes_index_lists
        self.classes_num = len(classes_index_lists)
        self.contrasts = contrasts
        self.views = views
        self.proccesses = []
        self.queues = []
        self.workers = []
        self.batch_size = batch_size
        self.w = w
        self.event = Event()
        self.workers_num = workers_num
        self.random_batch = random_batches
        self.concat_batch = concat_batch

        if aug_args is None:
            self.aug_args = {'flip': True, 'flip_p': 0.5, 'rescale': True, 'rescale_lowbound': 0.8,
                             'rescale_highbound': 1.2, 'rotate': True, 'rot_angle': 5}
        self.__initialize_proccesses()

    def get_generator(self):
        batch_p = batch_generator(self.queues[0], self.event,self.concat_batch)
        for i in self.proccesses:
            i.daemon = True
            i.start()
        return batch_p

    def close(self):
        self.event.set()
        for i in self.workers:
            i.finish()
        for i in self.proccesses:
            i.terminate()
        logger.info("finished")




def generator_gate(classes_index_set,data, contrasts, views,batch_size, w,predictor_num):
    aug_args = {'flip': True, 'flip_p': 0.5, 'rescale': True, 'rescale_lowbound': 0.8,
                     'rescale_highbound': 1.2, 'rotate': True, 'rot_angle': 5}
    patcher = AugmentationWorker2(data, contrasts, views, w,aug_args)
    while True:
        batch_indexes,labels = random_batch_indexes(classes_index_set,batch_size)
        batch_patches = collect_batch(batch_indexes,predictor_num,patcher)
        labels = to_categorical(labels)
        yield (batch_patches,labels)

if __name__ == "__main__":

    from data_containers import load_all_images,separate_classes_indexes
    import time
    import numpy as np
    from logging_tools import get_logger
    import sys
    run_dir = get_run_dir()
    logger = get_logger(run_dir)

    logger.info("start")


    def my_handler(type, value, tb):
        logger.exception("Uncaught exception: {0}".format(str(value)))


    # Install exception handler
    sys.excepthook = my_handler


    contrasts = ['FLAIR', 'T2', 'MPRAGE', 'PD']
    views = ['axial', 'coronal', 'sagittal']
    PersonTrainList = [(1,2)]
    index_path = '/media/sf_shared/src/medicalImaging/stats/test1_gate_indexes.npy'
    np.random.seed(42)

    data = load_all_images(PersonTrainList,contrasts)
    classes_indexes = np.load(index_path)
    indexes_per_class = separate_classes_indexes(classes_indexes,3)
    w=16
    batch_size = 16
    gen = generator_gate(indexes_per_class,data,  contrasts, views, batch_size, w,3)
    for i in range(3000):
        start = time.time()
        print "round {}".format(i)
        batch, labels = gen.next()
        end = time.time()
        print "round time {}".format(end - start)
    gen.close()