import numpy as np
from multiprocessing import JoinableQueue, Event, Process
from augmentations import AugmentationWorker
from paths import *
import logging

run_dir = get_run_dir()
logger =logging.getLogger('root')

def batch_generator(batch_queue,event):
    logger.debug("start batch generator")
    while not event.is_set():
        batch, labels = batch_queue.get()
        if batch == -1:
            batch_queue.task_done()
            break
            batch_queue.task_done()
        yield (batch,labels)
    batch_queue.close()
    logger.info("finish batch generator")

def random_batch_indexes(positive_list,negative_list,pos_queue,neg_queue,event):
    logger.debug("start random_batch_indexes")
    while not event.is_set():
        positive_list_np = np.random.permutation(positive_list).tolist()
        negative_list_np = np.random.permutation(negative_list).tolist()
        for index in range(len(positive_list_np)):
            pos_queue.put(positive_list_np[index])
            neg_queue.put(negative_list_np[index])
    pos_queue.close()
    neg_queue.close()
    logger.info("finish random_batch_indexes")

def collect_batch(pos_queue,neg_queue,batch_queue,batch_size,event,predictor_num):
    logger.debug("start collect_batch")
    size = batch_size/2
    while not event.is_set():
        batch_pos = []
        batch_neg = []
        while len(batch_pos) < size or len(batch_neg) < size:
            if len(batch_pos) < size:
                batch_pos.append((pos_queue.get(),1))
                pos_queue.task_done()
            if len(batch_neg) < size:
                batch_neg.append((neg_queue.get(),0))
                neg_queue.task_done()
        mix_batch = np.random.permutation(batch_pos + batch_neg)
        samples = [patches for patches, _ in mix_batch]
        batch = [[] for i in range(predictor_num)]
        for sample in samples:
            for num,patch in enumerate(sample):
                batch[num].append(patch)
        batch = [np.array(x) for x in batch]
        labels = [labels for _, labels in mix_batch]
        batch_queue.put((batch,labels))
    batch_queue.close()
    logger.info("finish collect_batch")


class TrainGenerator(object):

    def __initialize_proccesses(self):
        max_size = 100
        batch_q = JoinableQueue(max_size)
        pos_index_q = JoinableQueue(max_size)
        neg_index_q = JoinableQueue(max_size)
        index_q = JoinableQueue(max_size)
        pos_patch_q = JoinableQueue(max_size)
        neg_patch_q = JoinableQueue(max_size)
        self.queues = [batch_q, pos_index_q, neg_index_q, index_q, pos_patch_q, neg_patch_q]

        # create proccesses
        patch_p = Process(target=random_batch_indexes,
                          args=(self.positive_list, self.negative_list, pos_index_q, neg_index_q, self.event),
                          name='random_batch_indexes')
        collector_p = Process(target=collect_batch, args=(
        pos_patch_q, neg_patch_q, batch_q, self.batch_size, self.event,  len(self.views)),
                              name='collect_batch')
        self.proccesses.append(patch_p)
        self.proccesses.append(collector_p)
        for i in range(16):
            pos_worker = AugmentationWorker(pos_index_q, pos_patch_q, self.data, self.contrasts, self.views, self.w, self.event,
                                             self.aug_args)
            neg_worker = AugmentationWorker(neg_index_q, neg_patch_q, self.data, self.contrasts, self.views, self.w, self.event,
                                             self.aug_args)
            pos_p = Process(target=pos_worker.start_calc, name='positive worker{}'.format(i))
            neg_p = Process(target=neg_worker.start_calc, name='negative worker{}'.format(i))
            self.workers.append(pos_worker)
            self.workers.append(neg_worker)
            self.proccesses.append(pos_p)
            self.proccesses.append(neg_p)

    def __init__(self, data, positive_list, negative_list, contrasts, views, batch_size,w, aug_args=None):
        self.data = data
        self.positive_list = positive_list
        self.negative_list = negative_list
        self.contrasts = contrasts
        self.views = views
        self.proccesses = []
        self.queues = []
        self.workers = []
        self.batch_size = batch_size
        self.w = w
        self.event = Event()

        if aug_args is None:
            self.aug_args = {'flip': True, 'flip_p': 0.5, 'rescale': True, 'rescale_lowbound': 0.8,
                             'rescale_highbound': 1.2, 'rotate': True, 'rot_angle': 5}
        self.__initialize_proccesses()

    def get_generator(self):
        batch_p = batch_generator(self.queues[0], self.event)
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


if __name__ == "__main__":

    from data_containers import load_all_data
    from train_tools import combined_generator
    import time
    import numpy as np
    logger.info("start")
    contrasts = ['FLAIR', 'T2', 'MPRAGE', 'PD']
    views = ['axial', 'coronal', 'sagittal']
    PersonTrainList = [(1,3),(1,2)]

    np.random.seed(42)

    data,positive_list, negative_list = load_all_data(PersonTrainList,contrasts)
    w=16
    batch_size = 16
    gen = TrainGenerator(data,positive_list,negative_list,contrasts,views,batch_size,w)
    gen2 = gen.get_generator()
    old_gen = combined_generator(positive_list,negative_list,data,contrasts,views,batch_size=128)
    for i in range(10):
        start = time.time()
        print "round {}".format(i)
        batch, labels = gen2.next()
        end = time.time()
        print "round time {}".format(end - start)
    gen.close()
