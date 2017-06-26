import numpy as np
from paths import *
from prod.logging_tools import get_logger
import time as tm

run_dir = get_run_dir()
logger = get_logger(run_dir)

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

if __name__ == "__main__":
    from multiprocessing import JoinableQueue,Event,Process
    from augmentations import augmentation_worker
    from data_containers import load_all_data
    import time
    import numpy as np
    from train_tools import combined_generator
    logger.info("start")
    contrasts = ['FLAIR', 'T2', 'MPRAGE', 'PD']
    views = ['axial', 'coronal', 'sagittal']
    PersonTrainList = [(1,3),(1,2)]

    np.random.seed(42)

    data,positive_list, negative_list = load_all_data(PersonTrainList,contrasts)
    w=16
    max_size = 10000
    batch_q = JoinableQueue(max_size)
    pos_index_q = JoinableQueue(max_size)
    neg_index_q = JoinableQueue(max_size)
    index_q = JoinableQueue(max_size)
    pos_patch_q = JoinableQueue(max_size)
    neg_patch_q = JoinableQueue(max_size)
    batch_size = 128

    event = Event()
    aug_args = {'flip':True,'flip_p':0.5,'rescale':True,'rescale_lowbound':0.8,'rescale_highbound':1.2,'rotate':True,'rot_angle':5}
    collector_p = Process(target=collect_batch, args=(pos_patch_q, neg_patch_q, batch_q, batch_size, event,len(contrasts)*len(views)),
                          name='collect_batch')
    patch_p = Process(target=random_batch_indexes, args=(positive_list, negative_list, pos_index_q, neg_index_q, event),
                      name='random_batch_indexes')
    batch_p = batch_generator(batch_q,event)
    worker_pos_p = augmentation_worker(pos_index_q,pos_patch_q,data,contrasts,views,w,event,aug_args)
    worker_neg_p = augmentation_worker(neg_index_q, neg_patch_q, data, contrasts, views,w,event, aug_args)
    positive_p = Process(target=worker_pos_p.start_calc, name='positive worker')
    negative_p = Process(target=worker_neg_p.start_calc, name='negative worker')
    process_list = [collector_p,patch_p,positive_p,negative_p]#positive_p]#,negative_p]
    for i in process_list:
        i.daemon = True
        i.start()
    logger.info("start timing")
    for i in range(10):
        start = time.time()
        print "round {}".format(i)
        batch,labels = batch_p.next()
        end = time.time()
        print "round time {}".format(end-start)
    event.set()
    for i in process_list:
        i.terminate()

    raw_input("finished")
    from Queue import  Queue
    a = Queue()
    a.get()