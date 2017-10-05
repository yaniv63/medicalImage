# from keras.preprocessing.image import random_rotation
import numpy as np
import cv2
from collections import defaultdict
from itertools import product
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import logging
from multiprocessing import Pool
logger = logging.getLogger('root')
import time as tm

def plt_image_patch(image,patch,r1=None,r2=None):
    print r1, r2
    import matplotlib.pylab as plt
    import matplotlib
    plt.figure();
    plt.imshow(image, cmap=matplotlib.cm.gray)
    plt.figure();
    plt.imshow(patch, cmap=matplotlib.cm.gray)
    plt.show()

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def flip_patch(patch,p,randlr,randud):
    if randud < p:
        patch = flip_axis(patch,0)
    if randlr < p:
        patch = flip_axis(patch,1)
    return patch

def rescale(image,roi_mask,factor,SE):
    shape = image.shape
    out_shape = tuple(int(x * factor) for x in shape)
    if factor < 1:
        roi_mask = binary_dilation(roi_mask,SE).astype(roi_mask.dtype)
    # perform the actual resizing of the image
    interpolation = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    resized_image = cv2.resize(image, out_shape[::-1], interpolation=interpolation)
    resized_mask = cv2.resize(roi_mask, out_shape[::-1], interpolation=cv2.INTER_NEAREST)
    return resized_image,resized_mask


def rotate(image,roi_mask,rot_angle):
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot_angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    roi_mask = cv2.warpAffine(roi_mask, M, (cols, rows))
    return image,roi_mask

def crop_patch(image,r1,r2,w):
    x = np.arange(r1 - w, r1 + w +1, 1)
    y = np.arange(r2 - w, r2 + w +1, 1)
    indexes = np.ix_(x,y) #choose patch indices left for rows,right for columns
    try:
        patch = image[indexes]
    except IndexError as e:
        # a = np.pad(image, ((0, 0), (0, y[-1] - (image.shape[1] - 1))), 'edge')
        # patch = crop_patch(a, r1, r2, w)
        index = 1 if '1' in e.args[0].split() else 0
        if index == 1:
            a = np.pad(image, ((0, 0), (0, y[-1] - (image.shape[1] - 1))), 'edge')
        else:
            a = np.pad(image, ((0, x[-1] - (image.shape[0] - 1)), (0, 0)), 'edge')
        patch = crop_patch(a, r1, r2, w)
    return patch

def get_image(volume,view,i,j,k):
    if view == 'axial':
       image = volume[i,:,:]
       roi_mask = np.zeros(shape=image.shape)
       roi_mask[j,k] = 1
    elif view == 'coronal':
        image = volume[:,j,:]
        roi_mask = np.zeros(shape=image.shape)
        roi_mask[i,k] = 1
    elif view == 'sagittal':
        image = volume[:,:,k]
        roi_mask = np.zeros(shape=image.shape)
        roi_mask[i,j] = 1
    return image,roi_mask

class AugmentationWorker(object):

    def __init__(self,input_queue,output_queue,data,contrasts,views,w,event,aug_args):
        self.__input_queue = input_queue
        self.__output_queue = output_queue
        self.__data = data
        self.__contrasts = contrasts
        self.__views = views
        self.__w = w
        self.__event =event
        self.__binary_element = generate_binary_structure(2, 2)
        if aug_args['flip']:
            self.__flip = True
            self.__flip_chance = aug_args['flip_p']
        else:
            self.__flip = False
        if aug_args['rescale']:
            self.__rescale = True
            self.__lowbound,self.__highbound = aug_args['rescale_lowbound'],aug_args['rescale_highbound']
        else:
            self.__rescale = False

        if aug_args['rotate']:
            self.__rotate = True
            self.__angle = aug_args['rot_angle']
        else:
            self.__rotate = False
        self.__pool = Pool(1)

    def start_calc(self):
        self.__pool.apply_async(self.worker_augmentation())

    def finish(self):
        self.__pool.terminate()

   
    def worker_augmentation(self):
        while not self.__event.is_set():
            person, time, i, j, k = self.__input_queue.get()
            #print ("index {},{},{}".format(i,j,k))
            patch_dict = defaultdict(list)
            samples = []
            image_dict = []
            for contrast in self.__contrasts:
                volume = self.__data[person][time][contrast]
                if self.__rescale:
                    factor = np.random.uniform(self.__lowbound, self.__highbound)
                if self.__rotate:
                    rot_angle = np.random.randint(-self.__angle, self.__angle)
                if self.__flip:
                    randlr = np.random.random()
                    randud = np.random.random()
                for view in self.__views:
                    #volume = self.__data[person][time][contrast]
                    image,roi_mask = get_image(volume,view,i,j,k)
                    if self.__rescale:
                        image,roi_mask = rescale(image,roi_mask,factor,self.__binary_element)
                    if self.__rotate:
                        image, roi_mask = rotate(image,roi_mask,rot_angle)
                    try:
                        r1,r2 = np.transpose(np.nonzero(roi_mask))[0] #find roi in image after transformations
                    except Exception as e:
                        logger.error("error")
                    patch = crop_patch(image,r1,r2,self.__w)
                    if self.__flip:
                        patch = flip_patch(patch,self.__flip_chance,randlr,randud)
                    patch_dict[contrast].append(patch)
                    # image_dict.append(image)
                    # plt_image_patch(image, patch, r1, r2)
            for x in  self.__contrasts:
                sample = patch_dict[x]
                samples.append(sample)
            self.__input_queue.task_done()
            self.__output_queue.put(samples)
        self.__output_queue.close()

if __name__ == "__main__":
    from data_containers import load_image
    import matplotlib.pylab as plt
    import matplotlib

    # def plot_image_mask(image,mask):
    #     masked = np.ma.masked_where(mask == 0, mask)
    #     plt.figure();plt.imshow(image, cmap=matplotlib.cm.gray)
    #     plt.imshow(masked, 'jet', alpha=.7)
    #
    # def test_rescale(test_image,test_mask):
    #     plot_image_mask(test_image, test_mask)
    #     image, mask = rescale(test_image, test_mask, 1.0, 1.0)
    #     plot_image_mask(image, mask)
    #     image, mask = rescale(test_image, test_mask, 0.8, 0.8)
    #     plot_image_mask(image, mask)
    #     image, mask = rescale(test_image, test_mask, 1.2, 1.2)
    #     plot_image_mask(image, mask)
    #     image, mask = rescale(test_image, test_mask, 0.8, 1.2)
    #     plot_image_mask(image, mask)
    #     plt.show()

    # person = 1
    # time = 2
    # w=16
    # contrast  = 'FLAIR'
    # view = 'sagittal'
    # data = load_image(person,time,contrast)
    # z = 100;y=146;x=72
    # image,roi_mask = get_image(data,view,z,y,x)
    # #test_rescale(image,roi_mask)
    # patch = crop_patch(image,y,z,w)
    # plot_image_mask(image,roi_mask)
    # plt.figure();plt.imshow(patch, cmap=matplotlib.cm.gray)
    # plt.show()
    # raw_input("Press Enter to continue...")
    from multiprocessing import JoinableQueue,Event,Process
    from data_containers import load_all_data

    input_q = JoinableQueue(10)
    output_q = JoinableQueue(10)
    contrasts = ['FLAIR']#, 'T2', 'MPRAGE', 'PD']
    views = ['axial', 'coronal', 'sagittal']
    PersonTrainList = [(1, 2)]
    w=16
    index = [1,2,19,84,96]#[1,2,81,60,126]
    #np.random.seed(42)
    input_q.put(index)

    data, positive_list, negative_list = load_all_data(PersonTrainList, contrasts)
    aug_args = {'flip': True, 'flip_p': 0.5, 'rescale': True, 'rescale_lowbound': 0.8,
                'rescale_highbound': 1.2, 'rotate': True, 'rot_angle': 5}
    event = Event()
    worker = AugmentationWorker(input_q,output_q,data,contrasts,views,w,event,aug_args)
    pos_p = Process(target=worker.start_calc, name='positive worker')
    pos_p.daemon = True
    pos_p.start()
    patches = output_q.get()
    for i,contrast in enumerate(patches):
        for j,angle in enumerate(contrast):
            plt.figure()
            plt.imshow(angle,cmap=matplotlib.cm.gray)
            plt.title("angle is {} contrast {}".format(views[j],contrasts[i]))
    plt.show()
    raw_input("wait to see")
    worker.finish()
    pos_p.terminate()