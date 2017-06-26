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


def plt_image_patch(image,patch,r1,r2):
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

def flip_patch(patch,p):
    if np.random.random() < p:
        patch = flip_axis(patch,0)
    if np.random.random() < p:
        patch = flip_axis(patch,1)
    return patch

def rescale(image,roi_mask,low,high,SE):
    shape = image.shape
    factor = np.random.uniform(low,high)
    out_shape = tuple(int(x * factor) for x in shape)
    if factor < 1:
        roi_mask = binary_dilation(roi_mask,SE).astype(roi_mask.dtype)
    # perform the actual resizing of the image
    interpolation = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    resized_image = cv2.resize(image, out_shape[::-1], interpolation=interpolation)
    resized_mask = cv2.resize(roi_mask, out_shape[::-1], interpolation=cv2.INTER_NEAREST)
    return resized_image,resized_mask


def rotate(image,roi_mask,angle):
    rows, cols = image.shape
    rot_angle = np.random.randint(-angle,angle)
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
    except Exception as e:
        raise e
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

class augmentation_worker(object):

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
        self.__pool = Pool(5)

    def start_calc(self):
        self.__pool.apply_async(self.worker_augmentation())

    def finish(self):
        self.__pool.terminate()

    def worker_augmentation(self):
        last_cur =0
        while True:#not self.__event.is_set():
            start = tm.time()
            person, time, i, j, k = self.__input_queue.get()
            #cur = tm.time();print cur-last_cur;last_cur=cur
            #end = tm.time();print "queue get {}".format(end-start)
            patch_dict = defaultdict(list)
            samples = []
            for contrast in self.__contrasts:
                # start = tm.time()
                volume = self.__data[person][time][contrast]
                # end = tm.time();print "get data {}".format(end-start)
                for view in self.__views:
                    # start = tm.time()
                    image,roi_mask = get_image(volume,view,i,j,k)
                    # end = tm.time();print "get image {}".format(end-start)
                    if self.__rescale:
                        # start = tm.time()
                        image,roi_mask = rescale(image,roi_mask,self.__lowbound,self.__highbound,self.__binary_element)
                        # end = tm.time();print "rescall {}".format(end-start)
                    if self.__rotate:
                        # start = tm.time()
                        image, roi_mask = rotate(image,roi_mask,self.__angle)
                        # end = tm.time();print "rotate {}".format(end-start)
                    try:
                        # start = tm.time()
                        r1,r2 = np.transpose(np.nonzero(roi_mask))[0] #find roi in image after transformations
                        # end = tm.time(); print"transpose {}".format(end-start)
                    except Exception as e:
                        logger.error("error")
                    # start = tm.time()
                    patch = crop_patch(image,r1,r2,self.__w)
                    # end = tm.time();print "crop {}".format(end-start)
                    if self.__flip:
                        # start = tm.time()
                        patch = flip_patch(patch,self.__flip_chance)
                        # end = tm.time();print "flip {}".format(end-start)
                    patch_dict[contrast + '_' + view].append(patch)
                    #plt_image_patch(image, patch, r1, r2)
            #start = tm.time()
            for x, y in product(self.__contrasts, self.__views):
                sample = patch_dict[x + '_' + y]
                samples.append(sample)
            self.__input_queue.task_done()
            self.__output_queue.put(samples)
            end = tm.time();print "queue out {}".format(end-start)
        self.__output_queue.close()


if __name__ == "__main__":
    from data_containers import load_image
    import matplotlib.pylab as plt
    import matplotlib

    def plot_image_mask(image,mask):
        masked = np.ma.masked_where(mask == 0, mask)
        plt.figure();plt.imshow(image, cmap=matplotlib.cm.gray)
        plt.imshow(masked, 'jet', alpha=.7)

    def test_rescale(test_image,test_mask):
        plot_image_mask(test_image, test_mask)
        image, mask = rescale(test_image, test_mask, 1.0, 1.0)
        plot_image_mask(image, mask)
        image, mask = rescale(test_image, test_mask, 0.8, 0.8)
        plot_image_mask(image, mask)
        image, mask = rescale(test_image, test_mask, 1.2, 1.2)
        plot_image_mask(image, mask)
        image, mask = rescale(test_image, test_mask, 0.8, 1.2)
        plot_image_mask(image, mask)
        plt.show()

    person = 1
    time = 1
    w=16
    contrast  = 'FLAIR'
    view = 'sagittal'
    data = load_image(person,time,contrast)
    z = 100;y=146;x=72
    image,roi_mask = get_image(data,view,z,y,x)
    #test_rescale(image,roi_mask)
    patch = crop_patch(image,y,z,w)
    plot_image_mask(image,roi_mask)
    plt.figure();plt.imshow(patch, cmap=matplotlib.cm.gray)
    plt.show()
    raw_input("Press Enter to continue...")

