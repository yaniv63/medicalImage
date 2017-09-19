from prod.data_containers import load_lables
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle

contrasts = ['FLAIR', 'T2', 'MPRAGE', 'PD']
views = ['axial', 'coronal', 'sagittal']

def get_patches(indexes,person,time,w,MAX_SIZE=100):
    from multiprocessing import JoinableQueue, Event, Process
    from prod.data_containers import load_all_data
    from prod.augmentations import AugmentationWorker

    patches = []
    input_q = JoinableQueue(MAX_SIZE)
    output_q = JoinableQueue(MAX_SIZE)

    #index = [1, 2, 19, 84, 96]  # [1,2,81,60,126]
    # np.random.seed(42)
    for i,j,k in indexes:
        input_q.put((person,time,i,j,k))

    Person_data = [(person, time)]
    data, positive_list, negative_list = load_all_data(Person_data, contrasts)
    aug_args = {'flip': True, 'flip_p': 0.5, 'rescale': True, 'rescale_lowbound': 0.8,
                'rescale_highbound': 1.2, 'rotate': True, 'rot_angle': 5}
    event = Event()
    worker = AugmentationWorker(input_q, output_q, data, contrasts, views, w, event, aug_args)
    pos_p = Process(target=worker.start_calc, name='positive worker')
    pos_p.daemon = True
    pos_p.start()
    for _ in range(len(indexes)):
        patches.append(output_q.get())
    worker.finish()
    pos_p.terminate()
    return patches


person = 1
time = 2
w = 16

stats_path ='/media/sf_shared/src/medicalImaging/prod/runs/17_09_2017_11_18 - soft,batching exp=8/' #'/media/sf_shared/src/medicalImaging/prod/runs/22_08_2017_15_46 -  stats moe1 person 1 time 2/stats_{}_{}.npy'.format(person,time)
stats_path += 'stats_{}_{}.npy'.format(person,time)
stats = np.load(stats_path)
labels = load_lables(person,time,doc_num= 1)
error_stats = []
for stat in stats:
    index = stat[0]
    printable_index = tuple(x + 1 for x in index)
    label = labels[index]
    prob = stat[1][0]
    decisions = stat[1][1:4]
    prediction = np.dot(prob, decisions)
    hard_prediction = 0 if np.round(prediction) < 0.5 else 1
    if hard_prediction != label:
        error_stats.append((index, prediction, label,prob,decisions))
less_stats = [error_stats[i]  for i in range(len(error_stats)) if i%100==0 ]
indexes = [stat[0] for stat in less_stats]
# with open('error_indexes_1_2.lst','wb') as f:
#     pickle.dump(error_stats,f)

print "start patching"
patches = get_patches(indexes, person, time, w)
print "after patching"
for stat,patch in zip(less_stats, patches):
    index = stat[0]
    printable_index = tuple(x+1 for x in index)
    label = stat[2]
    prediction = stat[1]
    gate =stat[3]
    experts = stat[4]
    flair_patches =   [p[0] for p in patch]
    f, axarr = plt.subplots(1,3)
    for i in range(3):
        axarr[i].imshow(flair_patches[i],cmap=matplotlib.cm.gray)
        axarr[i].set_title('view: {} \n pred: {:.5f} \n  coeff: {:.5f}'.format(views[i],experts[i][0],gate[i]))
    f.suptitle("index {} \n label {} \n prediction {:.5f}".format(printable_index,label,prediction[0]), fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
raw_input("wait to see")
#
#
