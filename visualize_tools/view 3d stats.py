from prod.data_containers import load_lables
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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

stats_path ='/media/sf_shared/src/medicalImaging/prod/runs/19_09_2017_16_17 - hard batching combined with stats/'#'/media/sf_shared/src/medicalImaging/prod/runs/18_09_2017_12_47 - combined train after hard + batching/' #'/media/sf_shared/src/medicalImaging/prod/runs/22_08_2017_15_46 -  stats moe1 person 1 time 2/stats_{}_{}.npy'.format(person,time)
stats_path += 'stats_{}_{}.npy'.format(person,time)
stats = np.load(stats_path)
labels = load_lables(person,time,doc_num= 1)

less_stats = [stat for i,stat in enumerate(stats) if i%1500==0]
indexes = [stat[0] for stat in less_stats]
patches = get_patches(indexes,person,time,w)

for stat,patch in zip(less_stats,patches):
    index=  stat[0]
    printable_index = tuple(x+1 for x in index)
    label = labels[index]
    prob = stat[1][0]
    decisions = stat[1][1:4]
    prediction = np.dot(prob,decisions)
    flair_patches =   [p[0] for p in patch]
    f, axarr = plt.subplots(1,3)
    for i in range(3):
        axarr[i].imshow(flair_patches[i],cmap=matplotlib.cm.gray)
        axarr[i].set_title('view: {} \n pred: {:.5f} \n  coeff: {:.5f}'.format(views[i],decisions[i][0],prob[i]))
    f.suptitle("index {} \n label {} \n prediction {:.5f}".format(printable_index,label,prediction[0]), fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
raw_input("wait to see")



# f, axarr = plt.subplots(1, 3,figsize=(15,8))
# f.suptitle("tsne embedding")
# #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#
# less_stats2 = [stat for i,stat in enumerate(stats) if i%20==0]
#
# for i in range(3):
#     vectors = [stat[1][4+i] for stat in less_stats2]
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     Y = tsne.fit_transform(vectors)
#     axarr[i].scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
#     axarr[i].set_title('view: {}'.format(views[i]))



f, axarr = plt.subplots(1, 1,figsize=(15,8))
f.suptitle("pca embedding")
color_dict={(0,0):'b',(1,1):'r',(0,1):'y',(1,0):'g'}
probs=[]
decisions = []
prediction = []
stat_labels = []
for i,stat in enumerate(stats):
    stat_labels.append(labels[stat[0]])
    probs.append(stat[1][0])
    decisions.append(stat[1][1:4])
    prediction.extend(np.dot(probs[i],decisions))

final_predictions = [1 if x[0]>0.5 else 0 for x in prediction]
colors = [color_dict[x] for x in zip(stat_labels,final_predictions)]

vectors = [(stat[1][0][0],stat[1][0][1]) for stat in less_stats if labels[stat[0]] == 1]
xs = [x for x,y in vectors]
ys = [y for x,y in vectors]
# pca = PCA(n_components=2)
# Y = pca.fit_transform(vectors)
# print(pca.explained_variance_ratio_)
#axarr[i].scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral,color=colors)
axarr.scatter(xs,ys, cmap=plt.cm.Spectral, color=colors)
axarr.set_title('precentage axial coronal')
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='blue', label='TN')
red_patch = mpatches.Patch(color='red', label='TP')
yellow_patch = mpatches.Patch(color='yellow', label='FP')
green_patch = mpatches.Patch(color='green', label='FN')
plt.legend(handles=[red_patch,blue_patch,yellow_patch,green_patch])

plt.show()
# stats_append_label = []#[(index,stat,label) for index,stat in stats for label in ]
# for stat in stats:
#     index=  stat[0]
#     label = labels[index]
#     if label == 0 or  np.random.random() > 0.25 :
#        continue
#     prob = stat[1][0]
#     decisions = stat[1][1:]
#     stats_append_label.append((index,prob,decisions,label))

# plt.figure()
# ax = plt.subplot(111)
# for stat in stats_append_label:
#     x,y = stat[1][:2]
#     plt.text(x,y, str(stat[3]),
#                  color=plt.cm.Set1(stat[3] / 10.),
#                  fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([]), plt.yticks([])
# plt.show()
#
# for i, angle in enumerate(patches):
#     for j, contrast in enumerate(angle):
#         plt.figure()
#         plt.imshow(contrast, cmap=matplotlib.cm.gray)
#         plt.title("angle is {} contrast {}".format(views[i], contrasts[j]))
# plt.show()
# raw_input("wait to see")