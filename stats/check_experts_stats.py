import pandas as pd
import seaborn as sns
import numpy as np
from prod.data_containers import load_lables
import matplotlib.pyplot as plt
import ast


purpose = 'choose indexes'

person = 1
time = 2
w = 16
stats_path = '/media/sf_shared/src/medicalImaging/prod/runs/27_08_2017_10_39 - moe pretrain experts test 1/'  # '/media/sf_shared/src/medicalImaging/prod/runs/22_08_2017_15_46 -  stats moe1 person 1 time 2/stats_{}_{}.npy'.format(person,time)
stats_path += 'stats_{}_{}.npy'.format(person, time)
stats = np.load(stats_path)
labels = load_lables(person, time, doc_num=1)


if purpose == 'create csv':

    exp1 = []
    exp2 = []
    exp3 = []
    stat_labels = []
    indexes =  []
    for stat in stats:
        indexes.append(stat[0])
        stat_labels.append(labels[stat[0]])
        exp1.append(stat[1][1][0])
        exp2.append(stat[1][2][0])
        exp3.append(stat[1][3][0])

    df  =  pd.DataFrame()
    df['indexes'] = pd.Series(indexes)
    df['exp1'] = pd.Series(exp1)
    df['exp2'] = pd.Series(exp2)
    df['exp3'] = pd.Series(exp3)
    df['labels'] = pd.Series(stat_labels)
    df.to_csv('experts_stats.csv')




if purpose == 'plot stats':

    def heatmap_probs(series1, series2, title):
        t = np.zeros((11, 11))
        for index1, index2 in zip(series1, series2):
            index1 = int(10 * index1)
            index2 = int(10 * index2)
            t[index1][index2] += 1
        t = t / float(sum(sum(t)))
        sns.heatmap(t, annot=True, cmap="YlGnBu")
        plt.title(title)


    df = pd.read_csv('example.csv')
    q = df['norm 2']
    w = df['norm 1']
    title = 'normalized decisions'
    heatmap_probs(q, w, title)
    plt.figure()
    q = df['exp1']
    w = df['exp2']
    title = 'experts decisions'
    heatmap_probs(q, w, title)
    plt.show()

if purpose == 'choose indexes':

    def classify_expert(row):
        experts = [row['exp1'],row['exp2'],row['exp3']]
        if row['labels'] == 0:
            return np.argmin(experts)
        else:
            return np.argmax(experts)

    df = pd.read_csv('test{}_{}_stats.csv'.format(person,time))
    df2 = pd.DataFrame(df[['indexes','exp1','exp2','exp3','labels']][df['relevant']==1])
    df2['expert class'] = df2.apply(classify_expert,axis=1)

    indexes = []
    for i,row in df2.iterrows():
        index = ast.literal_eval(row['indexes'])
        exp_class = row['expert class']
        expended_index = (person,time,index[0],index[1],index[2])
        indexes.append((expended_index, exp_class))

    with open('test1_gate_indexes.npy','wb') as fp:
        np.save(fp, indexes)