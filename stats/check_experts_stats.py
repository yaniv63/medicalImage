import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import pandas as pd
import seaborn as sns
import numpy as np
from itertools import chain
from prod.data_containers import load_lables
import matplotlib.pyplot as plt
import ast

# person = 1
# time = 2
purpose = 'check if relevant'

if purpose == 'check if relevant':

    def experts_decided_correct(row):

        return np.round(row['exp1'], 0) == row['labels'] or \
                        np.round(row['exp2'], 0) == row['labels'] or \
                        np.round(row['exp3'], 0) == row['labels']


    def exist_difference_between_experts(row):
        experts = [row['exp1'], row['exp2'], row['exp3']]
        experts.sort()
        if row['labels'] == 0:
            exp1, exp2 = experts[0], experts[1]
        else:
            exp1, exp2 = experts[1], experts[2]
        return abs(exp1 - exp2) > 0.1

    # def part_bigger_than_thersh(num,den,thresh):
    #     return (abs(1-(float(num)/den)) < thresh)
    #
    # def exist_difference_between_experts(row):
    #     experts = [row['exp1'], row['exp2'], row['exp3']]
    #     experts.sort()
    #     return not (part_bigger_than_thersh(experts[0],experts[1],0.1) and
    #                 part_bigger_than_thersh(experts[1],experts[2],0.1))
    #
    #
    data = np.array([[(2,x) for x in range(1,5)],[(3,x) for x in range(1,6)],[(4,x) for x in range(1,5)],
            [(5,x) for x in range(1,5)]])
    data = [[(1,x) for x in range(1,5)]]
    for index_list in data:
        dfs = []
        for  person,time in index_list:

            w = 16
            stats_path = 'person_stats/'  # '/media/sf_shared/src/medicalImaging/prod/runs/22_08_2017_15_46 -  stats moe1 person 1 time 2/stats_{}_{}.npy'.format(person,time)
            stats_path += 'stats_{}_{}.npy'.format(person, time)
            stats = np.load(stats_path)
            labels = load_lables(person, time, doc_num=1)

            exp1 = []
            exp2 = []
            exp3 = []
            stat_labels = []
            indexes =  []
            for stat in stats:
                indexes.append((person,time,stat[0][0],stat[0][1],stat[0][2]))
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

            dfs.append(df)

        total_df = pd.concat(dfs)
    # total_df = pd.read_csv('/media/sf_shared/src/medicalImaging/stats/test1_2_stats.csv')
        total_df['exp_correct']  = total_df.apply(experts_decided_correct,axis=1)
        total_df['differece_between_exp'] = total_df.apply(exist_difference_between_experts,axis=1)
        total_df['relevant'] = total_df.apply(lambda row: row['exp_correct'] == 1 and row['differece_between_exp'] ==1 ,axis=1)

        def classify_expert(row):
            experts = [row['exp1'],row['exp2'],row['exp3']]
            if row['labels'] == 0:
                return np.argmin(experts)
            else:
                return np.argmax(experts)

        df2 = pd.DataFrame(total_df[['indexes','exp1','exp2','exp3','labels']][total_df['relevant']==True])
        df2['expert class'] = df2.apply(classify_expert,axis=1)

        indexes = []
        outer = False
        for i,row in df2.iterrows():
            if outer:
                index = ast.literal_eval(row['indexes'])
                index = (person,time,index[0],index[1],index[2])
            else:
                index = row['indexes']

            exp_class = row['expert class']
            indexes.append((index, exp_class))

        with open('gate_indexes_person_{}.npy'.format(person),'wb') as fp:
            np.save(fp, np.array(indexes))


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

