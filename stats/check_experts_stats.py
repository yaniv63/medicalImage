import pandas as pd
import seaborn as sns
import numpy as np
from itertools import chain
from prod.data_containers import load_lables
import matplotlib.pyplot as plt
import ast
from keras.utils.np_utils import to_categorical

# person = 1
# time = 2
purpose = 'check if relevant'
expert_labeling = 'hard'
exponent_soft = 10

if purpose == 'check if relevant':

    def experts_decided_correct(row):

        return np.round(row['exp1'], 0) == row['labels'] or \
                        np.round(row['exp2'], 0) == row['labels'] or \
                        np.round(row['exp3'], 0) == row['labels'] or \
                        np.round(row['exp4'], 0) == row['labels']


    def exist_difference_between_experts(row):
        experts = [row['exp1'], row['exp2'], row['exp3'],row['exp4']]
        experts.sort()
        if row['labels'] == 0:
            exp1, exp2 = experts[0], experts[1]
        else:
            exp1, exp2 = experts[3], experts[2]
        return abs(exp1 - exp2) > 0.1

    data = np.array([[(2,x) for x in range(1,5)],[(3,x) for x in range(1,6)],[(4,x) for x in range(1,5)],
            [(5,x) for x in range(1,5)],[(1,x) for x in range(1,5)],[(1,x) for x in range(1,5)]])
    all_dfs = []
    for index_list in data:
        dfs = []
        for  person,time in index_list:

            w = 16
            stats_path = '/media/sf_shared/src/medicalImaging/stats/person_stats/'  # '/media/sf_shared/src/medicalImaging/prod/runs/22_08_2017_15_46 -  stats moe1 person 1 time 2/stats_{}_{}.npy'.format(person,time)
            stats_path += 'stats_{}_{}.npy'.format(person, time)
            stats = np.load(stats_path)
            labels = load_lables(person, time, doc_num=1)

            exp1 = []
            exp2 = []
            exp3 = []
            exp4 = []
            exp1_vec = []
            exp2_vec = []
            exp3_vec = []
            exp4_vec = []
            stat_labels = []
            indexes =  []
            for stat in stats:
                indexes.append((person,time,stat[0][0],stat[0][1],stat[0][2]))
                stat_labels.append(labels[stat[0]])
                exp1.append(stat[1][2][0])
                exp2.append(stat[1][3][0])
                exp3.append(stat[1][4][0])
                exp4.append(stat[1][5][0])
                exp1_vec.append(stat[1][6])
                exp2_vec.append(stat[1][7])
                exp3_vec.append(stat[1][8])
                exp4_vec.append(stat[1][9])

            df  =  pd.DataFrame()
            df['indexes'] = pd.Series(indexes)
            df['exp1'] = pd.Series(exp1)
            df['exp2'] = pd.Series(exp2)
            df['exp3'] = pd.Series(exp3)
            df['exp4'] = pd.Series(exp4)
            df['exp1_vec'] = pd.Series(exp1_vec)
            df['exp2_vec'] = pd.Series(exp2_vec)
            df['exp3_vec'] = pd.Series(exp3_vec)
            df['exp4_vec'] = pd.Series(exp4_vec)
            df['labels'] = pd.Series(stat_labels)

            dfs.append(df)

        total_df = pd.concat(dfs)
    # total_df = pd.read_csv('/media/sf_shared/src/medicalImaging/stats/test1_2_stats.csv')
        total_df['exp_correct']  = total_df.apply(experts_decided_correct,axis=1)
        total_df['differece_between_exp'] = total_df.apply(exist_difference_between_experts,axis=1)
        total_df['relevant'] = total_df.apply(lambda row: row['exp_correct'] == 1 and row['differece_between_exp'] ==1 ,axis=1)
        #total_df.to_csv('/media/sf_shared/src/medicalImaging/stats/test1_stats.csv')
        def hard_decision_expert(row):
            experts = [row['exp1'],row['exp2'],row['exp3'],row['exp4']]
            if row['labels'] == 0:
                max_expert =  np.argmin(experts)
            else:
                max_expert  = np.argmax(experts)

            x= to_categorical([max_expert],4).tolist()
            return x

        def soft_decision_expert(row):
            experts = [row['exp1'],row['exp2'],row['exp3'],row['exp4']]
            if row['labels'] == 0:
                experts = [ 1-expert for expert in experts]
            exponent_experts = [expert**exponent_soft for expert in experts]
            normal_experts = [expert/float(sum(exponent_experts)) for expert in exponent_experts]
            # print  "experts " , experts
            # print "normal " , normal_experts
            return normal_experts

        df2 = pd.DataFrame(total_df[['indexes','exp1','exp2','exp3','exp4','exp1_vec','exp2_vec','exp3_vec','exp4_vec','labels']][total_df['relevant']==True])
        if expert_labeling == 'soft':
            method = soft_decision_expert
        elif expert_labeling == 'hard':
            method = hard_decision_expert
        df2['experts labels'] = df2.apply(method, axis=1)
        all_dfs.append(df2)
        indexes = []
        vectors=[]
        outer = False
        for i,row in df2.iterrows():
            if outer:
                index = ast.literal_eval(row['indexes'])
                index = (person,time,index[0],index[1],index[2])
            else:
                index = row['indexes']
            true_label = row['labels']

            exp_labels = row['experts labels']
            indexes.append((index, exp_labels, true_label))
            row_vec = (row['exp1_vec'],row['exp2_vec'],row['exp3_vec'],row['exp4_vec'])
            vectors.append((row_vec,exp_labels,true_label))
        with open('gate_indexes_expert_labels_{}_person_{}.npy'.format(expert_labeling,person),'wb') as fp:
            np.save(fp, np.array(indexes))

        with open('gate_parameters_samples_test1_set_{}.npy'.format(person), 'wb') as fp:
            np.save(fp, np.array(vectors))


    tot = pd.concat(all_dfs)
    tot.to_csv('/media/sf_shared/src/medicalImaging/all_train.csv')
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

