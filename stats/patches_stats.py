import numpy as np
np.random.seed(42)
import pickle

patches = r'./patches/'


def aggregate_val(personList, patchType):
    i = 1
    for index in personList:
        for index2 in range(1, 5):
            with open(patches + "patches_"+patchType+ "_val_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels__val_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                if i == 1:
                    samples_val = pickle.load(fp1)
                    labels_val = pickle.load(fp2)
                    i = 2
                else:
                    t1 = pickle.load(fp1)
                    t2 = pickle.load(fp2)
                    samples_val = np.append(samples_val, t1, axis=0)
                    labels_val = np.append(labels_val, t2, axis=0)

    samples_val = np.array(samples_val)
    labels_val = np.array(labels_val)

    samples_val = np.expand_dims(samples_val, 1)
    labels_val = np.expand_dims(labels_val, 1)

    return (samples_val, labels_val)

def aggregate_data(personList, patchType):
    i = 1
    for index in personList:
        for index2 in range(1, 5):
            with open(patches + "patches_"+patchType+ "_train_0{}_0{}.lst".format(index, index2), 'rb') as fp1, open(
                            patches + "labels_train_0{}_0{}.lst".format(index, index2), 'rb') as fp2:
                if i == 1:
                    samples_val = pickle.load(fp1)
                    labels_val = pickle.load(fp2)
                    i = 2
                else:
                    t1 = pickle.load(fp1)
                    t2 = pickle.load(fp2)
                    samples_val = np.append(samples_val, t1, axis=0)
                    labels_val = np.append(labels_val, t2, axis=0)

    samples_val = np.array(samples_val)
    labels_val = np.array(labels_val)

    samples_val = np.expand_dims(samples_val, 1)
    labels_val = np.expand_dims(labels_val, 1)

    return (samples_val, labels_val)

PersonTrainList = [2,3,4]
PersonValList = [1]
val_axial_set,val_axial_labels = aggregate_val(PersonValList,"axial")
data_axial_set,data_axial_labels = aggregate_data(PersonTrainList,"axial")

val_0_counter = 0
val_1_counter = 0
data_0_counter = 0
data_1_counter = 0
for i in range(len(val_axial_labels)):
    if val_axial_labels[i] == 0:
        val_0_counter += 1
    else :
        val_1_counter += 1
for i in range(len(data_axial_labels)):
    if data_axial_labels[i] == 0:
        data_0_counter += 1
    else :
        data_1_counter += 1

print("val_0_counter {}".format(val_0_counter))
print("val_1_counter {}".format(val_1_counter))
print("data_0_counter {}".format(data_0_counter))
print("data_1_counter {}".format(data_1_counter))
