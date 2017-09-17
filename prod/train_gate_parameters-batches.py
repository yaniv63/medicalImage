import numpy as np
from keras.optimizers import SGD
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from data_containers import load_index_list,separate_classes_indexes
from multi_predictors_combined import gating_model_logistic_regression,gating_model_use_parameters
from train_tools import create_callbacks,calc_epoch_size
from train_proccesses_gate import random_vectors

class_label_method = 'hard'
exponent = 5

if class_label_method =='hard':
    info_path = 'gate vectors - hard decision/'
else:
    info_path = 'gate vectors - soft decision - exponent={}/'.format(exponent)
    class_label_method  += '_{}'.format(exponent)


print "collect train data"
index_list_tr = [[(2,x) for x in range(1,5)],[(3,x) for x in range(1,6)],[(4,x) for x in range(1,5)]]
train_d = [item for sublist in index_list_tr for item in sublist]
indexes_tr = load_index_list(info_path +"gate_parameters_samples_test1_set",train_d)
indexes_set_for_class = separate_classes_indexes(indexes_tr,2)
batch_size = 16
class_num =2
smallest_set = min([len(set) for set in indexes_set_for_class])
features = []
labels =  []
gen  = random_vectors(indexes_set_for_class,16)

for feature, class_label, true_label in indexes_tr:
    features.append(np.concatenate(feature,axis=0))
    labels.append(class_label)
features = np.array(features)
labels = np.array(labels)
labels = labels.squeeze()

print "collect val data"
index_list_val =[(5,x) for x in range(1,5)]
indexes_val = load_index_list(info_path +"gate_parameters_samples_test1_set",index_list_val)
features_v = []
labels_v =  []
for feature, class_label, true_label in indexes_val:
    features_v.append(np.concatenate(feature,axis=0))
    labels_v.append(class_label)
features_v = np.array(features_v)
labels_v = np.array(labels_v)
labels_v = labels_v.squeeze()
name= "gate_paramters_model"
callbacks = create_callbacks(name, fold=0)


#model = gating_model_logistic_regression(N_exp=3)
model = gating_model_use_parameters(N_exp=3)
print model.input_shape
optimizer = SGD(lr=0.01, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy', 'fmeasure', 'precision', 'recall'])
#model.load_weights('/media/sf_shared/src/medicalImaging/runs/MOE runs/run9-pretrain gate parameters/temp/gate_parameters_test1.h5')
epoch_size = calc_epoch_size(indexes_set_for_class[1],16)
print epoch_size
print "train model"
epoch_size = class_num * smallest_set - ((class_num * smallest_set) % batch_size)

history = model.fit_generator(gen,nb_epoch=80,samples_per_epoch=epoch_size,validation_data=(features_v,labels_v),callbacks=callbacks)

#history = model.fit(features,labels,nb_epoch=50,validation_data=(features_v,labels_v),callbacks=callbacks)

predict = model.predict(features)
prediction = np.argmax(predict,axis=1)
true_labels = np.argmax(labels,axis=1)
print "classification \n \n {}".format(classification_report(true_labels,prediction))
print "accuracy \n \n {}".format(accuracy_score(true_labels,prediction))
print "confusion matrix  \n \n {}".format(confusion_matrix(true_labels,prediction))


index_list_test =[(1,x) for x in range(1,5)]
indexes_test = load_index_list(info_path + "gate_parameters_samples_test1_set",index_list_test)
features_t = []
labels_t =  []
for feature, class_label, true_label in indexes_test:
    features_t.append(np.concatenate(feature,axis=0))
    labels_t.append(class_label)
features_t = np.array(features_t)
labels_t = np.array(labels_t).squeeze()

predict = model.predict(features_t)
prediction = np.argmax(predict,axis=1)
true_labels = np.argmax(labels_t,axis=1)
print "classification \n \n {}".format(classification_report(true_labels,prediction))
print "accuracy \n \n {}".format(accuracy_score(true_labels,prediction))
print "confusion matrix  \n \n {}".format(confusion_matrix(true_labels,prediction))

model.save_weights('/media/sf_shared/src/medicalImaging/results/gate_batching_{}.h5 '.format(class_label_method))
