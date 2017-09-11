import numpy as np
from keras.optimizers import SGD
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from data_containers import load_index_list
from multi_predictors_combined import gating_model_logistic_regression,gating_model_use_parameters
from train_tools import create_callbacks


print "collect train data"
index_list_tr = [[(2,x) for x in range(1,5)],[(3,x) for x in range(1,6)],[(4,x) for x in range(1,5)]]
train_d = [item for sublist in index_list_tr for item in sublist]
indexes_tr = load_index_list("gate_parameters_samples_test1_set",train_d)

features = []
labels =  []
for index in indexes_tr:
    features.append(np.concatenate(index[0],axis=0))
    labels.append(index[1])
features = np.array(features)
labels = np.array(labels)
print "collect val data"
index_list_val =[(5,x) for x in range(1,5)]
indexes_val = load_index_list("gate_parameters_samples_test1_set",index_list_val)
features_v = []
labels_v =  []
for index in indexes_val:
    features_v.append(np.concatenate(index[0],axis=0))
    labels_v.append(index[1])
features_v = np.array(features_v)
labels_v = np.array(labels_v)
name= "gate_logistic_regression"
callbacks = create_callbacks(name, fold=0)

#model = gating_model_logistic_regression(N_exp=3)
model = gating_model_use_parameters(N_exp=3)
optimizer = SGD(lr=0.01, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy', 'fmeasure', 'precision', 'recall'])
print "train model"
history = model.fit(features,labels,nb_epoch=50,validation_data=(features_v,labels_v),callbacks=callbacks)

predict = model.predict(features)
prediction = np.argmax(predict,axis=1)
true_labels = np.argmax(labels,axis=1)
print "classification \n \n {}".format(classification_report(true_labels,prediction))
print "accuracy \n \n {}".format(accuracy_score(true_labels,prediction))
print "confusion matrix  \n \n {}".format(confusion_matrix(true_labels,prediction))


index_list_test =[(1,x) for x in range(1,5)]
indexes_test = load_index_list("gate_parameters_samples_test1_set",index_list_test)
features_t = []
labels_t =  []
for index in indexes_test:
    features_t.append(np.concatenate(index[0],axis=0))
    labels_t.append(index[1])
features_t = np.array(features_t)
labels_t = np.array(labels_t)

predict = model.predict(features_t)
prediction = np.argmax(predict,axis=1)
true_labels = np.argmax(labels_t,axis=1)
print "classification \n \n {}".format(classification_report(true_labels,prediction))
print "accuracy \n \n {}".format(accuracy_score(true_labels,prediction))
print "confusion matrix  \n \n {}".format(confusion_matrix(true_labels,prediction))

model.save_weights('/media/sf_shared/src/medicalImaging/runs/MOE runs/run9-pretrain gate parameters/gate_parameters_test1.h5 ')
