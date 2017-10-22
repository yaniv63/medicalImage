from keras.models import Model
from keras.optimizers import SGD
from multi_predictors_combined import n_experts_combined_model_gate_parameters,n_parameters_combined_model

def model_parameters_equal_model():
    w_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run15-contrast experts/'

    moe = n_parameters_combined_model(n=4,N_mod=3)
    moe.get_layer('Seq_0').load_weights(w_path + 'flair/model_test_1_fold_0_flair.h5', by_name=True)
    moe.load_weights(w_path + 'flair/model_test_1_fold_0_flair.h5', by_name=True)

    moe.get_layer('Seq_1').load_weights(w_path + 't2/model_test_1_fold_0_t2.h5', by_name=True)
    moe.load_weights(w_path + 't2/model_test_1_fold_0_t2.h5', by_name=True)

    moe.get_layer('Seq_2').load_weights(w_path + 'mprage/model_test_1_fold_0_mprage.h5', by_name=True)
    moe.load_weights(w_path + 'mprage/model_test_1_fold_0_mprage.h5', by_name=True)

    moe.get_layer('Seq_3').load_weights(w_path + 'pd/model_test_1_fold_0_pd.h5', by_name=True)
    moe.load_weights(w_path + 'pd/model_test_1_fold_0_pd.h5', by_name=True)

    optimizer = SGD(lr=0.001, nesterov=True)
    moe.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    return moe


def model_moe_pretrained_experts():
    w_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run15-contrast experts/'
    # w_path_gate = '/media/sf_shared/src/medicalImaging/results/'

    moe = n_experts_combined_model_gate_parameters(n=4, N_mod=3, img_rows=33, img_cols=33)
    moe.get_layer('Seq_0').load_weights(w_path + 'flair/model_test_1_fold_0_flair.h5', by_name=True)
    moe.load_weights(w_path + 'flair/model_test_1_fold_0_flair.h5', by_name=True)

    moe.get_layer('Seq_1').load_weights(w_path + 't2/model_test_1_fold_0_t2.h5', by_name=True)
    moe.load_weights(w_path + 't2/model_test_1_fold_0_t2.h5', by_name=True)

    moe.get_layer('Seq_2').load_weights(w_path + 'mprage/model_test_1_fold_0_mprage.h5', by_name=True)
    moe.load_weights(w_path + 'mprage/model_test_1_fold_0_mprage.h5', by_name=True)

    moe.get_layer('Seq_3').load_weights(w_path + 'pd/model_test_1_fold_0_pd.h5', by_name=True)
    moe.load_weights(w_path + 'pd/model_test_1_fold_0_pd.h5', by_name=True)

    # moe.load_weights(w_path_gate + 'gate_batching_hard.h5', by_name=True)

    layer_dict = dict([(layer.name, layer) for layer in moe.layers])

    for name, layer in layer_dict.items():
        if 'gate' not in name:
            layer.trainable = False
    optimizer = SGD(lr=0.001, nesterov=True)
    moe.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
    return moe


def get_model():
    return model_moe_pretrained_experts()



