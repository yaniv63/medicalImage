from keras.models import Model
from keras.optimizers import SGD
from multi_predictors_combined import n_experts_combined_model_gate_parameters



def get_model():
    w_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run5-moe with pretrained experts/'
    w_path_gate = '/media/sf_shared/src/medicalImaging/results/'

    moe = n_experts_combined_model_gate_parameters(n=3, N_mod=4, img_rows=33, img_cols=33)
    moe.get_layer('Seq_0').load_weights(w_path + 'model_test_1_axial_fold_0.h5', by_name=True)
    moe.load_weights(w_path + 'model_test_1_axial_fold_0.h5', by_name=True)

    moe.get_layer('Seq_1').load_weights(w_path + 'model_test_1_coronal_fold_0.h5', by_name=True)
    moe.load_weights(w_path + 'model_test_1_coronal_fold_0.h5', by_name=True)

    moe.get_layer('Seq_2').load_weights(w_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)
    moe.load_weights(w_path + 'model_test_1_sagittal_fold_0.h5', by_name=True)

    moe.load_weights(w_path_gate + 'gate_batching_hard.h5', by_name=True)

    layer_dict = dict([(layer.name, layer) for layer in moe.layers])

    for name, layer in layer_dict.items():
        if 'gate' not in name:
            layer.trainable = False
    optimizer = SGD(lr=0.001, nesterov=True)
    moe.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])

    # w_path = '/media/sf_shared/src/medicalImaging/runs/MOE runs/run13- multilabel moe/lr_0.01/model_test_1_fold_0_lr_0.01.h5'
    # w1= moe.get_weights()
    # moe.load_weights(w_path)
    # w2 =moe.get_weights()

    layer_name1 = 'out_gate'
    intermediate_layer_model = Model(input=moe.input,
                                     output=[moe.output,
                                             moe.get_layer(layer_name1).get_output_at(0),
                                             moe.get_layer('out0').output,
                                             moe.get_layer('out1').output,
                                             moe.get_layer('out2').output,
                                             moe.get_layer("perception_0").output,
                                             moe.get_layer("perception_1").output,
                                             moe.get_layer("perception_2").output])
    return intermediate_layer_model
