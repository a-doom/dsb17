import tensorflow as tf
import specific_models as sm
import dataset_dsb17
import sys

# trainable
# SGD all
# adam all


def _calc_model_params_count():
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    total_parameters = 0

    sess = tf.Session()
    sess.run(init)

    # iterating over all variables
    for variable in tf.trainable_variables():
        local_parameters = 1
        shape = variable.get_shape()  # getting shape of a variable
        for i in shape:
            local_parameters *= i.value  # mutiplying dimension values
        total_parameters += local_parameters
    print("total number of trainable parameters: {0}".format(total_parameters))

    # iterating over all variables
    # for variable in tf.all_variables():
    for variable in tf.global_variables():
        local_parameters = 1
        shape = variable.get_shape()  # getting shape of a variable
        for i in shape:
            local_parameters *= i.value  # mutiplying dimension values
        total_parameters += local_parameters
    print("total number of parameters: {0}".format(total_parameters))


def main():
    images, labels = dataset_dsb17.inputs(
        data_dir=sys.argv[1],
        mode=dataset_dsb17.DATASET_MODE.VALID,
        batch_size=1)
    predictions, loss, train_op = sm.res_net_pyramidal_model_d10_w64_k2_dr05(images, labels, "not_train")
    _calc_model_params_count()


if __name__ == '__main__':
    main()
