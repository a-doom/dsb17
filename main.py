import tensorflow as tf
import tensorflow.contrib.learn.python.learn as learn
import specific_models as sm
import os
import dataset_dsb17
import time
import logging
import datetime

tf.logging.set_verbosity(tf.logging.INFO)


def loginfo(msg):
    tf.logging._logger.info(msg)


models = {"rnp_d110_w350" : sm.res_net_pyramidal_model_d110_w350,
          "wrn_d28_w10": sm.res_net_wide_model_d28_w10,
          "rnp_d80_w256_k3_dr05": sm.res_net_pyramidal_model_d80_w256_k3_dr05,
          "rnp_d6_w32_k1_dr05": sm.res_net_pyramidal_model_d6_w32_k1_dr05,
          "rnp_d10_w64_k2_dr05": sm.res_net_pyramidal_model_d10_w64_k2_dr05}


def train(model, dataset_dir, model_dir, batch_size, train_steps,
          is_evaluate_accuracy, valid_every_n_steps, early_stopping_rounds):
    def input_fn_test():
        return dataset_dsb17.inputs(
            data_dir=dataset_dir,
            mode=dataset_dsb17.DATASET_MODE.TEST,
            batch_size=batch_size)

    def input_fn_train():
        return dataset_dsb17.inputs(
            data_dir=dataset_dir,
            mode=dataset_dsb17.DATASET_MODE.TRAIN,
            batch_size=batch_size)

    def input_fn_valid():
        return dataset_dsb17.inputs(
            data_dir=dataset_dir,
            mode=dataset_dsb17.DATASET_MODE.VALID,
            batch_size=batch_size)

    loginfo("Create validation monitor")
    # Monitors
    metrics = {
        'accuracy': learn.metric_spec.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key='accuracy')
    }

    validation_monitor = learn.monitors.ValidationMonitor(
        input_fn=input_fn_valid,
        every_n_steps=valid_every_n_steps,
        early_stopping_rounds=early_stopping_rounds,
        metrics=metrics)

    loginfo("Create classifier")
    start = time.time()
    classifier = learn.Estimator(model_fn=model, model_dir=model_dir)

    loginfo("Fit model")
    # Fit model.
    classifier.fit(input_fn=input_fn_train,
                   steps=train_steps,
                   monitors=[validation_monitor])
    loginfo("Fit model finished. Time: %.03f s" % (time.time() - start))
    start = time.time()

    if is_evaluate_accuracy:
        loginfo("Evaluate accuracy")
        # Evaluate accuracy.
        result = classifier.evaluate(
            input_fn=input_fn_test,
            metrics=metrics)
        loginfo('Accuracy: {0:f}'.format(result['accuracy']))
        loginfo("Time: %.03f s" % (time.time() - start))
    loginfo("done")


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'Directory for storing datasets')
tf.app.flags.DEFINE_string(
    'models_dir', None, 'Directory for storing models')
tf.app.flags.DEFINE_string('model_name', None, 'The model name.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, """int, the number of images in a batch.""")
tf.app.flags.DEFINE_integer(
    'train_steps', 10, """int, the number of training steps""")
tf.app.flags.DEFINE_bool(
    'is_evaluate_accuracy', False, """bool, Evaluate accuracy""")
tf.app.flags.DEFINE_integer(
    'valid_every_n_steps', 300, """int, run validation monitor every n steps""")
tf.app.flags.DEFINE_integer(
    'early_stopping_rounds', 500, """int, early stopping rounds""")
tf.app.flags.DEFINE_string(
    'optimizer_type', 'SGD', """Optimizer type (SGD|Adam)""")
tf.app.flags.DEFINE_float(
    'learning_rate', 0.001, """Learning rate )""")
tf.app.flags.DEFINE_string(
    'log_dir', None, 'Directory for storing logs')


def check_dir(path, name):
    if path is None:
        raise ValueError('You must supply the {0}'.format(name))
    if not os.path.isdir(path):
        raise ValueError('Wrong dataset directory {0}'.format(path))


def main(_):
    check_dir(FLAGS.dataset_dir, "dataset_dir")
    check_dir(FLAGS.models_dir, "models_dir")
    FLAGS.log_dir = FLAGS.log_dir or FLAGS.models_dir
    check_dir(FLAGS.log_dir, "log_dir")

    if not FLAGS.model_name:
        raise ValueError('You must supply the model name with --model_name')
    if FLAGS.model_name is not None and FLAGS.model_name not in models:
        raise ValueError('wrong model name')

    model_dir = os.path.join(FLAGS.models_dir, FLAGS.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = os.path.join(FLAGS.log_dir, FLAGS.model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = "log_" + datetime.datetime.now().strftime('%y_%m_%d_%H_%M') + ".log"

    fh = logging.FileHandler(os.path.join(log_dir, log_name))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    tf.logging._logger.addHandler(fh)

    model = sm.convert_model(
        model=models[FLAGS.model_name],
        optimizer_type=FLAGS.optimizer_type,
        learning_rate=FLAGS.learning_rate)

    train(model=model,
          dataset_dir=FLAGS.dataset_dir,
          model_dir=model_dir,
          batch_size=FLAGS.batch_size,
          train_steps=FLAGS.train_steps,
          is_evaluate_accuracy=FLAGS.is_evaluate_accuracy,
          valid_every_n_steps=FLAGS.valid_every_n_steps,
          early_stopping_rounds=FLAGS.early_stopping_rounds)


if __name__ == '__main__':
    tf.app.run()
