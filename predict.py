import tensorflow as tf
import tensorflow.contrib.learn.python.learn as learn
import specific_models as sm
import os
import dataset_dsb17
import datetime
import csv


def predict(model, dataset_dir, model_dir, result_name):
    def input_fn_train():
        return dataset_dsb17.inputs_for_predict(
            data_dir=dataset_dir)

    filename_queue = dataset_dsb17.get_filename_queues(dataset_dir, dataset_dsb17.DATASET_MODE.ALL)
    filename_queue.sort()
    filename_queue = list(filename_queue)
    filenames = [os.path.splitext(os.path.basename(os.path.normpath(f)))[0] for f in filename_queue]

    classifier = learn.Estimator(model_fn=model, model_dir=model_dir)
    print("predict...")
    results = []
    for result in classifier.predict(input_fn=input_fn_train, as_iterable=True):
        results.append(result["probabilities"][1])
    results = zip(filenames, results)
    results = [("id", "cancer")] + results

    with open(result_name, 'wb') as result_csv:
        wr = csv.writer(result_csv, quoting=csv.QUOTE_NONE)
        wr.writerows(results)

    print("save to {0}".format(result_name))
    print("done")


def check_dir(path, name):
    if path is None:
        raise ValueError('You must supply the {0}'.format(name))
    if not os.path.isdir(path):
        raise ValueError('Wrong directory {0}'.format(path))


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'Directory for storing datasets')
tf.app.flags.DEFINE_string(
    'models_dir', None, 'Directory for storing models')
tf.app.flags.DEFINE_string(
    'result_dir', None, 'Directory for storing result csv')
tf.app.flags.DEFINE_string('model_name', None, 'The model name.')


def main(_):
    check_dir(FLAGS.dataset_dir, "dataset_dir")
    check_dir(FLAGS.models_dir, "models_dir")
    check_dir(FLAGS.result_dir, "result_dir")

    if not FLAGS.model_name:
        raise ValueError('You must supply the model name with --model_name')
    if FLAGS.model_name is not None and FLAGS.model_name not in sm.models:
        raise ValueError('wrong model name')

    model_dir = os.path.join(FLAGS.models_dir, FLAGS.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    result_dir = os.path.join(FLAGS.result_dir, FLAGS.model_name + "_results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_name = "results_" + datetime.datetime.now().strftime('%y_%m_%d_%H_%M') + ".csv"
    result_name = os.path.join(result_dir, result_name)

    predict(model=sm.models[FLAGS.model_name],
            dataset_dir=FLAGS.dataset_dir,
            model_dir=model_dir,
            result_name=result_name)


if __name__ == '__main__':
    tf.app.run()
