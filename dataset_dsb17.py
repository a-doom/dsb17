import tensorflow.contrib.learn.python.learn as learn
import tensorflow as tf
import tarfile
import os
import tensorflow.contrib.learn.python.learn


DATASET_TRAIN = 'train.bin'
DATASET_VALID = 'valid.bin'
DATASET_TEST = 'test.bin'

IMAGE_LABEL_BYTES = 1
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
IMAGE_DEPTH = 400

NUM_READ_THREADS = 2

class DATASET_MODE:
    TRAIN = "Train"
    VALID = "Valid"
    TEST = "Test"

def get_filename_queues(dataset_dir, mode):
    filenames = []
    for fname in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, fname)
        if not os.path.isdir(path):
            filenames.append(path)

    test_percent = 0.1
    valid_percent = 0.1

    test_size = max(int(round(len(filenames) * test_percent)), 1)
    valid_size = max(int(round(len(filenames) * valid_percent)), 1)

    test = filenames[:test_size]
    valid = filenames[test_size:(test_size + valid_size)]
    train = filenames[(test_size + valid_size):]

    result = []
    if mode == DATASET_MODE.TRAIN:
        result = train
    elif mode == DATASET_MODE.TEST:
        result = test
    elif mode == DATASET_MODE.VALID:
        result = valid

    for f in result:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    return result


def read_data(filename_queue, batch_size, is_train):
    image_bytes = IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH * 2
    record_bytes = IMAGE_LABEL_BYTES + image_bytes

    examples = learn.read_batch_examples(
        file_pattern=filename_queue,
        batch_size=batch_size,
        reader=lambda: tf.FixedLengthRecordReader(record_bytes=record_bytes),
        num_threads=NUM_READ_THREADS,
        num_epochs=1 if not is_train else None)

    if(isinstance(examples, tuple) and len(examples) == 2):
        _, examples = examples

    examples = tf.decode_raw(examples, tf.int8)

    labels = tf.slice(examples, [0, 0], [-1, IMAGE_LABEL_BYTES])
    labels = tf.cast(labels, tf.int64)

    images = tf.slice(examples, [0, IMAGE_LABEL_BYTES], [-1, image_bytes]),
    images = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 2])
    images = tf.bitcast(images, tf.int16)
    # add channel dim
    images = tf.expand_dims(images, -1)
    images = tf.cast(images, tf.float32)

    return images, labels


def inputs(data_dir, mode, batch_size):
    filename_queue = get_filename_queues(data_dir, mode)
    return read_data(filename_queue, batch_size, mode == DATASET_MODE.TRAIN)
