import tensorflow.contrib.learn.python.learn as learn
import tensorflow as tf
import tarfile
import os
import tensorflow.contrib.learn.python.learn


DATASET_TRAIN = 'train.bin'
DATASET_VALID = 'valid.bin'
DATASET_TEST = 'test.bin'

IMAGE_LABEL_BYTES = 1
IMAGE_HEIGHT = IMAGE_WIDTH = IMAGE_LENGTH = 200

NUM_READ_THREADS = 2

class DATASET_MODE:
    TRAIN = "Train"
    VALID = "Valid"
    TEST = "Test"

def get_filename_queues(dataset_dir, mode):
    filenames = []
    for fname in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, fname)
        if not os.path.isdir(path) and path.endswith(".bin"):
            filenames.append(path)

    if len(filenames) == 0:
        raise ValueError("Wrong dataset dir: {0}".format(dataset_dir))

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
    image_bytes = IMAGE_LENGTH * IMAGE_HEIGHT * IMAGE_WIDTH * 2
    record_bytes = IMAGE_LABEL_BYTES + image_bytes

    image_float32 = IMAGE_LENGTH * IMAGE_HEIGHT * IMAGE_WIDTH
    record_float32 = IMAGE_LABEL_BYTES + image_float32

    def parse_fn(record):
        record = tf.decode_raw(record, tf.int8)
        record = tf.reshape(record, [record_bytes])

        label = tf.slice(record, [0], [IMAGE_LABEL_BYTES])
        label = tf.cast(label, tf.float32)

        image = tf.slice(record, [IMAGE_LABEL_BYTES], [image_bytes]),
        image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_LENGTH, IMAGE_WIDTH, 2])
        image = tf.bitcast(image, tf.int16)
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [image_float32])

        record = tf.concat(0, [label, image])
        return record

    examples = learn.read_batch_examples(
        file_pattern=filename_queue,
        batch_size=batch_size,
        reader=lambda: tf.FixedLengthRecordReader(record_bytes=record_bytes),
        num_threads=NUM_READ_THREADS,
        parse_fn=parse_fn,
        num_epochs=1 if not is_train else 10)

    if(isinstance(examples, tuple) and len(examples) == 2):
        _, examples = examples

    labels = tf.slice(examples, [0, 0], [-1, IMAGE_LABEL_BYTES])
    labels = tf.cast(labels, tf.int64)

    # add channel dim
    images = tf.slice(examples, [0, IMAGE_LABEL_BYTES], [-1, image_float32]),
    images = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_LENGTH, IMAGE_WIDTH])
    images = tf.expand_dims(images, -1)

    return images, labels


def inputs(data_dir, mode, batch_size):
    filename_queue = get_filename_queues(data_dir, mode)
    return read_data(filename_queue, batch_size, mode == DATASET_MODE.TRAIN)
