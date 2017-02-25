from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
import tensorflow as tf
from tensorflow.contrib import layers


@add_arg_scope
def add_multiplier(input, scope=None):
    with variable_scope.variable_scope(scope, 'add_multiplier', [input]) as sc:
        dtype = input.dtype.base_dtype

        weights_collections = utils.get_variable_collections(None, 'weights')
        weights = variables.model_variable('weights',
                                           shape=[1],
                                           dtype=dtype,
                                           initializer=initializers.xavier_initializer(),
                                           regularizer=None,
                                           collections=weights_collections,
                                           trainable=True)
        outputs = tf.mul(input, weights)

        biases_collections = utils.get_variable_collections(None, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[1],
                                          dtype=dtype,
                                          initializer=init_ops.zeros_initializer,
                                          regularizer=None,
                                          collections=biases_collections,
                                          trainable=True)
        outputs = tf.add(outputs, biases)
        return outputs


def residual_dropout(
        input,
        keep_prob=0.5,
        is_training=True,
        scope=None):
    with variable_scope.variable_scope(scope, 'residual_dropout', [input]) as sc:
        shape = tf.shape(input)
        shape_static = input.get_shape()
        noise_shape = [1] * (len(shape_static) - 1)
        noise_shape =  tf.concat(shape[0], [noise_shape])

        return layers.dropout(
            inputs=input,
            keep_prob=keep_prob,
            noise_shape=noise_shape,
            is_training=is_training)


def residual_v1(
        input,
        out_filter,
        is_training=True,
        is_half_size=False,
        scope=None):
        with tf.variable_scope(scope, 'residual_v1', [input]):
            net = layers.batch_norm(
                inputs=input,
                activation_fn=tf.nn.relu,
                is_training=is_training)
            net = layers.convolution(
                inputs=net,
                num_outputs=out_filter,
                kernel_size=[3, 3, 3],
                stride=2 if is_half_size else 1,
                normalizer_fn=layers.batch_norm,
                normalizer_params={'is_training': is_training},
                activation_fn=tf.nn.relu)
            net = layers.convolution(
                inputs=net,
                num_outputs=out_filter,
                kernel_size=[3, 3, 3],
                normalizer_fn=layers.batch_norm,
                normalizer_params={'is_training': is_training})
            return net