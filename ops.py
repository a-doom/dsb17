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


@add_arg_scope
def resnet_conv(
        input,
        kernel_size,
        is_training,
        stride=1,
        num_outputs=None,
        normalizer_fn=None,
        activation_fn=None,
        scope=None):
    with variable_scope.variable_scope(scope, 'resnet_conv', [input]) as sc:
        last_dim = utils.last_dimension(input.get_shape())
        num_outputs = num_outputs or last_dim

        output = layers.convolution(
            inputs=input,
            num_outputs=last_dim,
            kernel_size=[1, 1, 1])

        output = layers.convolution(
            inputs=output,
            num_outputs=last_dim,
            kernel_size=[1, 1, kernel_size])

        output = layers.convolution(
            inputs=output,
            num_outputs=last_dim,
            kernel_size=[1, kernel_size, 1])

        output = layers.convolution(
            inputs=output,
            num_outputs=num_outputs,
            kernel_size=[kernel_size, 1, 1],
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            normalizer_params={'is_training': is_training},
            stride=stride)

        return output


@add_arg_scope
def resnet_reduction(input, kernel_size, is_training, scope=None):
    with variable_scope.variable_scope(scope, 'resnet_reduction', [input]) as sc:
        concat_dim = len(input.get_shape().as_list()) - 1
        last_dim = utils.last_dimension(input.get_shape())

        result_pool = tf.nn.max_pool3d(
            input=input,
            ksize=[1, kernel_size, kernel_size, kernel_size, 1],
            strides=[1, 2, 2, 2, 1],
            padding='SAME',
            name=scope)

        result_resnet = resnet_conv(
            input=input,
            kernel_size=kernel_size,
            stride=2,
            num_outputs=last_dim,
            is_training=is_training)

        return tf.concat(concat_dim, [result_pool, result_resnet])


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
        num_outputs,
        is_training=True,
        is_half_size=False,
        scope=None):
    with tf.variable_scope(scope, 'residual_v1', [input]):
        net = layers.batch_norm(
            inputs=input,
            activation_fn=tf.nn.relu,
            is_training=is_training)

        net = resnet_conv(
            input=net,
            kernel_size=3,
            is_training=is_training)

        net = resnet_conv(
            input=net,
            kernel_size=3,
            stride=2 if is_half_size else 1,
            num_outputs=num_outputs,
            is_training=is_training)

        return net
