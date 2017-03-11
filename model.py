import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import layers, metrics
from tensorflow.contrib.learn.python.learn.estimators.prediction_key import PredictionKey
import pandas as pd
import numpy as np
import ops


def subsample(inputs, stride, scope=None):
    if stride == 1:
        return inputs
    else:
        return tf.nn.max_pool3d(
            input=inputs,
            ksize=[1, stride, stride, stride, 1],
            strides=[1, stride, stride, stride, 1],
            padding='SAME',
            name=scope)


def pad_last_dimension(tensor, new_last_dim):
    last_dim = utils.last_dimension(tensor.get_shape())
    if last_dim != new_last_dim:
        pad = int((new_last_dim - last_dim) // 2)
        return tf.pad(tensor, [[0, 0], [0, 0], [0, 0], [0, 0], [pad, pad]])
    return tensor


def res_net(
        features,
        res_blocks_size,
        num_classes,
        res_func,
        multi_k=1,
        is_training=True,
        keep_prob=1.0,
        is_add_multiplier=False,
        is_double_size=False,
        scope=None):
    with tf.variable_scope(scope, 'drl_mp', [features]):
        # 128 / 1
        net = features

        with tf.variable_scope('init'):
            first_output = np.array(res_blocks_size).flatten()[0]
            if(first_output < 16):
                raise ValueError("The first output size should be equal or greater than 16: {0}".format(first_output))

            net = layers.batch_norm(
                inputs=net,
                activation_fn=tf.nn.relu,
                is_training=is_training)

            # 64 / 4
            net = layers.convolution(
                inputs=net,
                num_outputs=4,
                kernel_size=[1, 1, 1],
                stride=2)

            if not is_double_size:
                # 32 / 8
                net = ops.resnet_reduction(
                    input=net,
                    kernel_size=3,
                    is_training=is_training)

            # 16 / 16
            net = ops.resnet_reduction(
                input=net,
                kernel_size=3,
                is_training=is_training)

        kpc = _KeepProbCalc(
            min_keep_prob=keep_prob,
            total_block_num=sum(len(x) for x in res_blocks_size))

        for block_num, block in enumerate(res_blocks_size):
            with tf.variable_scope(str.format('res_{0}', (block_num + 1))):
                for b_num, b in enumerate(block):
                    is_half_size = (block_num != 0 and b_num == 0)
                    residual_scope = str.format('res_{0}_{1}', (block_num + 1), (b_num + 1))
                    net = multi_residual(
                        inputs=net,
                        num_outputs=b,
                        multi_k=multi_k,
                        res_func=res_func,
                        scope=residual_scope,
                        keep_prob=kpc.get_next_decr_prob(),
                        is_training=is_training,
                        is_half_size=is_half_size,
                        is_add_multiplier=is_add_multiplier)

        with tf.variable_scope('unit_last'):
            net = layers.batch_norm(
                inputs=net,
                activation_fn=tf.nn.relu,
                is_training=is_training)
            net = tf.reduce_mean(net, [1, 2, 3])
            net = layers.fully_connected(
                inputs=net,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                biases_initializer=tf.constant_initializer())

        return net


class _KeepProbCalc:
    def __init__(self, min_keep_prob, total_block_num):
        if total_block_num <= 0 or min_keep_prob <= 0:
            raise ValueError()
        self.min_keep_prob = min_keep_prob
        self.total_block_num = total_block_num
        self.block_passed_num = 0

    def get_next_decr_prob(self):
        result = 1.0 - self.block_passed_num / self.total_block_num * (1 - self.min_keep_prob)
        self.block_passed_num += 1.0
        assert result >= self.min_keep_prob
        return result

    def reset(self):
        self.block_passed_num = 0


def multi_residual(
        inputs,
        num_outputs,
        multi_k,
        res_func,
        scope,
        keep_prob=1.0,
        is_training=True,
        is_half_size=False,
        is_add_multiplier=False):
    with tf.variable_scope(scope, 'multi_residual', [inputs]):
        orig_x = inputs
        stride = 2 if is_half_size else 1
        orig_x = subsample(orig_x, stride, 'shortcut')
        orig_x = pad_last_dimension(orig_x, num_outputs)
        if is_add_multiplier:
            orig_x = ops.add_multiplier(orig_x)

        result = orig_x
        for k in range(multi_k):
            res_func_result = res_func(
                input=inputs,
                num_outputs=num_outputs,
                is_training=is_training,
                is_half_size=is_half_size,
                scope=str.format('{0}_{1}', scope, (k + 1)))
            if is_add_multiplier:
                res_func_result = ops.add_multiplier(res_func_result)
            res_func_result = ops.residual_dropout(res_func_result, keep_prob, is_training)
            result += res_func_result
        return result


def res_net_model(
        features,
        targets,
        res_blocks_size,
        mode,
        num_classes,
        multi_k=1,
        keep_prob=1.0,
        optimizer_type='SGD',
        learning_rate=0.001,
        is_add_multiplier=False,
        is_double_size=False,
        scope=None):

    net = res_net(
        features=features,
        res_blocks_size=res_blocks_size,
        num_classes=num_classes,
        res_func=ops.residual_v1,
        multi_k=multi_k,
        is_training=mode == tf.contrib.learn.ModeKeys.TRAIN,
        keep_prob=keep_prob,
        is_add_multiplier=is_add_multiplier,
        is_double_size=is_double_size,
        scope=scope)

    predictions = _logits_to_predictions(net)

    targets = tf.one_hot(targets, depth=num_classes)
    targets = tf.reshape(targets, [-1, num_classes])
    loss = tf.contrib.losses.softmax_cross_entropy(net, targets)

    train_op = layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer=optimizer_type,
        learning_rate=learning_rate)

    return predictions, loss, train_op


def _logits_to_predictions(logits):
    return {
        PredictionKey.LOGITS: logits,
        PredictionKey.PROBABILITIES: tf.nn.softmax(logits, name=PredictionKey.PROBABILITIES),
        PredictionKey.CLASSES: tf.argmax(logits, 1, name=PredictionKey.CLASSES)}


def _calc_pyramidal_resnet_blocks(num_start, num_end, num_blocks):
    result = np.floor(np.arange(num_blocks - 1) * (num_end - num_start) / (num_blocks - 1)) + num_start
    result = np.concatenate((result, [num_end]))
    result = pd.Series(result)
    result[result % 2 != 0] = result - 1
    result = result.astype(np.int32)
    return result.tolist()


def res_net_pyramidal_model(
        features,
        targets,
        mode,
        num_classes,
        num_blocks = 10,
        multi_k=1,
        keep_prob=1.0,
        optimizer_type='SGD',
        learning_rate=0.001,
        is_add_multiplier=False,
        groups=None,
        is_double_size=False,
        scope=None):
    """ Deep Pyramidal Residual Networks
    From https://arxiv.org/abs/1610.02915
    """
    groups = groups or [16, 160, 320, 640]
    res_blocks_size = []

    for i in range(len(groups) - 1):
        res_blocks_size.append(_calc_pyramidal_resnet_blocks(
            groups[i],
            groups[i + 1],
            num_blocks))

    return res_net_model(
        features=features,
        targets=targets,
        res_blocks_size=res_blocks_size,
        mode=mode,
        num_classes=num_classes,
        multi_k=multi_k,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        keep_prob=keep_prob,
        is_add_multiplier=is_add_multiplier,
        is_double_size=is_double_size,
        scope=scope)


def res_net_wide_model(
        features,
        targets,
        mode,
        num_classes,
        num_blocks = 10,
        multi_k=1,
        keep_prob=1.0,
        optimizer_type='SGD',
        learning_rate=0.001,
        is_add_multiplier=False,
        groups=None,
        scope=None):
    """ Wide Residual Networks
    from https://arxiv.org/pdf/1605.07146v1.pdf
    """
    groups = groups or [160, 320, 640]
    res_blocks_size = []

    for g in groups:
        res_blocks_size.append([g] * num_blocks)

    return res_net_model(
        features=features,
        targets=targets,
        res_blocks_size=res_blocks_size,
        mode=mode,
        num_classes=num_classes,
        multi_k=multi_k,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        keep_prob=keep_prob,
        is_add_multiplier=is_add_multiplier,
        scope=scope)
