import tensorflow as tf
from tensorflow.contrib.layers import layer_norm


def dense_layer(output_size, signal):
    layer_shape = [signal.get_shape()[1].value, output_size]
    layer = tf.get_variable(
        'layer', layer_shape, tf.float32,
        tf.contrib.layers.xavier_initializer())
    signal = tf.matmul(signal, layer)
    return signal


def bias_layer(signal):
    bias = tf.get_variable(
        'bias', signal.get_shape()[1:], tf.float32,
        tf.zeros_initializer())
    return signal + bias


def activation_layer(signal):
    return tf.nn.relu(signal)


def dropout_layer(rng, training, keep_prob, signal):
    random = rng.random_uniform(tf.shape(signal), signal.dtype)
    mask = tf.to_float(random < keep_prob)
    return tf.cond(
        training,
        lambda: (signal * mask) / tf.sqrt(keep_prob),
        lambda: signal)


def batch_normalization_layer(training, signal):
    signal = tf.layers.batch_normalization(signal, training=training)
    with tf.control_dependencies(
            tf.get_collection(
                tf.GraphKeys.UPDATE_OPS,
                tf.get_variable_scope().name)):
        return tf.identity(signal)


def layer_normalization_layer(signal):
    return layer_norm(signal)

def dense_neural_network(
        rng, training, keep_prob, output_sizes, signal, normalize):
    for idx, output_size in enumerate(output_sizes):
        with tf.variable_scope('dense_layer_{}'.format(idx)):
            signal = dense_layer(output_size, signal)
            signal = bias_layer(signal)
            if normalize:
                signal = layer_normalization_layer(signal)
            signal = activation_layer(signal)
            signal = dropout_layer(rng, training, keep_prob, signal)
            print(signal.get_shape())
    return signal


def convolutional_layer(signal, filter_shape):
    filter_shape = [filter_shape[0], filter_shape[1],
                    signal.get_shape()[3].value, filter_shape[2]]
    filter = tf.get_variable(
        'filter', filter_shape, tf.float32,
        tf.contrib.layers.xavier_initializer())
    signal = tf.nn.conv2d(
        input=signal, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
    return signal


def max_pool_layer(signal, window_shape):
    return tf.nn.max_pool(
        signal, ksize=window_shape, strides=window_shape, padding='SAME')


def convolutional_neural_network(
        training, filter_shapes, window_shapes, signal, normalize):
    for idx, (filter_shape, window_shape) in enumerate(
            zip(filter_shapes, window_shapes)):
        with tf.variable_scope('convolutional_layer_{}'.format(idx)):
            signal = convolutional_layer(signal, filter_shape)
            signal = activation_layer(signal)
            if normalize:
                signal = batch_normalization_layer(training, signal)
            print(signal.get_shape())
            signal = max_pool_layer(signal, window_shape)
            print(signal.get_shape())
    return signal
