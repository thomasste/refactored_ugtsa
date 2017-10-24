import tensorflow as tf
import numpy as np
from model_builders import model_builder
from model_builders.common.neural_networks import *


class BasicModelBuilder(model_builder.ModelBuilder):
    def __init__(self, statistic_size, update_size,
                 board_shape, game_state_info_size,
                 payoff_size, player_count, worker_count,
                 statistic_filter_shapes,
                 statistic_window_shapes,
                 statistic_hidden_output_sizes,
                 update_hidden_output_sizes,
                 modified_statistic_lstm_state_sizes,
                 modified_update_hidden_output_sizes,
                 move_rate_hidden_output_sizes):
        super().__init__(
            statistic_size, update_size,
            board_shape, game_state_info_size,
            payoff_size, player_count, worker_count)

        self.statistic_filter_shapes = statistic_filter_shapes
        self.statistic_window_shapes = statistic_window_shapes
        self.statistic_hidden_output_sizes = \
            statistic_hidden_output_sizes
        self.update_hidden_output_sizes = \
            update_hidden_output_sizes
        self.modified_statistic_lstm_state_sizes = \
            modified_statistic_lstm_state_sizes
        self.modified_update_hidden_output_sizes = \
            modified_update_hidden_output_sizes
        self.move_rate_hidden_output_sizes = \
            move_rate_hidden_output_sizes

    def statistic(self, rng, training, board, game_state_info):
        signal = tf.expand_dims(board, -1)
        print(signal.get_shape())

        signal = convolutional_neural_network(
            self.statistic_filter_shapes,
            self.statistic_window_shapes, signal)

        signal = tf.reshape(
            signal, (-1, np.prod(signal.get_shape().as_list()[1:])))
        print(signal.get_shape())

        signal = tf.concat([signal, game_state_info], axis=1)
        print(signal.get_shape())

        return dense_neural_network(
            rng, training, 0.5,
            self.statistic_hidden_output_sizes + [self.statistic_size],
            signal)

    def update(self, rng, training, payoff):
        return dense_neural_network(
            rng, training, 0.5,
            self.update_hidden_output_sizes + [self.update_size],
            payoff)

    def modified_statistic(
            self, rng, training, statistic, updates_count, updates):
        max_i = tf.reduce_max(updates_count)

        def cond(i):
            return i < max_i

        def body(i, cs, hs, updates):
            input = updates[:i * self.update_size:(i+1) * self.update_size]

            ncs = []
            nhs = []

            for i, (state_size, c, h) in enumerate(
                    zip(self.modified_statistic_lstm_state_sizes, cs, hs)):
                with tf.variable_scope('lstm_layer_{}'.format(i)):
                    cell = tf.contrib.rnn.LSTMCell(state_size)
                    input, lstm_state = cell(input, [c, h])
                    ncs += [lstm_state.c]
                    nhs += [lstm_state.h]

            return [i+1, ncs, nhs, updates]

        split = tf.split(statistic, self.modified_statistic_lstm_state_sizes * 2, 1)
        cs0 = split[:len(self.modified_statistic_lstm_state_sizes)]  # c
        hs0 = split[len(self.modified_statistic_lstm_state_sizes):]  # h

        _, ncs, nhs, _ = tf.while_loop(cond, body, loop_vars=[0, cs0, hs0, updates])

        return tf.concat(ncs + nhs, axis=1)

    def modified_update(self, rng, training, update, statistic):
        signal = tf.concat([update, statistic], axis=1)
        print(signal.get_shape())
        return dense_neural_network(
            rng, training, 0.5,
            self.modified_update_hidden_output_sizes + [self.update_size],
            signal)

    # ! trzeba aplikować softmax przed użyciem
    def move_rate(self, rng, training, parent_statistic, child_statistic):
        signal = tf.concat([parent_statistic, child_statistic], axis=1)
        print(signal.get_shape())
        signal = dense_neural_network(
            rng, training, 0.5, self.move_rate_hidden_output_sizes, signal)
        signal = dense_layer(signal, self.player_count)
        signal = bias_layer(signal)
        print(signal.get_shape())
        return signal

    def cost_function(self, rng, training, logits, labels):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.nn.softmax(labels)))

    def apply_gradients(self, global_step, grads_and_vars):
        return tf.train.AdamOptimizer().apply_gradients(
            grads_and_vars, global_step, 'apply_gradients')