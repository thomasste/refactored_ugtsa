import tensorflow as tf
from model_builders.model_builder import ModelBuilder
from tensorflow.contrib.layers import layer_norm


class VerticalLSTMModelBuilder(ModelBuilder):
    def __init__(self, model_builder, normalize, modified_update_lstm_state_sizes):
        super().__init__(
            model_builder.statistic_size, model_builder.update_size,
            model_builder.board_shape, model_builder.game_state_info_size,
            model_builder.payoff_size, model_builder.player_count,
            model_builder.worker_count)

        self.model_builder = model_builder
        self.normalize = normalize
        self.modified_update_lstm_state_sizes = \
            modified_update_lstm_state_sizes

    def statistic(self, rng, training, board, game_state_info):
        return self.model_builder.statistic(
            rng, training, board, game_state_info)

    def update(self, rng, training, payoff):
        return self.model_builder.update(rng, training, payoff)

    def modified_statistic(
            self, rng, training, statistic, updates_count, updates):
        return self.model_builder.modified_statistic(
            rng, training, statistic, updates_count, updates)

    def modified_update(self, rng, training, update, statistic):
        split = tf.split(
            update, self.modified_update_lstm_state_sizes * 2, 1)

        cs = split[:len(self.modified_update_lstm_state_sizes)]
        hs = split[len(self.modified_update_lstm_state_sizes):]

        input = statistic
        ncs = []
        nhs = []

        for layer_idx, (state_size, c, h) in enumerate(
                zip(self.modified_update_lstm_state_sizes, cs, hs)):
            with tf.variable_scope('lstm_layer_{}'.format(layer_idx)):
                cell = tf.contrib.rnn.LSTMCell(state_size)
                print(input.get_shape(), c.get_shape(), h.get_shape())
                input, lstm_state = cell(input, [c, h])
                ncs += [layer_norm(lstm_state.c) if self.normalize else lstm_state.c]
                nhs += [layer_norm(lstm_state.h) if self.normalize else lstm_state.h]

        return tf.concat(ncs + nhs, axis=1)

    def move_rate(self, rng, training, parent_statistic, child_statistic):
        return self.model_builder.move_rate(
            rng, training, parent_statistic, child_statistic)

    def cost_function(self, rng, training, logits, labels):
        return self.model_builder.cost_function(
            rng, training, logits, labels)

    def apply_gradients(self, global_step, grads_and_vars):
        return self.model_builder.apply_gradients(global_step, grads_and_vars)
