from model_builders.model_builder import ModelBuilder
import tensorflow as tf


class UCBModelBuilder(ModelBuilder):
    def __init__(self, board_shape, game_state_info_size,
                 player_count, worker_count):
        self.statistic_size = player_count + 1
        self.update_size = player_count
        self.board_shape = \
            board_shape
        self.game_state_info_size = \
            game_state_info_size
        self.payoff_size = player_count
        self.player_count = player_count
        self.worker_count = worker_count

    def statistic(self, rng, training, board, game_state_info):
        # make get set untrainable variables happy
        tf.Variable(0., trainable=False)

        zeros_dims = tf.stack(
            [tf.shape(game_state_info)[0], self.player_count + 1])
        return tf.fill(zeros_dims, 0.0)

    def update(self, rng, training, payoff):
        return payoff

    def modified_statistic(
            self, rng, training, statistic, updates_count, updates):
        def cond(i, statistic, updates):
            return i < self.worker_count
        #
        def body(i, statistic, updates):
            input = tf.reshape(
                updates[:, i * self.update_size:(i+1) * self.update_size],
                (-1, self.update_size))
            argmax = tf.argmax(input, 1)
            one_hot = tf.one_hot(argmax, self.player_count + 1)

            new_statistic = \
                tf.where(
                    updates_count > i,
                    statistic + one_hot +
                        tf.one_hot([self.player_count], self.player_count + 1),
                    statistic)

            return [i + 1, new_statistic, updates]

        _, new_statistic, _ = tf.while_loop(
            cond, body, loop_vars=[tf.constant(0), statistic, updates])

        return new_statistic

    def modified_update(self, rng, training, update, statistic):
        return update

    def move_rate(self, rng, training, parent_statistic, child_statistic):
        _, pn = tf.split(parent_statistic, [self.player_count, 1], 1)
        w, cn = tf.split(child_statistic, [self.player_count, 1], 1)

        return (w / (cn + 1.)) + tf.sqrt(2.) * tf.sqrt(tf.log(pn + 1.) / (cn + 1.))

    def cost_function(self, rng, training, logits, labels):
        return tf.ones(shape=[])

    def apply_gradients(self, global_step, grads_and_vars):
        pass
