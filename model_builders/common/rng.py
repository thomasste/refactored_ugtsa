import tensorflow as tf
from tensorflow.contrib.stateless import stateless_random_uniform


class RNG:
    def __init__(self, state):
        self.state = state

    def random_uniform(self, shape, dtype=None, name=None):
        self.state, seed = tf.split(
            tf.bitcast(stateless_random_uniform([4], self.state, tf.float64), tf.int64), 2)

        return stateless_random_uniform(shape, seed, dtype, name)
