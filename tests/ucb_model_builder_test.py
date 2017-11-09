import numpy as np
import tensorflow as tf
from model_builders.ucb_model_builder import UCBModelBuilder

model_builder = UCBModelBuilder(
    board_shape=[1, 1],
    game_state_info_size=1,
    player_count=2,
    worker_count=5)

model_builder.build()

graph = tf.get_default_graph()

# statistic
statistic_seed = graph.get_tensor_by_name('statistic/seed:0')
statistic_training = graph.get_tensor_by_name('statistic/training:0')
statistic_board = graph.get_tensor_by_name('statistic/board:0')
statistic_game_state_info = graph.get_tensor_by_name(
    'statistic/game_state_info:0')

statistic_output = graph.get_tensor_by_name('statistic/output:0')

# update
update_seed = graph.get_tensor_by_name('update/seed:0')
update_training = graph.get_tensor_by_name('update/training:0')
update_payoff = graph.get_tensor_by_name('update/payoff:0')

update_output = graph.get_tensor_by_name('update/output:0')

# modified_statistic
modified_statistic_seed = graph.get_tensor_by_name(
    'modified_statistic/seed:0')
modified_statistic_training = graph.get_tensor_by_name(
    'modified_statistic/training:0')
modified_statistic_statistic = graph.get_tensor_by_name(
    'modified_statistic/statistic:0')
modified_statistic_updates_count = graph.get_tensor_by_name(
    'modified_statistic/updates_count:0')
modified_statistic_updates = graph.get_tensor_by_name(
    'modified_statistic/updates:0')

modified_statistic_output = graph.get_tensor_by_name(
    'modified_statistic/output:0')

# modified_update
modified_update_seed = graph.get_tensor_by_name('modified_update/seed:0')
modified_update_training = graph.get_tensor_by_name(
    'modified_update/training:0')
modified_update_update = graph.get_tensor_by_name('modified_update/update:0')
modified_update_statistic = graph.get_tensor_by_name(
    'modified_update/statistic:0')

modified_update_output = graph.get_tensor_by_name('modified_update/output:0')

# move_rate
move_rate_seed = graph.get_tensor_by_name('move_rate/seed:0')
move_rate_training = graph.get_tensor_by_name('move_rate/training:0')
move_rate_parent_statistic = graph.get_tensor_by_name(
    'move_rate/parent_statistic:0')
move_rate_child_statistic = graph.get_tensor_by_name(
    'move_rate/child_statistic:0')

move_rate_output = graph.get_tensor_by_name('move_rate/output:0')

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    assert np.array_equal(
        session.run(statistic_output, {
            statistic_board: [[[1]], [[2]]],
            statistic_game_state_info: [[1], [2]]
        }), [[0., 0., 0.], [0., 0., 0.]])

    assert np.array_equal(
        session.run(update_output, {
            update_payoff: [[1, 2], [3, 1], [-1, 2]]
        }), [[1, 2], [3, 1], [-1, 2]])

    assert np.array_equal(
        session.run(modified_statistic_output, {
            modified_statistic_statistic: [[0., 1., 2.], [3., 4., 5.]],
            modified_statistic_updates_count: [1, 2],
            modified_statistic_updates: [[1, 2] + ([1] * 8), [1, 0, 2, 1] + ([1] * 6)]
        }), [[0.,  2.,  3.], [5., 4., 7.]])

    assert np.array_equal(
        session.run(modified_update_output, {
            modified_update_update: [[1, 2], [3, 1], [-1, 2]]
        }), [[1, 2], [3, 1], [-1, 2]])

    assert np.allclose(
        session.run(move_rate_output, {
            move_rate_parent_statistic: [[0.,  2.,  3.], [5., 4., 7.]],
            move_rate_child_statistic: [[1.,  1.,  3.], [4., 2., 6.]]
        }), [[1.08255458, 1.08255458], [1.34222436, 1.05651009]])

