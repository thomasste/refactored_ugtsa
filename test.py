import tensorflow as tf
import numpy as np
from model_builders.basic_model_builder import BasicModelBuilder


def assert_equal(a, b):
    np.testing.assert_equal(a, b)


def assert_unequal(a, b):
    assert np.any(np.not_equal(a, b))


def assert_zero(a):
    assert not np.any(a)


def assert_nonzero(a):
    assert np.any(a)


def test_output(output, feed_dict, training):
    o0 = session.run(output, {**feed_dict, training: False})
    o1 = session.run(output, {**feed_dict, training: True})

    assert_nonzero(o0)
    assert_nonzero(o1)


def test_batch_normalization(output, feed_dict, training):
    o0 = session.run(output, {**feed_dict, training: False})
    o1 = session.run(output, {**feed_dict, training: True})
    o2 = session.run(output, {**feed_dict, training: True})
    o3 = session.run(output, {**feed_dict, training: False})

    assert_equal(o1, o2)
    assert_unequal(o0, o3)


def test_preserving_batch_normalization_state(output, feed_dict, training):
    s0 = session.run(collected_untrainable_variables)
    o0 = session.run(output, {**feed_dict, training: False})
    s1 = session.run(collected_untrainable_variables)
    o1 = session.run(output, {**feed_dict, training: True})
    s2 = session.run(collected_untrainable_variables)
    o2 = session.run(output, {**feed_dict, training: False})
    session.run(set_untrainable_variables, {set_untrainable_variables_input: s0})
    s3 = session.run(collected_untrainable_variables)
    o3 = session.run(output, {**feed_dict, training: False})

    assert_equal(s0, s1)
    assert_unequal(o0, o1)
    assert_unequal(s1, s2)
    assert_unequal(o0, o2)
    assert_equal(s0, s3)
    assert_equal(o0, o3)


def test_input_gradients(feed_dict, output_gradient, output_shape, input_gradients, training):
    os = session.run(input_gradients, {**feed_dict, output_gradient: np.ones(output_shape), training: True})

    for o in os:
        assert_nonzero(o)


def test_gradient_accumulators(feed_dict, output_gradient, output_shape, training, gradient_accumulators, update_gradient_accumulators, zero_gradient_accumulators):
    session.run(zero_gradient_accumulators)
    ga0 = session.run(gradient_accumulators)
    session.run(update_gradient_accumulators, {**feed_dict, training: True, output_gradient: np.ones(output_shape)})
    ga1 = session.run(gradient_accumulators)
    session.run(zero_gradient_accumulators)
    ga2 = session.run(gradient_accumulators)

    for ga in ga0:
        assert_zero(ga)

    for ga in ga1:
        print(np.any(ga))
        assert_nonzero(ga)

    for ga in ga2:
        assert_zero(ga)


model_builder = BasicModelBuilder(
    statistic_size=150,
    update_size=150,
    board_shape=[7, 7],
    game_state_info_size=2,
    payoff_size=2,
    player_count=2,
    worker_count=5,
    statistic_filter_shapes=[(2, 2, 16), (2, 2, 32)],
    statistic_window_shapes=[(1, 2, 2, 1), (1, 2, 2, 1)],
    statistic_hidden_output_sizes=[25, 25, 25],
    update_hidden_output_sizes=[25, 25, 25],
    modified_statistic_lstm_state_sizes=[25, 25, 25],
    modified_update_hidden_output_sizes=[25, 25, 25],
    move_rate_hidden_output_sizes=[25, 25, 25])

model_builder.build()

graph = tf.get_default_graph()

collected_untrainable_variables = graph.get_tensor_by_name('collected_untrainable_variables:0')
set_untrainable_variables_input = graph.get_tensor_by_name('set_untrainable_variables_input:0')
set_untrainable_variables = graph.get_operation_by_name('set_untrainable_variables')

# statistic
statistic_seed = graph.get_tensor_by_name('statistic/seed:0')
statistic_training = graph.get_tensor_by_name('statistic/training:0')
statistic_board = graph.get_tensor_by_name('statistic/board:0')
statistic_game_state_info = graph.get_tensor_by_name('statistic/game_state_info:0')

statistic_input_gradients = []

statistic_output = graph.get_tensor_by_name('statistic/output:0')
statistic_output_gradient = graph.get_tensor_by_name('statistic/output_gradient:0')

statistic_gradient_accumulators = graph.get_collection('statistic/gradient_accumulators')
statistic_update_gradient_accumulators = graph.get_operation_by_name('statistic/update_gradient_accumulators')
statistic_zero_gradient_accumulators = graph.get_operation_by_name('statistic/zero_gradient_accumulators')

statistic_feed_dict = {
    statistic_seed: [1, 2],
    statistic_training: True,
    statistic_board: [
        [[1 if x % 2 else 0 for x in range(7)] for y in range(7)],
        [[1 if x % 3 else 0 for x in range(7)] for y in range(7)]],
    statistic_game_state_info: [[0, 0], [1, 1]],
}

# update
update_seed = graph.get_tensor_by_name('update/seed:0')
update_training = graph.get_tensor_by_name('update/training:0')
update_payoff = graph.get_tensor_by_name('update/payoff:0')

update_input_gradients = []

update_output = graph.get_tensor_by_name('update/output:0')
update_output_gradient = graph.get_tensor_by_name('update/output_gradient:0')

update_gradient_accumulators = graph.get_collection('update/gradient_accumulators')
update_update_gradient_accumulators = graph.get_operation_by_name('update/update_gradient_accumulators')
update_zero_gradient_accumulators = graph.get_operation_by_name('update/zero_gradient_accumulators')

update_feed_dict = {
    update_seed: [3, 4],
    update_training: True,
    update_payoff: [[-10, 23], [13, 12]],
}

# modified_statistic
modified_statistic_seed = graph.get_tensor_by_name('modified_statistic/seed:0')
modified_statistic_training = graph.get_tensor_by_name('modified_statistic/training:0')
modified_statistic_statistic = graph.get_tensor_by_name('modified_statistic/statistic:0')
modified_statistic_updates_count = graph.get_tensor_by_name('modified_statistic/updates_count:0')
modified_statistic_updates = graph.get_tensor_by_name('modified_statistic/updates:0')

modified_statistic_input_gradients = [
    graph.get_tensor_by_name('modified_statistic/statistic_gradient:0'),
    graph.get_tensor_by_name('modified_statistic/updates_gradient:0'),
]

modified_statistic_output = graph.get_tensor_by_name('modified_statistic/output:0')
modified_statistic_output_gradient = graph.get_tensor_by_name('modified_statistic/output_gradient:0')

modified_statistic_gradient_accumulators = graph.get_collection('modified_statistic/gradient_accumulators')
modified_statistic_update_gradient_accumulators = graph.get_operation_by_name('modified_statistic/update_gradient_accumulators')
modified_statistic_zero_gradient_accumulators = graph.get_operation_by_name('modified_statistic/zero_gradient_accumulators')

modified_statistic_feed_dict = {
    modified_statistic_seed: [5, 6],
    modified_statistic_training: True,
    modified_statistic_statistic: [[(x + j) % 7 for x in range(150)] for j in range(2)],
    modified_statistic_updates_count: [3, 4],
    modified_statistic_updates: [[(x + j) % 3 for x in range(150 * 5)] for j in range(2)]
}

# modified_update
modified_update_seed = graph.get_tensor_by_name('modified_update/seed:0')
modified_update_training = graph.get_tensor_by_name('modified_update/training:0')
modified_update_update = graph.get_tensor_by_name('modified_update/update:0')
modified_update_statistic = graph.get_tensor_by_name('modified_update/statistic:0')

modified_update_input_gradients = [
    graph.get_tensor_by_name('modified_update/update_gradient:0'),
    graph.get_tensor_by_name('modified_update/statistic_gradient:0'),
]

modified_update_output = graph.get_tensor_by_name('modified_update/output:0')
modified_update_output_gradient = graph.get_tensor_by_name('modified_update/output_gradient:0')

modified_update_gradient_accumulators = graph.get_collection('modified_update/gradient_accumulators')
modified_update_update_gradient_accumulators = graph.get_operation_by_name('modified_update/update_gradient_accumulators')
modified_update_zero_gradient_accumulators = graph.get_operation_by_name('modified_update/zero_gradient_accumulators')

modified_update_feed_dict = {
    modified_update_seed: [7, 8],
    modified_update_training: True,
    modified_update_update: [[(x + j) % 7 for x in range(150)] for j in range(2)],
    modified_update_statistic: [[(x + j) % 3 for x in range(150)] for j in range(2)],
}

# move_rate
move_rate_seed = graph.get_tensor_by_name('move_rate/seed:0')
move_rate_training = graph.get_tensor_by_name('move_rate/training:0')
move_rate_parent_statistic = graph.get_tensor_by_name('move_rate/parent_statistic:0')
move_rate_child_statistic = graph.get_tensor_by_name('move_rate/child_statistic:0')

move_rate_input_gradients = [
    graph.get_tensor_by_name('move_rate/parent_statistic_gradient:0'),
    graph.get_tensor_by_name('move_rate/child_statistic_gradient:0'),
]

move_rate_output = graph.get_tensor_by_name('move_rate/output:0')
move_rate_output_gradient = graph.get_tensor_by_name('move_rate/output_gradient:0')

move_rate_gradient_accumulators = graph.get_collection('move_rate/gradient_accumulators')
move_rate_update_gradient_accumulators = graph.get_operation_by_name('move_rate/update_gradient_accumulators')
move_rate_zero_gradient_accumulators = graph.get_operation_by_name('move_rate/zero_gradient_accumulators')

move_rate_feed_dict = {
    move_rate_seed: [9, 10],
    move_rate_training: True,
    move_rate_parent_statistic: [[(x + j) % 7 for x in range(150)] for j in range(2)],
    move_rate_child_statistic: [[(x + j) % 3 for x in range(150)] for j in range(2)],
}

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # statistic
    test_output(statistic_output, statistic_feed_dict, statistic_training)
    test_batch_normalization(statistic_output, statistic_feed_dict, statistic_training)
    test_preserving_batch_normalization_state(statistic_output, statistic_feed_dict, statistic_training)
    test_input_gradients(statistic_feed_dict, statistic_output_gradient, [2, 150], statistic_input_gradients, statistic_training)
    test_gradient_accumulators(
        statistic_feed_dict, statistic_output_gradient, [2, 150], statistic_training, statistic_gradient_accumulators,
        statistic_update_gradient_accumulators, statistic_zero_gradient_accumulators)

    # update
    test_output(update_output, update_feed_dict, update_training)
    test_batch_normalization(update_output, update_feed_dict, update_training)
    test_preserving_batch_normalization_state(update_output, update_feed_dict, update_training)
    test_input_gradients(update_feed_dict, update_output_gradient, [2, 150], update_input_gradients,
                         update_training)
    test_gradient_accumulators(
        update_feed_dict, update_output_gradient, [2, 150], update_training, update_gradient_accumulators,
        update_update_gradient_accumulators, update_zero_gradient_accumulators)

    # modified_statistic
    test_output(modified_statistic_output, modified_statistic_feed_dict, modified_statistic_training)
    # test_batch_normalization(modified_statistic_output, modified_statistic_feed_dict, modified_statistic_training)
    # test_preserving_batch_normalization_state(modified_statistic_output, modified_statistic_feed_dict, modified_statistic_training)
    test_input_gradients(modified_statistic_feed_dict, modified_statistic_output_gradient, [2, 150], modified_statistic_input_gradients,
                         modified_statistic_training)
    test_gradient_accumulators(
        modified_statistic_feed_dict, modified_statistic_output_gradient, [2, 150], modified_statistic_training, modified_statistic_gradient_accumulators,
        modified_statistic_update_gradient_accumulators, modified_statistic_zero_gradient_accumulators)

    # modified_update
    test_output(modified_update_output, modified_update_feed_dict, modified_update_training)
    test_batch_normalization(modified_update_output, modified_update_feed_dict, modified_update_training)
    test_preserving_batch_normalization_state(modified_update_output, modified_update_feed_dict, modified_update_training)
    test_input_gradients(modified_update_feed_dict, modified_update_output_gradient, [2, 150], modified_update_input_gradients,
                         modified_update_training)
    test_gradient_accumulators(
        modified_update_feed_dict, modified_update_output_gradient, [2, 150], modified_update_training, modified_update_gradient_accumulators,
        modified_update_update_gradient_accumulators, modified_update_zero_gradient_accumulators)

    # move_rate
    test_output(move_rate_output, move_rate_feed_dict, move_rate_training)
    test_batch_normalization(move_rate_output, move_rate_feed_dict, move_rate_training)
    test_preserving_batch_normalization_state(move_rate_output, move_rate_feed_dict, move_rate_training)
    test_input_gradients(move_rate_feed_dict, move_rate_output_gradient, [2, 2], move_rate_input_gradients,
                         move_rate_training)
    test_gradient_accumulators(
        move_rate_feed_dict, move_rate_output_gradient, [2, 2], move_rate_training, move_rate_gradient_accumulators,
        move_rate_update_gradient_accumulators, move_rate_zero_gradient_accumulators)



