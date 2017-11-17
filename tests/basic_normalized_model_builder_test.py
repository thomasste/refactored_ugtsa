import tensorflow as tf
from tests.common import *
from model_builders.basic_model_builder import BasicModelBuilder

model_builder = BasicModelBuilder(
    statistic_size=150,
    update_size=150,
    board_shape=[7, 7],
    game_state_info_size=2,
    payoff_size=2,
    player_count=2,
    worker_count=5,
    normalize=True,
    statistic_filter_shapes=[(2, 2, 16), (2, 2, 32)],
    statistic_window_shapes=[(1, 2, 2, 1), (1, 2, 2, 1)],
    statistic_hidden_output_sizes=[25, 25, 25],
    update_hidden_output_sizes=[25, 25, 25],
    modified_statistic_lstm_state_sizes=[25, 25, 25],
    modified_update_hidden_output_sizes=[25, 25, 25],
    move_rate_hidden_output_sizes=[25, 25, 25])

model_builder.build()

graph = tf.get_default_graph()

collected_untrainable_variables = graph.get_tensor_by_name(
    'collected_untrainable_variables:0')
set_untrainable_variables_input = graph.get_tensor_by_name(
    'set_untrainable_variables_input:0')
set_untrainable_variables = graph.get_operation_by_name(
    'set_untrainable_variables')

zero_gradient_accumulators = graph.get_operation_by_name(
    'zero_gradient_accumulators')

# statistic
statistic_seed = graph.get_tensor_by_name('statistic/seed:0')
statistic_training = graph.get_tensor_by_name('statistic/training:0')
statistic_board = graph.get_tensor_by_name('statistic/board:0')
statistic_game_state_info = graph.get_tensor_by_name(
    'statistic/game_state_info:0')

statistic_input_gradients = []

statistic_output = graph.get_tensor_by_name('statistic/output:0')
statistic_output_gradient = graph.get_tensor_by_name(
    'statistic/output_gradient:0')

statistic_gradient_accumulators = graph.get_collection(
    'statistic/gradient_accumulators')
statistic_update_gradient_accumulators = graph.get_operation_by_name(
    'statistic/update_gradient_accumulators')

statistic_feed_dict1 = {
    statistic_seed: [1, 2],
    statistic_training: True,
    statistic_board: [
        [[1 if x % 2 else 0 for x in range(7)] for y in range(7)]],
    statistic_game_state_info: [[0, 0.3]],
}

statistic_feed_dict2 = {
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

update_gradient_accumulators = graph.get_collection(
    'update/gradient_accumulators')
update_update_gradient_accumulators = graph.get_operation_by_name(
    'update/update_gradient_accumulators')

update_feed_dict1 = {
    update_seed: [3, 4],
    update_training: True,
    update_payoff: [[-10, 23]],
}

update_feed_dict2 = {
    update_seed: [3, 4],
    update_training: True,
    update_payoff: [[-10, 23], [13, 12]],
}

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

modified_statistic_input_gradients = [
    graph.get_tensor_by_name('modified_statistic/statistic_gradient:0'),
    graph.get_tensor_by_name('modified_statistic/updates_gradient:0'),
]

modified_statistic_output = graph.get_tensor_by_name(
    'modified_statistic/output:0')
modified_statistic_output_gradient = graph.get_tensor_by_name(
    'modified_statistic/output_gradient:0')

modified_statistic_gradient_accumulators = graph.get_collection(
    'modified_statistic/gradient_accumulators')
modified_statistic_update_gradient_accumulators = graph.get_operation_by_name(
    'modified_statistic/update_gradient_accumulators')

modified_statistic_feed_dict1 = {
    modified_statistic_seed: [5, 6],
    modified_statistic_training: True,
    modified_statistic_statistic:
        [[x % 7 for x in range(150)]],
    modified_statistic_updates_count: [3],
    modified_statistic_updates:
        [[x % 3 for x in range(150 * 5)]]
}

modified_statistic_feed_dict2 = {
    modified_statistic_seed: [5, 6],
    modified_statistic_training: True,
    modified_statistic_statistic:
        [[(x + j) % 7 for x in range(150)] for j in range(2)],
    modified_statistic_updates_count: [3, 4],
    modified_statistic_updates:
        [[(x + j) % 3 for x in range(150 * 5)] for j in range(2)]
}

# modified_update
modified_update_seed = graph.get_tensor_by_name('modified_update/seed:0')
modified_update_training = graph.get_tensor_by_name(
    'modified_update/training:0')
modified_update_update = graph.get_tensor_by_name('modified_update/update:0')
modified_update_statistic = graph.get_tensor_by_name(
    'modified_update/statistic:0')

modified_update_input_gradients = [
    graph.get_tensor_by_name('modified_update/update_gradient:0'),
    graph.get_tensor_by_name('modified_update/statistic_gradient:0'),
]

modified_update_output = graph.get_tensor_by_name('modified_update/output:0')
modified_update_output_gradient = graph.get_tensor_by_name(
    'modified_update/output_gradient:0')

modified_update_gradient_accumulators = graph.get_collection(
    'modified_update/gradient_accumulators')
modified_update_update_gradient_accumulators = graph.get_operation_by_name(
    'modified_update/update_gradient_accumulators')

modified_update_feed_dict1 = {
    modified_update_seed: [7, 8],
    modified_update_training: True,
    modified_update_update:
        [[x % 7 for x in range(150)]],
    modified_update_statistic:
        [[x % 3 for x in range(150)]],
}

modified_update_feed_dict2 = {
    modified_update_seed: [7, 8],
    modified_update_training: True,
    modified_update_update:
        [[(x + j) % 7 for x in range(150)] for j in range(2)],
    modified_update_statistic:
        [[(x + j) % 3 for x in range(150)] for j in range(2)],
}

# move_rate
move_rate_seed = graph.get_tensor_by_name('move_rate/seed:0')
move_rate_training = graph.get_tensor_by_name('move_rate/training:0')
move_rate_parent_statistic = graph.get_tensor_by_name(
    'move_rate/parent_statistic:0')
move_rate_child_statistic = graph.get_tensor_by_name(
    'move_rate/child_statistic:0')

move_rate_input_gradients = [
    graph.get_tensor_by_name('move_rate/parent_statistic_gradient:0'),
    graph.get_tensor_by_name('move_rate/child_statistic_gradient:0'),
]

move_rate_output = graph.get_tensor_by_name('move_rate/output:0')
move_rate_output_gradient = graph.get_tensor_by_name(
    'move_rate/output_gradient:0')

move_rate_gradient_accumulators = graph.get_collection(
    'move_rate/gradient_accumulators')
move_rate_update_gradient_accumulators = graph.get_operation_by_name(
    'move_rate/update_gradient_accumulators')

move_rate_feed_dict1 = {
    move_rate_seed: [9, 10],
    move_rate_training: True,
    move_rate_parent_statistic:
        [[x % 7 for x in range(150)]],
    move_rate_child_statistic:
        [[x % 3 for x in range(150)]],
}

move_rate_feed_dict2 = {
    move_rate_seed: [9, 10],
    move_rate_training: True,
    move_rate_parent_statistic:
        [[(x + j) % 7 for x in range(150)] for j in range(2)],
    move_rate_child_statistic:
        [[(x + j) % 3 for x in range(150)] for j in range(2)],
}

# cost_function
cost_function_seed = graph.get_tensor_by_name('cost_function/seed:0')
cost_function_training = graph.get_tensor_by_name('cost_function/training:0')
cost_function_logits = graph.get_tensor_by_name('cost_function/logits:0')
cost_function_labels = graph.get_tensor_by_name('cost_function/labels:0')

cost_function_input_gradients = [
    graph.get_tensor_by_name('cost_function/logits_gradient:0'),
]

cost_function_output = graph.get_tensor_by_name('cost_function/output:0')
cost_function_output_gradient = graph.get_tensor_by_name(
    'cost_function/output_gradient:0')

cost_function_gradient_accumulators = graph.get_collection(
    'cost_function/gradient_accumulators')
cost_function_update_gradient_accumulators = graph.get_operation_by_name(
    'cost_function/update_gradient_accumulators')

cost_function_feed_dict1 = {
    cost_function_seed: [11, 12],
    cost_function_training: True,
    cost_function_logits: [[-1, 2]],
    cost_function_labels: [[0, 1]],
}

cost_function_feed_dict2 = {
    cost_function_seed: [11, 12],
    cost_function_training: True,
    cost_function_logits: [[-1, 2], [3, 4]],
    cost_function_labels: [[0, 1], [4, 3]],
}

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # statistic
    test_output(
        session, statistic_output, statistic_feed_dict1, statistic_training)
    test_batch_normalization(
        session, statistic_output, statistic_feed_dict1, statistic_training)
    test_preserving_batch_normalization_state(
        session, statistic_output, statistic_feed_dict1, statistic_training,
        collected_untrainable_variables, set_untrainable_variables,
        set_untrainable_variables_input)
    test_input_gradients(
        session, statistic_feed_dict1, statistic_output_gradient, [1, 150],
        statistic_input_gradients, statistic_training)
    test_gradient_accumulators(
        session, statistic_feed_dict1, statistic_output_gradient, [1, 150],
        statistic_training, statistic_gradient_accumulators,
        statistic_update_gradient_accumulators,
        zero_gradient_accumulators)

    test_output(
        session, statistic_output, statistic_feed_dict2, statistic_training)
    test_batch_normalization(
        session, statistic_output, statistic_feed_dict2, statistic_training)
    test_preserving_batch_normalization_state(
        session, statistic_output, statistic_feed_dict2, statistic_training,
        collected_untrainable_variables, set_untrainable_variables,
        set_untrainable_variables_input)
    test_input_gradients(
        session, statistic_feed_dict2, statistic_output_gradient, [2, 150],
        statistic_input_gradients, statistic_training)
    test_gradient_accumulators(
        session, statistic_feed_dict2, statistic_output_gradient, [2, 150],
        statistic_training, statistic_gradient_accumulators,
        statistic_update_gradient_accumulators,
        zero_gradient_accumulators)

    # update
    test_output(session, update_output, update_feed_dict1, update_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, update_feed_dict1, update_output_gradient, [1, 150],
        update_input_gradients, update_training)
    test_gradient_accumulators(
        session, update_feed_dict1, update_output_gradient, [1, 150],
        update_training, update_gradient_accumulators,
        update_update_gradient_accumulators,
        zero_gradient_accumulators)

    test_output(session, update_output, update_feed_dict2, update_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, update_feed_dict2, update_output_gradient, [2, 150],
        update_input_gradients, update_training)
    test_gradient_accumulators(
        session, update_feed_dict2, update_output_gradient, [2, 150],
        update_training, update_gradient_accumulators,
        update_update_gradient_accumulators,
        zero_gradient_accumulators)

    # modified_statistic
    test_output(
        session, modified_statistic_output, modified_statistic_feed_dict1,
        modified_statistic_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, modified_statistic_feed_dict1,
        modified_statistic_output_gradient, [1, 150],
        modified_statistic_input_gradients, modified_statistic_training)
    test_gradient_accumulators(
        session, modified_statistic_feed_dict1,
        modified_statistic_output_gradient, [1, 150],
        modified_statistic_training, modified_statistic_gradient_accumulators,
        modified_statistic_update_gradient_accumulators,
        zero_gradient_accumulators)

    test_output(
        session, modified_statistic_output, modified_statistic_feed_dict2,
        modified_statistic_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, modified_statistic_feed_dict2,
        modified_statistic_output_gradient, [2, 150],
        modified_statistic_input_gradients, modified_statistic_training)
    test_gradient_accumulators(
        session, modified_statistic_feed_dict2,
        modified_statistic_output_gradient, [2, 150],
        modified_statistic_training, modified_statistic_gradient_accumulators,
        modified_statistic_update_gradient_accumulators,
        zero_gradient_accumulators)

    # modified_update
    test_output(
        session, modified_update_output, modified_update_feed_dict1,
        modified_update_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, modified_update_feed_dict1, modified_update_output_gradient,
        [1, 150], modified_update_input_gradients, modified_update_training)
    test_gradient_accumulators(
        session, modified_update_feed_dict1, modified_update_output_gradient,
        [1, 150], modified_update_training,
        modified_update_gradient_accumulators,
        modified_update_update_gradient_accumulators,
        zero_gradient_accumulators)

    test_output(
        session, modified_update_output, modified_update_feed_dict2,
        modified_update_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, modified_update_feed_dict2, modified_update_output_gradient,
        [2, 150], modified_update_input_gradients, modified_update_training)
    test_gradient_accumulators(
        session, modified_update_feed_dict2, modified_update_output_gradient,
        [2, 150], modified_update_training,
        modified_update_gradient_accumulators,
        modified_update_update_gradient_accumulators,
        zero_gradient_accumulators)

    # move_rate
    test_output(
        session, move_rate_output, move_rate_feed_dict1, move_rate_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, move_rate_feed_dict1, move_rate_output_gradient, [1, 2],
        move_rate_input_gradients, move_rate_training)
    test_gradient_accumulators(
        session, move_rate_feed_dict1, move_rate_output_gradient, [1, 2],
        move_rate_training, move_rate_gradient_accumulators,
        move_rate_update_gradient_accumulators,
        zero_gradient_accumulators)

    test_output(
        session, move_rate_output, move_rate_feed_dict2, move_rate_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, move_rate_feed_dict2, move_rate_output_gradient, [2, 2],
        move_rate_input_gradients, move_rate_training)
    test_gradient_accumulators(
        session, move_rate_feed_dict2, move_rate_output_gradient, [2, 2],
        move_rate_training, move_rate_gradient_accumulators,
        move_rate_update_gradient_accumulators,
        zero_gradient_accumulators)

    # cost_function
    test_output(
        session, cost_function_output, cost_function_feed_dict1,
        cost_function_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, cost_function_feed_dict1, cost_function_output_gradient, [],
        cost_function_input_gradients, cost_function_training)
    test_gradient_accumulators(
        session, cost_function_feed_dict1, cost_function_output_gradient, [],
        cost_function_training, cost_function_gradient_accumulators,
        cost_function_update_gradient_accumulators,
        zero_gradient_accumulators)

    test_output(
        session, cost_function_output, cost_function_feed_dict2,
        cost_function_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, cost_function_feed_dict2, cost_function_output_gradient, [],
        cost_function_input_gradients, cost_function_training)
    test_gradient_accumulators(
        session, cost_function_feed_dict2, cost_function_output_gradient, [],
        cost_function_training, cost_function_gradient_accumulators,
        cost_function_update_gradient_accumulators,
        zero_gradient_accumulators)