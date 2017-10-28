import tensorflow as tf
from tests.common import *
from model_builders.basic_model_builder import BasicModelBuilder
from model_builders.vertical_lstm_model_builder import \
    VerticalLSTMModelBuilder

model_builder = VerticalLSTMModelBuilder(
    model_builder=BasicModelBuilder(
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
        move_rate_hidden_output_sizes=[25, 25, 25]),
    modified_update_lstm_state_sizes=[25, 25, 25])

model_builder.build()

graph = tf.get_default_graph()

zero_gradient_accumulators = graph.get_operation_by_name(
    'zero_gradient_accumulators')

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

modified_update_feed_dict = {
    modified_update_seed: [7, 8],
    modified_update_training: True,
    modified_update_update:
        [[(x + j) % 7 for x in range(150)] for j in range(2)],
    modified_update_statistic:
        [[(x + j) % 3 for x in range(150)] for j in range(2)],
}

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # modified_update
    test_output(
        session, modified_update_output, modified_update_feed_dict,
        modified_update_training)
    # test_batch_normalization
    # test_preserving_batch_normalization_state
    test_input_gradients(
        session, modified_update_feed_dict, modified_update_output_gradient,
        [2, 150], modified_update_input_gradients, modified_update_training)
    test_gradient_accumulators(
        session, modified_update_feed_dict, modified_update_output_gradient,
        [2, 150], modified_update_training,
        modified_update_gradient_accumulators,
        modified_update_update_gradient_accumulators,
        zero_gradient_accumulators)
