from model_builders.basic_model_builder import BasicModelBuilder
from model_builders.ucb_model_builder import UCBModelBuilder
from model_builders.vertical_lstm_model_builder import \
    VerticalLSTMModelBuilder

games = {
    'omringa': {
        'board_shape': [7, 7],
        'game_state_info_size': 2,
        'payoff_size': 2,
        'player_count': 2,
    }
}

models = {
    'ucb':
        lambda game_info, worker_count: UCBModelBuilder(
            board_shape=game_info['board_shape'],
            game_state_info_size=game_info['game_state_info_size'],
            player_count=game_info['player_count'],
            worker_count=worker_count),
    '25102017_basic':
        lambda game_info, worker_count: BasicModelBuilder(
            statistic_size=150,
            update_size=150,
            board_shape=game_info['board_shape'],
            game_state_info_size=game_info['game_state_info_size'],
            payoff_size=game_info['payoff_size'],
            player_count=game_info['player_count'],
            worker_count=worker_count,
            statistic_filter_shapes=[(2, 2, 16), (2, 2, 32)],
            statistic_window_shapes=[(1, 2, 2, 1), (1, 2, 2, 1)],
            statistic_hidden_output_sizes=[150, 150, 150],
            update_hidden_output_sizes=[150, 150, 150],
            modified_statistic_lstm_state_sizes=[25, 25, 25],
            modified_update_hidden_output_sizes=[150, 150, 150],
            move_rate_hidden_output_sizes=[150, 150, 150]),
    '25102017_vertical':
        lambda game_info, worker_count: VerticalLSTMModelBuilder(
            model_builder=BasicModelBuilder(
                statistic_size=150,
                update_size=150,
                board_shape=game_info['board_shape'],
                game_state_info_size=game_info['game_state_info_size'],
                payoff_size=game_info['payoff_size'],
                player_count=game_info['player_count'],
                worker_count=worker_count,
                statistic_filter_shapes=[(2, 2, 16), (2, 2, 32)],
                statistic_window_shapes=[(1, 2, 2, 1), (1, 2, 2, 1)],
                statistic_hidden_output_sizes=[150, 150, 150],
                update_hidden_output_sizes=[150, 150, 150],
                modified_statistic_lstm_state_sizes=[25, 25, 25],
                modified_update_hidden_output_sizes=None,
                move_rate_hidden_output_sizes=[150, 150, 150]),
            modified_update_lstm_state_sizes=[25, 25, 25]),
    '10112017_basic':
        lambda game_info, worker_count: BasicModelBuilder(
            statistic_size=50,
            update_size=50,
            board_shape=game_info['board_shape'],
            game_state_info_size=game_info['game_state_info_size'],
            payoff_size=game_info['payoff_size'],
            player_count=game_info['player_count'],
            worker_count=worker_count,
            statistic_filter_shapes=[(2, 2, 16), (2, 2, 32)],
            statistic_window_shapes=[(1, 2, 2, 1), (1, 2, 2, 1)],
            statistic_hidden_output_sizes=[150],
            update_hidden_output_sizes=[150],
            modified_statistic_lstm_state_sizes=[25],
            modified_update_hidden_output_sizes=[150],
            move_rate_hidden_output_sizes=[]),
}
