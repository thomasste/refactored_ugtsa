import tensorflow as tf
from model_builders.common.rng import RNG


class ModelBuilder:
    def __init__(self, statistic_size, update_size,
                 board_shape, game_state_info_size,
                 payoff_size, player_count, worker_count):
        self.statistic_size = statistic_size
        self.update_size = update_size
        self.board_shape = \
            board_shape
        self.game_state_info_size = \
            game_state_info_size
        self.payoff_size = payoff_size
        self.player_count = player_count
        self.worker_count = worker_count

    def statistic(self, rng, training, board, game_state_info):
        raise NotImplementedError

    def update(self, rng, training, payoff):
        raise NotImplementedError

    def modified_statistic(
            self, rng, training, statistic, updates_count, updates):
        raise NotImplementedError

    def modified_update(self, rng, training, update, statistic):
        raise NotImplementedError

    def move_rate(self, rng, training, parent_statistic, child_statistic):
        raise NotImplementedError

    def cost_function(self, rng, training, logits, labels):
        raise NotImplementedError

    def apply_gradients(self, global_step, grads_and_vars):
        raise NotImplementedError

    @classmethod
    def build_graph(cls, method, inputs, output_):
        print(method.__name__)
        with tf.variable_scope(method.__name__) as variable_scope:
            placeholders = {
                input['name']: tf.placeholder(
                    input['dtype'], input['shape'], input['name'])
                for input in inputs}

            rng = RNG(placeholders['seed'])

            with tf.variable_scope('transformation') as \
                    transformation_variable_scope:
                parameters = {
                    'rng': rng,
                    **{
                        k: v
                        for k, v in placeholders.items()
                        if k != 'seed'
                    }
                }
                signal = method(**parameters)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                output_['dtype'], output_['shape'], 'output_gradient')

            # gradient accumulators
            trainable_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                transformation_variable_scope.name)

            gradients = tf.gradients(
                output, trainable_variables, output_gradient)

            for trainable_variable in trainable_variables:
                tf.add_to_collection(
                    '{}/test_info_trainable_variables'.format(
                        variable_scope.name),
                    trainable_variable)

            for gradient in gradients:
                tf.add_to_collection(
                    '{}/test_info_gradients'.format(variable_scope.name),
                    gradient)

            gradient_accumulators = [
                tf.Variable(
                    tf.zeros(gradient.get_shape(), gradient.dtype),
                    trainable=False)
                for gradient in gradients]

            for gradient_accumulator in gradient_accumulators:
                tf.add_to_collection(
                    '{}/gradient_accumulators'.format(variable_scope.name),
                    gradient_accumulator)

            update_gradient_accumulators = [
                tf.assign_add(gradient_accumulator, gradient).op
                for gradient, gradient_accumulator in zip(
                    gradients, gradient_accumulators)]

            with tf.control_dependencies(update_gradient_accumulators):
                tf.no_op('update_gradient_accumulators')

            tf.variables_initializer(
                gradient_accumulators,
                'zero_gradient_accumulators')

            # placeholders gradients
            trainable_placeholders = [
                (input['name'], placeholders[input['name']])
                for input in inputs if input['trainable']]

            gradients = tf.gradients(
                output, [x[1] for x in trainable_placeholders],
                output_gradient)

            print(gradients)

            for (input_name, _), gradient in zip(
                    trainable_placeholders, gradients):
                tf.identity(gradient, '{}_gradient'.format(input_name))

    def build(self):
        methods = {
            self.statistic: {
                'inputs': [
                    {
                        'name': 'seed',
                        'dtype': tf.int64,
                        'shape': [2],
                        'trainable': False,
                    },
                    {
                        'name': 'training',
                        'dtype': tf.bool,
                        'shape': [],
                        'trainable': False,
                    },
                    {
                        'name': 'board',
                        'dtype': tf.float32,
                        'shape': [None,
                                  self.board_shape[0],
                                  self.board_shape[1]],
                        'trainable': False,
                    },
                    {
                        'name': 'game_state_info',
                        'dtype': tf.float32,
                        'shape': [None,
                                  self.game_state_info_size],
                        'trainable': False,
                    },
                ],
                'output': {
                    'dtype': tf.float32,
                    'shape': [None, self.statistic_size],
                },
            },
            self.update: {
                'inputs': [
                    {
                        'name': 'seed',
                        'dtype': tf.int64,
                        'shape': [2],
                        'trainable': False,
                    },
                    {
                        'name': 'training',
                        'dtype': tf.bool,
                        'shape': [],
                        'trainable': False,
                    },
                    {
                        'name': 'payoff',
                        'dtype': tf.float32,
                        'shape': [None,
                                  self.payoff_size],
                        'trainable': False,
                    },
                ],
                'output': {
                    'dtype': tf.float32,
                    'shape': [None, self.update_size],
                },
            },
            self.modified_statistic: {
                'inputs': [
                    {
                        'name': 'seed',
                        'dtype': tf.int64,
                        'shape': [2],
                        'trainable': False,
                    },
                    {
                        'name': 'training',
                        'dtype': tf.bool,
                        'shape': [],
                        'trainable': False,
                    },
                    {
                        'name': 'statistic',
                        'dtype': tf.float32,
                        'shape': [None, self.statistic_size],
                        'trainable': True,
                    },
                    {
                        'name': 'updates_count',
                        'dtype': tf.int32,
                        'shape': [None],
                        'trainable': False,
                    },
                    {
                        'name': 'updates',
                        'dtype': tf.float32,
                        'shape': [None, self.worker_count * self.update_size],
                        'trainable': True,
                    },
                ],
                'output': {
                    'dtype': tf.float32,
                    'shape': [None, self.statistic_size],
                },
            },
            self.modified_update: {
                'inputs': [
                    {
                        'name': 'seed',
                        'dtype': tf.int64,
                        'shape': [2],
                        'trainable': False,
                    },
                    {
                        'name': 'training',
                        'dtype': tf.bool,
                        'shape': [],
                        'trainable': False,
                    },
                    {
                        'name': 'update',
                        'dtype': tf.float32,
                        'shape': [None, self.update_size],
                        'trainable': True,
                    },
                    {
                        'name': 'statistic',
                        'dtype': tf.float32,
                        'shape': [None, self.statistic_size],
                        'trainable': True,
                    },
                ],
                'output': {
                    'dtype': tf.float32,
                    'shape': [None, self.update_size],
                },
            },
            self.move_rate: {
                'inputs': [
                    {
                        'name': 'seed',
                        'dtype': tf.int64,
                        'shape': [2],
                        'trainable': False,
                    },
                    {
                        'name': 'training',
                        'dtype': tf.bool,
                        'shape': [],
                        'trainable': False,
                    },
                    {
                        'name': 'parent_statistic',
                        'dtype': tf.float32,
                        'shape': [None, self.statistic_size],
                        'trainable': True,
                    },
                    {
                        'name': 'child_statistic',
                        'dtype': tf.float32,
                        'shape': [None, self.statistic_size],
                        'trainable': True,
                    },
                ],
                'output': {
                    'dtype': tf.float32,
                    'shape': [None, self.player_count],
                },
            },
            self.cost_function: {
                'inputs': [
                    {
                        'name': 'seed',
                        'dtype': tf.int64,
                        'shape': [2],
                        'trainable': False,
                    },
                    {
                        'name': 'training',
                        'dtype': tf.bool,
                        'shape': [],
                        'trainable': False,
                    },
                    {
                        'name': 'logits',
                        'dtype': tf.float32,
                        'shape': [None, self.player_count],
                        'trainable': True,
                    },
                    {
                        'name': 'labels',
                        'dtype': tf.float32,
                        'shape': [None, self.player_count],
                        'trainable': False,
                    },
                ],
                'output': {
                    'dtype': tf.float32,
                    'shape': [],
                },
            },
        }

        # save properties
        for property_name in [
                'statistic_size',
                'update_size',
                'board_shape',
                'game_state_info_size',
                'payoff_size',
                'player_count',
                'worker_count']:
            tf.Variable(
                initial_value=getattr(self, property_name),
                trainable=False,
                dtype=tf.int32,
                name=property_name)

        # save shapes
        for key, value in methods.items():
            for input in value['inputs']:
                tf.Variable(
                    initial_value=len(input['shape']),
                    trainable=False,
                    dtype=tf.int32,
                    name='{}_{}_dims'.format(key.__name__, input['name']))
                tf.Variable(
                    initial_value=[x if x is not None else -1
                                   for x in input['shape']],
                    trainable=False,
                    dtype=tf.int32,
                    name='{}_{}_shape'.format(key.__name__, input['name']))
            tf.Variable(
                initial_value=len(value['output']['shape']),
                trainable=False,
                dtype=tf.int32,
                name='{}_output_dims'.format(key.__name__))
            tf.Variable(
                initial_value=[x if x is not None else -1
                               for x in value['output']['shape']],
                trainable=False,
                dtype=tf.int32,
                name='{}_output_shape'.format(key.__name__))

        # build methods
        for key, value in methods.items():
            self.build_graph(key, value['inputs'], value['output'])

        # getter and setter of untrainable variables
        #  - used for preserving moving averages
        untrainable_variables = [
            variable
            for key in methods
            for variable in tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                '{}/transformation'.format(key.__name__))
            if variable not in tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES)]

        print(untrainable_variables)

        flattened_untrainable_variables = [
            tf.reshape(variable, [-1])
            for variable in untrainable_variables]

        tf.concat(
            flattened_untrainable_variables, 0,
            name='collected_untrainable_variables')

        set_untrainable_variables_input = tf.placeholder(
            tf.float32,
            sum([variable.get_shape().as_list()[0]
                 for variable in flattened_untrainable_variables]),
            'set_untrainable_variables_input')

        set_untrainable_variables = [
            tf.assign(variable, tf.reshape(value, variable.get_shape())).op
            for variable, value in zip(
                untrainable_variables,
                tf.split(
                    set_untrainable_variables_input,
                    [variable.get_shape().as_list()[0]
                     for variable in flattened_untrainable_variables]))]

        with tf.control_dependencies(set_untrainable_variables):
            tf.no_op('set_untrainable_variables')

        # build apply gradients graph
        # ! Adam creates untrainable variables ->
        # needs to be run as last command
        global_step = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.int32,
            name='global_step')
        grads_and_vars = [
            (gradient, variable)
            for key in methods
            for gradient, variable in zip(
                tf.get_collection(
                    '{}/gradient_accumulators'.format(key.__name__)),
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    '{}/transformation'.format(key.__name__)))]
        self.apply_gradients(global_step, grads_and_vars)
