import argparse
import tensorflow as tf
from config import games, models

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('game', type=str)
argument_parser.add_argument('model', type=str)
argument_parser.add_argument('worker_count', type=int)
argument_parser.add_argument('suffix', type=str)

args = argument_parser.parse_args()

game_info = games[args.game]
model = models[args.model]
model_builder = model(game_info, args.worker_count)

model_builder.build()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    graph_name = '{}__{}__{}__{}'.format(
        args.game, args.model, args.worker_count, args.suffix)
    model_name = '{}.0'.format(graph_name)

    saver = tf.train.Saver()
    saver_def = saver.as_saver_def()
    saver_save_op = session.graph.get_operation_by_name(
        saver_def.save_tensor_name[:-2])
    saver_filename = session.graph.get_tensor_by_name(
        saver_def.filename_tensor_name)

    print(saver_def.save_tensor_name[:-2])
    print(saver_def.restore_op_name)
    print(saver_def.filename_tensor_name)

    session.run(
        saver_save_op,
        {saver_filename: 'build/models/{}'.format(model_name)})

    tf.train.write_graph(
        session.graph.as_graph_def(), 'build/graphs/',
        '{}.pb'.format(graph_name),
        as_text=False)
