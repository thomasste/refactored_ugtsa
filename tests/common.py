import numpy as np


def assert_equal(a, b):
    np.testing.assert_equal(a, b)


def assert_unequal(a, b):
    assert np.any(np.not_equal(a, b))


def assert_zero(a):
    assert not np.any(a)


def assert_nonzero(a):
    assert np.any(a)


def test_output(session, output, feed_dict, training):
    o0 = session.run(output, {**feed_dict, training: False})
    o1 = session.run(output, {**feed_dict, training: True})

    assert_nonzero(o0)
    assert_nonzero(o1)


def test_batch_normalization(session, output, feed_dict, training):
    o0 = session.run(output, {**feed_dict, training: False})
    o1 = session.run(output, {**feed_dict, training: True})
    o2 = session.run(output, {**feed_dict, training: True})
    o3 = session.run(output, {**feed_dict, training: False})

    assert_equal(o1, o2)
    assert_unequal(o0, o3)


def test_preserving_batch_normalization_state(session, output, feed_dict, training, collected_untrainable_variables, set_untrainable_variables, set_untrainable_variables_input):
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


def test_input_gradients(session, feed_dict, output_gradient, output_shape, input_gradients, training):
    os = session.run(input_gradients, {**feed_dict, output_gradient: np.ones(output_shape), training: True})

    for o in os:
        assert_nonzero(o)


def test_gradient_accumulators(session, feed_dict, output_gradient, output_shape, training, gradient_accumulators, update_gradient_accumulators, zero_gradient_accumulators):
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
