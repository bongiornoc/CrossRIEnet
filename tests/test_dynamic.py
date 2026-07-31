"""Dynamic-dimension and tracing contract."""

from __future__ import annotations

import tensorflow as tf

from crient import CrossRIEnetLayer


def _inputs(batch, n_x, n_y, value):
    return [
        tf.eye(n_x, batch_shape=[batch]),
        tf.eye(n_y, batch_shape=[batch]),
        tf.random.stateless_normal((batch, n_x, n_y), seed=[value, 3]),
        tf.fill((batch,), tf.cast(20 + value, tf.float32)),
    ]


def test_one_signature_handles_tall_wide_square_and_variable_batch():
    layer = CrossRIEnetLayer(
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec((None, None, None), tf.float32),
            tf.TensorSpec((None, None, None), tf.float32),
            tf.TensorSpec((None, None, None), tf.float32),
            tf.TensorSpec((None,), tf.float32),
        ],
        reduce_retracing=True,
    )
    def apply(correlation_x, correlation_y, cross_correlation, n_observations):
        return layer(
            [correlation_x, correlation_y, cross_correlation, n_observations],
            training=False,
        )

    for batch, n_x, n_y, seed in (
        (1, 2, 5, 1),
        (3, 6, 2, 2),
        (2, 4, 4, 3),
        (1, 1, 7, 4),
    ):
        output = apply(*_inputs(batch, n_x, n_y, seed))
        assert output.shape == (batch, n_x, n_y)
        assert bool(tf.reduce_all(tf.math.is_finite(output)))

    assert len(apply._list_all_concrete_functions_for_serialization()) == 1
