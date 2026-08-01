"""Dynamic-dimension and tracing contract."""

from __future__ import annotations

import tensorflow as tf

from crient import CrossRIEnetLayer
from crient.spectral import svd_via_eigh_full


def _inputs(batch, n_x, n_y, value):
    return [
        tf.eye(n_x, batch_shape=[batch]),
        tf.eye(n_y, batch_shape=[batch]),
        tf.random.stateless_normal((batch, n_x, n_y), seed=[value, 3]),
        tf.fill((batch,), tf.cast(20 + value, tf.float32)),
    ]


def test_one_signature_handles_tall_wide_square_and_variable_batch():
    # Exercise the helper's broader leading-batch contract first.  TensorFlow's
    # reduce_retracing cache can otherwise make this test order-dependent by
    # generalizing the helper outputs to an unknown static rank.
    svd_via_eigh_full(tf.ones((1, 2, 3), tf.float32))
    svd_via_eigh_full(tf.ones((1, 1, 2, 3), tf.float32))

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
    def apply(correlation_x, correlation_y, cross_correlation, sample_size):
        return layer(
            [correlation_x, correlation_y, cross_correlation, sample_size],
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
