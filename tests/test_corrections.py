"""Correction semantics, token order and training propagation."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from crienet import CrossRIEnetLayer
from crienet.trainable_layers import TwoStreamEncoderLayer
from tests.conftest import make_valid_blocks


@pytest.mark.parametrize(
    ("mode", "expected_correction"),
    [
        ("additive", 0.0),
        ("bounded_multiplicative", 0.5),
        ("positive_multiplicative", np.log(2.0)),
    ],
)
def test_correction_mode_semantics(mode, expected_correction):
    inputs = make_valid_blocks(batch=1, n_x=2, n_y=3)
    layer = CrossRIEnetLayer(
        output_type=(
            "spectral_coefficients",
            "empirical_singular_values",
            "correction",
        ),
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
        correction_mode=mode,
    )
    layer(inputs)
    for weight in layer.weights:
        weight.assign(tf.zeros_like(weight))
    outputs = layer(inputs, training=False)

    np.testing.assert_allclose(
        outputs["correction"],
        expected_correction,
        atol=1e-7,
    )
    if mode == "additive":
        expected = outputs["empirical_singular_values"]
    else:
        expected = outputs["empirical_singular_values"] * expected_correction
    np.testing.assert_allclose(
        outputs["spectral_coefficients"],
        expected,
        atol=1e-7,
    )


class _CaptureEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.captured = None

    def call(self, inputs, training=None):
        self.captured = tuple(tf.identity(value) for value in inputs)
        return tf.zeros(tf.shape(inputs[0])[:2], dtype=inputs[0].dtype)


def test_equation_four_token_order_and_padding_semantics():
    inputs = make_valid_blocks(batch=1, n_x=2, n_y=4)
    layer = CrossRIEnetLayer(
        output_type="spectral_coefficients",
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )
    capture = _CaptureEncoder()
    layer.encoder = capture
    outputs = layer(inputs)
    stream_x, stream_y = capture.captured

    assert stream_x.shape == (1, 4, 3)
    assert stream_y.shape == (1, 4, 3)
    # Channel 1 contains the empirical singular-value sequence.
    np.testing.assert_allclose(stream_x[:, :2, 1], outputs)
    np.testing.assert_allclose(stream_x[:, 2:, 1], 0)
    # Channel 2 contains q, with legacy padding semantics retained.
    assert bool(tf.reduce_all(stream_x[:, :2, 2] > 0))
    np.testing.assert_allclose(stream_x[:, 2:, 2], 0)
    # Larger-side marginal directions reach the recurrent encoder.
    assert bool(tf.reduce_any(tf.abs(stream_y[:, 2:, 0]) > 0))


def test_training_flag_reaches_dropout_layers():
    tf.keras.utils.set_random_seed(31)
    layer = TwoStreamEncoderLayer(
        encoder_layer_sizes=(8, 4),
        recurrent_layer_sizes=(4,),
        head_layer_sizes=(4,),
        dropout_rate=0.5,
    )
    stream = tf.ones((2, 5, 3))
    inference_a = layer([stream, stream], training=False)
    inference_b = layer([stream, stream], training=False)
    training = layer([stream, stream], training=True)

    np.testing.assert_allclose(inference_a, inference_b, atol=0, rtol=0)
    assert not np.allclose(training, inference_a)
