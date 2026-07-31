"""Characterization tests for the CrossRIEnet 0.1 implementation.

These tests intentionally describe the legacy behavior.  They protect the
conversion path while the canonical 0.2 API moves to :mod:`crient`.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from crossrie import CrossRIEnetLayer
from crossrie.custom_layers import SpectralSVDLayer


def _inputs(dtype: tf.dtypes.DType = tf.float32):
    return [
        tf.eye(2, batch_shape=[1], dtype=dtype),
        tf.eye(3, batch_shape=[1], dtype=dtype),
        tf.constant(
            [[[0.40, -0.10, 0.05], [0.02, 0.30, -0.08]]],
            dtype=dtype,
        ),
        tf.constant([20.0], dtype=dtype),
    ]


def test_legacy_default_parameter_count_matches_paper():
    layer = CrossRIEnetLayer()
    output = layer(_inputs())

    assert output.shape == (1, 2, 3)
    assert layer.count_params() == 331_355


def test_legacy_float64_input_is_autocast_by_default_policy():
    layer = CrossRIEnetLayer()
    output = layer(_inputs(tf.float64))

    assert layer.compute_dtype == "float32"
    assert output.dtype == tf.float32
    assert {weight.dtype for weight in layer.weights} == {"float32"}


def test_legacy_spectral_eps_is_inert():
    matrix = _inputs()[2]
    reference = SpectralSVDLayer(eps=None)(matrix)
    configured = SpectralSVDLayer(eps=0.25)(matrix)

    for actual, expected in zip(configured, reference):
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=0, rtol=0)


def test_legacy_rejects_string_outputs():
    with pytest.raises(TypeError, match="list or tuple"):
        CrossRIEnetLayer(outputs="Cxy")


def test_legacy_token_order_is_gamma_q_s():
    layer = CrossRIEnetLayer(
        encoding_units=[2],
        lstm_units=[2],
        final_hidden_layer_sizes=[2],
    )
    captured = []

    def capture(inputs):
        captured.extend(tf.identity(value) for value in inputs)
        return tf.zeros(tf.shape(inputs[0])[:2], dtype=inputs[0].dtype)

    layer.two_stream_encoder.call = capture
    _ = layer(_inputs())

    empirical_s, _, _ = layer.svd_layer(_inputs()[2])
    np.testing.assert_allclose(
        captured[0][..., -1].numpy(),
        tf.pad(empirical_s[..., None], [[0, 0], [0, 1], [0, 0]])[..., 0].numpy(),
        atol=1e-6,
    )
    # The q feature is the middle channel in 0.1.
    assert captured[0].shape[-1] == 3
    assert np.all(captured[0].numpy()[:, :2, 1] > 0)


def test_legacy_zero_and_negative_t_are_not_rejected():
    layer = CrossRIEnetLayer()
    base = _inputs()

    for value in (0.0, -1.0):
        inputs = [*base[:3], tf.constant([value], tf.float32)]
        output = layer(inputs)
        assert bool(tf.reduce_all(tf.math.is_finite(output)))
