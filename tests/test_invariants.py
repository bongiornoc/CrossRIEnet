"""Scientific invariants and explicitly documented limitations."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from crient import CrossRIEnetLayer
from tests.conftest import make_valid_blocks


def _small_layer(dtype="float64"):
    return CrossRIEnetLayer(
        encoder_layer_sizes=(4, 2),
        recurrent_layer_sizes=(4,),
        head_layer_sizes=(3,),
        dtype=dtype,
    )


def test_zero_additive_head_reconstructs_empirical_cross_correlation():
    inputs = make_valid_blocks(batch=1, n_x=3, n_y=5, dtype=tf.float64)
    layer = _small_layer()
    layer(inputs)
    for weight in layer.weights:
        weight.assign(tf.zeros_like(weight))
    output = layer(inputs, training=False)
    np.testing.assert_allclose(output, inputs[2], atol=1e-7, rtol=1e-7)


def test_permutation_equivariance_for_distinct_spectrum():
    inputs = make_valid_blocks(
        batch=1,
        n_x=3,
        n_y=4,
        sample_size=64,
        dtype=tf.float64,
        seed=19,
    )
    layer = _small_layer()
    reference = layer(inputs, training=False)
    permutation_x = tf.constant([2, 0, 1])
    permutation_y = tf.constant([3, 1, 0, 2])
    transformed = (
        tf.gather(
            tf.gather(inputs[0], permutation_x, axis=1),
            permutation_x,
            axis=2,
        ),
        tf.gather(
            tf.gather(inputs[1], permutation_y, axis=1),
            permutation_y,
            axis=2,
        ),
        tf.gather(
            tf.gather(inputs[2], permutation_x, axis=1),
            permutation_y,
            axis=2,
        ),
        inputs[3],
    )
    actual = layer(transformed, training=False)
    expected = tf.gather(
        tf.gather(reference, permutation_x, axis=1),
        permutation_y,
        axis=2,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_xy_swap_symmetry_for_distinct_spectrum():
    inputs = make_valid_blocks(
        batch=1,
        n_x=3,
        n_y=4,
        sample_size=64,
        dtype=tf.float64,
        seed=23,
    )
    layer = _small_layer()
    reference = layer(inputs, training=False)
    swapped = layer(
        (
            inputs[1],
            inputs[0],
            tf.linalg.matrix_transpose(inputs[2]),
            inputs[3],
        ),
        training=False,
    )
    np.testing.assert_allclose(
        swapped,
        tf.linalg.matrix_transpose(reference),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.xfail(
    strict=True,
    reason="Sequential corrections are not subspace-invariant at degeneracy.",
)
def test_degenerate_spectrum_equivariance_limitation():
    tf.keras.utils.set_random_seed(17)
    identity = tf.eye(3, batch_shape=[1], dtype=tf.float64)
    inputs = (identity, identity, identity, tf.constant([20.0], tf.float64))
    layer = _small_layer()
    reference = layer(inputs, training=False)
    permutation = tf.constant([1, 2, 0])
    expected = tf.gather(
        tf.gather(reference, permutation, axis=1),
        permutation,
        axis=2,
    )
    np.testing.assert_allclose(reference, expected, atol=1e-10, rtol=1e-10)
