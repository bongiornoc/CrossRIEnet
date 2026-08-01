"""Numerical tests for the Gram/eigh spectral backend."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from crienet.ops_layers import reconstruct_from_full_svd
from crienet.spectral import SpectralSVDLayer, svd_via_eigh_full


@pytest.mark.parametrize("shape", [(2, 4, 6), (2, 6, 4), (2, 5, 5)])
def test_reconstruction_and_orthogonality(shape):
    matrix = tf.random.normal(shape, seed=7, dtype=tf.float64)
    singular, left, right = svd_via_eigh_full(matrix)
    reconstructed = reconstruct_from_full_svd(singular, left, right)
    np.testing.assert_allclose(reconstructed, matrix, atol=1e-7, rtol=1e-7)

    left_identity = tf.eye(shape[-2], batch_shape=[shape[0]], dtype=tf.float64)
    right_identity = tf.eye(shape[-1], batch_shape=[shape[0]], dtype=tf.float64)
    np.testing.assert_allclose(
        tf.matmul(left, left, transpose_a=True),
        left_identity,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        tf.matmul(right, right, transpose_a=True),
        right_identity,
        atol=1e-7,
    )


def test_zero_matrix_has_complete_orthonormal_bases():
    matrix = tf.zeros((2, 3, 5), tf.float64)
    singular, left, right = svd_via_eigh_full(matrix)
    assert not np.any(singular.numpy())
    np.testing.assert_allclose(
        tf.matmul(left, left, transpose_a=True),
        tf.eye(3, batch_shape=[2], dtype=tf.float64),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tf.matmul(right, right, transpose_a=True),
        tf.eye(5, batch_shape=[2], dtype=tf.float64),
        atol=1e-12,
    )


def test_rank_deficient_gradient_is_finite():
    vector_x = tf.constant([[[1.0], [2.0], [3.0]]])
    vector_y = tf.constant([[[1.0, -1.0, 0.5, 2.0]]])
    matrix = tf.Variable(tf.matmul(vector_x, vector_y))
    with tf.GradientTape() as tape:
        singular, left, right = svd_via_eigh_full(matrix)
        reconstructed = reconstruct_from_full_svd(singular, left, right)
        loss = tf.reduce_sum(singular) + tf.reduce_sum(reconstructed)
    gradient = tape.gradient(loss, matrix)
    assert gradient is not None
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)))


@pytest.mark.parametrize("scale", [1e-20, 1e20])
def test_extreme_scales_remain_finite(scale):
    matrix = tf.random.normal((2, 3, 5), seed=2, dtype=tf.float64) * scale
    outputs = svd_via_eigh_full(matrix)
    assert all(bool(tf.reduce_all(tf.math.is_finite(value))) for value in outputs)


def test_helper_supports_arbitrary_leading_batch_dimensions():
    matrix = tf.random.normal((2, 3, 4, 5), seed=3)
    singular, left, right = svd_via_eigh_full(matrix)
    assert singular.shape == (2, 3, 4)
    assert left.shape == (2, 3, 4, 4)
    assert right.shape == (2, 3, 5, 5)


@pytest.mark.parametrize(
    ("policy_name", "expected_dtype"),
    [
        ("mixed_float16", tf.float16),
        ("mixed_bfloat16", tf.bfloat16),
    ],
)
def test_public_spectral_layer_upcasts_low_precision(policy_name, expected_dtype):
    previous = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy(policy_name)
        layer = SpectralSVDLayer()
        outputs = layer(tf.random.normal((1, 3, 5)))
        assert all(output.dtype == expected_dtype for output in outputs)
        assert all(bool(tf.reduce_all(tf.math.is_finite(value))) for value in outputs)
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
