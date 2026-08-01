"""Shared scientific fixtures."""

from __future__ import annotations

import pytest
import tensorflow as tf


def make_valid_blocks(
    batch=2,
    n_x=3,
    n_y=4,
    sample_size=32,
    dtype=tf.float32,
    seed=13,
):
    generator = tf.random.Generator.from_seed(seed)
    returns = generator.normal(
        (batch, sample_size, n_x + n_y),
        dtype=dtype,
    )
    returns -= tf.reduce_mean(returns, axis=1, keepdims=True)
    covariance = tf.matmul(returns, returns, transpose_a=True)
    covariance /= tf.cast(sample_size, dtype)
    scale = tf.sqrt(tf.linalg.diag_part(covariance))
    full = covariance / scale[..., :, None] / scale[..., None, :]
    full = 0.5 * (full + tf.linalg.matrix_transpose(full))
    full = tf.linalg.set_diag(full, tf.ones_like(scale))
    return (
        full[:, :n_x, :n_x],
        full[:, n_x:, n_x:],
        full[:, :n_x, n_x:],
        tf.fill((batch,), tf.cast(sample_size, dtype)),
    )


@pytest.fixture
def valid_blocks():
    return make_valid_blocks()
