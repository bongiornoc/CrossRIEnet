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
    scale = tf.math.reduce_std(returns, axis=1, keepdims=True)
    standardized = returns / scale
    full = tf.matmul(standardized, standardized, transpose_a=True)
    full /= tf.cast(sample_size, dtype)
    return (
        full[:, :n_x, :n_x],
        full[:, n_x:, n_x:],
        full[:, :n_x, n_x:],
        tf.fill((batch,), tf.cast(sample_size, dtype)),
    )


@pytest.fixture
def valid_blocks():
    return make_valid_blocks()
