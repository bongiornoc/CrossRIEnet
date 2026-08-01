"""Correlation-domain and feasibility diagnostics."""

from __future__ import annotations

import tensorflow as tf

from crienet.diagnostics import feasibility_diagnostics


def test_valid_block_is_reported_feasible(valid_blocks):
    diagnostics = feasibility_diagnostics(*valid_blocks[:3])
    assert bool(tf.reduce_all(diagnostics["max_canonical_singular_value"] <= 1.001))
    assert bool(tf.reduce_all(diagnostics["violation_count"] == 0))
    assert bool(tf.reduce_all(diagnostics["min_eigenvalue_full_block"] >= -1e-5))


def test_infeasible_block_is_reported_without_projection():
    correlation_x = tf.eye(2, batch_shape=[1])
    correlation_y = tf.eye(2, batch_shape=[1])
    cross_correlation = tf.constant([[[1.2, 0.0], [0.0, 1.1]]])
    diagnostics = feasibility_diagnostics(
        correlation_x,
        correlation_y,
        cross_correlation,
    )
    assert diagnostics["violation_count"][0] == 2
    assert diagnostics["max_canonical_singular_value"][0] > 1
    assert diagnostics["min_eigenvalue_full_block"][0] < 0
