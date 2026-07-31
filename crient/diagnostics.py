"""Non-mutating scientific diagnostics for fixed-marginal feasibility."""

from __future__ import annotations

import tensorflow as tf

from .dtype_utils import machine_epsilon


def _inverse_sqrt_psd(matrix: tf.Tensor):
    eigenvalues, eigenvectors = tf.linalg.eigh(
        0.5 * (matrix + tf.linalg.matrix_transpose(matrix))
    )
    dtype = matrix.dtype
    maximum = tf.reduce_max(
        tf.maximum(eigenvalues, tf.zeros_like(eigenvalues)),
        axis=-1,
        keepdims=True,
    )
    dimension = tf.cast(tf.shape(matrix)[-1], dtype)
    tolerance = tf.sqrt(dimension * machine_epsilon(dtype)) * maximum
    resolved = eigenvalues > tolerance
    inverse_sqrt = tf.where(
        resolved,
        tf.math.rsqrt(tf.where(resolved, eigenvalues, tf.ones_like(eigenvalues))),
        tf.zeros_like(eigenvalues),
    )
    transform = tf.matmul(
        eigenvectors * inverse_sqrt[..., None, :],
        eigenvectors,
        transpose_b=True,
    )
    projector = tf.matmul(
        eigenvectors * tf.cast(resolved[..., None, :], dtype),
        eigenvectors,
        transpose_b=True,
    )
    positive_minimum = tf.reduce_min(
        tf.where(
            resolved,
            eigenvalues,
            tf.fill(tf.shape(eigenvalues), tf.cast(float("inf"), dtype)),
        ),
        axis=-1,
    )
    condition = tf.where(
        tf.math.is_finite(positive_minimum),
        tf.squeeze(maximum, axis=-1) / positive_minimum,
        tf.fill(tf.shape(positive_minimum), tf.cast(float("inf"), dtype)),
    )
    return transform, projector, eigenvalues[..., 0], condition


def feasibility_diagnostics(
    correlation_x: tf.Tensor,
    correlation_y: tf.Tensor,
    cross_correlation: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Diagnose whether a cross-correlation is feasible under fixed marginals.

    This function never projects, clips or otherwise changes the estimator.
    Rank-aware inverse square roots are used only to calculate diagnostics.

    Parameters
    ----------
    correlation_x : tf.Tensor
        Left marginal correlation with shape ``(batch, n_x, n_x)``.
    correlation_y : tf.Tensor
        Right marginal correlation with shape ``(batch, n_y, n_y)``.
    cross_correlation : tf.Tensor
        Rectangular cross-correlation with shape ``(batch, n_x, n_y)``.

    Returns
    -------
    dict[str, tf.Tensor]
        Canonical singular values, feasibility violations, margins, support
        residuals, minimum eigenvalues and condition estimates.
    """
    correlation_x = tf.convert_to_tensor(correlation_x)
    correlation_y = tf.convert_to_tensor(correlation_y)
    cross_correlation = tf.convert_to_tensor(cross_correlation)

    inverse_x, projector_x, min_x, condition_x = _inverse_sqrt_psd(correlation_x)
    inverse_y, projector_y, min_y, condition_y = _inverse_sqrt_psd(correlation_y)
    whitened = tf.matmul(
        tf.matmul(inverse_x, cross_correlation),
        inverse_y,
    )
    canonical_values = tf.linalg.svd(whitened, compute_uv=False)
    maximum = tf.reduce_max(canonical_values, axis=-1)
    dtype = cross_correlation.dtype
    tolerance = tf.sqrt(
        tf.cast(
            tf.maximum(
                tf.shape(cross_correlation)[-2],
                tf.shape(cross_correlation)[-1],
            ),
            dtype,
        )
        * machine_epsilon(dtype)
    )
    violations = tf.reduce_sum(
        tf.cast(canonical_values > 1 + tolerance, tf.int32),
        axis=-1,
    )

    supported = tf.matmul(
        tf.matmul(projector_x, cross_correlation),
        projector_y,
    )
    support_residual = tf.norm(
        cross_correlation - supported,
        axis=(-2, -1),
    )

    top = tf.concat([correlation_x, cross_correlation], axis=-1)
    bottom = tf.concat(
        [tf.linalg.matrix_transpose(cross_correlation), correlation_y],
        axis=-1,
    )
    full_block = tf.concat([top, bottom], axis=-2)
    min_full = tf.reduce_min(tf.linalg.eigvalsh(full_block), axis=-1)

    return {
        "canonical_singular_values": canonical_values,
        "max_canonical_singular_value": maximum,
        "violation_count": violations,
        "feasibility_margin": 1 - maximum,
        "support_residual": support_residual,
        "min_eigenvalue_x": min_x,
        "min_eigenvalue_y": min_y,
        "min_eigenvalue_full_block": min_full,
        "condition_estimate_x": condition_x,
        "condition_estimate_y": condition_y,
    }
