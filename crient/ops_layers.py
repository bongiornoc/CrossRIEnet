"""Deterministic tensor operations and Keras wrappers for CRIENT."""

from __future__ import annotations

import keras
import tensorflow as tf


def projected_variance_diagonal(
    correlation: tf.Tensor,
    basis: tf.Tensor,
) -> tf.Tensor:
    """Return ``diag(basis.T @ correlation @ basis)``.

    Parameters
    ----------
    correlation
        Batched square matrices with shape ``(..., n, n)``.
    basis
        Batched bases with shape ``(..., n, r)``.
    """
    projected = tf.matmul(correlation, basis)
    return tf.reduce_sum(basis * projected, axis=-2)


def pad_sequence_to(
    sequence: tf.Tensor,
    target_length: tf.Tensor,
) -> tf.Tensor:
    """Right-pad a rank-3 token sequence to ``target_length``."""
    current_length = tf.shape(sequence)[1]
    padding = target_length - current_length
    tf.debugging.assert_greater_equal(
        padding,
        0,
        message="target_length must not be shorter than the input sequence",
    )
    return tf.pad(sequence, [[0, 0], [0, padding], [0, 0]])


def reconstruct_from_full_svd(
    spectral_coefficients: tf.Tensor,
    left_vectors: tf.Tensor,
    right_vectors: tf.Tensor,
) -> tf.Tensor:
    """Reconstruct a rectangular matrix from full singular-vector bases."""
    rank = tf.shape(spectral_coefficients)[-1]
    left = left_vectors[..., :, :rank]
    right = right_vectors[..., :, :rank]
    scaled_right = spectral_coefficients[..., :, None] * tf.linalg.matrix_transpose(
        right
    )
    return tf.matmul(left, scaled_right)


@keras.saving.register_keras_serializable(package="crient")
class ProjectedVarianceDiagonalLayer(keras.layers.Layer):
    """Compute a marginal variance diagonal in a supplied spectral basis."""

    def call(self, inputs):
        correlation, basis = inputs
        return projected_variance_diagonal(correlation, basis)


@keras.saving.register_keras_serializable(package="crient")
class SequencePaddingLayer(keras.layers.Layer):
    """Right-pad a rank-3 sequence to a supplied scalar length."""

    def call(self, inputs):
        sequence, target_length = inputs
        return pad_sequence_to(sequence, target_length)


@keras.saving.register_keras_serializable(package="crient")
class SVDReconstructionLayer(keras.layers.Layer):
    """Reconstruct a matrix from signed spectral coefficients and full bases."""

    def call(self, inputs):
        spectral_coefficients, left_vectors, right_vectors = inputs
        return reconstruct_from_full_svd(
            spectral_coefficients,
            left_vectors,
            right_vectors,
        )


@keras.saving.register_keras_serializable(package="crient")
class CrossCovarianceRescalingLayer(keras.layers.Layer):
    """Convert a cross-correlation matrix to a cross-covariance matrix.

    The operation is

    ``cross_covariance[i, j] = std_x[i] * correlation[i, j] * std_y[j]``.

    Notes
    -----
    Inputs are ``cross_correlation`` with shape ``(batch, n_x, n_y)``,
    ``std_x`` with shape ``(batch, n_x)`` and ``std_y`` with shape
    ``(batch, n_y)``.
    """

    def call(self, inputs):
        """Rescale a cross-correlation by two marginal standard deviations.

        Parameters
        ----------
        inputs : sequence of tf.Tensor
            Cross-correlation, left standard deviations and right standard
            deviations.

        Returns
        -------
        tf.Tensor
            Cross-covariance with the same matrix shape.
        """
        cross_correlation, std_x, std_y = inputs
        tf.debugging.assert_rank(cross_correlation, 3)
        tf.debugging.assert_rank(std_x, 2)
        tf.debugging.assert_rank(std_y, 2)
        tf.debugging.assert_equal(
            tf.shape(cross_correlation)[0],
            tf.shape(std_x)[0],
            message="std_x batch dimension must match cross_correlation",
        )
        tf.debugging.assert_equal(
            tf.shape(cross_correlation)[0],
            tf.shape(std_y)[0],
            message="std_y batch dimension must match cross_correlation",
        )
        tf.debugging.assert_equal(
            tf.shape(cross_correlation)[1],
            tf.shape(std_x)[1],
            message="std_x length must match the left matrix dimension",
        )
        tf.debugging.assert_equal(
            tf.shape(cross_correlation)[2],
            tf.shape(std_y)[1],
            message="std_y length must match the right matrix dimension",
        )
        tf.debugging.assert_all_finite(
            cross_correlation,
            "cross_correlation must contain only finite values",
        )
        tf.debugging.assert_all_finite(
            std_x,
            "std_x must contain only finite values",
        )
        tf.debugging.assert_all_finite(
            std_y,
            "std_y must contain only finite values",
        )
        tf.debugging.assert_non_negative(
            std_x,
            message="std_x must be non-negative",
        )
        tf.debugging.assert_non_negative(
            std_y,
            message="std_y must be non-negative",
        )
        return std_x[..., :, None] * cross_correlation * std_y[..., None, :]
