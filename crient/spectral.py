"""Spectral operations used by CRIENT.

The current backend constructs Gram matrices and uses ``tf.linalg.eigh`` so
that full left and right bases are available for rectangular inputs.  This
module deliberately does not expose a native-SVD backend until TensorFlow
supports the required full-basis gradients for all rectangular shapes.
"""

from __future__ import annotations

import keras
import tensorflow as tf

from .dtype_utils import (
    cast_to_work_dtype,
    machine_epsilon,
    resolve_work_dtype,
    restore_compute_dtype,
)


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def automatic_rank_rtol(
    m: tf.Tensor,
    n: tf.Tensor,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Return the Gram-aware default relative rank tolerance.

    The Gram construction squares the condition number.  Singular values below
    ``sqrt(max(m, n) * eps) * s_max`` are therefore treated as numerically
    unresolved by default.
    """
    dimension = tf.cast(tf.maximum(m, n), dtype)
    return tf.sqrt(dimension * machine_epsilon(dtype))


@tf.function(reduce_retracing=True)
def svd_via_eigh_full(
    matrix: tf.Tensor,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute a full batched SVD through symmetric eigendecompositions.

    Parameters
    ----------
    matrix
        Floating tensor with shape ``(..., m, n)``.
    rank_rtol
        Relative numerical-rank tolerance.  ``None`` selects
        ``sqrt(max(m, n) * eps(dtype))``.
    rank_atol
        Absolute numerical-rank tolerance in the original matrix scale.

    Returns
    -------
    singular_values, left_vectors, right_vectors
        Tensors with shapes ``(..., min(m,n))``, ``(...,m,m)`` and
        ``(...,n,n)``.

    Notes
    -----
    The exact-zero branch uses a safe scale of one without perturbing nonzero
    inputs.  Rank decisions are relative to each matrix in the batch.
    """
    matrix = tf.convert_to_tensor(matrix)
    dtype = matrix.dtype
    if not dtype.is_floating:
        raise TypeError("svd_via_eigh_full requires a floating tensor")

    m = tf.shape(matrix)[-2]
    n = tf.shape(matrix)[-1]
    scale = tf.reduce_max(tf.abs(matrix), axis=(-2, -1))
    safe_scale = tf.where(scale == 0, tf.ones_like(scale), scale)
    scaled = matrix / safe_scale[..., None, None]

    if rank_rtol is None:
        relative_tolerance = automatic_rank_rtol(m, n, dtype)
    else:
        relative_tolerance = tf.cast(rank_rtol, dtype)
    absolute_tolerance = tf.cast(rank_atol, dtype) / safe_scale

    if m <= n:
        gram_left = _symmetrize(tf.matmul(scaled, scaled, transpose_b=True))
        eigenvalues, left_vectors = tf.linalg.eigh(gram_left)
        eigenvalues = tf.reverse(eigenvalues, axis=[-1])
        left_vectors = tf.reverse(left_vectors, axis=[-1])

        eigenvalues = tf.maximum(eigenvalues, tf.zeros_like(eigenvalues))
        safe_eigenvalues = tf.where(
            eigenvalues > 0,
            eigenvalues,
            tf.ones_like(eigenvalues),
        )
        singular_scaled = tf.where(
            eigenvalues > 0,
            tf.sqrt(safe_eigenvalues),
            tf.zeros_like(eigenvalues),
        )
        s_max = tf.reduce_max(singular_scaled, axis=-1, keepdims=True)
        tolerance = absolute_tolerance[..., None] + relative_tolerance * s_max
        resolved = singular_scaled > tolerance
        singular_scaled = tf.where(
            resolved,
            singular_scaled,
            tf.zeros_like(singular_scaled),
        )
        safe_singular = tf.where(
            resolved,
            singular_scaled,
            tf.ones_like(singular_scaled),
        )

        right_resolved = tf.matmul(scaled, left_vectors, transpose_a=True)
        right_resolved = right_resolved / safe_singular[..., None, :]
        right_resolved = tf.where(
            resolved[..., None, :],
            right_resolved,
            tf.zeros_like(right_resolved),
        )

        gram_right = _symmetrize(tf.matmul(scaled, scaled, transpose_a=True))
        _, right_full = tf.linalg.eigh(gram_right)
        right_full = tf.reverse(right_full, axis=[-1])
        replacement = right_full[..., :, :m]
        right_resolved = tf.where(
            resolved[..., None, :],
            right_resolved,
            replacement,
        )
        right_vectors = tf.concat(
            [right_resolved, right_full[..., :, m:]],
            axis=-1,
        )
    else:
        gram_right = _symmetrize(tf.matmul(scaled, scaled, transpose_a=True))
        eigenvalues, right_vectors = tf.linalg.eigh(gram_right)
        eigenvalues = tf.reverse(eigenvalues, axis=[-1])
        right_vectors = tf.reverse(right_vectors, axis=[-1])

        eigenvalues = tf.maximum(eigenvalues, tf.zeros_like(eigenvalues))
        safe_eigenvalues = tf.where(
            eigenvalues > 0,
            eigenvalues,
            tf.ones_like(eigenvalues),
        )
        singular_scaled = tf.where(
            eigenvalues > 0,
            tf.sqrt(safe_eigenvalues),
            tf.zeros_like(eigenvalues),
        )
        s_max = tf.reduce_max(singular_scaled, axis=-1, keepdims=True)
        tolerance = absolute_tolerance[..., None] + relative_tolerance * s_max
        resolved = singular_scaled > tolerance
        singular_scaled = tf.where(
            resolved,
            singular_scaled,
            tf.zeros_like(singular_scaled),
        )
        safe_singular = tf.where(
            resolved,
            singular_scaled,
            tf.ones_like(singular_scaled),
        )

        left_resolved = tf.matmul(scaled, right_vectors)
        left_resolved = left_resolved / safe_singular[..., None, :]
        left_resolved = tf.where(
            resolved[..., None, :],
            left_resolved,
            tf.zeros_like(left_resolved),
        )

        gram_left = _symmetrize(tf.matmul(scaled, scaled, transpose_b=True))
        _, left_full = tf.linalg.eigh(gram_left)
        left_full = tf.reverse(left_full, axis=[-1])
        replacement = left_full[..., :, :n]
        left_resolved = tf.where(
            resolved[..., None, :],
            left_resolved,
            replacement,
        )
        left_vectors = tf.concat(
            [left_resolved, left_full[..., :, n:]],
            axis=-1,
        )

    singular_values = singular_scaled * scale[..., None]
    return singular_values, left_vectors, right_vectors


@keras.saving.register_keras_serializable(package="crient")
class SpectralSVDLayer(keras.layers.Layer):
    """Full rectangular SVD layer with an explicit dtype and rank policy.

    Parameters
    ----------
    spectral_work_dtype
        ``"auto"``, ``"float32"`` or ``"float64"``.
    rank_rtol
        Relative rank tolerance. ``None`` selects the Gram-aware default.
    rank_atol
        Absolute rank tolerance in the input matrix scale.

    Returns
    -------
    tuple of tf.Tensor
        Singular values and complete left/right bases.  Public outputs follow
        the layer compute dtype.

    Notes
    -----
    The input shape is ``(batch, m, n)``.
    """

    def __init__(
        self,
        spectral_work_dtype: str = "auto",
        rank_rtol: float | None = None,
        rank_atol: float = 0.0,
        **kwargs,
    ):
        """Initialize the spectral backend and its rank policy."""
        super().__init__(**kwargs)
        if rank_rtol is not None and rank_rtol < 0:
            raise ValueError("rank_rtol must be non-negative or None")
        if rank_atol < 0:
            raise ValueError("rank_atol must be non-negative")
        # Validate eagerly even though the policy is resolved again in call.
        resolve_work_dtype(self.compute_dtype, spectral_work_dtype)
        self.spectral_work_dtype = spectral_work_dtype
        self.rank_rtol = rank_rtol
        self.rank_atol = float(rank_atol)

    def call(self, matrix):
        """Compute singular values and complete bases.

        Parameters
        ----------
        matrix : tf.Tensor
            Rank-3 floating input ``(batch, m, n)``.

        Returns
        -------
        tuple of tf.Tensor
            Singular values, left basis and right basis in compute dtype.
        """
        work_dtype = resolve_work_dtype(
            self.compute_dtype,
            self.spectral_work_dtype,
        )
        matrix_work = cast_to_work_dtype(matrix, work_dtype)
        outputs = svd_via_eigh_full(
            matrix_work,
            rank_rtol=self.rank_rtol,
            rank_atol=self.rank_atol,
        )
        singular_values, left_vectors, right_vectors = tuple(
            restore_compute_dtype(output, self.compute_dtype) for output in outputs
        )

        # ``svd_via_eigh_full`` also supports arbitrary leading batch
        # dimensions.  After TensorFlow has traced both rank-3 and higher-rank
        # calls, reduce_retracing may therefore generalize its cached output to
        # an unknown rank.  This layer's public contract is rank 3, so restore
        # that static rank before Keras performs input-shape inspection in the
        # downstream projection layers.
        singular_values = tf.ensure_shape(singular_values, (None, None))
        left_vectors = tf.ensure_shape(left_vectors, (None, None, None))
        right_vectors = tf.ensure_shape(right_vectors, (None, None, None))
        return singular_values, left_vectors, right_vectors

    def get_config(self):
        """Return the serializable dtype and rank-policy configuration."""
        return {
            **super().get_config(),
            "spectral_work_dtype": self.spectral_work_dtype,
            "rank_rtol": self.rank_rtol,
            "rank_atol": self.rank_atol,
        }

    @classmethod
    def from_config(cls, config):
        """Construct a spectral layer from serialized configuration."""
        return cls(**config)
