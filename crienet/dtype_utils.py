"""Dtype policy helpers.

Sensitive spectral operations run in at least float32.  Float64 policies stay
in float64, while low-precision policies restore their declared compute dtype
at the public layer boundary.
"""

from __future__ import annotations

import tensorflow as tf

LOW_PRECISION_DTYPES = (tf.float16, tf.bfloat16)


def resolve_work_dtype(
    compute_dtype: tf.dtypes.DType | str,
    spectral_work_dtype: str | tf.dtypes.DType = "auto",
) -> tf.dtypes.DType:
    """Resolve the dtype used by sensitive spectral operations.

    Parameters
    ----------
    compute_dtype
        Keras compute dtype of the owning layer.
    spectral_work_dtype
        ``"auto"``, ``"float32"`` or ``"float64"``.  ``"auto"`` keeps
        float64 policies in float64 and otherwise uses float32.

    Returns
    -------
    tf.dtypes.DType
        Floating dtype used internally.
    """
    compute_dtype = tf.dtypes.as_dtype(compute_dtype)
    if spectral_work_dtype == "auto":
        return tf.float64 if compute_dtype == tf.float64 else tf.float32

    work_dtype = tf.dtypes.as_dtype(spectral_work_dtype)
    if work_dtype not in (tf.float32, tf.float64):
        raise ValueError("spectral_work_dtype must be 'auto', 'float32', or 'float64'")
    if compute_dtype == tf.float64 and work_dtype != tf.float64:
        raise ValueError(
            "A float64 compute policy requires spectral_work_dtype='auto' or 'float64'."
        )
    return work_dtype


def machine_epsilon(dtype: tf.dtypes.DType | str) -> tf.Tensor:
    """Return the machine epsilon of a supported work dtype."""
    dtype = tf.dtypes.as_dtype(dtype)
    if dtype == tf.float64:
        value = 2.220446049250313e-16
    elif dtype == tf.float32:
        value = 1.1920928955078125e-7
    elif dtype == tf.float16:
        value = 9.765625e-4
    elif dtype == tf.bfloat16:
        value = 7.8125e-3
    else:
        raise TypeError(f"Unsupported floating dtype: {dtype.name}")
    return tf.cast(value, dtype)


def cast_to_work_dtype(
    tensor: tf.Tensor,
    work_dtype: tf.dtypes.DType | str,
) -> tf.Tensor:
    """Cast a tensor to the selected work dtype."""
    return tf.cast(tensor, tf.dtypes.as_dtype(work_dtype))


def restore_compute_dtype(
    tensor: tf.Tensor,
    compute_dtype: tf.dtypes.DType | str,
) -> tf.Tensor:
    """Cast a public output to the owning Keras layer's compute dtype."""
    return tf.cast(tensor, tf.dtypes.as_dtype(compute_dtype))
