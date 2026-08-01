"""Input and scientific-domain validation for CRIENet."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import tensorflow as tf

INPUT_NAMES = (
    "correlation_x",
    "correlation_y",
    "cross_correlation",
    "sample_size",
)


def unpack_inputs(inputs):
    """Normalize the supported sequence and mapping input structures."""
    if isinstance(inputs, Mapping):
        missing = [name for name in INPUT_NAMES if name not in inputs]
        extra = [name for name in inputs if name not in INPUT_NAMES]
        if missing or extra:
            raise ValueError(
                "Input mapping must contain exactly "
                f"{INPUT_NAMES}; missing={missing}, extra={extra}"
            )
        return tuple(inputs[name] for name in INPUT_NAMES)

    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise TypeError(
            "inputs must be a four-element sequence or a mapping with "
            f"keys {INPUT_NAMES}"
        )
    if len(inputs) != 4:
        raise ValueError(f"Expected four inputs, received {len(inputs)}")
    return tuple(inputs)


def _require_floating(name: str, tensor: tf.Tensor):
    if not tensor.dtype.is_floating:
        raise TypeError(f"{name} must have a floating dtype")


def validate_basic_inputs(inputs, compute_dtype):
    """Validate and canonicalize the public main-layer input contract."""
    correlation_x, correlation_y, cross_correlation, sample_size = unpack_inputs(inputs)
    correlation_x = tf.convert_to_tensor(correlation_x)
    correlation_y = tf.convert_to_tensor(correlation_y)
    cross_correlation = tf.convert_to_tensor(cross_correlation)
    sample_size = tf.convert_to_tensor(sample_size)

    for name, matrix in (
        ("correlation_x", correlation_x),
        ("correlation_y", correlation_y),
        ("cross_correlation", cross_correlation),
    ):
        _require_floating(name, matrix)
        if matrix.shape.rank is not None and matrix.shape.rank != 3:
            raise ValueError(f"{name} must have rank 3")
        tf.debugging.assert_rank(matrix, 3, message=f"{name} must have rank 3")
        tf.debugging.assert_all_finite(
            matrix,
            f"{name} must contain only finite values",
        )

    if not (sample_size.dtype.is_integer or sample_size.dtype.is_floating):
        raise TypeError("sample_size must have an integer or floating dtype")
    if sample_size.shape.rank == 2:
        if sample_size.shape[-1] is not None and sample_size.shape[-1] != 1:
            raise ValueError("sample_size rank-2 form must have shape (batch, 1)")
        tf.debugging.assert_equal(
            tf.shape(sample_size)[1],
            1,
            message="sample_size rank-2 form must have shape (batch, 1)",
        )
        sample_size = tf.squeeze(sample_size, axis=-1)
    elif sample_size.shape.rank != 1:
        raise ValueError("sample_size must have shape (batch,) or (batch, 1)")

    batch = tf.shape(cross_correlation)[0]
    n_x = tf.shape(cross_correlation)[1]
    n_y = tf.shape(cross_correlation)[2]
    assertions = (
        tf.debugging.assert_equal(
            tf.shape(correlation_x)[0],
            batch,
            message="correlation_x batch dimension must match cross_correlation",
        ),
        tf.debugging.assert_equal(
            tf.shape(correlation_y)[0],
            batch,
            message="correlation_y batch dimension must match cross_correlation",
        ),
        tf.debugging.assert_equal(
            tf.shape(sample_size)[0],
            batch,
            message="sample_size batch dimension must match the matrices",
        ),
        tf.debugging.assert_equal(
            tf.shape(correlation_x)[1],
            tf.shape(correlation_x)[2],
            message="correlation_x must be square",
        ),
        tf.debugging.assert_equal(
            tf.shape(correlation_y)[1],
            tf.shape(correlation_y)[2],
            message="correlation_y must be square",
        ),
        tf.debugging.assert_equal(
            tf.shape(correlation_x)[1],
            n_x,
            message="correlation_x size must match cross_correlation rows",
        ),
        tf.debugging.assert_equal(
            tf.shape(correlation_y)[1],
            n_y,
            message="correlation_y size must match cross_correlation columns",
        ),
        tf.debugging.assert_positive(
            sample_size,
            message="sample_size must be strictly positive",
        ),
    )
    with tf.control_dependencies(assertions):
        dtype = tf.dtypes.as_dtype(compute_dtype)
        return (
            tf.cast(tf.identity(correlation_x), dtype),
            tf.cast(tf.identity(correlation_y), dtype),
            tf.cast(tf.identity(cross_correlation), dtype),
            tf.cast(tf.identity(sample_size), dtype),
        )


def domain_tolerance(matrix: tf.Tensor) -> tf.Tensor:
    """Return a scale-aware tolerance for strict diagnostic assertions."""
    dtype = matrix.dtype
    if dtype == tf.float64:
        eps = tf.cast(2.220446049250313e-16, dtype)
    else:
        eps = tf.cast(1.1920928955078125e-7, dtype)
    dimension = tf.cast(tf.shape(matrix)[-1], dtype)
    scale = tf.maximum(
        tf.reduce_max(tf.abs(matrix), axis=(-2, -1)),
        tf.ones(tf.shape(matrix)[:-2], dtype=dtype),
    )
    return 100.0 * dimension * eps * scale


def validate_correlation_domain(
    correlation_x: tf.Tensor,
    correlation_y: tf.Tensor,
    cross_correlation: tf.Tensor,
):
    """Assert symmetry, unit diagonal and positive-semidefinite full blocks."""
    for name, matrix in (
        ("correlation_x", correlation_x),
        ("correlation_y", correlation_y),
    ):
        tolerance = domain_tolerance(matrix)
        symmetry_error = tf.reduce_max(
            tf.abs(matrix - tf.linalg.matrix_transpose(matrix)),
            axis=(-2, -1),
        )
        tf.debugging.assert_less_equal(
            symmetry_error,
            tolerance,
            message=f"{name} must be symmetric within dtype-aware tolerance",
        )
        diagonal_error = tf.reduce_max(
            tf.abs(tf.linalg.diag_part(matrix) - 1),
            axis=-1,
        )
        tf.debugging.assert_less_equal(
            diagonal_error,
            tolerance,
            message=f"{name} must have a unit diagonal",
        )

    top = tf.concat([correlation_x, cross_correlation], axis=-1)
    bottom = tf.concat(
        [tf.linalg.matrix_transpose(cross_correlation), correlation_y],
        axis=-1,
    )
    full_block = tf.concat([top, bottom], axis=-2)
    minimum_eigenvalue = tf.reduce_min(tf.linalg.eigvalsh(full_block), axis=-1)
    tolerance = domain_tolerance(full_block)
    tf.debugging.assert_greater_equal(
        minimum_eigenvalue,
        -tolerance,
        message="The complete correlation block must be positive semidefinite",
    )
