"""Public CrossRIEnet layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import keras
import tensorflow as tf

from .dtype_utils import cast_to_work_dtype, resolve_work_dtype
from .ops_layers import (
    ProjectedVarianceDiagonalLayer,
    SequencePaddingLayer,
    SVDReconstructionLayer,
)
from .spectral import SpectralSVDLayer
from .trainable_layers import TwoStreamEncoderLayer
from .typing import ALL_OUTPUTS
from .validation import validate_basic_inputs, validate_correlation_domain


def _normalize_output_type(output_type):
    if isinstance(output_type, str):
        if output_type == "all":
            return ALL_OUTPUTS, False, "all"
        if output_type not in ALL_OUTPUTS:
            raise ValueError(
                f"Unknown output_type {output_type!r}; expected one of "
                f"{(*ALL_OUTPUTS, 'all')}"
            )
        return (output_type,), True, output_type

    if not isinstance(output_type, Sequence):
        raise TypeError("output_type must be a string or a sequence of strings")
    requested = tuple(output_type)
    if not requested:
        raise ValueError("output_type must not be empty")
    expanded: list[str] = []
    for value in requested:
        if value == "all":
            expanded.extend(ALL_OUTPUTS)
        elif value in ALL_OUTPUTS:
            expanded.append(value)
        else:
            raise ValueError(
                f"Unknown output_type {value!r}; expected one of "
                f"{(*ALL_OUTPUTS, 'all')}"
            )
    components = tuple(dict.fromkeys(expanded))
    return components, False, requested


@keras.saving.register_keras_serializable(package="crient")
class CrossRIEnetLayer(keras.layers.Layer):
    """Clean a rectangular cross-correlation in its empirical SVD basis.

    The layer implements equations 1--7 of arXiv:2601.07687.  It receives two
    marginal correlation matrices, their empirical cross-correlation and a
    positive sample size.  It completes both singular-vector bases,
    projects the marginal correlations, pads to ``max(n_x, n_y)``, encodes the
    two streams with shared weights, fuses them by summation and applies a
    recurrent point-wise spectral correction.

    Parameters
    ----------
    output_type
        Output name, sequence of names or ``"all"``.  A single string returns
        a tensor; a sequence and ``"all"`` return a dictionary.
    encoder_layer_sizes
        Widths of the shared position-wise encoder.
    recurrent_layer_sizes
        Widths of the recurrent aggregator.
    head_layer_sizes
        Hidden widths of the point-wise correction head.
    correction_mode
        ``"additive"``, ``"bounded_multiplicative"`` or
        ``"positive_multiplicative"``.
    additive_activation
        ``"linear"`` or ``"tanh"`` for additive corrections.  It must remain
        ``"linear"`` for multiplicative modes.
    recurrent_cell
        ``"LSTM"`` or ``"GRU"``.
    recurrent_direction
        ``"bidirectional"``, ``"forward"`` or ``"backward"``.
    validation_mode
        ``"basic"`` validates structure and finite values. ``"strict"`` also
        checks symmetry, unit diagonals and positive-semidefiniteness of the
        complete correlation block.
    spectral_work_dtype
        ``"auto"``, ``"float32"`` or ``"float64"``.
    rank_rtol, rank_atol
        Relative and absolute numerical-rank tolerances used by the Gram/eigh
        SVD backend.

    Notes
    -----
    Inputs are supplied as a four-element sequence or mapping:
    ``correlation_x`` has shape ``(batch, n_x, n_x)``, ``correlation_y`` has
    shape ``(batch, n_y, n_y)``, ``cross_correlation`` has shape
    ``(batch, n_x, n_y)`` and ``sample_size`` has shape ``(batch,)`` or
    ``(batch, 1)``.

    Padding tokens represent real marginal directions from the larger block
    and are intentionally not masked.  Signed additive outputs are spectral
    coefficients, not necessarily mathematical singular values.  Equivariance
    is not guaranteed inside exactly degenerate singular subspaces.
    """

    def __init__(
        self,
        output_type: str | Sequence[str] = "cross_correlation",
        encoder_layer_sizes: Sequence[int] = (16, 2),
        recurrent_layer_sizes: Sequence[int] = (128, 64),
        head_layer_sizes: Sequence[int] = (252,),
        correction_mode: str = "additive",
        additive_activation: str = "linear",
        recurrent_cell: str = "LSTM",
        recurrent_direction: str = "bidirectional",
        validation_mode: str = "basic",
        spectral_work_dtype: str = "auto",
        rank_rtol: float | None = None,
        rank_atol: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize a CrossRIEnet cleaner.

        Parameters
        ----------
        output_type : str or sequence of str
            Requested output, outputs or ``"all"``.
        encoder_layer_sizes : sequence of int
            Shared encoder widths.
        recurrent_layer_sizes : sequence of int
            Recurrent aggregator widths.
        head_layer_sizes : sequence of int
            Point-wise correction-head hidden widths.
        correction_mode : str
            Additive, bounded-multiplicative or positive-multiplicative mode.
        additive_activation : str
            ``"linear"`` or ``"tanh"`` in additive mode.
        recurrent_cell : str
            ``"LSTM"`` or ``"GRU"``.
        recurrent_direction : str
            Bidirectional, forward or backward aggregation.
        validation_mode : str
            ``"basic"`` or ``"strict"``.
        spectral_work_dtype : str
            ``"auto"``, ``"float32"`` or ``"float64"``.
        rank_rtol, rank_atol : float, optional
            Relative and absolute numerical-rank tolerances.
        **kwargs
            Standard Keras layer arguments, including ``name`` and ``dtype``.
        """
        super().__init__(**kwargs)
        (
            self.output_components,
            self._single_output,
            self._output_config,
        ) = _normalize_output_type(output_type)

        correction_mode = correction_mode.strip().lower()
        if correction_mode not in {
            "additive",
            "bounded_multiplicative",
            "positive_multiplicative",
        }:
            raise ValueError(
                "correction_mode must be 'additive', "
                "'bounded_multiplicative', or 'positive_multiplicative'"
            )
        if additive_activation not in {"linear", "tanh"}:
            raise ValueError("additive_activation must be 'linear' or 'tanh'")
        if correction_mode != "additive" and additive_activation != "linear":
            raise ValueError(
                "additive_activation is only configurable in additive mode"
            )
        if validation_mode not in {"basic", "strict"}:
            raise ValueError("validation_mode must be 'basic' or 'strict'")

        self.encoder_layer_sizes = tuple(encoder_layer_sizes)
        self.recurrent_layer_sizes = tuple(recurrent_layer_sizes)
        self.head_layer_sizes = tuple(head_layer_sizes)
        self.correction_mode = correction_mode
        self.additive_activation = additive_activation
        self.recurrent_cell = recurrent_cell.strip().upper()
        self.recurrent_direction = recurrent_direction.strip().lower()
        self.validation_mode = validation_mode
        self.spectral_work_dtype = spectral_work_dtype
        self.rank_rtol = rank_rtol
        self.rank_atol = float(rank_atol)

        self.svd = SpectralSVDLayer(
            spectral_work_dtype=spectral_work_dtype,
            rank_rtol=rank_rtol,
            rank_atol=rank_atol,
            dtype=self.dtype_policy,
            name="spectral_svd",
        )
        self.project_x = ProjectedVarianceDiagonalLayer(
            dtype=self.dtype_policy,
            name="project_x",
        )
        self.project_y = ProjectedVarianceDiagonalLayer(
            dtype=self.dtype_policy,
            name="project_y",
        )
        self.padding = SequencePaddingLayer(
            dtype=self.dtype_policy,
            name="sequence_padding",
        )
        self.encoder = TwoStreamEncoderLayer(
            encoder_layer_sizes=self.encoder_layer_sizes,
            recurrent_layer_sizes=self.recurrent_layer_sizes,
            head_layer_sizes=self.head_layer_sizes,
            recurrent_cell=self.recurrent_cell,
            recurrent_direction=self.recurrent_direction,
            dtype=self.dtype_policy,
            name="two_stream_encoder",
        )
        self.reconstruction = SVDReconstructionLayer(
            dtype=self.dtype_policy,
            name="reconstruction",
        )
        self.supports_masking = False

    def build(self, input_shape: Any) -> None:
        """Build the shape-independent trainable stack.

        Parameters
        ----------
        input_shape
            Keras structure containing the four public input shapes.
        """
        self.encoder.build(
            [
                (None, None, 3),
                (None, None, 3),
            ]
        )
        super().build(input_shape)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        """Apply the spectral cleaner.

        Parameters
        ----------
        inputs
            Sequence or mapping described in the class-level input contract.
        training
            Keras training flag propagated to recurrent and dropout layers.

        Returns
        -------
        tf.Tensor or dict[str, tf.Tensor]
            One tensor for a string output request, otherwise an ordered
            dictionary.

        Raises
        ------
        TypeError
            If matrix dtypes or the input structure are invalid.
        ValueError
            If ranks or the sample-size shape are invalid.
        tf.errors.InvalidArgumentError
            If runtime shapes, values or strict domain checks fail.
        """
        (
            correlation_x,
            correlation_y,
            cross_correlation,
            sample_size,
        ) = validate_basic_inputs(inputs, self.compute_dtype)

        if self.validation_mode == "strict":
            work_dtype = resolve_work_dtype(
                self.compute_dtype,
                self.spectral_work_dtype,
            )
            validate_correlation_domain(
                cast_to_work_dtype(correlation_x, work_dtype),
                cast_to_work_dtype(correlation_y, work_dtype),
                cast_to_work_dtype(cross_correlation, work_dtype),
            )

        empirical_s, left_vectors, right_vectors = self.svd(cross_correlation)
        projected_x = self.project_x([correlation_x, left_vectors])
        projected_y = self.project_y([correlation_y, right_vectors])

        n_x = tf.shape(cross_correlation)[1]
        n_y = tf.shape(cross_correlation)[2]
        rank = tf.shape(empirical_s)[1]
        target_length = tf.maximum(n_x, n_y)

        gamma_x = projected_x[..., None]
        gamma_y = projected_y[..., None]
        q_x = tf.cast(n_x, self.compute_dtype) / sample_size
        q_y = tf.cast(n_y, self.compute_dtype) / sample_size
        q_x = tf.ones_like(gamma_x) * q_x[:, None, None]
        q_y = tf.ones_like(gamma_y) * q_y[:, None, None]
        singular_tokens = empirical_s[..., None]

        gamma_x = self.padding([gamma_x, target_length])
        gamma_y = self.padding([gamma_y, target_length])
        q_x = self.padding([q_x, target_length])
        q_y = self.padding([q_y, target_length])
        singular_tokens = self.padding([singular_tokens, target_length])

        # Equation 4 channel order: [projected variance, singular value, q].
        stream_x = tf.concat([gamma_x, singular_tokens, q_x], axis=-1)
        stream_y = tf.concat([gamma_y, singular_tokens, q_y], axis=-1)
        raw_correction = self.encoder(
            [stream_x, stream_y],
            training=training,
        )[:, :rank]

        if self.correction_mode == "additive":
            correction = (
                tf.tanh(raw_correction)
                if self.additive_activation == "tanh"
                else raw_correction
            )
            spectral_coefficients = empirical_s + correction
        elif self.correction_mode == "bounded_multiplicative":
            correction = tf.sigmoid(raw_correction)
            spectral_coefficients = empirical_s * correction
        else:
            correction = tf.nn.softplus(raw_correction)
            spectral_coefficients = empirical_s * correction

        available: dict[str, Any] = {
            "spectral_coefficients": spectral_coefficients,
            "empirical_singular_values": empirical_s,
            "correction": correction,
            "left_singular_vectors": left_vectors,
            "right_singular_vectors": right_vectors,
            "projected_variance_x": projected_x,
            "projected_variance_y": projected_y,
        }
        if "cross_correlation" in self.output_components:
            available["cross_correlation"] = self.reconstruction(
                [spectral_coefficients, left_vectors, right_vectors]
            )

        if self._single_output:
            return available[self.output_components[0]]
        return {name: available[name] for name in self.output_components}

    def get_config(self) -> dict[str, Any]:
        """Return the complete serializable constructor configuration."""
        return {
            **super().get_config(),
            "output_type": self._output_config,
            "encoder_layer_sizes": self.encoder_layer_sizes,
            "recurrent_layer_sizes": self.recurrent_layer_sizes,
            "head_layer_sizes": self.head_layer_sizes,
            "correction_mode": self.correction_mode,
            "additive_activation": self.additive_activation,
            "recurrent_cell": self.recurrent_cell,
            "recurrent_direction": self.recurrent_direction,
            "validation_mode": self.validation_mode,
            "spectral_work_dtype": self.spectral_work_dtype,
            "rank_rtol": self.rank_rtol,
            "rank_atol": self.rank_atol,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CrossRIEnetLayer:
        """Construct a layer from a canonical CRIENT configuration."""
        return cls(**config)
