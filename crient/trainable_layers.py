"""Trainable building blocks for CRIENT."""

from __future__ import annotations

from collections.abc import Sequence

import keras
import tensorflow as tf


def _positive_sizes(name: str, values: Sequence[int], *, allow_empty: bool):
    values = tuple(values)
    if not allow_empty and not values:
        raise ValueError(f"{name} must contain at least one positive integer")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in values
    ):
        raise ValueError(f"{name} must contain only positive integers")
    return values


@keras.saving.register_keras_serializable(package="crient")
class DeepLayer(keras.layers.Layer):
    """Position-wise multilayer perceptron with explicit training propagation."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activation: str = "leaky_relu",
        output_activation: str = "linear",
        dropout_rate: float = 0.0,
        use_hidden_bias: bool = True,
        use_output_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_sizes = _positive_sizes(
            "layer_sizes",
            layer_sizes,
            allow_empty=False,
        )
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = float(dropout_rate)
        self.use_hidden_bias = bool(use_hidden_bias)
        self.use_output_bias = bool(use_output_bias)
        self.kernel_initializer = kernel_initializer

        self.hidden_layers = tuple(
            keras.layers.Dense(
                size,
                activation=hidden_activation,
                use_bias=self.use_hidden_bias,
                kernel_initializer=kernel_initializer,
                dtype=self.dtype_policy,
                name=f"hidden_{index}",
            )
            for index, size in enumerate(self.layer_sizes[:-1])
        )
        self.dropouts = tuple(
            keras.layers.Dropout(
                self.dropout_rate,
                dtype=self.dtype_policy,
                name=f"dropout_{index}",
            )
            for index in range(len(self.layer_sizes) - 1)
        )
        self.output_layer = keras.layers.Dense(
            self.layer_sizes[-1],
            activation=output_activation,
            use_bias=self.use_output_bias,
            kernel_initializer=kernel_initializer,
            dtype=self.dtype_policy,
            name="output",
        )

    def build(self, input_shape):
        shape = tuple(input_shape)
        for dense, dropout in zip(
            self.hidden_layers,
            self.dropouts,
            strict=True,
        ):
            dense.build(shape)
            shape = dense.compute_output_shape(shape)
            dropout.build(shape)
        self.output_layer.build(shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        outputs = inputs
        for dense, dropout in zip(
            self.hidden_layers,
            self.dropouts,
            strict=True,
        ):
            outputs = dense(outputs)
            outputs = dropout(outputs, training=training)
        return self.output_layer(outputs)

    def get_config(self):
        return {
            **super().get_config(),
            "layer_sizes": self.layer_sizes,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "dropout_rate": self.dropout_rate,
            "use_hidden_bias": self.use_hidden_bias,
            "use_output_bias": self.use_output_bias,
            "kernel_initializer": self.kernel_initializer,
        }


@keras.saving.register_keras_serializable(package="crient")
class DeepRecurrentLayer(keras.layers.Layer):
    """Stacked recurrent aggregator followed by a point-wise output head."""

    def __init__(
        self,
        recurrent_layer_sizes: Sequence[int],
        head_layer_sizes: Sequence[int],
        recurrent_cell: str = "LSTM",
        recurrent_direction: str = "bidirectional",
        dropout_rate: float = 0.0,
        recurrent_dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.recurrent_layer_sizes = _positive_sizes(
            "recurrent_layer_sizes",
            recurrent_layer_sizes,
            allow_empty=False,
        )
        self.head_layer_sizes = _positive_sizes(
            "head_layer_sizes",
            head_layer_sizes,
            allow_empty=True,
        )
        recurrent_cell = recurrent_cell.strip().upper()
        if recurrent_cell not in {"LSTM", "GRU"}:
            raise ValueError("recurrent_cell must be 'LSTM' or 'GRU'")
        recurrent_direction = recurrent_direction.strip().lower()
        if recurrent_direction not in {"bidirectional", "forward", "backward"}:
            raise ValueError(
                "recurrent_direction must be 'bidirectional', 'forward', or 'backward'"
            )
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        if not 0 <= recurrent_dropout_rate < 1:
            raise ValueError("recurrent_dropout_rate must be in [0, 1)")

        self.recurrent_cell = recurrent_cell
        self.recurrent_direction = recurrent_direction
        self.dropout_rate = float(dropout_rate)
        self.recurrent_dropout_rate = float(recurrent_dropout_rate)

        recurrent_class = getattr(keras.layers, recurrent_cell)
        layers = []
        for index, units in enumerate(self.recurrent_layer_sizes):
            cell = recurrent_class(
                units,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout_rate,
                return_sequences=True,
                go_backwards=recurrent_direction == "backward",
                dtype=self.dtype_policy,
                name=f"{recurrent_cell.lower()}_{index}",
            )
            if recurrent_direction == "bidirectional":
                cell = keras.layers.Bidirectional(
                    cell,
                    dtype=self.dtype_policy,
                    name=f"bidirectional_{index}",
                )
            layers.append(cell)
        self.recurrent_layers = tuple(layers)
        self.head = DeepLayer(
            (*self.head_layer_sizes, 1),
            output_activation="linear",
            dropout_rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="head",
        )

    def build(self, input_shape):
        shape = tuple(input_shape)
        for recurrent_layer in self.recurrent_layers:
            recurrent_layer.build(shape)
            shape = recurrent_layer.compute_output_shape(shape)
        self.head.build(shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        outputs = inputs
        for recurrent_layer in self.recurrent_layers:
            outputs = recurrent_layer(outputs, training=training)
        outputs = self.head(outputs, training=training)
        return tf.squeeze(outputs, axis=-1)

    def get_config(self):
        return {
            **super().get_config(),
            "recurrent_layer_sizes": self.recurrent_layer_sizes,
            "head_layer_sizes": self.head_layer_sizes,
            "recurrent_cell": self.recurrent_cell,
            "recurrent_direction": self.recurrent_direction,
            "dropout_rate": self.dropout_rate,
            "recurrent_dropout_rate": self.recurrent_dropout_rate,
        }


@keras.saving.register_keras_serializable(package="crient")
class TwoStreamEncoderLayer(keras.layers.Layer):
    """Apply a shared encoder, sum fusion and recurrent spectral aggregator.

    Parameters
    ----------
    encoder_layer_sizes : sequence of int
        Widths of the shared point-wise encoder.
    recurrent_layer_sizes : sequence of int
        Widths of the recurrent stack.
    head_layer_sizes : sequence of int
        Hidden widths of the scalar output head.
    recurrent_cell : str
        ``"LSTM"`` or ``"GRU"``.
    recurrent_direction : str
        ``"bidirectional"``, ``"forward"`` or ``"backward"``.
    dropout_rate, recurrent_dropout_rate : float
        Keras dropout rates.
    **kwargs
        Standard Keras layer arguments.
    """

    def __init__(
        self,
        encoder_layer_sizes: Sequence[int],
        recurrent_layer_sizes: Sequence[int],
        head_layer_sizes: Sequence[int],
        recurrent_cell: str = "LSTM",
        recurrent_direction: str = "bidirectional",
        dropout_rate: float = 0.0,
        recurrent_dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_layer_sizes = _positive_sizes(
            "encoder_layer_sizes",
            encoder_layer_sizes,
            allow_empty=True,
        )
        self.recurrent_layer_sizes = _positive_sizes(
            "recurrent_layer_sizes",
            recurrent_layer_sizes,
            allow_empty=False,
        )
        self.head_layer_sizes = _positive_sizes(
            "head_layer_sizes",
            head_layer_sizes,
            allow_empty=True,
        )
        self.recurrent_cell = recurrent_cell
        self.recurrent_direction = recurrent_direction
        self.dropout_rate = float(dropout_rate)
        self.recurrent_dropout_rate = float(recurrent_dropout_rate)

        self.encoder = (
            DeepLayer(
                self.encoder_layer_sizes,
                dtype=self.dtype_policy,
                name="shared_encoder",
            )
            if self.encoder_layer_sizes
            else None
        )
        self.aggregator = DeepRecurrentLayer(
            recurrent_layer_sizes=self.recurrent_layer_sizes,
            head_layer_sizes=self.head_layer_sizes,
            recurrent_cell=recurrent_cell,
            recurrent_direction=recurrent_direction,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate,
            dtype=self.dtype_policy,
            name="aggregator",
        )

    def build(self, input_shape):
        first_shape = tuple(input_shape[0])
        if self.encoder is not None:
            self.encoder.build(first_shape)
            encoded_shape = (*first_shape[:-1], self.encoder_layer_sizes[-1])
        else:
            encoded_shape = first_shape
        self.aggregator.build(encoded_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Encode, fuse and aggregate two equal-shape token streams."""
        stream_x, stream_y = inputs
        if self.encoder is not None:
            encoded_x = self.encoder(stream_x, training=training)
            encoded_y = self.encoder(stream_y, training=training)
        else:
            encoded_x, encoded_y = stream_x, stream_y
        return self.aggregator(
            encoded_x + encoded_y,
            training=training,
        )

    def get_config(self):
        """Return the serializable trainable-stack configuration."""
        return {
            **super().get_config(),
            "encoder_layer_sizes": self.encoder_layer_sizes,
            "recurrent_layer_sizes": self.recurrent_layer_sizes,
            "head_layer_sizes": self.head_layer_sizes,
            "recurrent_cell": self.recurrent_cell,
            "recurrent_direction": self.recurrent_direction,
            "dropout_rate": self.dropout_rate,
            "recurrent_dropout_rate": self.recurrent_dropout_rate,
        }
