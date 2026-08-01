"""Public API and validation contract."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import tensorflow as tf

import crient
from crient import CrossRIEnetLayer, custom_layers
from crient.typing import ALL_OUTPUTS


def test_root_api_and_version():
    assert crient.__version__ == "0.2.0"
    assert crient.__all__ == [
        "CrossRIEnetLayer",
        "CrossRIEnetOutput",
        "__version__",
        "print_citation",
    ]
    assert custom_layers.SpectralSVDLayer.__module__ == "crient.spectral"


def test_removed_crossrie_package_is_not_importable():
    assert importlib.util.find_spec("crossrie") is None


def test_single_sequence_and_all_outputs(valid_blocks):
    single = CrossRIEnetLayer(
        output_type="spectral_coefficients",
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )(valid_blocks)
    assert isinstance(single, tf.Tensor)

    requested = CrossRIEnetLayer(
        output_type=(
            "cross_correlation",
            "spectral_coefficients",
            "cross_correlation",
        ),
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )(valid_blocks)
    assert tuple(requested) == (
        "cross_correlation",
        "spectral_coefficients",
    )

    all_outputs = CrossRIEnetLayer(
        output_type="all",
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )(valid_blocks)
    assert tuple(all_outputs) == ALL_OUTPUTS


@pytest.mark.parametrize("value", ["unknown", (), 3])
def test_invalid_output_type(value):
    with pytest.raises((TypeError, ValueError)):
        CrossRIEnetLayer(output_type=value)


def test_legacy_constructor_names_are_not_accepted():
    with pytest.raises(ValueError, match="Unrecognized keyword"):
        CrossRIEnetLayer(outputs=["Cxy"])


def test_sequence_and_mapping_inputs_match(valid_blocks):
    layer = CrossRIEnetLayer(
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )
    sequence = layer(valid_blocks, training=False)
    mapping = layer(
        {
            "correlation_x": valid_blocks[0],
            "correlation_y": valid_blocks[1],
            "cross_correlation": valid_blocks[2],
            "sample_size": valid_blocks[3],
        },
        training=False,
    )
    np.testing.assert_allclose(mapping, sequence, atol=0, rtol=0)


def test_sample_size_vector_and_column_match(valid_blocks):
    layer = CrossRIEnetLayer(
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )
    vector = layer(valid_blocks, training=False)
    column_inputs = (*valid_blocks[:3], valid_blocks[3][:, None])
    column = layer(column_inputs, training=False)
    np.testing.assert_allclose(column, vector, atol=0, rtol=0)


@pytest.mark.parametrize(
    "sample_size",
    [tf.constant(10.0), tf.ones((2, 2)), tf.constant([10.0, 0.0])],
)
def test_invalid_sample_size_is_rejected(valid_blocks, sample_size):
    layer = CrossRIEnetLayer(
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        layer((*valid_blocks[:3], sample_size))


def test_integer_matrices_are_rejected(valid_blocks):
    layer = CrossRIEnetLayer()
    with pytest.raises(TypeError, match="floating"):
        layer(
            (
                tf.cast(valid_blocks[0], tf.int32),
                valid_blocks[1],
                valid_blocks[2],
                tf.cast(valid_blocks[3], tf.int32),
            )
        )


def test_batch_broadcast_is_rejected(valid_blocks):
    layer = CrossRIEnetLayer()
    with pytest.raises(tf.errors.InvalidArgumentError, match="batch"):
        layer(
            (
                valid_blocks[0][:1],
                valid_blocks[1],
                valid_blocks[2],
                valid_blocks[3],
            )
        )


def test_strict_validation_accepts_valid_blocks(valid_blocks):
    output = CrossRIEnetLayer(
        validation_mode="strict",
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )(valid_blocks)
    assert bool(tf.reduce_all(tf.math.is_finite(output)))


def test_strict_validation_rejects_non_symmetric_marginal(valid_blocks):
    invalid_x = tf.tensor_scatter_nd_add(
        valid_blocks[0],
        indices=[[0, 0, 1]],
        updates=[0.1],
    )
    layer = CrossRIEnetLayer(
        validation_mode="strict",
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match="symmetric"):
        layer((invalid_x, *valid_blocks[1:]))


def test_strict_validation_rejects_infeasible_full_block():
    layer = CrossRIEnetLayer(
        validation_mode="strict",
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
    )
    inputs = (
        tf.eye(2, batch_shape=[1]),
        tf.eye(2, batch_shape=[1]),
        tf.constant([[[1.2, 0.0], [0.0, 1.1]]]),
        tf.constant([20.0]),
    )
    with pytest.raises(
        tf.errors.InvalidArgumentError,
        match="positive semidefinite",
    ):
        layer(inputs)


def test_default_parameter_count_matches_paper(valid_blocks):
    layer = CrossRIEnetLayer()
    layer(valid_blocks)
    assert layer.count_params() == 331_355
