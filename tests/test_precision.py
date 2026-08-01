"""Keras dtype-policy tests."""

from __future__ import annotations

import pytest
import tensorflow as tf

from crient import CrossRIEnetLayer


def _inputs(dtype):
    return [
        tf.eye(2, batch_shape=[1], dtype=dtype),
        tf.eye(3, batch_shape=[1], dtype=dtype),
        tf.constant([[[0.4, -0.1, 0.05], [0.02, 0.3, -0.08]]], dtype),
        tf.constant([20.0], dtype),
    ]


@pytest.mark.parametrize(
    ("policy_name", "expected_output", "expected_variable"),
    [
        ("float32", tf.float32, "float32"),
        ("float64", tf.float64, "float64"),
        ("mixed_float16", tf.float16, "float32"),
        ("mixed_bfloat16", tf.bfloat16, "float32"),
    ],
)
def test_main_layer_dtype_contract(
    policy_name,
    expected_output,
    expected_variable,
):
    previous = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy(policy_name)
        layer = CrossRIEnetLayer(
            encoder_layer_sizes=(3,),
            recurrent_layer_sizes=(3,),
            head_layer_sizes=(2,),
        )
        output = layer(_inputs(tf.float64))
        assert output.dtype == expected_output
        assert {weight.dtype for weight in layer.weights} == {expected_variable}
        assert bool(tf.reduce_all(tf.math.is_finite(output)))
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)


@pytest.mark.parametrize(
    "policy_name",
    ["float32", "float64", "mixed_float16", "mixed_bfloat16"],
)
def test_forward_and_backward_are_finite(policy_name):
    previous = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy(policy_name)
        dtype = tf.float64 if policy_name == "float64" else tf.float32
        layer = CrossRIEnetLayer(
            encoder_layer_sizes=(3,),
            recurrent_layer_sizes=(3,),
            head_layer_sizes=(2,),
        )
        inputs = _inputs(dtype)
        with tf.GradientTape() as tape:
            output = layer(inputs, training=True)
            loss = tf.reduce_sum(tf.square(tf.cast(output, tf.float32)))
        gradients = tape.gradient(loss, layer.trainable_variables)
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(
            bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients
        )
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
