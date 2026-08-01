"""Keras serialization tests for the canonical CRIENet API."""

import numpy as np
import pytest
import tensorflow as tf

from crienet import CrossRIEnetLayer


@pytest.mark.parametrize(
    "output_type",
    [
        "cross_correlation",
        ("cross_correlation", "spectral_coefficients"),
        "all",
    ],
)
def test_model_keras_roundtrip_preserves_outputs(tmp_path, output_type):
    tf.keras.utils.set_random_seed(11)
    input_x = tf.keras.Input(shape=(None, None), name="correlation_x")
    input_y = tf.keras.Input(shape=(None, None), name="correlation_y")
    input_xy = tf.keras.Input(shape=(None, None), name="cross_correlation")
    input_t = tf.keras.Input(shape=(), name="sample_size")
    output = CrossRIEnetLayer(
        encoder_layer_sizes=(3,),
        recurrent_layer_sizes=(3,),
        head_layer_sizes=(2,),
        output_type=output_type,
    )([input_x, input_y, input_xy, input_t])
    model = tf.keras.Model(
        inputs=[input_x, input_y, input_xy, input_t],
        outputs=output,
    )

    inputs = [
        tf.eye(2, batch_shape=[1], dtype=tf.float32),
        tf.eye(3, batch_shape=[1], dtype=tf.float32),
        tf.random.normal((1, 2, 3), dtype=tf.float32),
        tf.constant([10.0], dtype=tf.float32),
    ]
    before = model(inputs, training=False)
    path = tmp_path / "crienet_roundtrip.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path)
    after = loaded(inputs, training=False)

    if isinstance(before, dict):
        assert after.keys() == before.keys()
        for name in before:
            np.testing.assert_allclose(after[name], before[name], atol=0, rtol=0)
    else:
        np.testing.assert_allclose(after, before, atol=0, rtol=0)

    second_inputs = [
        tf.eye(4, batch_shape=[2], dtype=tf.float32),
        tf.eye(2, batch_shape=[2], dtype=tf.float32),
        tf.random.normal((2, 4, 2), dtype=tf.float32),
        tf.constant([12.0, 18.0], dtype=tf.float32),
    ]
    second = loaded(second_inputs, training=False)
    second_matrix = second["cross_correlation"] if isinstance(second, dict) else second
    if output_type == "spectral_coefficients":
        assert second_matrix.shape == (2, 2)
    else:
        assert second_matrix.shape == (2, 4, 2)


def test_layer_config_roundtrip_preserves_tuples():
    layer = CrossRIEnetLayer(
        output_type=("cross_correlation", "correction"),
        encoder_layer_sizes=(5, 2),
        recurrent_layer_sizes=(7,),
        head_layer_sizes=(4,),
        correction_mode="positive_multiplicative",
        recurrent_cell="GRU",
        recurrent_direction="forward",
        spectral_work_dtype="float32",
        rank_rtol=1e-4,
        rank_atol=1e-8,
    )

    restored = CrossRIEnetLayer.from_config(layer.get_config())

    assert restored.output_components == layer.output_components
    assert restored.encoder_layer_sizes == (5, 2)
    assert restored.recurrent_layer_sizes == (7,)
    assert restored.head_layer_sizes == (4,)
    assert restored.correction_mode == "positive_multiplicative"


@pytest.mark.parametrize(
    ("policy_name", "expected_dtype"),
    [("float64", tf.float64), ("mixed_float16", tf.float16)],
)
def test_dtype_policy_survives_keras_roundtrip(
    tmp_path,
    policy_name,
    expected_dtype,
):
    previous = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy(policy_name)
        dtype = tf.float64 if policy_name == "float64" else tf.float32
        inputs = [
            tf.keras.Input((None, None), dtype=dtype),
            tf.keras.Input((None, None), dtype=dtype),
            tf.keras.Input((None, None), dtype=dtype),
            tf.keras.Input((), dtype=dtype),
        ]
        output = CrossRIEnetLayer(
            encoder_layer_sizes=(3,),
            recurrent_layer_sizes=(3,),
            head_layer_sizes=(2,),
        )(inputs)
        model = tf.keras.Model(inputs, output)
        path = tmp_path / f"{policy_name}.keras"
        model.save(path)
        loaded = tf.keras.models.load_model(path)

        values = [
            tf.eye(2, batch_shape=[1], dtype=dtype),
            tf.eye(3, batch_shape=[1], dtype=dtype),
            tf.random.normal((1, 2, 3), dtype=dtype),
            tf.constant([10.0], dtype=dtype),
        ]
        assert loaded(values).dtype == expected_dtype
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
