# CRIENT: CrossRIEnet for Rectangular Cross-Correlation Cleaning

[![Python 3.10--3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**This library implements the neural estimator introduced in:**

- **Manolakis, E., Bongiorno, C., & Mantegna, R. N. (2026).
  Physics-Informed Singular-Value Learning for Cross-Covariances Forecasting
  in Financial Markets. [arXiv:2601.07687v3](https://arxiv.org/abs/2601.07687).**

CRIENT is a TensorFlow/Keras research implementation for cleaning rectangular
cross-correlation matrices. Given two marginal correlation matrices, their
empirical cross-correlation and the corresponding sample size, the model
learns corrections to the empirical spectral coefficients and reconstructs a
cross-correlation matrix in the empirical singular-vector basis.

The package operates on correlation matrices. Estimation of marginal
volatilities, conversion to cross-covariances and preprocessing of raw returns
are outside the core API.

Version 0.2 is an unreleased API revision. It does not provide compatibility
aliases for the earlier `crossrie` package.

## Relationship to RIEnet

[RIEnet](https://github.com/bongiornoc/RIEnet) applies learned spectral
corrections to square covariance and correlation matrices through an
eigendecomposition. CrossRIEnet applies a related construction to rectangular
cross-correlation matrices through a singular-value decomposition.

CRIENT is currently an independent package. It does not import RIEnet or rely
on RIEnet private symbols. Matrix outputs from the two packages can be composed
in an external workflow when marginal covariance or precision estimates are
also required.

## What this package provides

- A Keras layer for learned cleaning of rectangular cross-correlations
- Additive, bounded-multiplicative and positive-multiplicative corrections
- Access to the reconstructed matrix, spectral coefficients and intermediate
  spectral quantities
- Support for varying batch size and matrix dimensions in one traced function
- Float32, float64 and Keras mixed-precision policies
- Structural validation and optional validation of the full correlation block
- Keras `.keras` serialization and training through `Model.fit`

## Module organization

- `crient.layer`: the public `CrossRIEnetLayer`.
- `crient.trainable_layers`: the shared encoder, recurrent aggregator and
  correction head.
- `crient.ops_layers`: deterministic projection, padding and reconstruction
  layers.
- `crient.spectral`: the full-basis spectral backend.
- `crient.validation`: input and correlation-domain validation.
- `crient.diagnostics`: non-mutating feasibility diagnostics.

## Installation

CRIENT has not been published to PyPI. Install it from a local checkout:

```bash
git clone https://github.com/Efstratios7/CrossRIE.git
cd CrossRIE
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

The supplied Conda environment can be created with:

```bash
conda env update --file environment.yml --prune
conda activate crient_env
python -m pip install -e ".[dev]"
```

## Quick start

The examples below construct the three input matrices from the same joint
return sample. This ensures that the marginal matrices and cross-correlation
belong to one valid correlation block.

```python
import tensorflow as tf

from crient import CrossRIEnetLayer


def correlation_blocks(returns, n_x):
    """Compute correlation blocks from returns shaped (batch, time, variables)."""
    centered = returns - tf.reduce_mean(returns, axis=1, keepdims=True)
    standardized = centered / tf.math.reduce_std(
        centered,
        axis=1,
        keepdims=True,
    )
    sample_size = tf.shape(standardized)[1]
    correlation = tf.matmul(
        standardized,
        standardized,
        transpose_a=True,
    ) / tf.cast(sample_size, standardized.dtype)
    return (
        correlation[:, :n_x, :n_x],
        correlation[:, n_x:, n_x:],
        correlation[:, :n_x, n_x:],
    )


batch_size = 32
sample_size = 60
n_x = 10
n_y = 8

returns = tf.random.stateless_normal(
    (batch_size, sample_size, n_x + n_y),
    seed=(1, 2),
)
correlation_x, correlation_y, cross_correlation = correlation_blocks(
    returns,
    n_x,
)

inputs = {
    "correlation_x": correlation_x,
    "correlation_y": correlation_y,
    "cross_correlation": cross_correlation,
    "sample_size": tf.fill((batch_size,), tf.cast(sample_size, tf.float32)),
}

cleaned = CrossRIEnetLayer(output_type="cross_correlation")(
    inputs,
    training=False,
)

print(cleaned.shape)  # (32, 10, 8)
```

The input may also be supplied as a four-element sequence in this order:

```python
cleaned = CrossRIEnetLayer()(
    [
        correlation_x,
        correlation_y,
        cross_correlation,
        inputs["sample_size"],
    ],
    training=False,
)
```

`sample_size` accepts shape `(batch,)` or `(batch, 1)`. It may use an integer
or floating dtype, must be strictly positive, and may differ between samples in
the same batch.

## Training

The example below generates one regularized Wishart correlation population per
training example. Empirical blocks are estimated from finite Gaussian samples;
the corresponding population cross-correlation is the target. This illustrates
the `Model.fit` interface, not the training protocol used in the paper.

```python
import tensorflow as tf

from crient import CrossRIEnetLayer


def generate_populations(count, dimension, seed=(1, 2)):
    factors = tf.random.stateless_normal((count, dimension, dimension), seed)
    covariance = tf.matmul(factors, factors, transpose_b=True)
    covariance += 0.5 * tf.eye(dimension)
    scale = tf.sqrt(tf.linalg.diag_part(covariance))
    return covariance / scale[..., :, None] / scale[..., None, :]


def make_dataset(populations, n_x, sample_size, batch_size, seed=(3, 4)):
    count, dimension = tf.shape(populations)[0], tf.shape(populations)[1]
    noise = tf.random.stateless_normal(tf.stack([count, sample_size, dimension]), seed)
    returns = tf.matmul(noise, tf.linalg.cholesky(populations), transpose_b=True)
    returns -= tf.reduce_mean(returns, axis=1, keepdims=True)
    returns /= tf.math.reduce_std(returns, axis=1, keepdims=True)
    empirical = tf.matmul(returns, returns, transpose_a=True) / sample_size
    inputs = {
        "correlation_x": empirical[:, :n_x, :n_x],
        "correlation_y": empirical[:, n_x:, n_x:],
        "cross_correlation": empirical[:, :n_x, n_x:],
        "sample_size": tf.fill([count], tf.cast(sample_size, tf.float32)),
    }
    targets = populations[:, :n_x, n_x:]
    return tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(batch_size)


n_x, n_y = 6, 8
dataset = make_dataset(
    generate_populations(256, n_x + n_y),
    n_x=n_x,
    sample_size=64,
    batch_size=32,
)
inputs = {
    "correlation_x": tf.keras.Input((None, None), name="correlation_x"),
    "correlation_y": tf.keras.Input((None, None), name="correlation_y"),
    "cross_correlation": tf.keras.Input((None, None), name="cross_correlation"),
    "sample_size": tf.keras.Input((), name="sample_size"),
}
model = tf.keras.Model(inputs, CrossRIEnetLayer()(inputs))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss="mse",
)
model.fit(dataset, epochs=10)
```

Dense tensors in one call to `fit` have fixed `n_x` and `n_y`. Training across
different matrix dimensions requires a `tf.data.Dataset` that yields batches
with the corresponding dynamic `TensorSpec` objects. One batch must still
contain matrices with a common shape.

## Selecting outputs

`output_type` accepts one output name, a sequence of names or `"all"`.

```python
# One output name returns a tensor.
cleaned = CrossRIEnetLayer(
    output_type="cross_correlation",
)(inputs)

# A sequence returns a dictionary.
outputs = CrossRIEnetLayer(
    output_type=("cross_correlation", "spectral_coefficients"),
)(inputs)
cleaned = outputs["cross_correlation"]
coefficients = outputs["spectral_coefficients"]

# Intermediate quantities for inspection.
spectral = CrossRIEnetLayer(
    output_type=(
        "empirical_singular_values",
        "correction",
        "left_singular_vectors",
        "right_singular_vectors",
        "projected_variance_x",
        "projected_variance_y",
    ),
)(inputs)
```

Stable outputs are:

- `cross_correlation`;
- `spectral_coefficients`.

The remaining outputs expose internal quantities for diagnostics and research.
With additive correction, spectral coefficients can be negative and are not
strict mathematical singular values.

## Correction modes

```python
additive = CrossRIEnetLayer(
    correction_mode="additive",
    additive_activation="linear",
)

bounded = CrossRIEnetLayer(
    correction_mode="bounded_multiplicative",
)

positive = CrossRIEnetLayer(
    correction_mode="positive_multiplicative",
)
```

The corresponding definitions are:

```text
additive:
    coefficient = empirical + activation(delta)

bounded_multiplicative:
    coefficient = empirical * sigmoid(delta)

positive_multiplicative:
    coefficient = empirical * softplus(delta)
```

## Feasibility diagnostics

`feasibility_diagnostics` evaluates compatibility of a proposed
cross-correlation with fixed marginal correlation matrices. It reports values;
it does not modify or project the matrix.

```python
from crient.diagnostics import feasibility_diagnostics

diagnostics = feasibility_diagnostics(
    correlation_x,
    correlation_y,
    cleaned,
)

maximum_canonical_value = diagnostics["max_canonical_singular_value"]
violation_count = diagnostics["violation_count"]
```


## Dtype and mixed precision

The Keras dtype policy controls computation, variables and public outputs:

| Policy | Spectral work dtype | Variables | Public output |
| --- | --- | --- | --- |
| `float64` | `float64` | `float64` | `float64` |
| `float32` | `float32` | `float32` | `float32` |
| `mixed_float16` | `float32` | `float32` | `float16` |
| `mixed_bfloat16` | `float32` | `float32` | `bfloat16` |

Passing float64 inputs does not override a float32 layer policy. Use
`dtype="float64"` on the layer or select the global float64 policy when that
precision is required.


## Requirements

- Python 3.10--3.12
- TensorFlow 2.16.1--2.21
- Keras 3

## Development

Run the complete test suite from the repository root:

```bash
pytest
```

The suite covers public API validation, dynamic tracing, float32, float64,
mixed precision, reconstruction, gradients, valid correlation blocks,
non-degenerate equivariance, diagnostics and `.keras` serialization.

## Citation

Use `print_citation()` or the repository `CITATION.cff`:

```python
import crient

crient.print_citation()
```

```bibtex
@article{manolakis2026crossrienet,
  title   = {Physics-Informed Singular-Value Learning for Cross-Covariances
             Forecasting in Financial Markets},
  author  = {Manolakis, Efstratios and Bongiorno, Christian and
             Mantegna, Rosario N.},
  year    = {2026},
  eprint  = {2601.07687},
  archivePrefix = {arXiv}
}
```

## Support

For questions, issues, or contributions,

- Open an issue on [GitHub](https://github.com/bongiornoc/CrossRIE/issues)
- Check the documentation
- Contact Prof. Christian Bongiorno (<christian.bongiorno@centralesupelec.fr>) for calibrated model weights or collaboration requests


