# CrossRIEnet / CRIENT

CRIENT is the Python package for CrossRIEnet, a dimension-aware neural
estimator that cleans rectangular **cross-correlation** matrices in their
empirical singular-vector basis.

It implements the architecture described in:

> Efstratios Manolakis, Christian Bongiorno, and Rosario N. Mantegna (2026),
> *Physics-Informed Singular-Value Learning for Cross-Covariances Forecasting
> in Financial Markets*, [arXiv:2601.07687v3](https://arxiv.org/abs/2601.07687).

Version 0.2 is a pre-release API redesign. It intentionally provides no
compatibility package or aliases for the earlier `crossrie` API.

## Relationship to RIEnet

CrossRIEnet is a companion extension of
[RIEnet](https://github.com/bongiornoc/RIEnet). RIEnet learns rotationally
invariant corrections for square covariance and correlation matrices through
eigendecomposition. CrossRIEnet extends the same dimension-aware spectral
design to rectangular cross-correlations through singular-value decomposition.

CRIENT remains an independent package. It does not import RIEnet or depend on
private RIEnet symbols. The public matrix-level seam is designed for future
composition with RIEnet preprocessing, marginal-volatility estimation and
marginal precision cleaning.

## Scientific domain

The core layer receives:

- a marginal correlation matrix for block X;
- a marginal correlation matrix for block Y;
- their empirical cross-correlation matrix;
- the positive observation count used to estimate them.

It returns a cleaned cross-correlation. Converting that result into a
cross-covariance is a separate, deterministic operation:

```text
cross_covariance = diag(std_x) @ cross_correlation @ diag(std_y)
```

Use `CrossCovarianceRescalingLayer` for this step.

## Installation

CRIENT has not been published to PyPI. Install the current checkout:

```bash
git clone https://github.com/Efstratios7/CrossRIE.git
cd CrossRIE
python -m pip install -e .
```

Supported runtime:

- Python 3.10–3.12;
- TensorFlow 2.16.1–2.21;
- Keras 3.

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

The Conda definition uses the environment name `crient_env`:

```bash
conda env update --file environment.yml --prune
conda activate crient_env
python -m pip install -e ".[dev]"
```

`setup_env.py` is retained temporarily but deprecated.

## Quick start with valid correlation blocks

The example derives all three blocks from the same joint standardized return
sample. This guarantees symmetry, unit diagonals up to floating-point error,
and consistency of the complete correlation block.

```python
import tensorflow as tf

from crient import CrossRIEnetLayer

batch, n_observations, n_x, n_y = 4, 128, 6, 8
returns = tf.random.normal((batch, n_observations, n_x + n_y))
returns -= tf.reduce_mean(returns, axis=1, keepdims=True)
standardized = returns / tf.math.reduce_std(
    returns,
    axis=1,
    keepdims=True,
)
correlation = tf.matmul(
    standardized,
    standardized,
    transpose_a=True,
) / tf.cast(n_observations, standardized.dtype)

correlation_x = correlation[:, :n_x, :n_x]
correlation_y = correlation[:, n_x:, n_x:]
cross_correlation = correlation[:, :n_x, n_x:]

layer = CrossRIEnetLayer(
    output_type=("cross_correlation", "spectral_coefficients"),
)
outputs = layer(
    {
        "correlation_x": correlation_x,
        "correlation_y": correlation_y,
        "cross_correlation": cross_correlation,
        "n_observations": tf.fill((batch,), float(n_observations)),
    },
    training=False,
)

cleaned_cross_correlation = outputs["cross_correlation"]
cleaned_coefficients = outputs["spectral_coefficients"]
```

## Public API

The package root exports:

```python
from crient import (
    CrossCovarianceRescalingLayer,
    CrossRIEnetLayer,
    CrossRIEnetOutput,
    __version__,
    print_citation,
)
```

Advanced layers and numerical helpers are available from `crient.spectral`,
`crient.ops_layers`, `crient.trainable_layers`, `crient.validation` and
`crient.diagnostics`.

### Output selection

`output_type` accepts a string, a sequence or `"all"`.

Stable outputs:

- `"cross_correlation"`;
- `"spectral_coefficients"`.

Advanced diagnostic outputs:

- `"empirical_singular_values"`;
- `"correction"`;
- `"left_singular_vectors"`;
- `"right_singular_vectors"`;
- `"projected_variance_x"`;
- `"projected_variance_y"`.

A single string returns one tensor. A sequence or `"all"` returns a
deduplicated dictionary.

### Correction modes

```text
additive:
    coefficient = empirical + activation(delta)

bounded_multiplicative:
    coefficient = empirical * sigmoid(delta)

positive_multiplicative:
    coefficient = empirical * softplus(delta)
```

Additive coefficients may be negative. They are therefore called spectral
coefficients rather than singular values.

## Architecture and paper mapping

| Stage | Paper | CRIENT |
| --- | --- | --- |
| Empirical full SVD | Eq. 1 | `SpectralSVDLayer` |
| Marginal projections | Eq. 2 | `ProjectedVarianceDiagonalLayer` |
| Complete bases and padding | Eq. 3 | `SequencePaddingLayer` |
| `[gamma, s, q]` tokens | Eq. 4 | `CrossRIEnetLayer` |
| Shared encoder | Eq. 5 | `TwoStreamEncoderLayer` |
| Sum fusion and recurrent aggregation | Eq. 6 | `TwoStreamEncoderLayer` |
| Point-wise spectral correction | Eq. 7 | `CrossRIEnetLayer` |
| Correlation-to-covariance rescaling | Eq. 8 | `CrossCovarianceRescalingLayer` |

Padding to `max(n_x, n_y)` is scientifically meaningful: it preserves
marginal directions that exist only in the larger block. CRIENT does not apply
an RNN mask to these tokens.

## Dynamic dimensions

The main layer accepts rank-3 matrices:

```text
(batch, n_x, n_x)
(batch, n_y, n_y)
(batch, n_x, n_y)
```

Batch size, `n_x` and `n_y` may vary in the same traced function. Arbitrary
leading batch dimensions are not part of the 0.2 main-layer contract.

`n_observations` must have shape `(batch,)` or `(batch, 1)` and must be
strictly positive.

## Dtype and mixed precision

The Keras dtype policy is authoritative:

| Policy | Spectral work dtype | Variables | Public output |
| --- | --- | --- | --- |
| float64 | float64 | float64 | float64 |
| float32 | float32 | float32 | float32 |
| mixed_float16 | float32 | float32 | float16 |
| mixed_bfloat16 | float32 | float32 | bfloat16 |

Passing a float64 tensor does not override a float32 layer policy. Set
`dtype="float64"` on the layer or select the corresponding global policy.

## Numerical policy

- Invalid domains raise errors; they are not repaired with epsilon.
- `n_observations` is validated and division is exact.
- The exactly-zero SVD input uses `safe_scale = 1`.
- Numerical rank uses
  `rank_atol + rank_rtol * max_singular_value`.
- The automatic Gram-aware default is
  `sqrt(max(n_x, n_y) * machine_epsilon(work_dtype))`.
- No default jitter, clipping, absolute value, pseudoinverse or feasibility
  projection changes the estimator.

The current backend uses Gram matrices and `tf.linalg.eigh` to obtain complete
left and right bases. Forming Gram matrices squares the condition number.
`tf.linalg.svd` is not yet a drop-in training backend because TensorFlow does
not provide every required gradient for full rectangular bases.

## Validation and feasibility

`validation_mode="basic"` checks:

- input structure and rank;
- floating matrix dtype;
- exact batch and matrix-dimension compatibility;
- finite values;
- square marginal matrices;
- positive observation counts.

`validation_mode="strict"` additionally checks symmetry, unit diagonals and
positive-semidefiniteness of the complete correlation block.

`crient.diagnostics.feasibility_diagnostics` reports canonical singular values,
violations, feasibility margin, support residuals and condition estimates. It
never projects or modifies the supplied estimator.

## Training and serialization

`training` is propagated explicitly to the shared encoder, recurrent stack and
dropout layers. Registered public and advanced layers use the Keras
serialization namespace `crient` and support `.keras` round trips.

## Testing

Run the full suite:

```bash
pytest
```

The suite covers API validation, dynamic shape tracing, float32/float64/mixed
precision, full-basis reconstruction, gradients, correlation-domain fixtures,
equivariance for non-degenerate spectra, feasibility diagnostics and
serialization.

## Limitations

- Exact or nearly repeated singular values define non-unique subspaces. The
  current sequential aggregator is not guaranteed to be invariant to every
  basis rotation inside a degenerate subspace.
- The Gram/eigh backend loses accuracy for sufficiently ill-conditioned
  matrices.
- The core starts from correlation matrices; preprocessing raw returns is
  outside the 0.2 scope.
- No direct RIEnet dependency or integrated raw-return pipeline is included.
- CRIENT 0.2 intentionally does not load the earlier `crossrie` API or its
  checkpoints.

## Citation

Use `print_citation()` or the repository `CITATION.cff`.

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
