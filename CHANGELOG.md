# Changelog

All notable changes are documented in this file.

## 0.2.0 - 2026-08-01

- Rename the Python package and distribution to `crient`.
- Replace the 0.1 API with `output_type` and descriptive output names.
- Define the core domain as marginal correlations plus cross-correlation.
- Add explicit additive, bounded-multiplicative and positive-multiplicative
  correction modes.
- Add Keras dtype-policy support and float32 spectral work under mixed
  precision.
- Add structural and optional strict domain validation.
- Add feasibility diagnostics.
- Name the public sample-count input `sample_size`.
- Align token channel order with equation 4 of arXiv:2601.07687v3.
- Drop all `crossrie` compatibility aliases and checkpoint migration support.
