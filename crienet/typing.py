"""Public typing helpers for CRIENet."""

from __future__ import annotations

from typing import Literal, TypeAlias

CrossRIEnetOutput: TypeAlias = Literal[
    "cross_correlation",
    "spectral_coefficients",
    "empirical_singular_values",
    "correction",
    "left_singular_vectors",
    "right_singular_vectors",
    "projected_variance_x",
    "projected_variance_y",
    "all",
]

STABLE_OUTPUTS = (
    "cross_correlation",
    "spectral_coefficients",
)

ADVANCED_OUTPUTS = (
    "empirical_singular_values",
    "correction",
    "left_singular_vectors",
    "right_singular_vectors",
    "projected_variance_x",
    "projected_variance_y",
)

ALL_OUTPUTS = STABLE_OUTPUTS + ADVANCED_OUTPUTS
