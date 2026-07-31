"""Advanced layer imports.

Prefer the focused modules for new code.  This module is a convenience
namespace within the canonical :mod:`crient` package, not a legacy shim.
"""

from .ops_layers import (
    CrossCovarianceRescalingLayer,
    ProjectedVarianceDiagonalLayer,
    SequencePaddingLayer,
    SVDReconstructionLayer,
)
from .spectral import SpectralSVDLayer
from .trainable_layers import (
    DeepLayer,
    DeepRecurrentLayer,
    TwoStreamEncoderLayer,
)

__all__ = [
    "CrossCovarianceRescalingLayer",
    "DeepLayer",
    "DeepRecurrentLayer",
    "ProjectedVarianceDiagonalLayer",
    "SVDReconstructionLayer",
    "SequencePaddingLayer",
    "SpectralSVDLayer",
    "TwoStreamEncoderLayer",
]
