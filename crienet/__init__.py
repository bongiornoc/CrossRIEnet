"""CRIENet: neural cleaning of rectangular cross-correlations."""

from .layer import CrossRIEnetLayer
from .typing import CrossRIEnetOutput
from .version import __version__

__author__ = "Efstratios Manolakis, Christian Bongiorno, Rosario N. Mantegna"


def print_citation() -> None:
    """Print the paper and software citation for the installed version."""
    print(
        f"""Please cite:

Manolakis, E., Bongiorno, C., & Mantegna, R. N. (2026).
Physics-Informed Singular-Value Learning for Cross-Covariances Forecasting
in Financial Markets. arXiv:2601.07687v3.

Software: CRIENet {__version__}
https://github.com/bongiornoc/CrossRIEnet
"""
    )


__all__ = [
    "CrossRIEnetLayer",
    "CrossRIEnetOutput",
    "__version__",
    "print_citation",
]
