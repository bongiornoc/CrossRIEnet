"""Smoke checks for the interactive help contract."""

from __future__ import annotations

import inspect

from crient import CrossRIEnetLayer
from crient.diagnostics import feasibility_diagnostics
from crient.spectral import SpectralSVDLayer, svd_via_eigh_full


def test_public_docstrings_are_present_and_describe_core_contracts():
    objects = (
        CrossRIEnetLayer,
        CrossRIEnetLayer.call,
        SpectralSVDLayer,
        svd_via_eigh_full,
        feasibility_diagnostics,
    )
    for value in objects:
        assert inspect.getdoc(value)

    main_help = inspect.getdoc(CrossRIEnetLayer)
    assert "cross-correlation" in main_help
    assert "sample_size" in main_help
    assert "dtype" in main_help
    assert "degenerate" in main_help
