Limitations
===========

The current architecture is permutation-equivariant for generic,
non-degenerate spectra in the tested cases.  Exact degeneracy leaves a
non-unique singular basis, while the sequential recurrent aggregator may
produce position-dependent corrections.  Full subspace invariance is not
claimed for that case.

The public main-layer contract is rank 3.  Arbitrary leading batch dimensions,
a native-SVD training backend, feasibility enforcement, raw-return
preprocessing and direct RIEnet integration are outside version 0.2.

Version 0.2 has no compatibility alias for the earlier ``crossrie`` package.
