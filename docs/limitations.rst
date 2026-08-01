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

Version 0.2 does not claim ``jit_compile=True`` support.  In the tested
TensorFlow 2.20 and Keras 3.12 environment, Keras disables JIT for cuDNN-backed
recurrent layers on GPU, while CPU XLA compilation fails in the dynamic
rectangular spectral branch.  Standard ``Model.fit`` execution without JIT is
supported on CPU and GPU.

The documented ``mixed_bfloat16`` dtype contract does not guarantee finite
gradients for every initialization.  An intermittent non-finite GPU backward
pass was observed during release testing; ``float32`` is the conservative
training policy when finite-gradient guarantees are required.

Version 0.2 has no compatibility alias for the earlier ``crossrie`` package.
