Numerical and dtype policy
==========================

The active Keras policy defines variables, compute dtype and public output
dtype.  Float16 and bfloat16 policies use float32 for the spectral backend.
Float64 policies remain float64 end to end.

The Gram/eigh backend supplies complete singular-vector bases but squares the
condition number.  Its automatic numerical-rank tolerance is

.. math::

   rtol = \sqrt{\max(n_x,n_y)\,\epsilon_{\mathrm{work}}}.

The effective threshold is ``rank_atol + rank_rtol * s_max``.

Exact cases and invalid domains
-------------------------------

An exactly zero matrix is divided by a safe scale of one.  Nonzero matrices are
not perturbed.  Observation counts must be strictly positive and are used
without epsilon.  CRIENT does not silently add jitter, clip coefficients,
apply absolute values, use pseudoinverses or project feasibility.
