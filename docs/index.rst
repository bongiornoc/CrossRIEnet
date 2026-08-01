CRIENT documentation
====================

CRIENT is the Python distribution of CrossRIEnet.  Its core operates on
marginal correlations and a rectangular cross-correlation.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   numerics
   api
   limitations

Scientific reference
--------------------

The architecture follows equations 1--7 of arXiv:2601.07687v3.  Padding to
``max(n_x, n_y)`` retains scientific marginal directions and is never treated
as disposable sequence padding.
