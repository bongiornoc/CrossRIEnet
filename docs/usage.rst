Usage
=====

The canonical import is:

.. code-block:: python

   from crient import CrossRIEnetLayer

   layer = CrossRIEnetLayer(
       output_type=("cross_correlation", "spectral_coefficients"),
   )

Inputs can be passed as a four-element sequence or as a mapping with the keys
``correlation_x``, ``correlation_y``, ``cross_correlation`` and
``sample_size``.

Output selection
----------------

A single output string returns a tensor.  A sequence and ``"all"`` return an
ordered dictionary.  ``cross_correlation`` and ``spectral_coefficients`` are
the stable outputs; all other intermediate outputs are advanced diagnostics.

Training
--------

The ``training`` argument is propagated explicitly through recurrent and
dropout layers.  Standard Keras Functional models, ``fit`` and ``.keras``
serialization are supported.
