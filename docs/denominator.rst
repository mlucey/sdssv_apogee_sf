The 2MASS Denominator
=====================

The denominator is a pre-computed count of **2MASS PSC sources** per HEALPix
pixel, H-magnitude bin, and G−H colour bin.  It is the same for any APOGEE
programme that selects from the 2MASS PSC.

Coverage
--------

The provided denominator covers:

============  ===========================  ===========
Quantity      Range                        Native step
============  ===========================  ===========
H magnitude   −3.0 to 18.0 mag            0.5 mag
G − H colour  −2.5 to 14.5 mag            0.5 mag
HEALPix nside 64 (pixel size ≈ 55 arcmin) —
============  ===========================  ===========

Stars in your observed catalogue that fall outside these ranges will trigger a
``UserWarning`` and be excluded from the selection function.

By default, ``from_observed`` restricts the magnitude axis to the min/max of
your data, so you only use the denominator bins that are actually relevant.
You can coarsen the binning in 0.5-mag steps using ``h_bin_size`` and
``gh_bin_size``.

Bundled file
-----------

``twomass_denominator.npz`` is shipped with the package and loaded
automatically when ``denominator_path=None`` (the default).  No additional
setup is required.

Regenerating the denominator
-----------------------------

If you need to rebuild the denominator (e.g. with a different nside or
magnitude range), you will need access to the SDSS-V database (Utah cluster):

.. code-block:: bash

   python gg_selfunc_healpix.py --use-color

This writes ``twomass_denominator.npz`` in the current directory.  Pass its
path via ``denominator_path=`` to use it instead of the bundled file.

File format
-----------

The ``.npz`` file contains:

===========  =====================================  ===========
Key          Shape                                  Description
===========  =====================================  ===========
``hist_all`` ``(n_H, n_pix, n_GH)``                2MASS source counts
``H_BINS``   ``(n_H + 1,)``                        H-magnitude bin edges
``GH_BINS``  ``(n_GH + 1,)``                       G−H bin edges
``nside``    scalar                                 HEALPix nside
===========  =====================================  ===========
