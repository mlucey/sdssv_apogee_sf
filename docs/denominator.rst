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

.. warning::

   The 2MASS PSC is not complete at all sky positions for H > 12.  In
   crowded regions such as the Galactic bulge, source confusion causes the
   catalogue to become incomplete at faint magnitudes, which means the
   denominator undercounts 2MASS stars there and the selection function will
   be **overestimated**.  A ``UserWarning`` is issued automatically when your
   sample contains stars with H > 12.

Magnitude range and binning
---------------------------

By default, ``from_observed`` restricts the magnitude axes to the min/max of
your observed data (rounded outward to the nearest 0.5-mag step), so you only
use the denominator bins that are actually relevant.

You can override this with explicit limits using ``h_range`` and ``gh_range``::

    sf = APOGEESelectionFunction.from_observed(
        observed,
        h_col="h_mag",
        h_range=(7.0, 13.5),   # must be within the denominator bounds
        gh_range=(0.0, 8.0),   # colour mode only
    )

Both limits must fall within the denominator's coverage shown in the table
above.  A ``ValueError`` is raised if they do not.

You can also coarsen the bin size in steps of 0.5 mag using ``h_bin_size``
and ``gh_bin_size`` (e.g. ``h_bin_size=1.0`` gives 1-mag bins).  The
minimum bin size is always 0.5 mag (the native denominator resolution).

Bundled file
-----------

``twomass_denominator.npz`` is shipped with the package and loaded
automatically when ``denominator_path=None`` (the default).  No additional
setup is required.

File format
-----------

The ``.npz`` file contains:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Key
     - Shape
     - Description
   * - ``hist_all``
     - ``(n_H, n_pix, n_GH)``
     - 2MASS source counts
   * - ``H_BINS``
     - ``(n_H + 1,)``
     - H-magnitude bin edges
   * - ``GH_BINS``
     - ``(n_GH + 1,)``
     - G−H bin edges
   * - ``nside``
     - scalar
     - HEALPix nside
