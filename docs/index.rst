sdssv-apogee-sf
===============

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   denominator
   api

Overview
--------

``sdssv_apogee_sf`` computes selection functions for SDSS-V APOGEE survey
programmes.  The selection function :math:`S(\ell, b, H, G{-}H)` is the
probability that a 2MASS star at a given sky position and magnitude was
observed by APOGEE:

.. math::

   S = \frac{N_\mathrm{observed} + 1}{N_\mathrm{2MASS} + 2}

using a Bayesian Beta(1, 1) prior by default.  The sky is pixelised with
HEALPix and pixels with too few 2MASS sources are adaptively merged to
coarser resolution.

Pass ``estimator='map'`` to :meth:`~sdssv_apogee_sf.APOGEESelectionFunction.from_observed`
to report the posterior mode instead of the mean. For a uniform Beta(1, 1)
prior the mode coincides with the raw ratio :math:`S = N_\mathrm{observed} / N_\mathrm{2MASS}`,
without the +1/+2 shrinkage toward 0.5.

Small-scale structure from merged H bins
----------------------------------------

In sparsely populated :math:`(H, G{-}H)` bins the adaptive scheme has to
coarsen the HEALPix resolution until each cell holds ``min_count`` 2MASS
sources, which can wash out real small-scale structure in the footprint.
By default, :meth:`~sdssv_apogee_sf.APOGEESelectionFunction.from_observed`
therefore recovers this structure from wider magnitude bins, assuming that the
small-scale angular pattern of the completeness varies only weakly with
:math:`H` over broad magnitude ranges:

1. The native :math:`H` bins are merged into wider bins (by default
   :math:`H < 7` and :math:`H \ge 7` over the :math:`H` range in use).
   With more stars, the merged bins can be estimated at finer resolution.
2. Within each native cell, the adaptive completeness is redistributed
   across the finer sub-pixels in proportion to the merged bin's
   completeness, normalised so that the cell's expected number of observed
   stars is exactly preserved.

The result is therefore identical to the adaptive selection function on
the original coarse scale and only adds angular detail within each cell.
Cells where the merged bin is no better resolved, or that were
unconstrained, are unchanged, and values are capped at 1.

Control it with ``h_merge_breaks``::

    # default: merged bins split at H = 7
    sf = APOGEESelectionFunction.from_observed(observed_table, h_merge_breaks=(7.0,))

    # different merged bins
    sf = APOGEESelectionFunction.from_observed(observed_table, h_merge_breaks=(6.0, 9.0))

    # turn merging off: plain adaptive selection function
    sf = APOGEESelectionFunction.from_observed(observed_table, h_merge_breaks=None)

Breaks are snapped to the nearest :math:`H` bin edge.  The merged-bin edges
used are stored with the selection function (``sf.h_merge_edges``, saved by
:meth:`~sdssv_apogee_sf.APOGEESelectionFunction.write`), and ``sf.method``
reports ``'merged_h'`` (merged bins used) or ``'adaptive'`` (merging off).

Quick start::

    from sdssv_apogee_sf import APOGEESelectionFunction

    sf = APOGEESelectionFunction.from_observed(
        observed_table,
        h_col="h_mag",
        g_col="g_mag",
        denominator_path="twomass_denominator.npz",
    )

    completeness = sf.query(coords, H=h_mag, GH=g_mag - h_mag)
