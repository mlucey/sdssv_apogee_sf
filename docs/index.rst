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

Pass ``use_prior=False`` to :meth:`~sdssv_apogee_sf.APOGEESelectionFunction.from_observed`
to instead use the raw MLE ratio :math:`S = N_\mathrm{observed} / N_\mathrm{2MASS}`,
without the +1/+2 shrinkage toward 0.5.

Quick start::

    from sdssv_apogee_sf import APOGEESelectionFunction

    sf = APOGEESelectionFunction.from_observed(
        observed_table,
        h_col="h_mag",
        g_col="g_mag",
        denominator_path="twomass_denominator.npz",
    )

    completeness = sf.query(coords, H=h_mag, GH=g_mag - h_mag)
