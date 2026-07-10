"""
sdssv_apogee_sf — selection functions for SDSS-V APOGEE surveys.

Quick start::

    from sdssv_apogee_sf import APOGEESelectionFunction

    sf = APOGEESelectionFunction.from_observed(
        observed_table,   # astropy Table or pandas DataFrame
        h_col="h_m",
        g_col="g_mag",    # optional; enables G-H colour axis
    )

    completeness = sf.query(coords, H=h_mag, GH=g_mag - h_mag)
"""

from ._selfunc import APOGEESelectionFunction, H_BINS_DEFAULT, GH_BINS_DEFAULT, GAIA_G_LIMIT

__all__ = [
    "APOGEESelectionFunction",
    "H_BINS_DEFAULT",
    "GH_BINS_DEFAULT",
    "GAIA_G_LIMIT",
]

__version__ = "0.1.0"
