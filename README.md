# sdssv_apogee_sf

Selection functions for SDSS-V APOGEE surveys.

## Documentation

Full documentation, including installation instructions, API reference, and denominator file details, is available at:

**https://sdssv-apogee-sf.readthedocs.io**

## Quick start

```python
from sdssv_apogee_sf import APOGEESelectionFunction

sf = APOGEESelectionFunction.from_observed(
    observed_table,   # astropy Table or pandas DataFrame
    h_col="h_mag",
    g_col="g_mag",    # optional; enables G−H colour axis
)

completeness = sf.query(coords, H=h_mag, GH=g_mag - h_mag)
```

See the [tutorial notebook](tutorial.ipynb) for a worked example using the public SDSS DR19 ASPCAP 0.6.0 catalogue.
