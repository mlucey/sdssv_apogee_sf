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

By default, sparse cells are coarsened on the sky as usual and then given
finer angular detail from wider merged H bins
(split at H = 7) while keeping each cell's expected number of observed stars.
Pass `h_merge_breaks=(...)` to choose different merged bins, or
`h_merge_breaks=None` for the plain adaptive selection function. See the
[documentation](https://sdssv-apogee-sf.readthedocs.io) for details.

See the [tutorial notebook](tutorial.ipynb) for a worked example using the public SDSS DR19 ASPCAP 0.6.0 catalogue.
