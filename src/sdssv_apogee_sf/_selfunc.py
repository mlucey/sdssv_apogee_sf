"""
Core selection function class for SDSS-V APOGEE surveys.
"""
from __future__ import annotations

import importlib.resources
import warnings
from pathlib import Path
from typing import Union

import healpy as hp
import numpy as np

# Native denominator resolution — bin edges are multiples of this step.
_NATIVE_STEP = 0.5

# Full denominator magnitude ranges (set when the denominator is built).
H_BINS_DEFAULT  = np.arange(-3.0, 18.01, _NATIVE_STEP)
GH_BINS_DEFAULT = np.arange(-2.5, 14.51, _NATIVE_STEP)

# Gaia DR3 faint detection limit used when placing no-Gaia 2MASS stars in GH bins.
GAIA_G_LIMIT = 21.0


def _coarsen(hist: np.ndarray, bins: np.ndarray, factor: int,
             axis: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Coarsen *hist* along *axis* by summing *factor* adjacent bins.

    Trailing bins that don't fill a complete group are dropped.

    Returns
    -------
    hist_out : ndarray — coarsened histogram
    bins_out : ndarray — new bin edges
    """
    n = hist.shape[axis]
    n_keep = (n // factor) * factor
    hist = np.take(hist, range(n_keep), axis=axis)
    shape = list(hist.shape)
    shape[axis] = n_keep // factor
    shape.insert(axis + 1, factor)
    hist_out = hist.reshape(shape).sum(axis=axis + 1)
    bins_out = bins[::factor][:n_keep // factor + 1]
    return hist_out, bins_out


def _restrict_range(hist: np.ndarray, bins: np.ndarray,
                    lo: float, hi: float, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice *hist* along *axis* to the bins that overlap [lo, hi].

    Returns the sliced histogram and the corresponding bin edges.
    """
    i_start = max(0, np.searchsorted(bins[1:], lo, side='right'))
    i_end   = min(len(bins) - 1, np.searchsorted(bins[:-1], hi, side='left'))
    slc = [slice(None)] * hist.ndim
    slc[axis] = slice(i_start, i_end)
    return hist[tuple(slc)], bins[i_start:i_end + 1]


def _degrade_sum(arr1d: np.ndarray, nside_in: int, nside_out: int) -> np.ndarray:
    """Degrade a 1-D HEALPix map from nside_in to nside_out by summing child pixels."""
    n_children = (nside_in // nside_out) ** 2
    # hp.ud_grade preserves input dtype: for integer maps (e.g. observed-star
    # counts) it truncates the intermediate per-pixel mean, silently zeroing
    # out sparse counts that don't divide evenly among the children before
    # they get multiplied back up. Cast to float first so the mean is exact.
    return hp.ud_grade(arr1d.astype(float), nside_out) * n_children


def _build_adaptive(
    hist_all: np.ndarray,
    hist_obs: np.ndarray,
    nside: int,
    min_count: int = 5,
    nside_min: int = 8,
    use_prior: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SF with adaptive HEALPix coarsening.

    Starting at *nside*, each (H-bin, pixel) cell — or (H-bin, pixel, GH-bin)
    cell in colour mode — that has fewer than *min_count* 2MASS sources is
    deferred to the next coarser resolution via hp.ud_grade (which correctly
    sums ALL fine pixels, not just uncovered ones).  This repeats down to
    *nside_min*.  Cells still uncovered after that remain NaN.

    Parameters
    ----------
    hist_all : (n_H, n_pix) or (n_H, n_pix, n_GH) — denominator counts at nside
    hist_obs : same shape — numerator counts
    nside    : int — fine resolution (RING ordering)
    min_count: int
    nside_min: int
    use_prior: bool — if True (default), use the Bayesian Beta(1,1) estimate
        ``(N_obs + 1) / (N_2MASS + 2)``.  If False, use the raw MLE ratio
        ``N_obs / N_2MASS`` (undefined cells, where the coarsened denominator
        is still zero, stay NaN).

    Returns
    -------
    sf       : same shape as hist_all — SF values in [0,1] or NaN
    nside_map: (n_H, n_pix) — effective nside used per (H-bin, pixel)
    """
    use_color = hist_all.ndim == 3
    if use_color:
        n_h, npix_max, n_gh = hist_all.shape
    else:
        n_h, npix_max = hist_all.shape

    sf_out    = np.full(hist_all.shape, np.nan, dtype=float)
    nside_map = np.zeros((n_h, npix_max), dtype=np.int32)
    covered   = (np.zeros((n_h, npix_max, n_gh), dtype=bool) if use_color
                 else np.zeros((n_h, npix_max), dtype=bool))

    ns = nside
    while ns >= max(nside_min, 1):
        npix_ns = hp.nside2npix(ns)

        if ns == nside:
            ha, ht = hist_all, hist_obs
        else:
            if use_color:
                ha = np.zeros((n_h, npix_ns, n_gh))
                ht = np.zeros((n_h, npix_ns, n_gh))
                for hi in range(n_h):
                    for gi in range(n_gh):
                        ha[hi, :, gi] = _degrade_sum(hist_all[hi, :, gi], nside, ns)
                        ht[hi, :, gi] = _degrade_sum(hist_obs[hi, :, gi], nside, ns)
            else:
                ha = np.stack([_degrade_sum(hist_all[hi], nside, ns) for hi in range(n_h)])
                ht = np.stack([_degrade_sum(hist_obs[hi], nside, ns) for hi in range(n_h)])

        good_at_ns = ha >= max(min_count, 1)

        # Broadcast coarse mask back to fine grid
        if ns < nside:
            if use_color:
                good_at_max = np.zeros((n_h, npix_max, n_gh), dtype=bool)
                for hi in range(n_h):
                    for gi in range(n_gh):
                        good_at_max[hi, :, gi] = (
                            hp.ud_grade(good_at_ns[hi, :, gi].astype(float), nside) > 0.5
                        )
            else:
                good_at_max = np.zeros((n_h, npix_max), dtype=bool)
                for hi in range(n_h):
                    good_at_max[hi] = (
                        hp.ud_grade(good_at_ns[hi].astype(float), nside) > 0.5
                    )
        else:
            good_at_max = good_at_ns

        new_covered = good_at_max & ~covered

        if new_covered.any():
            if use_prior:
                sf = (ht + 1.0) / (ha + 2.0)
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    sf = np.where(ha > 0, ht / ha, np.nan)

            # Upgrade coarse SF values back to fine grid
            if ns < nside:
                if use_color:
                    sf_up = np.zeros((n_h, npix_max, n_gh))
                    for hi in range(n_h):
                        for gi in range(n_gh):
                            sf_up[hi, :, gi] = hp.ud_grade(sf[hi, :, gi], nside)
                else:
                    sf_up = np.stack([hp.ud_grade(sf[hi], nside) for hi in range(n_h)])
            else:
                sf_up = sf

            sf_out[new_covered] = sf_up[new_covered]

            newly_touched = new_covered.any(axis=2) if use_color else new_covered
            nside_map[newly_touched] = ns
            covered |= new_covered

        if covered.all():
            break

        ns //= 2

    return sf_out, nside_map


def _pix_for_coords(coords, nside: int) -> np.ndarray:
    """Return RING HEALPix pixel indices (equatorial) for an astropy SkyCoord."""
    ra  = np.atleast_1d(np.asarray(coords.icrs.ra.deg,  dtype=float))
    dec = np.atleast_1d(np.asarray(coords.icrs.dec.deg, dtype=float))
    theta = np.radians(90.0 - dec)
    phi   = np.radians(ra)
    return hp.ang2pix(nside, theta, phi, nest=False)


class APOGEESelectionFunction:
    """
    SDSS-V APOGEE selection function computed against a pre-built 2MASS denominator.

    The denominator (2MASS PSC counts binned by HEALPix pixel + H magnitude, and
    optionally G-H colour) can be downloaded automatically from Zenodo or supplied
    locally.

    Typical usage::

        from sdssv_apogee_sf import APOGEESelectionFunction
        import astropy.table as tbl

        observed = tbl.Table.read("my_apogee_stars.fits")

        sf = APOGEESelectionFunction.from_observed(
            observed,
            h_col="h_m",
            g_col="g_mag",   # optional; enables G-H colour axis
        )

        import astropy.coordinates as coord
        c = coord.SkyCoord(ra=observed["ra"], dec=observed["dec"], unit="deg")
        completeness = sf.query(c, H=observed["h_m"], GH=observed["g_mag"] - observed["h_m"])
    """

    def __init__(
        self,
        selfunc: np.ndarray,
        hist_all: np.ndarray,
        hist_obs: np.ndarray,
        H_BINS: np.ndarray,
        nside: int,
        GH_BINS: np.ndarray | None = None,
        nside_map: np.ndarray | None = None,
        use_prior: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        selfunc   : (n_H, n_pix) or (n_H, n_pix, n_GH) — SF values in [0, 1] or NaN
        hist_all  : same shape — 2MASS denominator counts
        hist_obs  : same shape — observed star counts (numerator)
        H_BINS    : (n_H + 1,) — H-magnitude bin edges
        nside     : int — HEALPix nside (RING ordering)
        GH_BINS   : (n_GH + 1,) or None — G-H bin edges if colour axis is used
        nside_map : (n_pix,) or None — effective nside per pixel after adaptive coarsening
        use_prior : bool — whether *selfunc* was computed with the Bayesian
            Beta(1,1) prior (``True``) or as the raw MLE ratio (``False``).
            Recorded for provenance; does not affect querying.
        """
        self._selfunc   = selfunc
        self._hist_all  = hist_all
        self._hist_obs  = hist_obs
        self._H_BINS    = H_BINS
        self._GH_BINS   = GH_BINS
        self._nside     = nside
        self._nside_map = nside_map
        self._use_prior = use_prior

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def nside(self) -> int:
        return self._nside

    @property
    def H_BINS(self) -> np.ndarray:
        return self._H_BINS

    @property
    def GH_BINS(self) -> np.ndarray | None:
        return self._GH_BINS

    @property
    def use_color(self) -> bool:
        return self._GH_BINS is not None

    @property
    def use_prior(self) -> bool:
        """Whether the SF was computed with the Bayesian Beta(1,1) prior."""
        return self._use_prior

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_observed(
        cls,
        observed,
        ra_col: str = "ra",
        dec_col: str = "dec",
        h_col: str = "h_m",
        g_col: str | None = None,
        nside: int = 64,
        min_count: int = 5,
        nside_min: int = 8,
        use_prior: bool = True,
        h_bin_size: float | None = None,
        gh_bin_size: float | None = None,
        h_range: tuple[float, float] | None = None,
        gh_range: tuple[float, float] | None = None,
        denominator_path: str | Path | None = None,
    ) -> "APOGEESelectionFunction":
        """
        Build a selection function from a table of observed APOGEE stars.

        Parameters
        ----------
        observed : astropy Table, pandas DataFrame, or any dict-like with column access.
        ra_col, dec_col : column names for right ascension and declination (degrees).
        h_col : column name for H-band magnitude.
        g_col : column name for Gaia G magnitude.  When provided, a colour (H × G−H)
            selection function is computed.  Stars missing G are excluded from the
            colour numerator but are still used in H-only mode.
        nside : HEALPix nside for the sky pixelisation (default 64).
        min_count : minimum 2MASS sources per adaptive cell before merging to a
            coarser resolution (default 5).
        nside_min : coarsest HEALPix resolution allowed by adaptive binning (default 8).
        use_prior : if True (default), compute the SF with a Bayesian Beta(1,1)
            prior, ``S = (N_observed + 1) / (N_2MASS + 2)``.  If False, compute
            the raw MLE ratio ``S = N_observed / N_2MASS`` instead — cells are
            still merged to coarser resolution via *min_count* / *nside_min*,
            but the SF value itself is not shrunk toward 0.5.
        h_bin_size : H-magnitude bin width in magnitudes.  Must be a positive multiple
            of the denominator's native 0.5-mag step (i.e. 0.5, 1.0, 1.5, …).
            Defaults to the native resolution (0.5 mag).
        gh_bin_size : G−H bin width, same rules as *h_bin_size*.  Only used in colour mode.
        h_range : ``(lo, hi)`` tuple setting explicit H-magnitude limits.  Both values
            must lie within the denominator's H range.  Defaults to the min/max of
            the observed H values, rounded outward to the nearest 0.5-mag step.
        gh_range : ``(lo, hi)`` tuple setting explicit G−H limits, same rules as
            *h_range*.  Only used in colour mode.
        denominator_path : path to the pre-computed 2MASS denominator .npz.
            When *None* the bundled file is used.

        Returns
        -------
        APOGEESelectionFunction

        Warns
        -----
        UserWarning
            If any observed stars fall outside the denominator's magnitude range.
            Those stars are excluded from the selection function.

        Notes
        -----
        By default the magnitude axes are restricted to the min/max of the input
        table, rounded outward to the nearest 0.5-mag step.  Use *h_range* /
        *gh_range* to set explicit limits instead.  Either way the limits must
        fall within the denominator's coverage.
        """
        use_color = g_col is not None

        # ── Load denominator — bins and nside come from the file ──────────────
        denom = cls._load_denominator(denominator_path)
        H_BINS_denom  = denom["H_BINS"]
        GH_BINS_denom = denom["GH_BINS"]
        nside_denom   = int(denom["nside"])
        hist_all_full = denom["hist_all"]   # (n_H, n_pix, n_GH) at full range + native res

        if nside_denom != nside:
            raise ValueError(
                f"Denominator file has nside={nside_denom}, but nside={nside} was requested."
            )

        # ── Validate bin sizes ────────────────────────────────────────────────
        def _check_bin_size(size, name):
            if size is None:
                return 1  # factor of 1 = no coarsening
            factor = round(size / _NATIVE_STEP)
            if abs(factor * _NATIVE_STEP - size) > 1e-9 or factor < 1:
                raise ValueError(
                    f"{name}={size} is not a positive multiple of the native "
                    f"step {_NATIVE_STEP}. Use 0.5, 1.0, 1.5, 2.0, …"
                )
            return factor

        h_factor  = _check_bin_size(h_bin_size,  "h_bin_size")
        gh_factor = _check_bin_size(gh_bin_size, "gh_bin_size")

        # ── Extract magnitudes and warn about out-of-range stars ──────────────
        ra  = np.asarray(observed[ra_col],  dtype=float)
        dec = np.asarray(observed[dec_col], dtype=float)
        h   = np.asarray(observed[h_col],   dtype=float)
        g   = np.asarray(observed[g_col],   dtype=float) if use_color else None

        h_denom_lo, h_denom_hi = H_BINS_denom[0], H_BINS_denom[-1]
        n_below_h = int(np.sum(h < h_denom_lo))
        n_above_h = int(np.sum(h >= h_denom_hi))
        if n_below_h:
            warnings.warn(
                f"{n_below_h} star(s) have H < {h_denom_lo:.2f} (below denominator range) "
                f"and will be excluded.",
                UserWarning, stacklevel=2,
            )
        if n_above_h:
            warnings.warn(
                f"{n_above_h} star(s) have H >= {h_denom_hi:.2f} (above denominator range) "
                f"and will be excluded.",
                UserWarning, stacklevel=2,
            )

        h_denom_hi_warn = 12.0
        if np.any(h[np.isfinite(h)] > h_denom_hi_warn):
            warnings.warn(
                f"Some stars have H > {h_denom_hi_warn:.0f}. At these faint magnitudes "
                "the 2MASS PSC may be incomplete in crowded regions (e.g. the Galactic "
                "bulge), which can cause the selection function to be overestimated there.",
                UserWarning, stacklevel=2,
            )

        if use_color:
            gh = g - h
            gh_denom_lo, gh_denom_hi = GH_BINS_denom[0], GH_BINS_denom[-1]
            both_valid = np.isfinite(h) & np.isfinite(gh)
            n_below_gh = int(np.sum(both_valid & (gh < gh_denom_lo)))
            n_above_gh = int(np.sum(both_valid & (gh >= gh_denom_hi)))
            if n_below_gh:
                warnings.warn(
                    f"{n_below_gh} star(s) have G−H < {gh_denom_lo:.2f} (below denominator "
                    f"range) and will be excluded from the colour numerator.",
                    UserWarning, stacklevel=2,
                )
            if n_above_gh:
                warnings.warn(
                    f"{n_above_gh} star(s) have G−H >= {gh_denom_hi:.2f} (above denominator "
                    f"range) and will be excluded from the colour numerator.",
                    UserWarning, stacklevel=2,
                )

        # ── Restrict denominator to requested or data range ───────────────────
        if h_range is not None:
            h_lo, h_hi = float(h_range[0]), float(h_range[1])
            if h_lo < h_denom_lo or h_hi > h_denom_hi:
                raise ValueError(
                    f"h_range=({h_lo}, {h_hi}) falls outside the denominator bounds "
                    f"[{h_denom_lo}, {h_denom_hi}]."
                )
        else:
            h_valid = h[np.isfinite(h)]
            h_lo = float(np.floor(h_valid.min() / _NATIVE_STEP) * _NATIVE_STEP) if len(h_valid) else h_denom_lo
            h_hi = float(np.ceil( h_valid.max() / _NATIVE_STEP) * _NATIVE_STEP) if len(h_valid) else h_denom_hi
            h_lo = max(h_lo, h_denom_lo)
            h_hi = min(h_hi, h_denom_hi)

        hist_all_full, H_BINS = _restrict_range(hist_all_full, H_BINS_denom, h_lo, h_hi, axis=0)

        if use_color:
            if gh_range is not None:
                gh_lo, gh_hi = float(gh_range[0]), float(gh_range[1])
                if gh_lo < GH_BINS_denom[0] or gh_hi > GH_BINS_denom[-1]:
                    raise ValueError(
                        f"gh_range=({gh_lo}, {gh_hi}) falls outside the denominator bounds "
                        f"[{GH_BINS_denom[0]}, {GH_BINS_denom[-1]}]."
                    )
            else:
                gh_arr = gh[np.isfinite(gh) & np.isfinite(h)]
                gh_lo = float(np.floor(gh_arr.min() / _NATIVE_STEP) * _NATIVE_STEP) if len(gh_arr) else GH_BINS_denom[0]
                gh_hi = float(np.ceil( gh_arr.max() / _NATIVE_STEP) * _NATIVE_STEP) if len(gh_arr) else GH_BINS_denom[-1]
                gh_lo = max(gh_lo, GH_BINS_denom[0])
                gh_hi = min(gh_hi, GH_BINS_denom[-1])
            hist_all_full, GH_BINS = _restrict_range(hist_all_full, GH_BINS_denom, gh_lo, gh_hi, axis=2)
        else:
            GH_BINS = GH_BINS_denom

        # ── Coarsen bins if requested ─────────────────────────────────────────
        if h_factor > 1:
            hist_all_full, H_BINS = _coarsen(hist_all_full, H_BINS, h_factor, axis=0)
        if use_color and gh_factor > 1:
            hist_all_full, GH_BINS = _coarsen(hist_all_full, GH_BINS, gh_factor, axis=2)

        # ── Build per-mode denominator ────────────────────────────────────────
        if use_color:
            hist_all = hist_all_full
        else:
            hist_all = hist_all_full.sum(axis=2)

        # ── Bin observed stars into numerator ─────────────────────────────────
        hist_obs = cls._bin_observed(
            ra, dec, h, g,
            nside=nside, H_BINS=H_BINS, GH_BINS=GH_BINS if use_color else None,
        )

        # ── Adaptive SF ───────────────────────────────────────────────────────
        sf, nside_map = _build_adaptive(
            hist_all, hist_obs, nside=nside,
            min_count=min_count, nside_min=nside_min, use_prior=use_prior,
        )

        return cls(
            selfunc=sf,
            hist_all=hist_all,
            hist_obs=hist_obs,
            H_BINS=H_BINS,
            nside=nside,
            GH_BINS=GH_BINS if use_color else None,
            nside_map=nside_map,
            use_prior=use_prior,
        )

    @classmethod
    def _load_denominator(cls, path: str | Path | None) -> dict:
        """
        Load the pre-computed 2MASS colour denominator npz.

        The file must contain:

        - ``hist_all`` — shape ``(n_H, n_pix, n_GH)``, 2MASS counts per bin
        - ``H_BINS`` — H-magnitude bin edges
        - ``GH_BINS`` — G−H bin edges
        - ``nside`` — HEALPix resolution

        Bin edges and nside are read from the file; callers do not need to
        specify them separately.
        """
        if path is None:
            pkg_data = importlib.resources.files("sdssv_apogee_sf") / "data" / "twomass_denominator.npz"
            with importlib.resources.as_file(pkg_data) as p:
                path = p

        d = np.load(path, allow_pickle=False)

        for key in ("hist_all", "H_BINS", "GH_BINS", "nside"):
            if key not in d:
                raise ValueError(
                    f"Denominator file is missing key '{key}'. "
                    "Re-generate it with: python gg_selfunc_healpix.py --use-color"
                )
        return dict(d)

    @staticmethod
    def _bin_observed(
        ra: np.ndarray,
        dec: np.ndarray,
        h: np.ndarray,
        g: np.ndarray | None,
        nside: int,
        H_BINS: np.ndarray,
        GH_BINS: np.ndarray | None,
    ) -> np.ndarray:
        """
        Bin observed stars into a HEALPix + H (+ G-H) histogram.

        Returns
        -------
        hist : (n_H, n_pix) or (n_H, n_pix, n_GH)
        """
        n_pix = hp.nside2npix(nside)
        n_H   = len(H_BINS) - 1

        theta = np.radians(90.0 - dec)
        phi   = np.radians(ra)
        pix   = hp.ang2pix(nside, theta, phi, nest=False)

        valid = np.isfinite(h)
        h_idx = np.searchsorted(H_BINS[1:], h[valid])
        h_idx = np.clip(h_idx, 0, n_H - 1)
        pix_v = pix[valid]

        if GH_BINS is None:
            hist = np.zeros((n_H, n_pix), dtype=np.int32)
            for hi in range(n_H):
                sel = pix_v[h_idx == hi]
                np.add.at(hist[hi], sel, 1)
            return hist

        n_GH = len(GH_BINS) - 1
        both    = np.isfinite(h) & np.isfinite(g)
        pix_b   = hp.ang2pix(nside, np.radians(90.0 - dec[both]),
                              np.radians(ra[both]), nest=False)
        h_b     = h[both]
        gh_b    = g[both] - h[both]
        h_idx_b = np.clip(np.searchsorted(H_BINS[1:], h_b), 0, n_H - 1)
        gh_idx  = np.clip(np.searchsorted(GH_BINS[1:], gh_b), 0, n_GH - 1)

        hist = np.zeros((n_H, n_pix, n_GH), dtype=np.int32)
        for hi in range(n_H):
            sel_h = h_idx_b == hi
            for gi in range(n_GH):
                sel = pix_b[sel_h & (gh_idx == gi)]
                np.add.at(hist[hi, :, gi], sel, 1)
        return hist

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        coords,
        H: Union[float, np.ndarray],
        GH: Union[float, np.ndarray, None] = None,
    ) -> np.ndarray:
        """
        Evaluate the selection function at sky coordinates and magnitude(s).

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            Sky position(s) of the query stars.
        H      : float or array-like
            H-band magnitude(s).
        GH     : float or array-like, optional
            G-H colour(s). Required when the SF was built with ``g_col``.

        Returns
        -------
        p : ndarray, shape (N,) or scalar
            Selection function value(s) in [0, 1].  NaN for out-of-range inputs.
        """
        scalar = np.ndim(H) == 0
        pix    = _pix_for_coords(coords, self._nside)
        H      = np.atleast_1d(np.asarray(H, dtype=float))
        n_H    = len(self._H_BINS) - 1

        h_idx  = np.clip(np.searchsorted(self._H_BINS[1:], H), 0, n_H - 1)
        out_h  = (H < self._H_BINS[0]) | (H >= self._H_BINS[-1])

        if not self.use_color:
            result = self._selfunc[h_idx, pix]
            out_of_range = out_h
        else:
            if GH is None:
                raise ValueError("GH must be provided for a colour selection function.")
            GH    = np.atleast_1d(np.asarray(GH, dtype=float))
            n_GH  = len(self._GH_BINS) - 1
            gh_idx = np.clip(np.searchsorted(self._GH_BINS[1:], GH), 0, n_GH - 1)
            out_gh = (GH < self._GH_BINS[0]) | (GH >= self._GH_BINS[-1])
            result = self._selfunc[h_idx, pix, gh_idx]
            out_of_range = out_h | out_gh

        result = result.copy()
        result[out_of_range] = np.nan
        return float(result[0]) if scalar else result

    # ── Persistence ───────────────────────────────────────────────────────────

    def write(self, path: str | Path) -> None:
        """Save to a compressed .npz file."""
        path = Path(path)
        arrays: dict[str, np.ndarray] = {
            "selfunc":  self._selfunc,
            "hist_all": self._hist_all,
            "hist_obs": self._hist_obs,
            "H_BINS":   self._H_BINS,
            "nside":    np.array(self._nside),
            "use_prior": np.array(self._use_prior),
        }
        if self._GH_BINS is not None:
            arrays["GH_BINS"] = self._GH_BINS
        if self._nside_map is not None:
            arrays["nside_map"] = self._nside_map
        np.savez_compressed(path, **arrays)

    @classmethod
    def read(cls, path: str | Path) -> "APOGEESelectionFunction":
        """Load from a .npz file written by :meth:`write`."""
        d = np.load(path, allow_pickle=False)
        return cls(
            selfunc   = d["selfunc"],
            hist_all  = d["hist_all"],
            hist_obs  = d["hist_obs"],
            H_BINS    = d["H_BINS"],
            nside     = int(d["nside"]),
            GH_BINS   = d["GH_BINS"]   if "GH_BINS"   in d else None,
            nside_map = d["nside_map"] if "nside_map" in d else None,
            use_prior = bool(d["use_prior"]) if "use_prior" in d else True,
        )
