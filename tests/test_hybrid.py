"""Tests for the hybrid (merged-H sky pattern) selection function."""
import warnings

import healpy as hp
import numpy as np
import pytest

from sdssv_apogee_sf import APOGEESelectionFunction
from sdssv_apogee_sf._selfunc import (
    _adaptive_nest,
    _build_adaptive,
    _build_hybrid,
    _degrade_nest,
    _h_merge_edges,
    _to_nest,
)

NSIDE, NSIDE_MIN, MIN_COUNT = 8, 2, 20
H_BINS = np.arange(5.0, 9.01, 0.5)      # 8 native H bins
GH_BINS = np.arange(1.0, 3.01, 0.5)     # 4 G-H bins
EDGES = np.array([5.0, 7.0, 9.0])


def _counts(seed=0, color=True):
    """Sparse 2MASS counts and a sky-varying observed fraction (< 1)."""
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(NSIDE)
    shape = (len(H_BINS) - 1, npix) + ((len(GH_BINS) - 1,) if color else ())
    ha = rng.poisson(3.0, size=shape).astype(float)
    theta, _ = hp.pix2ang(NSIDE, np.arange(npix))
    frac = 0.05 + 0.4 * np.cos(theta) ** 2
    frac = frac[None, :, None] if color else frac[None, :]
    ho = rng.binomial(ha.astype(int), np.broadcast_to(frac, shape)).astype(float)
    return ha, ho


@pytest.mark.parametrize("estimator", ["expectation", "map"])
@pytest.mark.parametrize("color", [True, False])
def test_nan_pattern_and_range(estimator, color):
    ha, ho = _counts(color=color)
    sf_a, _ = _build_adaptive(ha, ho, NSIDE, MIN_COUNT, NSIDE_MIN, estimator)
    sf_h, nmap = _build_hybrid(ha, ho, H_BINS, NSIDE, EDGES, MIN_COUNT, NSIDE_MIN, estimator)
    assert np.array_equal(np.isnan(sf_h), np.isnan(sf_a))
    finite = sf_h[np.isfinite(sf_h)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0
    assert nmap.shape == ha.shape[:2]
    assert not np.allclose(sf_h, sf_a, equal_nan=True)  # hybrid actually adds detail


@pytest.mark.parametrize("estimator", ["expectation", "map"])
def test_adaptive_nest_matches_build_adaptive(estimator):
    ha, ho = _counts()
    sf_a, _ = _build_adaptive(ha, ho, NSIDE, MIN_COUNT, NSIDE_MIN, estimator)
    sf_n, _ = _adaptive_nest(_to_nest(ha, NSIDE), _to_nest(ho, NSIDE),
                             NSIDE, MIN_COUNT, NSIDE_MIN, estimator)
    assert np.allclose(_to_nest(sf_a, NSIDE), sf_n, equal_nan=True)


@pytest.mark.parametrize("estimator", ["expectation", "map"])
def test_expected_counts_conserved(estimator):
    """Each original coarse cell keeps its expected number of observed stars."""
    ha, ho = _counts()
    sf_h, _ = _build_hybrid(ha, ho, H_BINS, NSIDE, EDGES, MIN_COUNT, NSIDE_MIN, estimator)
    han = _to_nest(ha, NSIDE)
    sf_n, ns_eff = _adaptive_nest(han, _to_nest(ho, NSIDE), NSIDE, MIN_COUNT, NSIDE_MIN, estimator)
    hyb_n = _to_nest(sf_h, NSIDE)
    assert hyb_n.max() < 1.0  # no clipping, so conservation is exact
    ns = NSIDE
    while ns >= NSIDE_MIN:
        m = ns_eff == ns
        exp_orig = _degrade_nest(np.where(m, han * sf_n, 0.0), NSIDE, ns)
        exp_hyb = _degrade_nest(np.where(m, han * np.nan_to_num(hyb_n), 0.0), NSIDE, ns)
        assert np.allclose(exp_hyb, exp_orig, rtol=1e-9, atol=1e-12)
        ns //= 2


@pytest.mark.parametrize("estimator", ["expectation", "map"])
def test_unmerged_edges_reduce_to_adaptive(estimator):
    """If every merged bin is a single native bin, nothing changes."""
    ha, ho = _counts()
    sf_a, nmap_a = _build_adaptive(ha, ho, NSIDE, MIN_COUNT, NSIDE_MIN, estimator)
    sf_h, nmap_h = _build_hybrid(ha, ho, H_BINS, NSIDE, H_BINS, MIN_COUNT, NSIDE_MIN, estimator)
    assert np.allclose(sf_h, sf_a, equal_nan=True)
    assert np.array_equal(nmap_h, nmap_a)


def test_misaligned_edges_raise():
    ha, ho = _counts()
    with pytest.raises(ValueError):
        _build_hybrid(ha, ho, H_BINS, NSIDE, [5.0, 7.2, 9.0], MIN_COUNT, NSIDE_MIN)


def test_h_merge_edges():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(_h_merge_edges(H_BINS, (7.0,)), [5.0, 7.0, 9.0])
        assert np.array_equal(_h_merge_edges(H_BINS, ()), [5.0, 9.0])
        # breaks at or outside the range are dropped
        assert np.array_equal(_h_merge_edges(H_BINS, (4.0, 5.0, 9.0, 12.0)), [5.0, 9.0])
    with pytest.warns(UserWarning, match="not an H bin edge"):
        assert np.array_equal(_h_merge_edges(H_BINS, (7.1,)), [5.0, 7.0, 9.0])


def test_write_read_roundtrip(tmp_path):
    ha, ho = _counts()
    sf_h, nmap = _build_hybrid(ha, ho, H_BINS, NSIDE, EDGES, MIN_COUNT, NSIDE_MIN)
    sf = APOGEESelectionFunction(sf_h, ha, ho, H_BINS, NSIDE, GH_BINS, nmap,
                                 h_merge_edges=EDGES)
    assert sf.method == "hybrid"
    sf.write(tmp_path / "hyb.npz")
    back = APOGEESelectionFunction.read(tmp_path / "hyb.npz")
    assert back.method == "hybrid"
    assert np.array_equal(back.h_merge_edges, EDGES)

    plain = APOGEESelectionFunction(sf_h, ha, ho, H_BINS, NSIDE, GH_BINS, nmap)
    plain.write(tmp_path / "plain.npz")
    assert APOGEESelectionFunction.read(tmp_path / "plain.npz").method == "adaptive"

    # Files from the earlier analysis/build_hybrid_selfunc.py script.
    d = dict(np.load(tmp_path / "plain.npz"))
    d["level_h_edges"] = EDGES
    np.savez(tmp_path / "legacy.npz", **d)
    legacy = APOGEESelectionFunction.read(tmp_path / "legacy.npz")
    assert np.array_equal(legacy.h_merge_edges, EDGES)


def test_from_observed_default_is_hybrid():
    """Uses the bundled 2MASS denominator."""
    rng = np.random.default_rng(1)
    n = 4000
    table = {
        "ra": rng.uniform(0, 360, n),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
        "h_m": rng.uniform(6.0, 8.0, n),
    }
    kw = dict(h_range=(6.0, 8.0), min_count=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hyb = APOGEESelectionFunction.from_observed(table, **kw)
        ada = APOGEESelectionFunction.from_observed(table, h_merge_breaks=None, **kw)
    assert hyb.method == "hybrid" and np.array_equal(hyb.h_merge_edges, [6.0, 7.0, 8.0])
    assert ada.method == "adaptive" and ada.h_merge_edges is None
    sf_a, _ = _build_adaptive(ada._hist_all, ada._hist_obs, ada.nside, 20, 8)
    assert np.allclose(ada._selfunc, sf_a, equal_nan=True)
    assert np.array_equal(np.isnan(hyb._selfunc), np.isnan(ada._selfunc))
