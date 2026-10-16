#!/usr/bin/env python3
"""
Build a "hybrid" version of an existing colour GG selection function npz.

Everything (bins, counts, nside, min_count, nside_min, estimator) is copied from
the input; only how ``selfunc`` is computed changes.  For a native cell
(H bin h, pixel p, G-H bin g) with h inside the merged "level" H bin J:

    S_hyb(h, p, g) = S_orig(h, g, P) * u(J, g, q) * N(h, g, P) / sum_{p' in P} N(h, g, p') u(p')

  P      : pixel containing p at the resolution the ORIGINAL adaptive SF used
           for (h, p, g); S_orig is the original value there
  q      : pixel containing p at the finest nside where the merged (J, g) bin
           holds >= min_count 2MASS stars (adaptive, as in _build_adaptive)
  u      : the merged bin's SF at q (prior pseudo-count split over q by 2MASS
           share for the 'expectation' estimator; raw ratio for 'map')
  N      : 2MASS counts of the native bin

So each original cell keeps exactly its expected number of observed stars,
S_orig * N(P), and the hybrid only redistributes it across sub-pixels following
the merged bin's sky pattern.  Where the merged bin is no finer than the native
one, S_hyb equals the original value.  Cells that were NaN in the original stay
NaN.  Results are clipped to [0, 1].

Extra keys written: method='hybrid', level_h_edges, nside_map_shape (the
original nside_map).  ``nside_map`` follows the package convention (coarsest
resolution over G-H bins per (H, pixel)) but now refers to the level term.

Usage:
  python analysis/build_hybrid_selfunc.py --input IN.npz --output OUT.npz
"""

import argparse
from pathlib import Path

import healpy as hp
import numpy as np

LEVEL_H_EDGES = (-2.0, 7.0, 11.0)


def to_nest(arr, nside):
    return arr[:, hp.nest2ring(nside, np.arange(hp.nside2npix(nside)))]


def to_ring(arr, nside):
    return arr[:, hp.ring2nest(nside, np.arange(hp.nside2npix(nside)))]


def degrade(arr, nside, ns):
    k = (nside // ns) ** 2
    return arr.reshape((arr.shape[0], arr.shape[1] // k, k) + arr.shape[2:]).sum(axis=2)


def estimate(o, a, estimator):
    if estimator == 'expectation':
        return (o + 1.0) / (a + 2.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(a > 0, o / a, np.nan)


def adaptive(ha, ho, nside, min_count, nside_min, estimator):
    """NESTED-order clone of _build_adaptive returning per-cell SF and nside."""
    sf = np.full(ha.shape, np.nan)
    ns_eff = np.zeros(ha.shape, dtype=np.int32)
    ns = nside
    while ns >= max(nside_min, 1):
        k = (nside // ns) ** 2
        a, o = degrade(ha, nside, ns), degrade(ho, nside, ns)
        good = np.repeat(a >= max(min_count, 1), k, axis=1)
        s = np.repeat(np.clip(estimate(o, a, estimator), 0.0, 1.0), k, axis=1)
        new = good & (ns_eff == 0)
        sf[new] = s[new]
        ns_eff[new] = ns
        ns //= 2
    return sf, ns_eff


def nside_map_convention(ns_eff):
    """Package convention: coarsest covered nside over G-H bins, 0 if none."""
    big = np.where(ns_eff > 0, ns_eff, np.iinfo(np.int32).max)
    m = big.min(axis=2)
    return np.where(m == np.iinfo(np.int32).max, 0, m).astype(np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--level-h-edges', type=float, nargs='+', default=list(LEVEL_H_EDGES))
    args = ap.parse_args()

    d = np.load(args.input)
    nside = int(d['nside'])
    min_count = int(d['min_count'])
    nside_min = int(d['nside_min'])
    estimator = str(d['estimator'])
    H_BINS = d['H_BINS']
    level_edges = np.asarray(args.level_h_edges, float)
    print(f"{args.input}: nside={nside} min_count={min_count} nside_min={nside_min} "
          f"estimator={estimator} level H edges={level_edges.tolist()}")

    ha = to_nest(d['hist_all'].astype(float), nside)
    ho = to_nest(d['hist_obs'].astype(float), nside)

    # 1. Original SF (and check it reproduces the stored file exactly)
    sf_fine, ns_fine = adaptive(ha, ho, nside, min_count, nside_min, estimator)
    same = np.allclose(to_ring(sf_fine, nside), d['selfunc'], equal_nan=True)
    same_map = np.array_equal(to_ring(nside_map_convention(ns_fine), nside), d['nside_map'])
    print(f"reproduces stored selfunc: {same}; stored nside_map: {same_map}")
    if not same:
        raise SystemExit("Could not reproduce the input selfunc; aborting.")

    # 2. Merged level bins
    starts = np.searchsorted(H_BINS, level_edges[:-1] - 1e-9)
    if (not np.allclose(H_BINS[starts], level_edges[:-1])
            or not np.isclose(level_edges[-1], H_BINS[-1])):
        raise SystemExit(f"level edges {level_edges} not aligned with H_BINS")
    parent = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, ha.shape[0])))
    la = np.add.reduceat(ha, starts, axis=0)
    lo = np.add.reduceat(ho, starts, axis=0)
    _, ns_lvl = adaptive(la, lo, nside, min_count, nside_min, estimator)
    # Level counts in each fine pixel's level pixel q (at the level's resolution)
    LA_q = np.zeros(la.shape); LO_q = np.zeros(la.shape)
    ns = nside
    while ns >= max(nside_min, 1):
        m = ns_lvl == ns
        if m.any():
            k = (nside // ns) ** 2
            LA_q[m] = np.repeat(degrade(la, nside, ns), k, axis=1)[m]
            LO_q[m] = np.repeat(degrade(lo, nside, ns), k, axis=1)[m]
        ns //= 2
    LA_q, LO_q, ns_L = LA_q[parent], LO_q[parent], ns_lvl[parent]

    # 3. Sky weights u(h, p, g) from the level bin, and count-conserving
    #    renormalisation within each native cell (h, P, g) at its own resolution:
    #      S_hyb(p) = S_orig(P) * u(p) * N2M(h, P) / sum_{p' in P} N2M(h, p') u(p')
    #    so sum_p N2M(h,p) S_hyb(p) == S_orig(P) N2M(h,P) exactly (before clipping).
    #    'expectation': the parent's single Beta(1,1) pseudo-count is split over
    #    level pixels q in proportion to their 2MASS share, w = N_lvl(q)/N_lvl(P).
    #    'map': u = raw level ratio.
    hyb = np.full(ha.shape, np.nan)
    ns = nside
    while ns >= max(nside_min, 1):
        m = ns_fine == ns
        if m.any():
            k = (nside // ns) ** 2
            # Like _build_adaptive, a coarse value is computed from all children
            # of P but only ASSIGNED to the children not covered at a finer
            # nside (mask m).  Conserve counts over the assigned children only.
            if estimator == 'expectation':
                LA_P = np.repeat(degrade(np.where(m, la[parent], 0.0), nside, ns), k, axis=1)
                w = LA_q / np.where(LA_P > 0, LA_P, 1.0)
                with np.errstate(invalid='ignore'):  # 0/0 only outside covered cells
                    u = (LO_q + w) / (LA_q + 2.0 * w)
            else:
                with np.errstate(invalid='ignore', divide='ignore'):
                    u = np.where(LA_q > 0, LO_q / LA_q, 0.0)
            u = np.where(m, u, 0.0)
            norm = np.repeat(degrade(ha * u, nside, ns), k, axis=1)
            ha_P = np.repeat(degrade(np.where(m, ha, 0.0), nside, ns), k, axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                s = sf_fine * u * ha_P / norm
            s = np.where(norm > 0, s, sf_fine)  # no level signal in P: keep original
            hyb[m] = s[m]
        ns //= 2

    hyb_unclipped = np.where(ns_fine == 0, np.nan, hyb)
    hyb = np.where(ns_fine == 0, np.nan, np.clip(hyb, 0.0, 1.0))
    print(f"cells clipped at S=1: {(hyb_unclipped > 1).sum():,} "
          f"of {np.isfinite(hyb).sum():,} finite")
    ns_used = np.where(ns_fine > 0, np.maximum(ns_L, ns_fine), 0).astype(np.int32)

    # Sanity: the children assigned to each original coarse value keep its expected
    # observed count.  Checked before and after clipping S to <= 1.
    def conservation(arr):
        bad = n_cells = 0
        ns = nside
        while ns >= max(nside_min, 1):
            m = ns_fine == ns
            cell = degrade(m.astype(float), nside, ns) > 0
            exp_o = degrade(np.where(m, ha * sf_fine, 0), nside, ns)
            exp_h = degrade(np.where(m, ha * np.nan_to_num(arr), 0), nside, ns)
            bad += (cell & ~np.isclose(exp_h, exp_o, rtol=1e-6, atol=1e-9)).sum()
            n_cells += cell.sum()
            ns //= 2
        return bad, n_cells
    bad_u, n_cells = conservation(hyb_unclipped)
    bad_c, _ = conservation(hyb)
    print(f"count conservation over {n_cells:,} original cells: "
          f"{bad_u:,} differ before clipping, {bad_c:,} after clipping to S <= 1")
    assert np.array_equal(np.isnan(hyb), np.isnan(sf_fine))

    w = ho
    n_obs = w.sum()
    for lab, nsarr in (('original', ns_fine), ('hybrid', ns_used)):
        fr = " / ".join(f"{100 * w[nsarr == n].sum() / n_obs:.1f}%" for n in (64, 32, 16, 8, 0))
        print(f"  {lab:8s} stars at nside 64/32/16/8/uncov: {fr}")

    payload = {k: d[k] for k in d.files}
    payload['selfunc'] = to_ring(hyb, nside)
    payload['nside_map'] = to_ring(nside_map_convention(ns_used), nside)
    payload['nside_map_shape'] = d['nside_map']
    payload['method'] = np.array('hybrid')
    payload['level_h_edges'] = level_edges
    np.savez_compressed(args.output, **payload)
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
