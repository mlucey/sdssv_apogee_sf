#!/usr/bin/env python3
"""
Can a hybrid SF put every (H, G-H) cell at full nside-64 resolution?

Hybrid model:  S(h, g, p) = L(J, G, p) * S_fine(h, g, P) / S_level(J, G, P)
  L      : "level" from a merged (wide-H J, wide-G-H G) bin, at fine pixel p
  shape  : native 0.5-mag (h, g) bin relative to its merged bin, at coarse pixel P

For several merge choices this reports
  1. fraction of observed stars whose level cell has >= min_count 2MASS stars at
     nside 64 (i.e. could be quoted at full resolution),
  2. noise-debiased rms error vs the native SF, evaluated at nside 16 with the
     shape taken from nside 8 (cells with >= 100 2MASS stars), and at nside 32
     with the shape from nside 8 (closer to the real 64:1 use case).

Usage:  python analysis/test_hybrid_full_res.py [--input ...npz] [--min-count 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hbin_width import DATA_DIR, degrade_nest, to_nest  # noqa: E402

H_EDGES = np.array([-2.0, 7.0, 11.0])  # level H bins: everything bright, then 7-11


def merge(arr, h_native, gh_native, h_edges, gh_edges):
    hs = np.searchsorted(h_native, h_edges[:-1] - 1e-9)
    gs = np.searchsorted(gh_native, gh_edges[:-1] - 1e-9)
    out = np.add.reduceat(np.add.reduceat(arr, hs, axis=0), gs, axis=2)
    hp_ = np.repeat(np.arange(len(hs)), np.diff(np.append(hs, arr.shape[0])))
    gp_ = np.repeat(np.arange(len(gs)), np.diff(np.append(gs, arr.shape[2])))
    return out, hp_, gp_


def beta_var(o, a):
    al, be = o + 1, a - o + 1
    return al * be / ((al + be) ** 2 * (al + be + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=str(DATA_DIR / 'gg_sf_color_dr19_mincount20.npz'))
    ap.add_argument('--min-count', type=int, default=20)
    ap.add_argument('--cmp-min-count', type=int, default=100)
    args = ap.parse_args()

    d = np.load(args.input)
    nside = int(d['nside'])
    H, GH = d['H_BINS'], d['GH_BINS']
    ha = to_nest(d['hist_all'].astype(float), nside)
    ho = to_nest(d['hist_obs'].astype(float), nside)
    n_obs = ho.sum()

    gh_lo, gh_hi = GH[0], GH[-1]
    schemes = {
        'H 2-mag, G-H native': (np.array([-2.0, 7.0, 9.0, 11.0]), GH),
        'H 7-11, G-H native': (H_EDGES, GH),
        'H 7-11, G-H 1-mag': (H_EDGES, np.append(np.arange(gh_lo, gh_hi, 1.0), gh_hi)),
        'H 7-11, G-H 2-mag': (H_EDGES, np.append(np.arange(gh_lo, gh_hi, 2.0), gh_hi)),
        'H 7-11, G-H 4-mag': (H_EDGES, np.append(np.arange(gh_lo, gh_hi, 4.0), gh_hi)),
        'H 7-11, G-H all': (H_EDGES, np.array([gh_lo, gh_hi])),
    }

    print(f"{args.input}  min_count={args.min_count}")
    print(f"{'level bins':24s} {'stars @64':>9s} {'stars @>=32':>11s}"
          f" {'err 16|8':>9s} {'err 32|8':>9s} {'cov 32|8':>9s}")
    for name, (he, ge) in schemes.items():
        la, hpar, gpar = merge(ha, H, GH, he, ge)
        lo, _, _ = merge(ho, H, GH, he, ge)
        good64 = (la >= args.min_count)[hpar][:, :, gpar]
        good32 = (degrade_nest(la, nside, 32) >= args.min_count)[hpar][:, :, gpar]
        good32 = good64 | np.repeat(good32, 4, axis=1)
        f64 = ho[good64].sum() / n_obs
        f32 = ho[good32].sum() / n_obs

        errs = []
        for ns_eval, ns_shape in ((16, 8), (32, 8)):
            k = (ns_eval // ns_shape) ** 2
            a_f = degrade_nest(ha, nside, ns_eval); o_f = degrade_nest(ho, nside, ns_eval)
            a_l = degrade_nest(la, nside, ns_eval); o_l = degrade_nest(lo, nside, ns_eval)
            a_fc = degrade_nest(ha, nside, ns_shape); o_fc = degrade_nest(ho, nside, ns_shape)
            a_lc = degrade_nest(la, nside, ns_shape); o_lc = degrade_nest(lo, nside, ns_shape)
            s_f = (o_f + 1) / (a_f + 2)
            s_l = ((o_l + 1) / (a_l + 2))[hpar][:, :, gpar]
            shape = ((o_fc + 1) / (a_fc + 2)) / ((o_lc + 1) / (a_lc + 2))[hpar][:, :, gpar]
            s_hyb = s_l * np.repeat(shape, k, axis=1)
            ok = a_f >= args.cmp_min_count
            w = o_f[ok]
            dS = (s_hyb - s_f)[ok]
            # Approximate debias: subtract the fine-cell counting variance only
            # (ignores hybrid noise and its covariance with the fine cell).
            noise = beta_var(o_f, a_f)[ok]
            ex = (w * (dS ** 2 - noise)).sum() / w.sum()
            s_mean = (w * s_f[ok]).sum() / w.sum()
            raw = np.sqrt((w * dS ** 2).sum() / w.sum()) / s_mean
            nrms = np.sqrt((w * noise).sum() / w.sum()) / s_mean
            errs.append((np.sqrt(max(ex, 0)) / s_mean, w.sum() / n_obs, raw, nrms))
        print(f"{name:24s} {100 * f64:8.1f}% {100 * f32:10.1f}%"
              f" {100 * errs[0][0]:8.1f}% {100 * errs[1][0]:8.1f}% {100 * errs[1][1]:8.0f}%"
              f"   raw/noise 16|8: {100 * errs[0][2]:.1f}/{100 * errs[0][3]:.1f}%"
              f"  32|8: {100 * errs[1][2]:.1f}/{100 * errs[1][3]:.1f}%")


if __name__ == '__main__':
    main()
