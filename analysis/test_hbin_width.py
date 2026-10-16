#!/usr/bin/env python3
"""
Test how widening the H-magnitude bins changes the on-sky resolution of the
GG selection function.

The adaptive scheme in ``_build_adaptive`` coarsens HEALPix pixels (nside 64 ->
32 -> 16 -> 8) until each (H, pixel[, G-H]) cell holds >= min_count 2MASS
sources.  Wider H bins put more 2MASS stars in each cell, so more cells should
stay at fine sky resolution.  The cost is that the SF is averaged over a wider
H range.

For each H bin width we
  1. sum adjacent 0.5-mag H bins of the stored hist_all / hist_obs (nside 64),
  2. rebuild the adaptive SF (vectorised clone of ``_build_adaptive``, verified
     against the package on the 0.5-mag baseline),
  3. GAIN: record the effective nside of the cell each observed star lands in,
  4. COST: compare the wide-H SF with the native 0.5-mag SF at a matched sky
     resolution (nside_cmp), in cells where the fine-bin SF is well measured,
     so the difference isolates the smoothing in H.

Usage:
  python analysis/test_hbin_width.py
  python analysis/test_hbin_width.py --input .../gg_sf_dr19.npz --min-count 50
"""

import argparse
import json
import sys
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from sdssv_apogee_sf._selfunc import _build_adaptive  # noqa: E402

DATA_DIR = Path('/Users/maddielucey/gg_selec_func/sdssv_apogee_sf')
# H-bin schemes: label -> bin edges (must be a subset of the native 0.5-mag edges).
# The GG selection has a hard bright limit at H = 7.0, so the 'var' schemes put an
# edge there, lump everything brighter into one bin, and vary the width over 7-11.
SCHEMES = {
    '0.5': np.arange(-2.0, 11.01, 0.5),
    '1.0': np.arange(-2.0, 11.01, 1.0),
    '1.5': np.append(np.arange(-2.0, 11.01, 1.5), 11.0),
    '2.0': np.append(np.arange(-2.0, 11.01, 2.0), 11.0),
    '3.0': np.append(np.arange(-2.0, 11.01, 3.0), 11.0),
    # (lumping H<7 alone changes nothing: <0.2% of stars are brighter than 7)
    'var-2.0': np.array([-2.0, 7.0, 9.0, 11.0]),
    # keep 0.5 mag where S(H) is steep (7-9), one wide bin where it is flat (9-11)
    'var-0.5|2': np.array([-2.0, 7.0, 7.5, 8.0, 8.5, 9.0, 11.0]),
}
NSIDE_LEVELS = (64, 32, 16, 8)

# Single-hue blue ramp (light -> dark) for ordered quantities
BLUES = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
ORANGES = ['#eb6834', '#a83c12']  # variable-width schemes
NSIDE_COLORS = {64: '#104281', 32: '#2a78d6', 16: '#6da7ec', 8: '#b7d3f6', 0: '#c3c2b7'}
INK, INK2 = '#0b0b0b', '#52514e'


# ---------------------------------------------------------------------------
# Rebinning / adaptive build (NESTED ordering makes degrading a reshape-sum)
# ---------------------------------------------------------------------------

def rebin_h(arr, h_edges, new_edges):
    """Sum native H bins along axis 0 into the bins defined by *new_edges*."""
    starts = np.searchsorted(h_edges, new_edges[:-1] - 1e-9)
    if not np.allclose(h_edges[starts], new_edges[:-1]) or new_edges[-1] != h_edges[-1]:
        raise ValueError(f"edges {new_edges} are not aligned with native H bins")
    out = np.add.reduceat(arr, starts, axis=0)
    return out, np.asarray(new_edges, float), starts


def to_nest(arr, nside):
    """Reorder pixel axis (axis 1) from RING to NESTED."""
    return arr[:, hp.nest2ring(nside, np.arange(hp.nside2npix(nside)))]


def adaptive_nest(ha, ho, nside, min_count, nside_min=8):
    """
    Vectorised equivalent of _build_adaptive operating on NESTED arrays.
    Returns sf (same shape as ha) and per-cell effective nside (0 = uncovered).
    """
    sf = np.full(ha.shape, np.nan)
    ns_eff = np.zeros(ha.shape, dtype=np.int16)
    ns = nside
    while ns >= nside_min:
        k = (nside // ns) ** 2
        shp = (ha.shape[0], ha.shape[1] // k, k) + ha.shape[2:]
        a = ha.reshape(shp).sum(axis=2)
        o = ho.reshape(shp).sum(axis=2)
        good = a >= max(min_count, 1)
        s = np.clip((o + 1.0) / (a + 2.0), 0.0, 1.0)
        good_f = np.repeat(good, k, axis=1)
        s_f = np.repeat(s, k, axis=1)
        new = good_f & (ns_eff == 0)
        sf[new] = s_f[new]
        ns_eff[new] = ns
        ns //= 2
    return sf, ns_eff


def degrade_nest(arr, nside, ns):
    k = (nside // ns) ** 2
    return arr.reshape((arr.shape[0], arr.shape[1] // k, k) + arr.shape[2:]).sum(axis=2)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=str(DATA_DIR / 'gg_sf_color_dr19_mincount20.npz'))
    ap.add_argument('--min-count', type=int, default=None,
                    help='Default: value stored in the input npz (else 20)')
    ap.add_argument('--nside-cmp', type=int, default=16,
                    help='Matched sky resolution for the H-smoothing cost metric')
    ap.add_argument('--cmp-min-count', type=int, default=100,
                    help='Fine-bin 2MASS count needed for a cell to enter the cost metric')
    ap.add_argument('--outdir', default=str(REPO / 'analysis' / 'hbin_width_out'))
    ap.add_argument('--skip-verify', action='store_true')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.input)
    hist_all = d['hist_all'].astype(float)
    hist_obs = d['hist_obs'].astype(float)
    H_BINS = d['H_BINS']
    nside = int(d['nside'])
    min_count = args.min_count or (int(d['min_count']) if 'min_count' in d else 20)
    color = hist_all.ndim == 3
    tag = f"{Path(args.input).stem}_mc{min_count}"
    print(f"Input {args.input}: shape={hist_all.shape}, nside={nside}, "
          f"min_count={min_count}, colour={color}")

    # -- Verify vectorised clone against the package on the baseline ---------
    ha_n, ho_n = to_nest(hist_all, nside), to_nest(hist_obs, nside)
    if not args.skip_verify:
        sel = slice(None) if not color else (slice(None), slice(None), slice(8, 14))
        ref_sf, _ = _build_adaptive(hist_all[sel], hist_obs[sel], nside,
                                    min_count=min_count, nside_min=8)
        my_sf, _ = adaptive_nest(ha_n[sel], ho_n[sel], nside, min_count)
        ring2nest = hp.ring2nest(nside, np.arange(hp.nside2npix(nside)))
        my_sf_ring = my_sf[:, ring2nest]
        same = np.allclose(ref_sf, my_sf_ring, equal_nan=True)
        print(f"Verify vs package _build_adaptive (subset): match={same}")
        if not same:
            raise SystemExit("Vectorised build disagrees with package; aborting.")

    n_obs_tot = hist_obs.sum()
    results = []
    per_h = {}
    for width, edges in SCHEMES.items():
        ha, h_edges, starts = rebin_h(ha_n, H_BINS, edges)
        ho, _, _ = rebin_h(ho_n, H_BINS, edges)
        sf, ns_eff = adaptive_nest(ha, ho, nside, min_count)

        # GAIN: star-weighted effective nside
        frac = {ns: float(ho[ns_eff == ns].sum() / n_obs_tot) for ns in (*NSIDE_LEVELS, 0)}
        w = ho[ns_eff > 0]
        pix_deg = np.degrees(hp.nside2resol(ns_eff[ns_eff > 0].astype(int)))
        mean_pix = float((w * pix_deg).sum() / w.sum())
        # Posterior std of each star's cell (statistical precision)
        a_eff = np.zeros_like(ha); o_eff = np.zeros_like(ho)
        for ns in NSIDE_LEVELS:
            m = ns_eff == ns
            if not m.any():
                continue
            k = (nside // ns) ** 2
            a_eff[m] = np.repeat(degrade_nest(ha, nside, ns), k, axis=1)[m]
            o_eff[m] = np.repeat(degrade_nest(ho, nside, ns), k, axis=1)[m]
        al, be = o_eff + 1, a_eff - o_eff + 1
        sig = np.sqrt(al * be / ((al + be) ** 2 * (al + be + 1)))
        cov = ns_eff > 0
        order = np.argsort(sig[cov])
        cw = np.cumsum(ho[cov][order])
        med_sig = float(sig[cov][order][np.searchsorted(cw, cw[-1] / 2)])

        # Per-H-bin star-weighted mean pixel size
        ph = []
        for i in range(ha.shape[0]):
            m = ns_eff[i] > 0
            wi = ho[i][m]
            ph.append(float((wi * np.degrees(hp.nside2resol(ns_eff[i][m].astype(int)))).sum()
                            / wi.sum()) if wi.sum() > 0 else np.nan)
        per_h[width] = dict(edges=h_edges.tolist(), mean_pix=ph)

        # COST: H smoothing at matched sky resolution nside_cmp
        a_c_fine = degrade_nest(ha_n, nside, args.nside_cmp)
        o_c_fine = degrade_nest(ho_n, nside, args.nside_cmp)
        s_fine = (o_c_fine + 1) / (a_c_fine + 2)
        a_c = degrade_nest(ha, nside, args.nside_cmp)
        o_c = degrade_nest(ho, nside, args.nside_cmp)
        s_wide = (o_c + 1) / (a_c + 2)
        parent = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, ha_n.shape[0])))
        s_wide_f = s_wide[parent]
        ok = a_c_fine >= args.cmp_min_count
        wc = o_c_fine[ok]
        dS = (s_wide_f - s_fine)[ok]
        rms_abs = float(np.sqrt((wc * dS ** 2).sum() / wc.sum()))
        rel = dS / s_fine[ok]
        med_rel = float(np.median(np.repeat(np.abs(rel), wc.astype(int)))) if wc.sum() else np.nan
        # noise floor of the fine-bin estimate itself, for reference
        al_f, be_f = o_c_fine[ok] + 1, a_c_fine[ok] - o_c_fine[ok] + 1
        sig_f = np.sqrt(al_f * be_f / ((al_f + be_f) ** 2 * (al_f + be_f + 1)))
        noise_rms = float(np.sqrt((wc * sig_f ** 2).sum() / wc.sum()))
        # Noise-debiased smoothing rms: for a member of a weighted mean,
        # var(S_wide - S_fine) ~ sig_fine^2 - sig_wide^2 from counting noise alone.
        al_w = o_c[parent][ok] + 1
        be_w = a_c[parent][ok] - o_c[parent][ok] + 1
        sig_w = np.sqrt(al_w * be_w / ((al_w + be_w) ** 2 * (al_w + be_w + 1)))
        excess = (wc * (dS ** 2 - (sig_f ** 2 - sig_w ** 2))).sum() / wc.sum()
        rms_debiased = float(np.sqrt(max(excess, 0.0)))
        s_mean = float((wc * s_fine[ok]).sum() / wc.sum())
        # Per fine-H-bin debiased smoothing rms (where in H does widening hurt?)
        h_idx = np.broadcast_to(np.arange(ha_n.shape[0]).reshape((-1,) + (1,) * (ha_n.ndim - 1)),
                                ok.shape)[ok]
        cost_h = []
        for j in range(ha_n.shape[0]):
            m = h_idx == j
            if wc[m].sum() < 100:
                cost_h.append(np.nan)
                continue
            ex = (wc[m] * (dS[m] ** 2 - (sig_f[m] ** 2 - sig_w[m] ** 2))).sum() / wc[m].sum()
            sm = (wc[m] * s_fine[ok][m]).sum() / wc[m].sum()
            cost_h.append(float(np.sqrt(max(ex, 0.0)) / sm))
        per_h[width]['cost_rel'] = cost_h

        # HYBRID: wide-bin sky pattern x fine-bin H shape measured one level coarser.
        # S_hyb(j, p) = S_wide(J, p) * S_fine(j, P) / S_wide(J, P),  P = parent of p
        ns_c = args.nside_cmp // 2
        k_c = 4
        sf_c = (degrade_nest(ho_n, nside, ns_c) + 1) / (degrade_nest(ha_n, nside, ns_c) + 2)
        sw_c = (degrade_nest(ho, nside, ns_c) + 1) / (degrade_nest(ha, nside, ns_c) + 2)
        ratio = np.repeat(sf_c / sw_c[parent], k_c, axis=1)
        dS_h = (s_wide_f * ratio - s_fine)[ok]
        ex_h = (wc * (dS_h ** 2 - (sig_f ** 2 - sig_w ** 2))).sum() / wc.sum()
        rms_hyb = float(np.sqrt(max(ex_h, 0.0)))
        raw_wide = float(np.sqrt((wc * dS ** 2).sum() / wc.sum()))
        raw_hyb = float(np.sqrt((wc * dS_h ** 2).sum() / wc.sum()))
        frac_stars_in_cmp = float(wc.sum() / n_obs_tot)

        r = dict(width=width, n_H=int(ha.shape[0]), frac=frac,
                 mean_pix_deg=mean_pix, median_post_sigma=med_sig,
                 cost_rms_dS=rms_abs, cost_median_rel_dS=med_rel,
                 cost_fine_noise_rms=noise_rms, cost_rms_debiased=rms_debiased,
                 cost_rel_debiased=rms_debiased / s_mean,
                 cost_rel_hybrid=rms_hyb / s_mean, cost_star_frac=frac_stars_in_cmp)
        results.append(r)
        print(f"\nH scheme {width} ({ha.shape[0]} bins): edges {np.round(h_edges, 1).tolist()}")
        print("  stars at nside 64/32/16/8/uncov: " +
              " / ".join(f"{100 * frac[k]:.1f}%" for k in (*NSIDE_LEVELS, 0)))
        print(f"  star-weighted mean pixel size: {mean_pix:.2f} deg; "
              f"median posterior sigma: {med_sig:.3f}")
        print(f"  H-smoothing cost @nside {args.nside_cmp}: rms dS={rms_abs:.4f}, "
              f"median |dS|/S={100 * med_rel:.1f}% "
              f"(fine-bin noise rms {noise_rms:.4f}; {100 * frac_stars_in_cmp:.0f}% of stars)")
        print(f"  noise-debiased smoothing rms dS={rms_debiased:.4f} "
              f"= {100 * rms_debiased / s_mean:.1f}% of star-weighted mean S ({s_mean:.3f}); "
              f"hybrid-corrected: {100 * rms_hyb / s_mean:.1f}%")
        print(f"  raw rms dS (no debias): wide={raw_wide:.4f}, hybrid={raw_hyb:.4f}, "
              f"fine-bin noise={noise_rms:.4f}")

    with open(outdir / f'summary_{tag}.json', 'w') as fh:
        json.dump(dict(input=args.input, min_count=min_count, nside_cmp=args.nside_cmp,
                       results=results, per_h=per_h), fh, indent=1)
    make_figures(results, per_h, outdir, tag, args.nside_cmp)


def scheme_color(label):
    uni = [k for k in SCHEMES if not k.startswith('var')]
    var = [k for k in SCHEMES if k.startswith('var')]
    return BLUES[uni.index(label)] if label in uni else ORANGES[var.index(label)]


def make_figures(results, per_h, outdir, tag, nside_cmp):
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    labels = [r['width'] for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.0), layout='constrained',
                             gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1]))

    # (a) where observed stars end up, stacked by effective nside
    ax = axes[0]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for ns in (*NSIDE_LEVELS, 0):
        v = np.array([100 * r['frac'][ns] for r in results])
        lab = f'nside {ns} ({np.degrees(hp.nside2resol(ns)):.1f}°)' if ns else 'uncovered'
        ax.bar(x, v, bottom=bottom, color=NSIDE_COLORS[ns], width=0.72,
               edgecolor='white', linewidth=1.5, label=lab)
        bottom += v
    ax.set_xticks(x, labels, rotation=30)
    ax.set_xlabel('H bin scheme [mag]  (var: one bin at H<7)')
    ax.set_ylabel('observed stars [%]')
    ax.set_title("(a) Sky resolution of each star's SF cell", loc='left', color=INK)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc='outside lower left', ncol=5)

    # (b) star-weighted pixel size vs H
    ax = axes[1]
    for lab in labels:
        ax.stairs(per_h[lab]['mean_pix'], np.array(per_h[lab]['edges']), baseline=None,
                  color=scheme_color(lab), linewidth=2, label=lab,
                  linestyle='--' if lab.startswith('var') else '-')
    ax.set_xlabel('H [mag]')
    ax.set_ylabel('star-weighted mean pixel size [deg]')
    ax.set_title('(b) Pixel size vs H', loc='left', color=INK)
    ax.set_xlim(6.5, 11)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # (c) the trade-off: gain vs H-smoothing cost
    ax = axes[2]
    for r in results:
        lab, c = r['width'], scheme_color(r['width'])
        p, q = r['mean_pix_deg'], 100 * r['cost_rel_debiased']
        qh = 100 * r['cost_rel_hybrid']
        ax.plot([p, p], [q, qh], color=c, linewidth=1, zorder=1)
        ax.scatter(p, q, s=70, color=c, edgecolor='white', linewidth=2, zorder=2,
                   marker='s' if lab.startswith('var') else 'o')
        ax.scatter(p, qh, s=50, facecolor='white', edgecolor=c, linewidth=1.5, zorder=2,
                   marker='s' if lab.startswith('var') else 'o')
        ax.annotate(lab, (p, q), textcoords='offset points', xytext=(6, 3),
                    fontsize=8, color=INK2)
    ax.set_xlabel('star-weighted mean pixel size [deg]  (← finer)')
    ax.set_ylabel('H-smoothing error, rms ΔS / ⟨S⟩ [%]')
    ax.set_title(f'(c) Trade-off (error at nside {nside_cmp})', loc='left', color=INK)
    ax.text(0.98, 0.97, 'filled: wide-bin SF\nopen: + fine H-shape correction',
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK2)

    # (d) where in H the smoothing error comes from
    ax = axes[3]
    e = np.array(per_h[labels[0]]['edges'])
    for lab in labels[1:]:
        ax.stairs(100 * np.array(per_h[lab]['cost_rel']), e, baseline=None,
                  color=scheme_color(lab), linewidth=2, label=lab,
                  linestyle='--' if lab.startswith('var') else '-')
    ax.set_xlabel('H [mag] (native 0.5-mag bin)')
    ax.set_ylabel('rms ΔS / ⟨S⟩ [%]')
    ax.set_title('(d) H-smoothing error vs H', loc='left', color=INK)
    ax.set_xlim(6.5, 11)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    path = outdir / f'hbin_width_{tag}.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'), dpi=130)
    print(f"\nSaved {path} (+ .png)")


if __name__ == '__main__':
    main()
