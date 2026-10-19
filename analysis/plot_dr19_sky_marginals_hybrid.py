#!/usr/bin/env python3
"""
Hybrid-SF versions of figures/dr19_sky_marginals.pdf (plot_dr19_sky_grid.py).

Figure 1 (--output): identical layout to dr19_sky_marginals.pdf, with every
  completeness panel from the hybrid selection function:
    rows 1-2 (G-H marginalised, H=[7,7.5) and [10.5,11)): hybrid rebuilt from the
             G-H-summed counts (same min_count / nside_min as the files)
    rows 3-4 (fixed G-H): read from the hybrid npz files
Figure 2 (--output-compare): the same four rows, original vs hybrid side by
  side for one estimator (--compare-estimator, default 'map').

dr19_sky_allsky and dr19_cmd_grid are not remade: both sum the counts over all
H (and G-H, or the whole sky) before estimating S, so the hybrid's merged-H sky
level adds nothing and they would be identical.

Usage:
  python analysis/plot_dr19_sky_marginals_hybrid.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = Path('/Users/maddielucey/gg_selec_func')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJ))
import plot_dr19_sky_grid as G  # noqa: E402
from build_hybrid_selfunc import LEVEL_H_EDGES, hybrid_ring  # noqa: E402
from plot_dr19_sky_grid import _build_adaptive as orig_build_adaptive  # noqa: E402

SF_DIR = PROJ / 'sdssv_apogee_sf'


def rows_for(sf_2d, sf_3d, H_BINS, GH_BINS):
    """The four (title, map) rows used by dr19_sky_marginals."""
    h_idx = [G.bin_index(H_BINS, h) for h in (7.0, 10.999)]
    gh_idx = [G.bin_index(GH_BINS, gh) for gh in (2.5, 3.5)]
    hf = h_idx[1]
    rows = [(f"H = [{H_BINS[i]:.1f}, {H_BINS[i+1]:.1f}), G-H marginalized", sf_2d[i])
            for i in h_idx]
    rows += [(f"H = [{H_BINS[hf]:.1f}, {H_BINS[hf+1]:.1f}), "
              f"G-H = [{GH_BINS[g]:.1f}, {GH_BINS[g+1]:.1f})", sf_3d[hf, :, g])
             for g in gh_idx]
    return rows


def two_column_figure(rows_left, rows_right, headers, gamma, cmap, output):
    """Layout copied from G.build_figure_marginals, with arbitrary columns."""
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.02,
                          left=0.03, right=0.97, top=0.93, bottom=0.05)
    axes = []
    for r, ((title, left), (_, right)) in enumerate(zip(rows_left, rows_right)):
        axes.append(G.draw_panel(fig, gs[r, 0], G.make_map(left, gamma), title, cmap))
        axes.append(G.draw_panel(fig, gs[r, 1], G.make_map(right, gamma), title, cmap,
                                 show_b=False))
    left_axes, right_axes = axes[0::2], axes[1::2]
    width = left_axes[0].get_position().width
    pair_cx = (left_axes[0].get_position().x0 + right_axes[0].get_position().x1) / 2
    gap = 0.055
    for ax in left_axes:
        pos = ax.get_position()
        ax.set_position([pair_cx - gap / 2 - width, pos.y0, width, pos.height])
    for ax in right_axes:
        pos = ax.get_position()
        ax.set_position([pair_cx + gap / 2, pos.y0, width, pos.height])
    for ax, label in zip((axes[0], axes[1]), headers):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.965, label, ha='center', va='bottom',
                 fontsize=G.TITLE_FS + 1, fontweight='bold')
    G.shared_completeness_colorbar(fig, axes, cmap, gamma, pad=0.07)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=str(SF_DIR / 'gg_sf_color_dr19_mincount20.npz'))
    ap.add_argument('--input-map', default=str(SF_DIR / 'gg_sf_color_dr19_mincount20_map.npz'))
    ap.add_argument('--input-hybrid',
                    default=str(SF_DIR / 'gg_sf_color_dr19_mincount20_hybrid.npz'))
    ap.add_argument('--input-hybrid-map',
                    default=str(SF_DIR / 'gg_sf_color_dr19_mincount20_hybrid_map.npz'))
    ap.add_argument('--output', default=str(PROJ / 'figures' / 'dr19_sky_marginals_hybrid.pdf'))
    ap.add_argument('--output-compare',
                    default=str(PROJ / 'figures' / 'dr19_sky_marginals_hybrid_vs_orig.pdf'))
    ap.add_argument('--compare-estimator', choices=('expectation', 'map'), default='map')
    ap.add_argument('--cmap', default='viridis')
    ap.add_argument('--gamma', type=float, default=0.4)
    args = ap.parse_args()

    d, dm = np.load(args.input), np.load(args.input_map)
    dh, dhm = np.load(args.input_hybrid), np.load(args.input_hybrid_map)
    for x in (dh, dhm):
        assert str(x['method']) == 'hybrid'
        assert np.allclose(x['level_h_edges'], LEVEL_H_EDGES)
    H_BINS, GH_BINS = d['H_BINS'], d['GH_BINS']
    nside, min_count, nside_min = int(d['nside']), int(d['min_count']), int(d['nside_min'])
    ha_2d = d['hist_all'].sum(axis=2)
    ho_2d = d['hist_obs'].sum(axis=2)
    print(f"nside={nside} min_count={min_count} nside_min={nside_min}")

    sf2d = {}
    for est in ('expectation', 'map'):
        sf2d[('orig', est)], _ = orig_build_adaptive(ha_2d, ho_2d, nside, min_count=min_count,
                                                     nside_min=nside_min, estimator=est)
        sf2d[('hyb', est)] = hybrid_ring(ha_2d, ho_2d, H_BINS, nside, min_count,
                                         nside_min, est)
        n_o = np.isfinite(sf2d[('orig', est)]).sum()
        n_h = np.isfinite(sf2d[('hyb', est)]).sum()
        assert n_o == n_h, (n_o, n_h)
    sf3d = {('orig', 'expectation'): d['selfunc'], ('orig', 'map'): dm['selfunc'],
            ('hyb', 'expectation'): dh['selfunc'], ('hyb', 'map'): dhm['selfunc']}

    # Figure 1: same as dr19_sky_marginals, hybrid throughout
    two_column_figure(
        rows_for(sf2d[('hyb', 'expectation')], sf3d[('hyb', 'expectation')], H_BINS, GH_BINS),
        rows_for(sf2d[('hyb', 'map')], sf3d[('hyb', 'map')], H_BINS, GH_BINS),
        ('Mean (hybrid)', 'MAP (hybrid)'), args.gamma, args.cmap, args.output)

    # Figure 2: original vs hybrid for one estimator
    est = args.compare_estimator
    lab = 'Mean' if est == 'expectation' else 'MAP'
    two_column_figure(
        rows_for(sf2d[('orig', est)], sf3d[('orig', est)], H_BINS, GH_BINS),
        rows_for(sf2d[('hyb', est)], sf3d[('hyb', est)], H_BINS, GH_BINS),
        (f'{lab} (original)', f'{lab} (hybrid)'), args.gamma, args.cmap, args.output_compare)


if __name__ == '__main__':
    main()
