#!/usr/bin/env python3
"""Closure + API checks for hybrid SF files vs their originals.

Closure: per H bin, predicted observed counts sum(S * N_2MASS) over covered
cells should match the actual observed counts in those cells.
API: files load with APOGEESelectionFunction.read and query() runs.

Usage: python analysis/check_hybrid_selfunc.py ORIG.npz HYBRID.npz
"""

import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sdssv_apogee_sf import APOGEESelectionFunction  # noqa: E402


def main(orig_path, hyb_path):
    o, h = np.load(orig_path), np.load(hyb_path)
    ha, hobs, H = o['hist_all'], o['hist_obs'].astype(float), o['H_BINS']
    print(f"{Path(hyb_path).name}: method={h['method']} estimator={h['estimator']}")
    print(f"{'H bin':>11s} {'N_obs':>8s} {'pred orig':>10s} {'pred hyb':>10s}")
    for i in range(len(H) - 1):
        if H[i] < 7:
            continue
        m = np.isfinite(o['selfunc'][i])
        n = hobs[i][m].sum()
        po = (o['selfunc'][i][m] * ha[i][m]).sum()
        ph = (h['selfunc'][i][m] * ha[i][m]).sum()
        print(f"{H[i]:5.1f}-{H[i + 1]:4.1f} {n:8.0f} {po:10.0f} {ph:10.0f}")

    sf_o = APOGEESelectionFunction.read(orig_path)
    sf_h = APOGEESelectionFunction.read(hyb_path)
    rng = np.random.default_rng(1)
    n = 100000
    c = SkyCoord(ra=rng.uniform(0, 360, n), dec=np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
                 unit='deg')
    Hq, GHq = rng.uniform(7, 11, n), rng.uniform(2, 6, n)
    qo, qh = sf_o.query(c, H=Hq, GH=GHq), sf_h.query(c, H=Hq, GH=GHq)
    print(f"query OK: NaN pattern identical={np.array_equal(np.isnan(qo), np.isnan(qh))}, "
          f"range=[{np.nanmin(qh):.3f}, {np.nanmax(qh):.3f}], "
          f"median |hyb-orig|={np.nanmedian(np.abs(qh - qo)):.4f}")


if __name__ == '__main__':
    main(*sys.argv[1:3])
