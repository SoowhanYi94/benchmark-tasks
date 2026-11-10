"""Dev only: read a TIG_DEBUG log and report, along the seam, the tungsten's height and the rod tip's position at each dab.
python dev/droop.py <log> [lift_mm]"""
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/dev/", 1)[0])
from tig_seam import mech as M  # noqa: E402

PAT = re.compile(r"TIG t=\s*([\d.]+) h\s+(-?[\d.]+) lat\s+([\d.]+) arc (\d) tip \[([-\d., ]+)\] tilt\s+([\d.]+) rtip \[[-\d., ]+\] rod ahead\s+(-?[\d.]+) rh\s+(-?[\d.]+) rlat\s+([\d.]+) gap\s+([\d.]+)")


def s_of(p):
    """Arc length along the seam of the nearest seam point."""
    (x0, y0), (xc, yc), (x1, y1) = M.SEAM
    la = x0 - xc
    sa = min(max(x0 - p[0], 0.0), la)
    da = np.hypot(p[0] - (x0 - sa), p[1] - y0)
    sb = min(max(yc - p[1], 0.0), yc - y1)
    db = np.hypot(p[0] - xc, p[1] - (yc - sb))
    return (sa, da) if da <= db else (la + sb, db)


def main():
    lift = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    rows = []
    for line in open(sys.argv[1], errors="replace"):
        m = PAT.search(line)
        if m:
            t, h, lat, arc = float(m[1]), float(m[2]), float(m[3]), int(m[4])
            tip = np.array([float(v) for v in m[5].split(",")]) / 1000
            rows.append((t, h, lat, arc, tip, float(m[6]), float(m[7]), float(m[8]), float(m[9]), float(m[10])))
    weld = [r for r in rows if r[2] < 6 and abs(r[1] - (3.5 + lift)) < 12 and M.SLAB["x"][0] < r[4][0] < M.SLAB["x"][1]]
    hs = np.array([r[1] for r in weld])
    print(f"{len(weld)} steps over the seam; tip height {hs.min():.2f} to {hs.max():.2f} mm (aim {3.5 + lift:.1f}), mean {hs.mean():.2f}, sd {hs.std():.2f}")
    bins = {}
    for r in weld:
        s, _ = s_of(r[4])
        n = r[4][1] - M.SEAM[0][1] if s < 0.12 else r[4][0] - M.SEAM[1][0]  # signed: +y on leg 1, +x on leg 2
        bins.setdefault(int(s * 1000 // 10), []).append((r[1] - lift, 1000 * n))
    print("s(mm)  height below aim? (mean h-lift, mm)  lateral max")
    for k in sorted(bins):
        v = np.array(bins[k])
        print(f"{10 * k:4d}-{10 * k + 10:<4d} h {v[:, 0].mean():5.2f} [{v[:, 0].min():5.2f},{v[:, 0].max():5.2f}] side {v[:, 1].mean():5.2f} [{v[:, 1].min():5.2f},{v[:, 1].max():5.2f}]")
    downs = [r for r in rows if r[7] <= 3.5 and r[2] < 6]
    if downs:
        a = np.array([r[6] for r in downs])
        rl = np.array([r[8] for r in downs])
        g = np.array([r[9] for r in downs])
        print(f"rod down {len(downs)} steps: ahead {a.min():.1f}..{a.max():.1f} mean {a.mean():.1f}; rlat max {rl.max():.1f} mean {rl.mean():.1f}; gap min {g.min():.1f}")
        for leg in (0, 1):
            sel = [r for r in downs if (s_of(r[4])[0] < 0.12) == (leg == 0)]
            if sel:
                print(f"  leg {leg + 1}: ahead mean {np.mean([r[6] for r in sel]):.1f} rlat mean {np.mean([r[8] for r in sel]):.1f} n {len(sel)}")


main()
