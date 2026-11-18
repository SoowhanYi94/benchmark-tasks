"""The drawn (visual) shape of the screwdriver: a Wera 367 TORX Stubby TX 30 x 25 mm (article 05008861001).

Our own mesh, built from the maker's published sizes (blade 25 mm, blade 5.6 mm across, handle 54 mm long; overall 81 mm and handle
34 mm across at its widest from the maker's data as resellers list it) and the tool's look in product photos: a two-component Kraftform
stubby handle, bulged, with three softly flattened sides, a hexagonal anti-roll section toward the end, a domed end cap, green hard zones
and black soft zones, no lettering; a round blade with a TX 30 six-lobe tip. No third-party file is used.

Frame: the bit's tip at the origin, the axis along +z (the handle up). Units: metres. Pure NumPy.
`parts()` gives (name, points, triangles, colour) for each coloured piece; `write_obj(path)` writes them as one OBJ file.
Collision is NOT taken from here: the simulation collides with a 29 mm cylinder over the handle and a 5.6 mm blade (mech.build_driver).
"""

import math

import numpy as np

BLADE, HANDLE, BLADE_R = 0.025, 0.054, 0.0028
GREEN, BLACK, STEEL = (0.20, 0.55, 0.22), (0.05, 0.05, 0.05), (0.72, 0.74, 0.77)
N = 48  # points round each section

# handle radius (m) at heights over the handle's bottom (m): neck collar, bulge (34 mm at its widest), anti-roll hex, domed cap
PROFILE = [(0.000, 0.0120), (0.002, 0.0135), (0.006, 0.0143), (0.012, 0.0150), (0.020, 0.0160), (0.028, 0.0168), (0.034, 0.0170),
           (0.040, 0.0166), (0.044, 0.0158), (0.050, 0.0150), (0.051, 0.0135), (0.0525, 0.0105), (0.0535, 0.0065), (0.054, 0.0)]
ZONES = [("handle_collar", 0.000, 0.006, GREEN), ("handle_grip", 0.006, 0.044, BLACK), ("handle_end", 0.044, 0.054, GREEN)]


def _radius(h, theta):
    r = float(np.interp(h, *zip(*PROFILE)))
    if 0.044 <= h <= 0.050:  # the hexagonal anti-roll section: a regular hexagon of this circumradius
        k = math.pi / 6
        a = (theta + k) % (2 * k) - k
        return r * math.cos(k) / math.cos(a)
    return r * (1.0 - 0.05 * (1.0 + math.cos(3 * theta)) / 2)  # three softly flattened sides


def _loft(hs, radius, z0):
    """Rings at heights hs (over z0), joined by triangles; closed at the bottom and top by fans."""
    pts, tris = [], []
    for h in hs:
        for k in range(N):
            t = 2 * math.pi * k / N
            r = radius(h, t)
            pts.append((r * math.cos(t), r * math.sin(t), z0 + h))
    for i in range(len(hs) - 1):
        for k in range(N):
            a, b = i * N + k, i * N + (k + 1) % N
            c, d = a + N, b + N
            tris += [(a, b, d), (a, d, c)]
    for ring_i, z, flip in ((0, hs[0], True), (len(hs) - 1, hs[-1], False)):
        ci = len(pts)
        pts.append((0.0, 0.0, z0 + z))
        for k in range(N):
            a, b = ring_i * N + k, ring_i * N + (k + 1) % N
            tris.append((ci, b, a) if flip else (ci, a, b))
    return np.array(pts), np.array(tris)


def parts():
    out = []
    for name, h0, h1, col in ZONES:
        hs = [h0] + [h for h, _ in PROFILE if h0 < h < h1] + [h1]
        hs = sorted(set(hs + list(np.linspace(h0, h1, max(2, int((h1 - h0) / 0.002) + 1)))))
        p, t = _loft(hs, _radius, BLADE)
        out.append((name, p, t, col))
    p, t = _loft([0.003, BLADE], lambda h, th: BLADE_R, 0.0)  # round blade
    out.append(("blade", p, t, STEEL))
    pts, tris = [], []
    for z in (0.0, 0.003):  # TX 30 tip, 3 mm long
        for k in range(24):
            th = 2 * math.pi * k / 24
            r = BLADE_R if (k // 2) % 2 == 0 else 0.0021
            pts.append((r * math.cos(th), r * math.sin(th), z))
    for k in range(24):
        a, b = k, (k + 1) % 24
        tris += [(a, b, b + 24), (a, b + 24, a + 24)]
    pts.append((0.0, 0.0, 0.0))
    for k in range(24):
        tris.append((48, (k + 1) % 24, k))
    out.append(("tip_tx30", np.array(pts), np.array(tris), STEEL))
    return out


def write_obj(path):
    lines, base = ["# Wera 367 TORX Stubby TX 30 x 25 mm (05008861001): our own mesh from the maker's sizes; metres; tip at origin, +z up"], 1
    for name, p, t, col in parts():
        lines.append(f"o {name}")
        lines.append(f"# colour {col}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in p]
        lines += [f"f {a + base} {b + base} {c + base}" for a, b, c in t]
        base += len(p)
    open(path, "w").write("\n".join(lines) + "\n")
