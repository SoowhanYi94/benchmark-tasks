"""CPU check of the scripted demo: plays both ArmPlans through the arena IK at 20 Hz and reports the worst IK error per
waypoint and the closest approach between the two arms' collision hulls (sampled every 0.5 s). No simulator.

    python dev/plan_check.py [variant]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import task as T  # noqa: E402
from demo_plan import plan  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS, ARM_JOINTS  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.analysis import fk_link_poses_world, world_hulls, hull_set_distance  # noqa: E402

import math as _m
import gsparts as _G
import demo_plan as _D


def _nominal(b, l, mouth):
    """Nominal socket tuples (c, R, jaw, loop, dW, Yw) for a strut from rod-end point b to body-end point l."""
    X = (l - b) / np.linalg.norm(l - b)
    Y = np.asarray(mouth, float); Y = Y - X * (Y @ X); Y /= np.linalg.norm(Y)
    Z = np.cross(X, Y); R = np.column_stack([X, Y, Z]); sj = 1 if Z[2] >= 0 else -1
    d = R @ np.array([0.0, -_m.sin(_G.JAW_BETA), sj * _m.cos(_G.JAW_BETA)])
    out = {}
    for end, c, e in (("body", b, -1.0), ("lid", l, 1.0)):
        loop = c + R @ np.array([e * _G.TAB_PAD, _G.JAW_Y + 0.0009, sj * (_G.JAW_R + 0.002)])
        out[end] = (c, R, 0.0, loop, d, Y)
    return out


def LID_AT(t):
    """Nominal lid angle at time t: shut until the lift starts, then 45 deg until the close is let go."""
    return 45.0 if T_LIFT[0] <= t < T_LIFT[1] else 0.0


T_LIFT = [9e9, 9e9]
VARIANT = sys.argv[1] if len(sys.argv) > 1 else ""
p = plan(SimpleNamespace(**vars(T)), VARIANT)
_tt = 0.0
for _w in p["right"]:
    _tt += _w.duration
    if "lift the lid 49" in _w.label and T_LIFT[0] > 1e9:
        T_LIFT[0] = _tt
    if "let go: it falls shut" in _w.label:
        T_LIFT[1] = _tt
for m in _D.MEASURES:
    if m["kind"] != "loop":
        continue
    side = m["side"]
    b, l = _G.stud_points(side)
    b, l = _G.stud_points(side, 0.0)          # removal happens with the lid shut
    if m["end"] == "body":
        socks = _nominal(b, l, (0, -side, 0))
    else:   # after the lower end was laid on the table outboard, near (0.30, 0.25, +0.0065)
        b2 = np.array([0.29, side * 0.245, _G.TOP + 0.0065])
        socks = _nominal(b2, l, (0, -side, 0))
    _D.apply_loop(m, {m["strut"]: socks})
arms = {"left": ArmPlan(p["left"], LEFT_BASE_POS), "right": ArmPlan(p["right"], RIGHT_BASE_POS)}
dur = max(a.duration for a in arms.values())
print(f"duration {dur:.1f} s = {int(dur * 20)} steps")
worst = {}
closest = (9.0, None, None)
import gsparts as G
from yam_arena.analysis import fk_link_poses_world as _fk
lid_hit = {}
WHOS = {}
WHO = ""


def lid_clear(hs, deg):
    """Smallest distance (m) from any hull vertex to the lid (an oriented 200 x 407 x 25 mm slab) and the body box."""
    th = np.radians(deg)
    d_ = np.array([np.cos(th), 0, np.sin(th)]); n = np.array([-np.sin(th), 0, np.cos(th)])
    o = np.array([G.X0, 0, G.TOP + G.H])
    best = 9.0
    global WHO
    for name, h in hs:
        v = h.vertices - o
        a, y, c = v @ d_, v[:, 1], v @ n
        da = np.maximum(np.maximum(0 - a, a - 0.20), 0); dy = np.maximum(np.abs(y) - 0.2035, 0)
        dc = np.maximum(np.maximum(-0.018 - c, c - 0.002), 0)
        dl = float(np.sqrt(da**2 + dy**2 + dc**2).min())
        if dl < best:
            best = dl; WHO = name + "/lid"
        lo, hi = np.array([G.X0, -G.HW, G.TOP]), np.array([G.X1, G.HW, G.TOP + G.H])
        dd = np.maximum(np.maximum(lo - h.vertices, h.vertices - hi), 0)
        db = float(np.linalg.norm(dd, axis=1).min())
        if db < best:
            best = db; WHO = name + "/body"
    return best
for k in range(int(dur * 20)):
    t = k / 20
    qs = {}
    for name, a in arms.items():
        before = (a.worst_ik_mm, a.worst_ik_deg)
        a.worst_ik_mm = a.worst_ik_deg = 0.0
        act, label = a.action7(t)
        key = (name, label)
        worst[key] = max(worst.get(key, (0, 0)), (a.worst_ik_mm, a.worst_ik_deg))
        a.worst_ik_mm, a.worst_ik_deg = max(before[0], a.worst_ik_mm), max(before[1], a.worst_ik_deg)
        qs[name] = act[:6]
    if k % 10 == 0:
        hs = {}
        for name, base in (("left", LEFT_BASE_POS), ("right", RIGHT_BASE_POS)):
            d = dict(zip(ARM_JOINTS, qs[name])); d["joint7"] = d["joint8"] = -0.01
            hs[name] = world_hulls(fk_link_poses_world(d, np.array(base)))
        labels = {n: a.target(t)[3] for n, a in arms.items()}
        holding = {n: ("lid" in labels[n] or "handle" in labels[n] or "bar" in labels[n]) for n in arms}
        for n in arms:
            if not holding[n]:
                c = lid_clear([(nm, h) for nm, h in hs[n] if not nm.startswith("tip")], LID_AT(t))
                key = (n, labels[n])
                if c < lid_hit.get(key, 9.0):
                    lid_hit[key] = c; WHOS[key] = WHO
        dist, links = hull_set_distance(hs["left"], hs["right"])
        if dist < closest[0]:
            closest = (dist, t, links)
for (name, label), (mm, deg) in worst.items():
    flag = "  <-- IK" if mm > 2 or deg > 2 else ""
    print(f"{name:5s} {label:60s} {mm:6.2f} mm {deg:5.2f} deg{flag}")
for (n, lab), c in lid_hit.items():
    if c < 0.02:
        print(f"LID/BODY {n:5s} {lab:55s} {c*1000:6.1f} mm {WHOS.get((n, lab))}")
print(f"closest arms {closest[0]*1000:.1f} mm at t={closest[1]} s {closest[2]}")
