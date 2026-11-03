"""On the station table: a granite surface plate in the middle of the bench, with a 30 cm precision aluminium
straightedge lying on a cloth on the bench in front of it, along y, and a support block standing on each side. A
precision straightedge is stored and used resting on two supports at its Bessel points, 22 % of its length in from
each end, where it sags least. The robot must stand the two supports on the plate at the right spacing and lay the
straightedge across them, level, with each support under one of its Bessel points.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/bessel-supports/bessel_supports.py \\
           --instruction "support the 30 cm straightedge at its Bessel points: stand the two support blocks on the surface plate and lay the straightedge across them, level, with each support 22 percent of its length in from its end" --scorer task_score ...
"""

import math

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

PLATE = dict(center=(0.29, 0.0), size=(0.16, 0.40, 0.02))
PLATE_Z = TOP + PLATE["size"][2]
BAR = (0.025, 0.30, 0.025)  # x, y, z: an aluminium straightedge, 0.5 kg
BAR_L = BAR[1]
BESSEL = 0.2203  # Bessel points: this fraction of the length in from each end
SUPPORT = (0.03, 0.03, 0.04)  # x, y, z: a support block
SUPPORT_START = {"support_a": (0.18, 0.24), "support_b": (0.18, -0.24)}  # one on each side
CLOTH = dict(center=(0.15, 0.0), size=(0.06, 0.34, 0.003))
BAR_START = (0.15, 0.0)
BESSEL_TOL = 0.01  # each support's centre within 1 cm of a Bessel point (6.6 cm in from an end)


OBJECTS = [
    Box("surface_plate", pos=(*PLATE["center"], TOP + PLATE["size"][2] / 2), size=PLATE["size"], color=(0.20, 0.20, 0.22)),
    Box("bar_cloth", pos=(*CLOTH["center"], TOP + CLOTH["size"][2] / 2), size=CLOTH["size"], color=(0.55, 0.20, 0.20)),
    Box("straightedge", pos=(*BAR_START, TOP + CLOTH["size"][2] + BAR[2] / 2), size=BAR, mass=0.5, color=(0.80, 0.82, 0.85)),
    *[Box(n, pos=(*xy, TOP + SUPPORT[2] / 2), size=SUPPORT, mass=0.25, color=(0.25, 0.40, 0.70)) for n, xy in SUPPORT_START.items()],
]


# -- scoring --------------------------------------------------------------------------------------------------------
def _axis_tilt(quat_xyzw):
    """Angle of the bar's long axis (its local y) from the horizontal (rad)."""
    x, y, z, w = quat_xyzw
    return math.asin(min(1.0, abs(2 * (y * z + w * x))))


def _tilt(quat_xyzw):
    x, y, z, w = quat_xyzw
    return math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))


def _support_up(name, objects):
    """Standing upright on the plate, still."""
    o = objects[name]
    (cx, cy), (sx, sy, _) = PLATE["center"], PLATE["size"]
    return bool(abs(o.pos[0] - cx) <= sx / 2 and abs(o.pos[1] - cy) <= sy / 2 and o.low[2] <= PLATE_Z + 0.004
                and _tilt(o.quat_xyzw) <= math.radians(5) and o.resting())


def bar_on_supports(objects):
    """The bar resting level (its length within 2 degrees of horizontal) across both supports, standing on the
    plate, and touching nothing else: its underside at the supports' height and over both of them."""
    b = objects["straightedge"]
    if not all(_support_up(n, objects) for n in SUPPORT_START) or not b.resting():
        return False
    under = all(abs(objects[n].pos[0] - b.pos[0]) <= (SUPPORT[0] + BAR[0]) / 2 and b.low[1] < objects[n].pos[1] < b.high[1]
                for n in SUPPORT_START)
    return bool(under and abs(b.low[2] - (PLATE_Z + SUPPORT[2])) <= 0.004 and _axis_tilt(b.quat_xyzw) <= math.radians(2))


def supports_at_bessel(objects):
    """How many supports stand within BESSEL_TOL of a Bessel point of the bar (one support per point)."""
    b = objects["straightedge"]
    ends = (b.pos[1] - BAR_L / 2, b.pos[1] + BAR_L / 2)
    points = (ends[0] + BESSEL * BAR_L, ends[1] - BESSEL * BAR_L)
    ys = sorted(objects[n].pos[1] for n in SUPPORT_START)
    return sum(abs(y - p) <= BESSEL_TOL for y, p in zip(ys, points))


def success(objects):
    return bar_on_supports(objects) and supports_at_bessel(objects) == 2


def score(objects):
    """0.5 for the bar resting level across both supports + 0.25 for each support within 1 cm of a Bessel point (6.6 cm
    in from an end). Supports and bar anywhere else score 0."""
    if not bar_on_supports(objects):
        return 0.0
    return float(0.5 + 0.25 * supports_at_bessel(objects))


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
PX = PLATE["center"][0]
SUPPORT_Y = BAR_L / 2 - BESSEL * BAR_L  # each support this far from the bar's middle (8.4 cm)
GRIP_Y = 0.12  # each arm holds the bar this far from its middle: outside the supports (2 cm clear), the two hands well
# apart (8 cm apart, they collide)
TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
BAR_TRIM_X = 0.0  # carried by both arms (held 12 cm either side of its middle), the straightedge comes down on target


class _Plan:
    """Waypoints for both arms in phases: within a phase each arm moves independently, then both wait for the slower."""

    def __init__(self):
        self.wps = {a: [] for a in SIDE}
        self.t = {a: 0.0 for a in SIDE}
        self.last = {}
        for a in SIDE:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None):
        p, f, y = self.last[arm]
        p = p if pos is None else pos
        f = f if finger is None else finger
        y = y if yaw is None else yaw
        self.wps[arm].append(Waypoint(dur, p, f, label, y))
        self.last[arm] = (p, f, y)
        self.t[arm] += dur

    def sync(self):
        end = max(self.t.values())
        for a in SIDE:
            if self.t[a] < end - 1e-9:
                self.go(a, end - self.t[a], label="wait")


def _yaw(arm, base):
    return min((base + k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def demo():
    plan = _Plan()
    # 1. the supports, one with each arm: the left one to +8.4 cm, the right one to -8.4 cm
    g = SUPPORT[2] - 0.01
    grip, ready = -FINGER_PER_M * (SUPPORT[1] - 0.004), -FINGER_PER_M * (SUPPORT[1] + 0.02)
    for arm, name, y in (("left", "support_a", SUPPORT_Y), ("right", "support_b", -SUPPORT_Y)):
        (sx, sy) = SUPPORT_START[name]
        qx, qy = PX + TRIM[arm][0], y + TRIM[arm][1]
        yaw = _yaw(arm, 0.0)
        z_up = PLATE_Z + SUPPORT[2] + 0.04
        plan.go(arm, 1.3, (sx, sy, z_up), ready, f"{name}: above it", yaw)
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 1.0, (sx, sy, TOP + g), label="descend")
        plan.go(arm, 0.6, finger=grip, label="grip")
        plan.go(arm, 0.9, (sx, sy, z_up), label="lift")
        plan.go(arm, 1.5, (qx, qy, z_up), label="over its place")
        plan.go(arm, 1.0, (qx, qy, PLATE_Z + g + 0.002), label="stand it on the plate")
        plan.go(arm, 0.6, finger=ready, label="release")
        plan.go(arm, 0.8, (qx, qy, z_up), label="up")
        plan.go(arm, 1.0, (0.18, SIDE[arm] * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
        plan.sync()
    # 2. both arms lay the bar across them
    bx, by = BAR_START
    gb = 0.015
    bgrip, bready = -FINGER_PER_M * (BAR[0] - 0.004), -FINGER_PER_M * (BAR[0] + 0.02)
    z_up = PLATE_Z + SUPPORT[2] + 0.03 + gb
    for a, s in SIDE.items():
        plan.go(a, 1.5, (bx, by + s * GRIP_Y, z_up), bready, "straightedge: above it", _yaw(a, math.pi / 2))
    plan.sync()
    for a, s in SIDE.items():
        plan.go(a, 1.0, (bx, by + s * GRIP_Y, TOP + CLOTH["size"][2] + gb), label="descend")
    plan.sync()
    for a in SIDE:
        plan.go(a, 0.6, finger=bgrip, label="grip")
    plan.sync()
    for a, s in SIDE.items():
        plan.go(a, 1.2, (bx, by + s * GRIP_Y, z_up), label="lift together")
        plan.go(a, 2.0, (PX + BAR_TRIM_X, s * GRIP_Y, z_up), label="carry over the supports")
        plan.go(a, 1.2, (PX + BAR_TRIM_X, s * GRIP_Y, PLATE_Z + SUPPORT[2] + gb + 0.002), label="lower onto them")
    plan.sync()
    for a, s in SIDE.items():
        plan.go(a, 0.8, finger=bready, label="release")
        plan.go(a, 0.8, (PX, s * GRIP_Y, z_up), label="up")
        plan.go(a, 1.0, (0.18, s * 0.32, TOP + 0.18), FINGER_OPEN, "back")
    plan.sync()
    return plan.wps


# -- self-check (pure Python): the plan fits the run limit and the scorer covers the obvious outcomes ----------------
if __name__ == "__main__":
    class _O:
        def __init__(self, x, y, z, sx, sy, sz):
            self.pos = np.array((x, y, z + sz / 2))
            self.low, self.high = np.array((x - sx / 2, y - sy / 2, z)), np.array((x + sx / 2, y + sy / 2, z + sz))
            self.quat_xyzw = np.array((0.0, 0.0, 0.0, 1.0))

        def resting(self):
            return True

    sup = lambda y: _O(PX, y, PLATE_Z, *SUPPORT)
    bar = lambda y, z=PLATE_Z + SUPPORT[2]: _O(PX, y, z, *BAR)
    start = {"straightedge": _O(*BAR_START, TOP + 0.003, *BAR), **{n: _O(*xy, TOP, *SUPPORT) for n, xy in SUPPORT_START.items()}}
    assert score(start) == 0.0
    done = {"straightedge": bar(0.0), "support_a": sup(SUPPORT_Y), "support_b": sup(-SUPPORT_Y)}
    assert score(done) == 1.0 and success(done)
    narrow = dict(done, support_a=sup(0.04), support_b=sup(-0.04))  # at a quarter of the length: 4.4 cm off
    assert score(narrow) == 0.5
    shifted = dict(done, straightedge=bar(0.03))  # bar laid 3 cm off-centre over correctly spaced supports
    assert score(shifted) == 0.5
    one_off = dict(done, support_b=sup(-SUPPORT_Y - 0.02))
    assert score(one_off) == 0.75
    on_plate = dict(done, straightedge=bar(0.0, PLATE_Z))
    assert score(on_plate) == 0.0
    t = max(sum(w.duration for w in wps) for wps in demo().values())
    assert t * 20 < 2400, t
    print(f"ok: demo {t:.1f} s ({t * 20:.0f} steps)")
