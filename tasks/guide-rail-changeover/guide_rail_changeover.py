"""On the station table: the end of a packaging line's conveyor runs away from the arms down the middle of the bench, a
flat bed with a white centre line and, at its far end, a green outfeed mark. Two loose aluminium guide rails lie on the
bed along it, set 14 cm apart for the last product. The next product, a carton 8 cm wide, waits on a staging table on
the left. The robot must change the conveyor over: move both guide rails in so the carton just fits between them
(a few millimetres of clearance), centred on the centre line and parallel to it, then set the carton between the
rails on the outfeed mark as a test.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/guide-rail-changeover/guide_rail_changeover.py \\
           --instruction "change the conveyor over for the new carton: move both guide rails in so the carton just fits between them, centred on the white line and parallel to it, then put the carton between the rails on the green outfeed mark" --scorer task_score ...
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
ALU = (0.78, 0.80, 0.83)

BED = dict(center=(0.30, 0.0), size=(0.26, 0.22, 0.03))
BED_Z = TOP + BED["size"][2]
RAIL = (0.24, 0.02, 0.025)  # x, y, z: an aluminium guide rail (along x), 0.32 kg
RAIL_START_Y = {"rail_left": 0.08, "rail_right": -0.08}  # centres: inner faces 14 cm apart
CARTON = (0.06, 0.08, 0.06)  # x, y, z: 8 cm wide (across the conveyor), 0.3 kg
CARTON_START = (0.17, 0.25)
STAGING = dict(center=(0.17, 0.25), size=(0.10, 0.12, 0.01))
OUTFEED = (0.38, 0.0)
GAP_RANGE = (CARTON[1] + 0.002, CARTON[1] + 0.012)  # between the rails' inner faces: 2-12 mm of clearance in all
CENTRE_TOL = 0.005


OBJECTS = [
    Box("conveyor_bed", pos=(*BED["center"], TOP + BED["size"][2] / 2), size=BED["size"], color=(0.18, 0.18, 0.20)),
    Box("centre_line", pos=(BED["center"][0], 0.0, BED_Z + 0.0003), size=(BED["size"][0], 0.004, 0.0006), color=(0.95, 0.95, 0.92)),
    Box("outfeed_mark", pos=(*OUTFEED, BED_Z + 0.0004), size=(0.04, 0.02, 0.0008), color=(0.20, 0.70, 0.30)),
    Box("staging_table", pos=(*STAGING["center"], TOP + STAGING["size"][2] / 2), size=STAGING["size"], color=(0.45, 0.46, 0.48)),
    *[Box(n, pos=(BED["center"][0], y, BED_Z + RAIL[2] / 2), size=RAIL, mass=0.32, color=ALU) for n, y in RAIL_START_Y.items()],
    Box("carton", pos=(*CARTON_START, TOP + STAGING["size"][2] + CARTON[2] / 2), size=CARTON, mass=0.3, color=(0.70, 0.55, 0.35)),
]


# -- scoring --------------------------------------------------------------------------------------------------------
def _yaw(quat_xyzw):
    x, y, z, w = quat_xyzw
    a = abs(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return min(a, math.pi - a)


def _tilt(quat_xyzw):
    x, y, z, w = quat_xyzw
    return math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))


def _sides(objects):
    """The two (identical) rails as (left, right) by where they lie, whichever one ends up on which side."""
    a, b = objects["rail_left"], objects["rail_right"]
    return (a, b) if a.pos[1] >= b.pos[1] else (b, a)


def rail_gap(objects):
    """Between the rails' inner faces (m)."""
    l, r = _sides(objects)
    return float(l.low[1] - r.high[1])


def rails_set(objects):
    """Both rails flat on the bed, parallel to the conveyor (within 2 degrees), the gap between them 8.2-9.2 cm (the
    carton plus 2-12 mm in all), and centred on the centre line within 5 mm; still."""
    l, r = _sides(objects)
    flat = all(o.low[2] <= BED_Z + 0.003 and _yaw(o.quat_xyzw) <= math.radians(2) and _tilt(o.quat_xyzw) <= math.radians(3)
               and o.resting() for o in (l, r))
    g = rail_gap(objects)
    mid = (l.low[1] + r.high[1]) / 2
    return bool(flat and GAP_RANGE[0] <= g <= GAP_RANGE[1] and abs(mid) <= CENTRE_TOL)


def carton_at_outfeed(objects):
    """Upright on the bed between the rails, its centre within 1.5 cm of the outfeed mark, still."""
    o, (l, r) = objects["carton"], _sides(objects)
    return bool(np.hypot(*(o.pos[:2] - np.array(OUTFEED))) <= 0.015 and o.low[2] <= BED_Z + 0.004
                and r.high[1] <= o.low[1] + 0.003 and o.high[1] <= l.low[1] + 0.003
                and _tilt(o.quat_xyzw) <= math.radians(5) and o.resting())


def success(objects):
    return rails_set(objects) and carton_at_outfeed(objects)


def score(objects):
    """0.6 rails set for the carton + 0.4 the carton between them on the outfeed mark (only with the rails set)."""
    s = rails_set(objects)
    return float(0.6 * s + 0.4 * (s and carton_at_outfeed(objects)))


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
RAIL_SET_Y = (CARTON[1] + 0.011) / 2 + RAIL[1] / 2  # aimed for 5.5 mm clearance a side (the rails settle about 1.7 mm
# closer each): tighter, the open-loop carton wedges between them
TRIM = {"left": -0.0057, "right": 0.0057}  # each arm set its rail down 5.7 mm short of its aim, outward (measured)


class _Plan:
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


def _yaw_for(arm, base):
    return min((base + k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def demo():
    plan = _Plan()
    x = BED["center"][0]
    g = 0.017  # rails gripped across their 2 cm width
    grip, ready = -FINGER_PER_M * (RAIL[1] - 0.004), -FINGER_PER_M * (RAIL[1] + 0.03)
    z_up = BED_Z + RAIL[2] + 0.03 + g
    # 1. each arm moves its rail in, one at a time (lifted and set down, not dragged)
    for arm, name in (("left", "rail_left"), ("right", "rail_right")):
        s = SIDE[arm]
        y0, y1 = RAIL_START_Y[name], s * RAIL_SET_Y + TRIM[arm]
        yaw = _yaw_for(arm, 0.0)  # fingers along y: across the rail
        plan.go(arm, 1.3, (x, y0, z_up), ready, f"{name}: above it", yaw)
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 1.0, (x, y0, BED_Z + g), label="descend")
        plan.go(arm, 0.6, finger=grip, label="grip")
        plan.go(arm, 0.8, (x, y0, z_up), label="lift")
        plan.go(arm, 1.0, (x, y1, z_up), label="over its new place")
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 1.0, (x, y1, BED_Z + g + 0.002), label="set it down")
        plan.go(arm, 0.6, finger=ready, label="release")
        plan.go(arm, 0.8, (x, y1, z_up), label="up")
        plan.go(arm, 1.0, (0.18, s * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
        plan.sync()
    # 2. left: the carton between the rails on the outfeed mark (gripped across its 6 cm, fingers along x)
    cg = CARTON[2] - 0.012
    cgrip, cready = -FINGER_PER_M * (CARTON[0] - 0.004), -FINGER_PER_M * (CARTON[0] + 0.02)
    cz = BED_Z + RAIL[2] + 0.03 + cg
    yaw = _yaw_for("left", math.pi / 2)
    plan.go("left", 1.3, (*CARTON_START, cz), cready, "carton: above it", yaw)
    plan.go("left", 0.4, label="settle")
    plan.go("left", 1.0, (*CARTON_START, TOP + STAGING["size"][2] + cg), label="descend")
    plan.go("left", 0.6, finger=cgrip, label="grip")
    plan.go("left", 0.8, (*CARTON_START, cz), label="lift")
    plan.go("left", 1.6, (*OUTFEED, cz), label="over the outfeed mark")
    plan.go("left", 0.4, label="settle")
    plan.go("left", 1.2, (*OUTFEED, BED_Z + cg + 0.003), label="down between the rails")
    plan.go("left", 0.6, finger=cready, label="release")
    plan.go("left", 0.8, (*OUTFEED, cz), label="up")
    plan.go("left", 1.0, (0.18, 0.32, TOP + 0.18), FINGER_OPEN, "back")
    plan.sync()
    return plan.wps


# -- self-check (pure Python): the plan fits the run limit and the scorer covers the obvious outcomes ----------------
if __name__ == "__main__":
    class _O:
        def __init__(self, x, y, z, s):
            self.pos = np.array((x, y, z + s[2] / 2))
            self.low, self.high = np.array((x - s[0] / 2, y - s[1] / 2, z)), np.array((x + s[0] / 2, y + s[1] / 2, z + s[2]))
            self.quat_xyzw = np.array((0.0, 0.0, 0.0, 1.0))

        def resting(self):
            return True

    x = BED["center"][0]
    rail = lambda y: _O(x, y, BED_Z, RAIL)
    start = {"rail_left": rail(0.08), "rail_right": rail(-0.08), "carton": _O(*CARTON_START, TOP + 0.01, CARTON)}
    assert score(start) == 0.0 and not success(start)
    done = {"rail_left": rail(RAIL_SET_Y), "rail_right": rail(-RAIL_SET_Y), "carton": _O(*OUTFEED, BED_Z, CARTON)}
    assert abs(score(done) - 1.0) < 1e-9 and success(done)
    too_tight = dict(done, rail_left=rail(RAIL_SET_Y - 0.006), rail_right=rail(-RAIL_SET_Y + 0.006))  # 7.9 cm: narrower than the carton
    assert score(too_tight) == 0.0
    off_centre = dict(done, rail_left=rail(RAIL_SET_Y + 0.008), rail_right=rail(-RAIL_SET_Y + 0.008))
    assert score(off_centre) == 0.0
    swapped = dict(done, rail_left=done["rail_right"], rail_right=done["rail_left"])  # exploit (council): identical rails swapped
    assert abs(score(swapped) - 1.0) < 1e-9, score(swapped)
    no_test = dict(done, carton=start["carton"])
    assert abs(score(no_test) - 0.6) < 1e-9
    assert abs(rail_gap(done) - (CARTON[1] + 0.011)) < 1e-9
    t = max(sum(w.duration for w in wps) for wps in demo().values())
    assert t * 20 < 2400, t
    print(f"ok: demo {t:.1f} s ({t * 20:.0f} steps)")
