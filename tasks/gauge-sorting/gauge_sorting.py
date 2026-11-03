"""On the station table: five machined aluminium blocks lie in a row across the near edge of the bench, each with its long
side pointing away from the arms. They look alike, but two of them are 4 mm too tall. A fixed go/no-go bridge gauge
stands in the middle of the bench beyond them: a steel bar across the bench on two posts, its underside 3.7 cm above
the table. A block that is in tolerance (3.5 cm tall) slides under the bar when pushed along the table; an oversize
one (3.9 cm) stops against the bar's face. A green ACCEPT tray stands on the left, a red REJECT tray on the right. The
robot must check every block under the gauge and put the good ones in ACCEPT and the oversize ones in REJECT.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/gauge-sorting/gauge_sorting.py \\
           --instruction "check each block with the bridge gauge: put the blocks that slide under the bar into the green ACCEPT tray and the ones that don't into the red REJECT tray" --scorer task_score ...
"""

import math

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
ALU = (0.75, 0.77, 0.80)
GAUGE_GREY = (0.30, 0.32, 0.35)
ALU_DENSITY = 2700.0  # kg/m^3: machined aluminium blocks

# -- the gauge: a fixed bar across y, 1 cm thick and 2 cm deep, its underside GAP_H above the table, on two posts --
GAP_H = 0.037
BAR = dict(x0=0.31, depth=0.02, half_span=0.10, thick=0.01)  # x0: the bar's near face

# -- the parts (m): x (long side), y, z (height: the gauged size); start (x, y) on the bench; the demo arm on the side
# of the block's tray. Lengths and widths vary so that only the height tells good from oversize. -------------------
GOOD_H, BAD_H = 0.035, 0.039
PARTS = {
    "block_1": dict(size=(0.060, 0.040, GOOD_H), start=(0.15, 0.20), arm="left"),
    "block_2": dict(size=(0.055, 0.042, BAD_H), start=(0.15, 0.10), arm="right"),
    "block_3": dict(size=(0.065, 0.038, GOOD_H), start=(0.15, 0.00), arm="left"),
    "block_4": dict(size=(0.058, 0.044, GOOD_H), start=(0.15, -0.10), arm="left"),
    "block_5": dict(size=(0.062, 0.040, BAD_H), start=(0.15, -0.20), arm="right"),
}
GOOD = {n for n, p in PARTS.items() if p["size"][2] < GAP_H}

# -- trays: inside size, wall thickness and height ----------------------------------------------------------------
TRAYS = {
    "accept": dict(center=(0.30, 0.27), inner=(0.16, 0.18), color=(0.15, 0.60, 0.25)),
    "reject": dict(center=(0.30, -0.27), inner=(0.16, 0.18), color=(0.80, 0.15, 0.12)),
}
TRAY_WALL, TRAY_H, TRAY_FLOOR = 0.01, 0.03, 0.004
WRONG_PENALTY = 1.5  # a wrongly sorted block costs this many correct ones: sorting everything into one tray scores 0


def _gauge():
    x0, d, s, t = BAR["x0"], BAR["depth"], BAR["half_span"], BAR["thick"]
    cx = x0 + d / 2
    return [
        Box("gauge_bar", pos=(cx, 0.0, TOP + GAP_H + t / 2), size=(d, 2 * s, t), color=GAUGE_GREY),
        *[Box(f"gauge_post_{n}", pos=(cx, sign * (s - 0.01), TOP + GAP_H / 2), size=(d, 0.02, GAP_H), color=GAUGE_GREY)
          for n, sign in (("left", 1), ("right", -1))],
    ]


def _tray(name, t):
    (cx, cy), (ix, iy), w, h = t["center"], t["inner"], TRAY_WALL, TRAY_H
    c = t["color"]
    return [
        Box(f"{name}_base", pos=(cx, cy, TOP + TRAY_FLOOR / 2), size=(ix + 2 * w, iy + 2 * w, TRAY_FLOOR), color=c),
        Box(f"{name}_far", pos=(cx + ix / 2 + w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=c),
        Box(f"{name}_near", pos=(cx - ix / 2 - w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=c),
        Box(f"{name}_left", pos=(cx, cy + iy / 2 + w / 2, TOP + h / 2), size=(ix, w, h), color=c),
        Box(f"{name}_right", pos=(cx, cy - iy / 2 - w / 2, TOP + h / 2), size=(ix, w, h), color=c),
    ]


OBJECTS = [
    *_gauge(),
    *_tray("accept_tray", TRAYS["accept"]),
    *_tray("reject_tray", TRAYS["reject"]),
    *[Box(n, pos=(*p["start"], TOP + p["size"][2] / 2), size=p["size"], mass=ALU_DENSITY * float(np.prod(p["size"])),
          color=ALU) for n, p in PARTS.items()],
]


# -- scoring --------------------------------------------------------------------------------------------------------
def in_tray(name, tray, objects):
    """Centre inside the tray's walls, resting on its floor (underside within 1.5 cm of it), still."""
    (cx, cy), (ix, iy) = TRAYS[tray]["center"], TRAYS[tray]["inner"]
    o = objects[name]
    return bool(abs(o.pos[0] - cx) <= ix / 2 and abs(o.pos[1] - cy) <= iy / 2
                and o.low[2] <= TOP + TRAY_FLOOR + 0.015 and o.resting())


def _sorted(objects):
    correct = wrong = 0
    for n in PARTS:
        right, other = ("accept", "reject") if n in GOOD else ("reject", "accept")
        correct += in_tray(n, right, objects)
        wrong += in_tray(n, other, objects)
    return correct, wrong


def success(objects):
    return _sorted(objects)[0] == len(PARTS)


def score(objects):
    """(blocks in the right tray - 1.5 x blocks in the wrong tray) / 5, at least 0. Blocks anywhere else count
    neither way. Sorting everything into ACCEPT scores 0 (3 - 1.5 x 2), as does leaving everything on the bench."""
    correct, wrong = _sorted(objects)
    return float(max(0.0, correct - WRONG_PENALTY * wrong) / len(PARTS))


# -- scripted demo ----------------------------------------------------------------------------------------------------
Z_UP = TOP + 0.10  # carry height: blocks clear the tray walls, the gauge and the other blocks
GRIP_Z = 0.018  # TCP this far above a block's underside when gripping it across its width
TEST_X = 0.22  # a block is slid toward the bar from here, centred on y = 0
PUSH_PAST = 0.015  # slide until a good block's front is this far under the bar (2 cm deep): the regrip stays clear of it
LAG = 0.015  # the arms reach this much short along x around the gauge (measured): the slide aims this much further
REAR_GRIP = 0.025  # blocks are held this far in from their rear end, so the fingers stay clear of the bar
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
PLACES = {"accept": [-0.055, 0.0, 0.055], "reject": [-0.04, 0.04]}  # y offsets in each tray, blocks along x


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


def _inspect(plan, name, place):
    """Pick a block by its rear end, slide it along the table toward the bar (a good one goes under, an oversize one
    stops at the bar's face), slide it back out and lay it in its tray. Held by the rear end, the fingers never reach
    the bar."""
    p = PARTS[name]
    arm, (L, _, _) = p["arm"], p["size"]
    yaw = min((k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))  # fingers across the width
    (px, py), (qx, qy) = p["start"], place
    off = L / 2 - REAR_GRIP  # the block's centre is this far ahead (+x) of the fingers
    gz = TOP + GRIP_Z
    gx = px - off
    plan.go(arm, 1.2, (gx, py, Z_UP), FINGER_OPEN, f"{name}: above its rear end", yaw)
    plan.go(arm, 1.0, (gx, py, gz), label="descend")
    plan.go(arm, 0.6, finger=FINGER_CLOSED, label="grip")
    plan.go(arm, 0.8, (gx, py, Z_UP), label="lift")
    start = TEST_X - off
    plan.go(arm, 1.3, (start, 0.0, Z_UP), label="to the gauge")
    plan.go(arm, 0.8, (start, 0.0, gz + 0.001), label="down onto the table")
    through = BAR["x0"] + PUSH_PAST - L + REAR_GRIP  # fingers here: a good block's front is PUSH_PAST under the bar
    plan.go(arm, 1.5, (through + LAG, 0.0, gz + 0.001), label="slide it under the bar")
    plan.go(arm, 0.4, label="does it go under?")
    plan.go(arm, 1.0, (start, 0.0, gz + 0.001), label="slide it back out")
    plan.go(arm, 0.8, (start, 0.0, Z_UP), label="lift")
    plan.go(arm, 1.4, (qx - off, qy, Z_UP), label="over its tray")
    plan.go(arm, 0.9, (qx - off, qy, TOP + TRAY_FLOOR + GRIP_Z + 0.004), label="lower")
    plan.go(arm, 0.4, finger=FINGER_OPEN, label="release")
    plan.go(arm, 0.6, (qx - off, qy, Z_UP), label="up")
    plan.go(arm, 1.0, (0.18, SIDE[arm] * 0.33, Z_UP + 0.08), label="clear of the gauge")


def demo():
    plan = _Plan()
    used = {"accept": 0, "reject": 0}
    # one arm at the gauge at a time; each block goes with the arm on its tray's side
    for name in PARTS:
        tray = "accept" if name in GOOD else "reject"
        (cx, cy), dy = TRAYS[tray]["center"], PLACES[tray][used[tray]]
        used[tray] += 1
        _inspect(plan, name, (cx, cy + dy))
        plan.sync()
    return plan.wps


# -- self-check (pure Python): the plan fits the run limit and the scorer covers the obvious outcomes ----------------
if __name__ == "__main__":
    class _O:
        def __init__(self, x, y, z, still=True):
            self.pos = np.array((x, y, z + 0.02))
            self.low = np.array((x - 0.03, y - 0.02, z))
            self._still = still

        def resting(self):
            return self._still

    def _at(tray):
        (cx, cy) = TRAYS[tray]["center"]
        return _O(cx, cy, TOP + TRAY_FLOOR)

    assert GOOD == {"block_1", "block_3", "block_4"}
    bench = {n: _O(*p["start"], TOP) for n, p in PARTS.items()}
    right = {n: _at("accept" if n in GOOD else "reject") for n in PARTS}
    assert score(bench) == 0.0 and not success(bench)
    assert score(right) == 1.0 and success(right)
    assert score({n: _at("accept") for n in PARTS}) == 0.0 and score({n: _at("reject") for n in PARTS}) == 0.0
    assert abs(score(dict(right, block_2=_at("accept"))) - (4 - 1.5) / 5) < 1e-9
    assert all((p["size"][2] < GAP_H) == (n in GOOD) for n, p in PARTS.items())
    t = max(sum(w.duration for w in wps) for wps in demo().values())
    assert t * 20 < 3600, t
    print(f"ok: demo {t:.1f} s ({t * 20:.0f} steps)")
