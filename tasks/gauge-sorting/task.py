"""On the station table: six machined aluminium blocks lie flat in two rows of three near the arms. They look alike,
but they come in three heights: two are in tolerance (35.5 mm), two are 3.5 mm oversize and two 3.5 mm undersize, and
their lengths and widths vary so that nothing else gives them away. A fixed two-limit bridge gauge stands beyond them:
a green GO bar on the left (its underside 37.5 mm above the table) and a red NO-GO bar on the right (34.0 mm). A good
block slides under the GO bar but stops at the NO-GO bar; an oversize one stops at the GO bar; an undersize one slides
under both. The robot must gauge every block and put the good ones in the green ACCEPT tray on the left, the oversize
ones in the red OVERSIZE tray and the undersize ones in the blue UNDERSIZE tray, both on the right.
Three layouts (A, B, C) say which block lies where; the environment variable GAUGE_LAYOUT picks one (A by default).

Run it:  GAUGE_LAYOUT=A inspect-robots run --embodiment yam_gauge -E task=tasks/gauge-sorting/task.py \\
           --instruction "Check each block, lying flat, with the two-bar gauge: put blocks that slide under the green GO bar but not under the red NO-GO bar in the green ACCEPT tray, blocks that don't fit under the GO bar in the red OVERSIZE tray, and blocks that also slide under the NO-GO bar in the blue UNDERSIZE tray." --scorer task_score ...

`yam_gauge` (gauge_sorting/embodiment.py) is the stock yam_arena embodiment that also reports `left_tcp` and
`right_tcp`, so the scorer can tell whether a block has been let go.
"""

import math
import os

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
ALU = (0.75, 0.77, 0.80)
STEEL = (0.30, 0.32, 0.35)
GREEN, RED, BLUE = (0.15, 0.60, 0.25), (0.80, 0.15, 0.12), (0.15, 0.35, 0.80)
ALU_DENSITY = 2700.0  # kg/m^3: machined aluminium blocks

# -- the gauge: two fixed bars side by side across y, each 2 cm deep (along x) and 1 cm thick, their near faces at
# x0. GO on the left (+y), NO-GO on the right, a post at each outer end and one between them. Each bar spans a 10 cm
# lane between posts; a block is gauged by sliding it along the table down its lane (+x) toward the bar. ----------
GO_H, NOGO_H = 0.0375, 0.0340  # the bars' undersides above the table
BAR = dict(x0=0.31, depth=0.02, thick=0.01)
LANE = {"go": 0.06, "nogo": -0.06}  # lane centres (y); each lane is 10 cm wide between posts 2 cm wide
POSTS = (0.12, 0.0, -0.12)  # post centres (y)

# -- the parts (m): size x (long side) along x, y (width), z (height, the gauged size, always the smallest);
# start (x, y). Heights: undersize 32.0 mm (2 mm under NO-GO), good 35.5 mm (1.5 mm over NO-GO, 2 mm under GO),
# oversize 39.0 mm (1.5 mm over GO). Lengths and widths vary so that footprint and mass don't tell the kinds apart.
UNDER_H, GOOD_H, OVER_H = 0.0320, 0.0355, 0.0390
FRONT_X, BACK_X = 0.10, 0.175
SIZES = {
    "block_1": (0.064, 0.044, UNDER_H),
    "block_2": (0.060, 0.042, GOOD_H),
    "block_3": (0.054, 0.041, OVER_H),
    "block_4": (0.056, 0.041, OVER_H),
    "block_5": (0.056, 0.040, UNDER_H),
    "block_6": (0.055, 0.040, GOOD_H),
}
# Six places, front row then back row, each left to right (+y first). Three layouts say which block goes where; each
# place holds an undersize, a good and an oversize block once over the three, so where a block lies says nothing.
# GAUGE_LAYOUT picks one for a run (A by default).
SLOTS = [(FRONT_X, 0.08), (FRONT_X, 0.0), (FRONT_X, -0.08), (BACK_X, 0.08), (BACK_X, 0.0), (BACK_X, -0.08)]
LAYOUTS = {
    "A": (1, 2, 3, 4, 5, 6),  # front: under, good, over; back: over, under, good
    "B": (2, 3, 1, 5, 6, 4),  # front: good, over, under; back: under, good, over
    "C": (3, 5, 6, 2, 4, 1),  # front: over, under, good; back: good, over, under
}
LAYOUT = os.environ.get("GAUGE_LAYOUT", "A").upper()
PARTS = {f"block_{k}": dict(size=SIZES[f"block_{k}"], start=xy) for k, xy in zip(LAYOUTS[LAYOUT], SLOTS)}
PARTS = {n: PARTS[n] for n in SIZES}  # in block-number order


def kind(name):
    """What the gauge says: 'accept', 'over' or 'under'."""
    h = PARTS[name]["size"][2]
    return "over" if h >= GO_H else "under" if h < NOGO_H else "accept"


# -- trays (fixed): ACCEPT on the left; on the right one tray split by a divider into OVERSIZE (inboard) and
# UNDERSIZE (outboard). inner = inside size (x, y); walls 1 cm thick and 3 cm tall, floor 4 mm. ------------------
TRAY_WALL, TRAY_H, TRAY_FLOOR = 0.01, 0.03, 0.004
TRAYS = {
    "accept": dict(center=(0.30, 0.27), inner=(0.16, 0.18), color=GREEN, walls=("far", "near", "left", "right")),
    "over": dict(center=(0.30, -0.2075), inner=(0.16, 0.105), color=RED, walls=("far", "near", "left", "right")),
    "under": dict(center=(0.30, -0.3225), inner=(0.16, 0.105), color=BLUE, walls=("far", "near", "right")),
}
WRONG_PENALTY = 3.0  # a block in the wrong tray, or off the table, cancels three correct ones
HANDS_CLEAR = 0.02  # a block counts only when both fingertip centres are this far from its bounding box


def _gauge():
    x0, d, t = BAR["x0"], BAR["depth"], BAR["thick"]
    cx = x0 + d / 2
    go_y0, go_y1 = POSTS[1] + 0.01, POSTS[0] + 0.01  # each bar runs from the middle post's face over its outer post
    ng_y0, ng_y1 = POSTS[2] - 0.01, POSTS[1] - 0.01
    return [
        Box("gauge_go_bar", pos=(cx, (go_y0 + go_y1) / 2, TOP + GO_H + t / 2), size=(d, go_y1 - go_y0, t),
            color=GREEN),
        Box("gauge_nogo_bar", pos=(cx, (ng_y0 + ng_y1) / 2, TOP + NOGO_H + t / 2), size=(d, ng_y1 - ng_y0, t),
            color=RED),
        Box("gauge_post_left", pos=(cx, POSTS[0], TOP + GO_H / 2), size=(d, 0.02, GO_H), color=STEEL),
        Box("gauge_post_middle", pos=(cx, POSTS[1], TOP + (GO_H + t) / 2), size=(d, 0.02, GO_H + t), color=STEEL),
        Box("gauge_post_right", pos=(cx, POSTS[2], TOP + NOGO_H / 2), size=(d, 0.02, NOGO_H), color=STEEL),
    ]


def _tray(name, t):
    (cx, cy), (ix, iy), w, h, c = t["center"], t["inner"], TRAY_WALL, TRAY_H, t["color"]
    walls = {
        "far": ((cx + ix / 2 + w / 2, cy), (w, iy + 2 * w)),
        "near": ((cx - ix / 2 - w / 2, cy), (w, iy + 2 * w)),
        "left": ((cx, cy + iy / 2 + w / 2), (ix, w)),
        "right": ((cx, cy - iy / 2 - w / 2), (ix, w)),
    }
    return [
        Box(f"{name}_tray_base", pos=(cx, cy, TOP + TRAY_FLOOR / 2), size=(ix + 2 * w, iy + 2 * w, TRAY_FLOOR),
            color=c),
        *[Box(f"{name}_tray_{k}", pos=(*walls[k][0], TOP + h / 2), size=(*walls[k][1], h), color=c) for k in t["walls"]],
    ]


OBJECTS = [
    *_gauge(),
    *[b for n, t in TRAYS.items() for b in _tray(n, t)],
    *[Box(n, pos=(*p["start"], TOP + p["size"][2] / 2), size=p["size"], mass=ALU_DENSITY * float(np.prod(p["size"])),
          color=ALU) for n, p in PARTS.items()],
]


# -- scoring --------------------------------------------------------------------------------------------------------
def hands_clear(name, objects):
    """Both fingertip centres at least HANDS_CLEAR from the block's bounding box: it has been let go."""
    o = objects[name]
    for arm in ("left_tcp", "right_tcp"):
        p = np.asarray(objects[arm].pos, float)
        if np.linalg.norm(p - np.clip(p, o.low, o.high)) < HANDS_CLEAR:
            return False
    return True


def in_tray(name, tray, objects):
    """Centre inside the tray's walls, resting on its floor (underside within 1.5 cm of it), still and let go."""
    (cx, cy), (ix, iy) = TRAYS[tray]["center"], TRAYS[tray]["inner"]
    o = objects[name]
    return bool(abs(o.pos[0] - cx) <= ix / 2 and abs(o.pos[1] - cy) <= iy / 2
                and o.low[2] <= TOP + TRAY_FLOOR + 0.015 and o.resting() and hands_clear(name, objects))


def off_table(name, objects):
    return bool(objects[name].pos[2] < TOP)


def _sorted(objects):
    correct = wrong = 0
    for n in PARTS:
        correct += in_tray(n, kind(n), objects)
        wrong += off_table(n, objects) or any(in_tray(n, t, objects) for t in TRAYS if t != kind(n))
    return correct, wrong


def success(objects):
    return _sorted(objects)[0] == len(PARTS)


def score(objects):
    """(blocks in the right tray - 3 x blocks in a wrong tray or off the table) / 6, at least 0. Blocks anywhere else
    on the table count neither way."""
    correct, wrong = _sorted(objects)
    return float(max(0.0, correct - WRONG_PENALTY * wrong) / len(PARTS))


# -- scripted demo ----------------------------------------------------------------------------------------------------
Z_UP = TOP + 0.10  # carry height: blocks clear the tray walls, the gauge and the other blocks
GRIP_Z = 0.018  # TCP this far above a block's underside when gripping it across its width
FRONT_GAP = 0.02  # each gauge pass starts with the block's front this far before the bar
PUSH_PAST = 0.015  # slide until a passing block's front is this far under the bar (2 cm deep)
LAG = 0.015  # the arms reach this much short along x around the gauge (measured): the slide aims this much further
REAR_GRIP = 0.025  # blocks are held this far in from their rear end, so the fingers stay clear of the bars
FINGER_PER_M = -FINGER_OPEN / 0.094  # finger joint per metre of opening (the gripper opens 94 mm)
SIDE = {"left": 1, "right": -1}
ARM = {"accept": "left", "over": "right", "under": "right"}  # each block goes with the arm on its tray's side
PLACES = [0.26, 0.34]  # x of the two blocks in each tray, on its centre line
PARK = {a: (0.15, SIDE[a] * 0.33, TOP + 0.18) for a in SIDE}  # out of the way, above the trays' near walls
NAMES = {"go": "GO", "nogo": "NO-GO"}
ORDER = sorted(PARTS, key=lambda n: (-PARTS[n]["start"][0], -PARTS[n]["start"][1]))  # back row first, left to right


def opening(w):
    """Finger target for an opening 2 cm wider than a block w wide (clear of its neighbours and the tray walls)."""
    return -FINGER_PER_M * (w + 0.02)


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
    """Pick a block by its rear end; slide it along the table toward the GO bar and back out; if it went under, slide
    it toward the NO-GO bar and back out; then lay it in its tray. Held by the rear end, the fingers never reach the
    bars. The demo is open loop: it knows each block's kind, and the gauge shows the result rather than deciding it."""
    p = PARTS[name]
    arm, (L, W, _) = ARM[kind(name)], p["size"]
    yaw = 0.0  # fingers close along y, across the block's width
    (px, py), (qx, qy) = p["start"], place
    off = L / 2 - REAR_GRIP  # the block's centre is this far ahead (+x) of the fingers
    gz = TOP + GRIP_Z
    plan.go(arm, 1.2, (px - off, py, Z_UP), opening(W), f"{name}: above its rear end", yaw)
    plan.go(arm, 1.0, (px - off, py, gz), label="descend")
    plan.go(arm, 0.6, finger=FINGER_CLOSED, label="grip")
    plan.go(arm, 0.8, (px - off, py, Z_UP), label="lift")
    start = BAR["x0"] - FRONT_GAP - L + REAR_GRIP  # fingers here: the block's front is FRONT_GAP before the bar
    through = BAR["x0"] + PUSH_PAST - L + REAR_GRIP  # fingers here: a passing block's front is PUSH_PAST under the bar
    plan.go(arm, 1.3, (start, LANE["go"], Z_UP), label="to the GO lane")
    plan.go(arm, 0.8, (start, LANE["go"], gz + 0.001), label="down onto the table")
    for bar in ("go", "nogo"):
        y = LANE[bar]
        if bar == "nogo":
            plan.go(arm, 0.5, (start, LANE["go"], gz + 0.011), label="ease it up 1 cm")
            plan.go(arm, 1.0, (start, y, gz + 0.011), label="across to the NO-GO lane")
            plan.go(arm, 0.5, (start, y, gz + 0.001), label="down onto the table")
        plan.go(arm, 1.5, (through + LAG, y, gz + 0.001), label=f"slide it toward the {NAMES[bar]} bar")
        plan.go(arm, 0.4, label="does it go under?")
        plan.go(arm, 1.0, (start, y, gz + 0.001), label="slide it back out")
        if kind(name) == "over":  # stopped at the GO bar: oversize, no NO-GO check needed
            break
    plan.go(arm, 0.8, (start, y, Z_UP), label="lift")
    plan.go(arm, 1.4, (qx - off, qy, Z_UP), label="over its tray")
    plan.go(arm, 0.9, (qx - off, qy, TOP + TRAY_FLOOR + GRIP_Z + 0.004), label="lower")
    plan.go(arm, 0.4, finger=opening(W), label="release")
    plan.go(arm, 0.6, (qx - off, qy, Z_UP), label="up")
    plan.go(arm, 1.0, PARK[arm], label="clear of the gauge")


def demo():
    plan = _Plan()
    for a in SIDE:
        plan.go(a, 1.0, PARK[a], FINGER_OPEN, "out of the way", 0.0)
    used = {t: 0 for t in TRAYS}
    for name in ORDER:  # one arm at the gauge at a time
        t = kind(name)
        _inspect(plan, name, (PLACES[used[t]], TRAYS[t]["center"][1]))
        used[t] += 1
        plan.sync()
    return plan.wps


def ik_errors(wps):
    """The largest IK position (m) and rotation (rad) error over every waypoint of both arms."""
    worst_p = worst_r = 0.0
    for arm, ws in wps.items():
        q = np.array(READY, float)
        for w in ws:
            q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q,
                                 base_pos=BASES[arm])
            worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
    return worst_p, worst_r


if __name__ == "__main__":  # self-check (no simulator): this layout's demo fits the run limit and IK reaches it
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re = ik_errors(wps)
    assert t * 20 < 3600 and pe < 1e-3 and re < 1e-2, (LAYOUT, t, pe, re)
    print(f"ok: layout {LAYOUT}: demo {t:.1f} s ({t * 20:.0f} steps), IK error <= {pe * 1000:.2f} mm / "
          f"{math.degrees(re):.2f} deg")
