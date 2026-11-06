"""On the station table: a 12 cm die block, too wide for a gripper to pick up, stands in a rubber-matted lane with low
yellow curbs either side. One of its faces is its ground datum face, painted red; each neighbouring face has a red band
along the edge it shares with the red face, so the top shows which way the red face lies, and no band means it is
underneath. The robot must tip the block over its edges along the lane until the red face is up, and move both hands away.
A push high on a face tips it; a push low only slides it. Lots A, B and C start with the red face to the left, to the
right and underneath; the environment variable TIP_LOT picks one (A by default).

Run it:  TIP_LOT=A inspect-robots run --embodiment yam_tip -E task=task.py -E object_jitter=0 --policy yam_scripted \\
           --instruction "Tip the die block over along the lane until its red datum face is on top, then move both hands away." --scorer task_score

`yam_tip` (tip_die_block/embodiment.py) spawns the lane and the marked block, and reports each gripper's fingertip midpoint.
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

SIDE = 0.12  # the block's edge: too wide for the 94 mm gripper
BLOCK_MASS = 0.5  # a real die block this size is about 13 kg; lightened to suit the arms
BAND = 0.03  # the red band, along the shared edge on each face touching the red one
X0, Y0 = 0.25, 0.0
MAT_T = 0.005
LANE_X_HALF, LANE_Y_HALF = 0.065, 0.32  # 13 cm between the curbs, 5 mm of play either side of the block
CURB_W, CURB_H = 0.01, 0.025


def _rx(deg):
    a = math.radians(deg) / 2
    return (math.sin(a), 0.0, 0.0, math.cos(a))


# The block's +z is the red face's outward normal. Each lot turns it about x: the red face to the left, right or underneath.
# tips: (the arm that pushes, the side of the block it stands on, +1 for +y), as the demo does them.
LOTS = {"A": dict(rot=-90.0, tips=(("left", 1),)),  # red on +y: one tip toward -y brings it up
        "B": dict(rot=90.0, tips=(("right", -1),)),  # red on -y: one tip toward +y
        "C": dict(rot=180.0, tips=(("left", 1), ("right", 1)))}  # underneath: two tips, either way
LOT = os.environ.get("TIP_LOT", "A").upper()
START_XYZW = _rx(LOTS[LOT]["rot"])
BLOCK_Z = TOP + MAT_T + SIDE / 2

OBJECTS = [
    # placeholders: the yam_tip embodiment spawns the lane and the block in their place
    Box("tip_lane", pos=(X0, Y0, TOP), size=(0.01, 0.01, 0.01), color=(0.1, 0.1, 0.1)),
    Box("die_block", pos=(X0, Y0, BLOCK_Z + 0.0002), size=(0.01, 0.01, 0.01), mass=BLOCK_MASS),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
UP = math.radians(10)  # the red face's normal within this of straight up
HANDS_CLEAR = 0.03  # each fingertip midpoint at least this far from the block's surface


def _rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def red_up_cos(block):
    """How straight up the red face points: 1 straight up, -1 underneath."""
    return float(_rot(block.quat_xyzw)[2, 2])


def red_up(block):
    return red_up_cos(block) >= math.cos(UP)


def in_lane(block):
    """On the mat between the curbs: its centre over the lane and no more than 1 cm above resting height."""
    return bool(abs(block.pos[0] - X0) <= LANE_X_HALF and abs(block.pos[1] - Y0) <= LANE_Y_HALF and abs(block.pos[2] - BLOCK_Z) <= 0.01)


def hands_clear(objects):
    b = objects["die_block"]
    R = _rot(b.quat_xyzw)
    for arm in ("left_tcp", "right_tcp"):
        p = R.T @ (np.asarray(objects[arm].pos, float) - b.pos)
        if np.linalg.norm(p - np.clip(p, -SIDE / 2, SIDE / 2)) < HANDS_CLEAR:
            return False
    return True


def score(objects):
    """1.0: the red face up (within 10 degrees), the block at rest in the lane, both hands clear. 0.5: the red face up with
    any of those missing. Otherwise 0."""
    b = objects["die_block"]
    if not red_up(b):
        return 0.0
    return 1.0 if in_lane(b) and b.resting() and hands_clear(objects) else 0.5


def success(objects):
    return score(objects) == 1.0


# -- scripted demo -----------------------------------------------------------------------------------------------------
PAD_TIP_PAST_TCP = 0.013  # closed pad tips, measured past the TCP
PAD_HALF = 0.012  # a closed finger's half thickness along the push (assumed from the earlier button pushes)
PUSH_H = 0.09  # pad tips this far above the mat: a push above about 6 cm tips rather than slides, at mu 1
START_GAP = 0.02  # the pad's face this far off the block before pushing
TRAVEL = 0.12  # aims well through the block: the soft arm falls short, and the error is what makes the force
HOVER = 0.15 + TOP  # TCP height while moving to the block's side
PARK = 0.18, 0.32, TOP + 0.18
SIDES = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}


class _Plan:
    def __init__(self):
        self.wps = {a: [] for a in SIDES}
        self.t = {a: 0.0 for a in SIDES}
        self.last = {}
        for a in SIDES:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None):
        p, f, y = self.last[arm]
        p = p if pos is None else tuple(float(v) for v in pos)
        f = f if finger is None else finger
        y = y if yaw is None else yaw
        self.wps[arm].append(Waypoint(dur, p, f, label, y))
        self.last[arm] = (p, f, y)
        self.t[arm] += dur

    def sync(self):
        end = max(self.t.values())
        for a in SIDES:
            if self.t[a] < end - 1e-9:
                self.go(a, end - self.t[a], label="wait")


def _yaw(arm):  # fingers close along x, so the pair is thin along the push
    return min((math.pi / 2 + k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def _tip(plan, arm, s, cy):
    """The arm comes down beside the block on its ``s`` side (+1 for +y), pushes high on the face, and lifts away."""
    face = cy + s * SIDE / 2
    y0, y1 = face + s * (PAD_HALF + START_GAP), face + s * (PAD_HALF - TRAVEL)
    z = TOP + MAT_T + PUSH_H + PAD_TIP_PAST_TCP
    plan.go(arm, 1.4, (X0, y0, HOVER), FINGER_CLOSED, "beside the block", _yaw(arm))
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 1.0, (X0, y0, z), label="down to push height")
    plan.go(arm, 2.4, (X0, y1, z), label="push it over")
    plan.go(arm, 0.8, label="let it land")
    plan.go(arm, 0.8, (X0, y1, HOVER), label="up")
    plan.go(arm, 1.2, (PARK[0], SIDES[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)
    plan.sync()
    return cy - s * SIDE


def demo(lot=None):
    tips = LOTS[LOT if lot is None else lot]["tips"]
    plan = _Plan()
    for a in SIDES:  # from READY to a top-down park pose
        plan.go(a, 1.0, (PARK[0], SIDES[a] * PARK[1], PARK[2]), FINGER_OPEN, "park", 0.0)
    plan.sync()
    cy = Y0
    for arm, s in tips:
        cy = _tip(plan, arm, s, cy)
    plan.go("left", 1.0, label="still")
    plan.sync()
    return plan.wps


def ik_errors(wps):
    worst_p = worst_r = 0.0
    for arm, ws in wps.items():
        q = np.array(READY, float)
        for w in ws:
            q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
            worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
    return worst_p, worst_r


# -- self-check (no simulator): every lot's demo fits the run limit and IK reaches every waypoint -----------------------
if __name__ == "__main__":
    for name in LOTS:
        wps = demo(name)
        t = max(sum(w.duration for w in ws) for ws in wps.values())
        pe, re = ik_errors(wps)
        assert t * 20 < 1800 and pe < 1e-3 and re < 1e-2, (name, t, pe, re)
        print(f"ok: lot {name}: demo {t:.1f} s ({t * 20:.0f} steps), IK error <= {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg")
