"""On the station table: a gravity feed chute slopes down across the bench, its low end over a catch bin, with six
parts queued in it nose to tail, one of them red. A slotted cover keeps the parts in the chute, so none can be lifted
out. Two stops meter the queue: the lower stop is sprung shut and opens when its orange knob is pressed down; the upper
stop is sprung open and drops between the first and the second part when its blue knob is pressed down. To release one
part, hold the upper stop down, press the lower one, let the lower one go, then let the upper one go: the queue moves up
one. With both open together, the whole queue runs out. The robot must release exactly the parts ahead of the red one
into the bin, and leave the red one in the chute. Four lots (A, B, C, D) put the red part second to fifth in the queue;
the environment variable ESCAPEMENT_LOT picks one (A by default).

Run it:  ESCAPEMENT_LOT=A inspect-robots run --embodiment yam_escapement -E task=task.py -E object_jitter=0 --policy yam_scripted \\
           --instruction "Use the two stops on the feed chute to release, one at a time, every part ahead of the red one into the bin, then leave the red part in the chute with both hands off the stops." --scorer task_score

`yam_escapement` (two_stop_escapement/embodiment.py) spawns the chute, the stops and the parts, and reports each
gripper's fingertip midpoint.
"""

import math
import os

import numpy as np

from two_stop_escapement import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
STEEL = (0.30, 0.32, 0.35)
BLUE = (0.12, 0.28, 0.65)

LOTS = {"A": 1, "B": 2, "C": 3, "D": 4}  # how many parts are ahead of the red one
LOT = os.environ.get("ESCAPEMENT_LOT", "A").upper()
AHEAD = LOTS[LOT]
NAMES = [f"part_{i + 1}" for i in range(6)]  # by place in the queue, part_1 first
RED_NAME = NAMES[AHEAD]

CHUTE_LOW = (0.24, -0.06, TOP + 0.06)  # the floor's low end: high enough that parts piled in the bin don't block it


def world(y, z=0.0, x=0.0):
    return M.to_world((x, y, z), CHUTE_LOW)


BIN = dict(center=(0.24, -0.135), inner=(0.12, 0.14), wall=0.01, height=0.04, floor=0.004)


def _bin():
    (cx, cy), (ix, iy), w, h, f = BIN["center"], BIN["inner"], BIN["wall"], BIN["height"], BIN["floor"]
    return [
        Box("bin_floor", pos=(cx, cy, TOP + f / 2), size=(ix + 2 * w, iy + 2 * w, f), color=BLUE),
        Box("bin_far", pos=(cx + ix / 2 + w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=BLUE),
        Box("bin_near", pos=(cx - ix / 2 - w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=BLUE),
        Box("bin_up", pos=(cx, cy + iy / 2 + w / 2, TOP + h / 2), size=(ix, w, h), color=BLUE),
        Box("bin_down", pos=(cx, cy - iy / 2 - w / 2, TOP + h / 2), size=(ix, w, h), color=BLUE),
    ]


def _legs():
    out = []
    for name, y in (("leg_low", 0.03), ("leg_high", 0.40)):
        p = world(y, -M.WALL)
        out.append(Box(name, pos=(float(p[0]), float(p[1]), float((TOP + p[2]) / 2)), size=(0.06, 0.02, float(p[2] - TOP)), color=STEEL))
    return out


OBJECTS = [
    # placeholders: the yam_escapement embodiment spawns the chute, the stops and the parts (mech) in their place
    Box("chute_frame", pos=tuple(CHUTE_LOW), size=(0.01, 0.01, 0.01), color=STEEL),
    Box("chute_stops", pos=tuple(CHUTE_LOW), size=(0.01, 0.01, 0.01), color=STEEL),
    *[Box(n, pos=tuple(float(v) for v in world(M.part_y(i), 0.0005)), size=(0.01, 0.01, 0.01), mass=M.PART_MASS) for i, n in enumerate(NAMES)],
    *_bin(),
    *_legs(),
]

KNOB_U_WORLD = world(M.UPPER_Y, M.KNOB_U_TOP, M.KNOB_X)
KNOB_L_WORLD = world(M.KNOB_L, M.KNOB_L_TOP, M.KNOB_X)

# -- scoring -----------------------------------------------------------------------------------------------------------
HANDS_CLEAR = 0.06  # each fingertip midpoint at least this far from both knobs
ONE_SHORT = 0.15
OVER_BIN = 0.05  # a part in the bin is at most this far above its rim


def chute_y(o):
    return M.to_local(o.pos, CHUTE_LOW)[1]


def in_bin(o):
    (cx, cy), (ix, iy) = BIN["center"], BIN["inner"]
    return bool(abs(o.pos[0] - cx) <= ix / 2 and abs(o.pos[1] - cy) <= iy / 2 and o.pos[2] <= TOP + BIN["height"] + OVER_BIN)


def released(objects):
    """How many parts are at rest in the bin."""
    return sum(in_bin(objects[n]) and objects[n].resting() for n in NAMES)


def red_held(objects):
    """The red part still in the chute, its block behind the lower stop's face (2 mm of play) and within the chute's width."""
    x, y, z = M.to_local(objects[RED_NAME].pos, CHUTE_LOW)
    return bool(abs(x) <= M.INNER_HALF and M.STOP_FACE + 0.025 - 0.002 <= y <= M.LENGTH and -0.002 <= z <= M.COVER_Z)


def hands_clear(objects):
    return all(np.linalg.norm(objects[a].pos - k) >= HANDS_CLEAR for a in ("left_tcp", "right_tcp")
               for k in (KNOB_U_WORLD, KNOB_L_WORLD))


STILL_S = 1.0  # every part at rest this long
_trial = {"still": 0.0}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset."""
    _trial.update(still=0.0)


def note_step(objects, dt):
    """Called by the embodiment after every control step: the stillness clock."""
    _trial["still"] = _trial["still"] + dt if all(objects[n].resting() for n in NAMES) else 0.0


def score(objects):
    """Read at the end of the run; there's no early end, or a plan could release one part at a time blind and be stopped
    at the right count. 1.0: every part ahead of the red one in the bin, the red one in the chute, every part at rest for
    1 s, both hands clear. 0.15: the red one in the chute, with at least one part in the bin and either one part still
    to come or those last conditions missing. Otherwise 0, including the red part out of the chute. Parts anywhere else
    count for nothing."""
    if not red_held(objects):
        return 0.0
    n = released(objects)
    if n == AHEAD and _trial["still"] >= STILL_S and hands_clear(objects):
        return 1.0
    return ONE_SHORT if n >= 1 and n >= AHEAD - 1 else 0.0


# -- scripted demo -----------------------------------------------------------------------------------------------------
PAD_TIP_PAST_TCP = 0.013  # closed pad tips, measured past the TCP
HOVER = 0.03  # TCP this far above a knob before pressing it
U_PRESS, L_PRESS = M.UPPER_TRAVEL + 0.006, 0.024  # TCP travel, past each limit (the lower pad drops about 19 mm)
PARK = 0.18, 0.32, TOP + 0.18  # x, |y|, z
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}


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
        p = p if pos is None else tuple(float(v) for v in pos)
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


def _yaw(arm):  # yaw 0 (or a multiple of pi): fingers close along y
    return min((k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def _over(knob, up):
    return (knob[0], knob[1], knob[2] + PAD_TIP_PAST_TCP + up)


def demo(lot=None):
    ahead = LOTS[LOT if lot is None else lot]
    plan = _Plan()
    for a in SIDE:  # from READY to a top-down park pose
        plan.go(a, 1.0, (PARK[0], SIDE[a] * PARK[1], PARK[2]), FINGER_OPEN, "park", 0.0)
    plan.go("left", 1.4, _over(KNOB_U_WORLD, HOVER), FINGER_CLOSED, "over the upper stop's knob", _yaw("left"))
    plan.go("right", 1.4, _over(KNOB_L_WORLD, HOVER), FINGER_CLOSED, "over the lower stop's knob", _yaw("right"))
    plan.sync()
    for _ in range(ahead):
        if os.environ.get("ESCAPEMENT_DEMO_UNHELD") != "1":  # debugging only: skip the upper stop, to show the queue runs out
            plan.go("left", 0.8, _over(KNOB_U_WORLD, -U_PRESS), label="hold the upper stop down")
        plan.go("left", 0.4, label="settle")
        plan.sync()
        plan.go("right", 0.6, _over(KNOB_L_WORLD, -L_PRESS), label="open the lower stop")
        plan.go("right", 1.0, label="the first part runs out")
        plan.go("right", 0.6, _over(KNOB_L_WORLD, HOVER), label="let the lower stop shut")
        plan.go("right", 0.4, label="settle")
        plan.sync()
        plan.go("left", 0.6, _over(KNOB_U_WORLD, HOVER), label="let the upper stop rise")
        plan.go("left", 1.2, label="the queue moves up one")
        plan.sync()
    for a in SIDE:
        plan.go(a, 1.2, (PARK[0], SIDE[a] * PARK[1], PARK[2]), FINGER_OPEN, "away from the stops", 0.0)
    plan.go("left", 2.0, label="let the bin settle")
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
        assert t * 20 < 2400 and pe < 1e-3 and re < 1e-2, (name, t, pe, re)
        print(f"ok: lot {name}: demo {t:.1f} s ({t * 20:.0f} steps), IK error <= {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg")
