"""On the station table: a machining plate with two 4 cm holes lies on a stand. Two 5 cm parallels and two pins wait at
either side. The robot must rest the plate on the two parallels on the fixture mat so that both pins drop fully through its
holes, then move both hands away. A machinist never leaves a parallel under a through-hole: a pin that meets one stops 3 cm
proud. Lots A, B, C and D put the holes in different places, and with the parallels at least 10 cm apart no one layout clears
every lot, so their places have to come from reading the plate; the environment variable PARALLELS_LOT picks one (A by default).

Run it:  PARALLELS_LOT=A inspect-robots run --embodiment yam_parallels -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Stand the two parallels on the mat at least 10 cm apart and clear of the plate's holes, rest the plate on them, drop a pin fully through each hole, and move both hands away." --scorer task_score

`yam_parallels` (plate_parallels/embodiment.py) spawns the plate and pins and reports each gripper's fingertip midpoint.
"""

import math
import os

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY  # noqa: F401
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
STEEL, DARK = (0.30, 0.32, 0.35), (0.14, 0.15, 0.17)
ALU, ORANGE, BLUE = (0.78, 0.80, 0.83), (0.85, 0.45, 0.10), (0.15, 0.35, 0.80)

X0 = 0.27  # the mat's centre line
MAT = dict(size=(0.16, 0.42), t=0.005)
BASE_Z = TOP + MAT["t"]

LANES = (-0.12, -0.06, 0.0, 0.06, 0.12)  # hole places along the plate, from its centre
LOTS = {"A": (-0.12, 0.12), "B": (-0.12, 0.06), "C": (-0.06, 0.12), "D": (-0.06, 0.06)}  # hole centres
LOT = os.environ.get("PARALLELS_LOT", "A").upper()
HOLES = LOTS[LOT]

# the plate, in its own frame (origin at the centre of its underside, y along it): 30 x 8 cm, 1 cm thick, holes 4 cm square
L, W, T, HOLE = 0.30, 0.08, 0.01, 0.04
GRIP = dict(half_x=0.015, half_y=0.0125, z1=0.05)  # a grip on top at its middle, near its balance point


def plate_boxes(holes):
    """The plate as boxes: two side rails, and webs between the holes."""
    b = {"rail_far": ((HOLE / 2, -L / 2, 0.0), (W / 2, L / 2, T)), "rail_near": ((-W / 2, -L / 2, 0.0), (-HOLE / 2, L / 2, T))}
    y = -L / 2
    for i, h in enumerate(sorted(holes)):
        b[f"web_{i}"] = ((-HOLE / 2, y, 0.0), (HOLE / 2, h - HOLE / 2, T))
        y = h + HOLE / 2
    b["web_end"] = ((-HOLE / 2, y, 0.0), (HOLE / 2, L / 2, T))
    b["grip"] = ((-GRIP["half_x"], -GRIP["half_y"], T), (GRIP["half_x"], GRIP["half_y"], GRIP["z1"]))
    return b


# a pin, in its own frame (origin at the centre of its collar's underside): a 1.6 cm shank 4 cm long under a 5 cm collar
SHANK, SHANK_L, COLLAR, COLLAR_T = 0.016, 0.04, 0.05, 0.01
PIN_GRIP = dict(half=0.012, z1=0.04)
PIN_BOXES = {
    "shank": ((-SHANK / 2, -SHANK / 2, -SHANK_L), (SHANK / 2, SHANK / 2, 0.0)),
    "collar": ((-COLLAR / 2, -COLLAR / 2, 0.0), (COLLAR / 2, COLLAR / 2, COLLAR_T)),
    "grip": ((-PIN_GRIP["half"], -PIN_GRIP["half"], COLLAR_T), (PIN_GRIP["half"], PIN_GRIP["half"], PIN_GRIP["z1"])),
}
PLATE_MASS, PIN_MASS, MU = 0.45, 0.08, 0.6
PARTS = {"plate": (plate_boxes(HOLES), PLATE_MASS, MU, ALU), "pin_a": (PIN_BOXES, PIN_MASS, MU, ORANGE), "pin_b": (PIN_BOXES, PIN_MASS, MU, ORANGE)}

PAR = dict(x=0.08, y=0.05, z=0.05, mass=0.3)  # a parallel: hollow steel, 5 cm wide along the plate
ROW_X = 0.15
PLATE_START = (ROW_X, 0.0, TOP + 0.03)
PAR_START = {"parallel_a": (ROW_X, 0.20), "parallel_b": (ROW_X, -0.20)}
RACK_H = 0.045  # the pins hang by their collars across two rails, this high
PIN_START = {"pin_a": (ROW_X, 0.28), "pin_b": (ROW_X, -0.28)}
STANDS = (-0.10, 0.10)  # the plate's stand bars, y


def _objects():
    out = [
        Box("mat", pos=(X0, 0.0, TOP + MAT["t"] / 2), size=(*MAT["size"], MAT["t"]), color=DARK),
        Box("plate", pos=PLATE_START, size=(0.01, 0.01, 0.01), mass=PLATE_MASS),
        Box("pin_a", pos=(*PIN_START["pin_a"], TOP + RACK_H), size=(0.01, 0.01, 0.01), mass=PIN_MASS),
        Box("pin_b", pos=(*PIN_START["pin_b"], TOP + RACK_H), size=(0.01, 0.01, 0.01), mass=PIN_MASS),
    ]
    for i, y in enumerate(STANDS):
        out.append(Box(f"stand_{i}", pos=(ROW_X, y, TOP + 0.015), size=(0.08, 0.02, 0.03), color=STEEL))
    for n, (x, y) in PAR_START.items():
        out.append(Box(n, pos=(x, y, TOP + PAR["z"] / 2), size=(PAR["x"], PAR["y"], PAR["z"]), mass=PAR["mass"], color=BLUE))
    for n, (x, y) in PIN_START.items():
        for s, sign in (("l", 1), ("r", -1)):
            out.append(Box(f"rack_{n}_{s}", pos=(x, y + sign * 0.02, TOP + RACK_H / 2), size=(0.06, 0.012, RACK_H), color=STEEL))
    return out


OBJECTS = _objects()
PARALLELS = tuple(PAR_START)
PINS = tuple(PIN_START)

# -- scoring -----------------------------------------------------------------------------------------------------------
FLAT = math.radians(2)
UPRIGHT = math.radians(5)
SQUARE = math.radians(10)
MIN_SPAN = 0.10  # the parallels at least this far apart, centre to centre: side by side in the middle they'd clear every lot's holes
GAP = 0.004  # a parallel's top, or a collar's underside, within this of what it rests on
HANDS_CLEAR = 0.08
STILL_S = 1.0
ONE_PIN = 0.2
_trial = {"still": 0.0}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset."""
    _trial.update(still=0.0)


def _rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def flat(plate):
    return bool(_rot(plate.quat_xyzw)[2, 2] >= math.cos(FLAT))


def on_mat(plate):
    return bool(abs(plate.pos[0] - X0) <= MAT["size"][0] / 2 and abs(plate.pos[1]) <= MAT["size"][1] / 2)


def supports(objects):
    """How many parallels hold the plate: upright on the mat, wholly under it along its length, across its width within
    2 cm, their tops at its underside, the two at least 10 cm apart, each wholly clear of every hole. Local to the plate, so it needn't be anywhere exact."""
    plate = objects["plate"]
    n = 0
    if np.linalg.norm(np.asarray(objects["parallel_a"].pos, float)[:2] - np.asarray(objects["parallel_b"].pos, float)[:2]) < MIN_SPAN:
        return 0
    for name in PARALLELS:
        o = objects[name]
        d = _rot(plate.quat_xyzw).T @ (np.asarray(o.pos, float) - plate.pos)
        on_the_mat = abs(o.low[2] - BASE_Z) <= 0.003 and abs(o.pos[0] - X0) <= MAT["size"][0] / 2 and abs(o.pos[1]) <= MAT["size"][1] / 2
        clear_of_holes = all(abs(d[1] - h) >= (PAR["y"] + HOLE) / 2 for h in HOLES)
        square = abs((_rot(plate.quat_xyzw).T @ _rot(o.quat_xyzw))[1, 1]) >= math.cos(SQUARE)  # its 5 cm side along the plate, or the footprint check means nothing
        n += bool(on_the_mat and clear_of_holes and square and abs(_rot(o.quat_xyzw)[2, 2]) >= math.cos(UPRIGHT) and abs(d[0]) <= 0.02
                  and abs(d[1]) <= (L - PAR["y"]) / 2 and abs(o.high[2] - plate.pos[2]) <= GAP)
    return n


def _pin_in(pin, plate):
    """Its shank through a hole, within the hole's edges, the collar's underside on the plate's top, upright."""
    d = _rot(plate.quat_xyzw).T @ (np.asarray(pin.pos, float) - plate.pos)
    margin = (HOLE - SHANK) / 2
    return bool(abs(d[0]) <= margin and any(abs(d[1] - h) <= margin for h in HOLES) and abs(d[2] - T) <= GAP
                and _rot(pin.quat_xyzw)[2, 2] >= math.cos(UPRIGHT))


def pins_in(objects):
    return sum(_pin_in(objects[n], objects["plate"]) for n in PINS)


def hands_clear(objects):
    plate = objects["plate"]
    R = _rot(plate.quat_xyzw)
    lo, hi = np.array((-W / 2, -L / 2, 0.0)), np.array((W / 2, L / 2, GRIP["z1"]))
    for arm in ("left_tcp", "right_tcp"):
        p = R.T @ (np.asarray(objects[arm].pos, float) - plate.pos)
        if np.linalg.norm(p - np.clip(p, lo, hi)) < HANDS_CLEAR:
            return False
        if any(np.linalg.norm(np.asarray(objects[arm].pos, float) - objects[n].pos) < HANDS_CLEAR for n in PINS):
            return False
    return True


def note_step(objects, dt):
    """Called by the embodiment after every control step: the clock for parts at rest with both hands clear."""
    still = objects["plate"].resting() and all(objects[n].resting() for n in (*PARALLELS, *PINS)) and hands_clear(objects)
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end, or a plan could try places in turn and be stopped when right. 0
    unless the plate is flat on the mat on both parallels, and everything has been at rest with both hands clear for the last
    1 s. Then 1.0 with both pins through, 0.2 with one. Otherwise 0."""
    plate = objects["plate"]
    if not (flat(plate) and on_mat(plate) and supports(objects) == len(PARALLELS)) or _trial["still"] < STILL_S:
        return 0.0
    n = pins_in(objects)
    return 1.0 if n == len(PINS) else ONE_PIN if n else 0.0


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
PARK = 0.18, 0.32, TOP + 0.18
PLACES = {"A": (0.05, -0.05), "B": (0.105, -0.05), "C": (0.05, -0.11), "D": (0.105, -0.11)}  # y of parallel_a and parallel_b, for each lot
PLATE_Y = 0.0
Z_CARRY = TOP + 0.13


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


def _yaw(arm, along_x=True):  # along_x: fingers close along x
    base = math.pi / 2 if along_x else 0.0
    return min((base + k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def _fingers(width):
    return -FINGER_PER_M * width


def _park(plan, arm):
    plan.go(arm, 1.2, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


def _place_parallel(plan, arm, name, y):
    sx, sy = PAR_START[name]
    grip_z = PAR["z"] - 0.012
    ready, grip = _fingers(PAR["y"] + 0.03), _fingers(PAR["y"] - 0.004)
    yaw = _yaw(arm, along_x=False)  # fingers close along y, across its 5 cm
    plan.go(arm, 1.4, (sx, sy, TOP + PAR["z"] + 0.05), ready, f"{name}: above it", yaw)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 1.0, (sx, sy, TOP + grip_z), label="descend")
    plan.go(arm, 0.6, finger=grip, label="grip")
    plan.go(arm, 1.0, (sx, sy, TOP + PAR["z"] + 0.04), label="lift")
    plan.go(arm, 2.6, (X0, y, TOP + PAR["z"] + 0.04), label="over its place")
    plan.go(arm, 1.0, label="settle")
    plan.go(arm, 1.0, (X0, y, BASE_Z + grip_z + 0.001), label="stand it on the mat")
    plan.go(arm, 0.5, finger=ready, label="let go")
    plan.go(arm, 0.8, (X0, y, TOP + PAR["z"] + 0.06), label="up")
    _park(plan, arm)


def _place_plate(plan, arm):
    gz = GRIP["z1"] - 0.008
    ready, grip = _fingers(2 * GRIP["half_x"] + 0.03), _fingers(2 * GRIP["half_x"] - 0.004)
    px, py, pz = PLATE_START
    yaw = _yaw(arm)
    plan.go(arm, 1.4, (px, py, pz + gz + 0.05), ready, "plate: above its grip", yaw)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.9, (px, py, pz + gz), label="descend")
    plan.go(arm, 0.6, finger=grip, label="grip")
    plan.go(arm, 1.2, (px, py, Z_CARRY + 0.02), label="lift")
    plan.go(arm, 2.8, (X0, PLATE_Y, Z_CARRY + 0.02), label="over the parallels")
    plan.go(arm, 1.2, label="settle")
    plan.go(arm, 1.8, (X0, PLATE_Y, BASE_Z + PAR["z"] + gz + 0.002), label="down onto them")
    plan.go(arm, 0.6, finger=ready, label="let go")
    plan.go(arm, 0.8, (X0, PLATE_Y, Z_CARRY + 0.02), label="up")
    _park(plan, arm)


def _place_pin(plan, arm, name, hole_y):
    sx, sy = PIN_START[name]
    gz = PIN_GRIP["z1"] - 0.006
    ready, grip = _fingers(2 * PIN_GRIP["half"] + 0.03), _fingers(2 * PIN_GRIP["half"] - 0.004)
    yaw = _yaw(arm)
    plate_top = BASE_Z + PAR["z"] + T
    plan.go(arm, 1.4, (sx, sy, TOP + RACK_H + gz + 0.05), ready, f"{name}: above it", yaw)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.9, (sx, sy, TOP + RACK_H + gz), label="descend")
    plan.go(arm, 0.6, finger=grip, label="grip")
    plan.go(arm, 1.1, (sx, sy, plate_top + SHANK_L + gz + 0.015), label="lift it out")
    plan.go(arm, 2.8, (X0, PLATE_Y + hole_y, plate_top + SHANK_L + gz + 0.015), label="over its hole")
    plan.go(arm, 1.2, label="settle")
    plan.go(arm, 1.6, (X0, PLATE_Y + hole_y, plate_top + gz + 0.002), label="down through the hole")
    plan.go(arm, 0.6, label="settle")
    plan.go(arm, 0.6, finger=ready, label="let go")
    plan.go(arm, 0.7, (X0, PLATE_Y + hole_y, plate_top + SHANK_L + gz + 0.015), label="up")
    _park(plan, arm)


def demo(lot=None):
    lot = LOT if lot is None else lot
    holes = LOTS[lot]
    a_y, b_y = PLACES[os.environ.get("PARALLELS_DEMO_PLACES", lot)]  # debugging only: another lot's places, to show pins stop
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    plan.sync()
    _place_parallel(plan, "left", "parallel_a", a_y)  # one arm at a time: their wrists meet over the mat
    plan.sync()
    _place_parallel(plan, "right", "parallel_b", b_y)
    plan.sync()
    _place_plate(plan, "right")
    plan.sync()
    _place_pin(plan, "left", "pin_a", max(holes))
    plan.sync()
    _place_pin(plan, "right", "pin_b", min(holes))
    plan.go("right", 2.0, label="still")
    plan.sync()
    return plan.wps


def ik_errors(wps):
    worst_p = worst_r = 0.0
    bad = []
    for arm, ws in wps.items():
        q = np.array(READY, float)
        for w in ws:
            q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
            worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
            if pe > 1e-3 or re > 1e-2:
                bad.append((arm, w.label, tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1)))
    return worst_p, worst_r, bad


# -- self-check (no simulator): every lot's demo fits the run limit and IK reaches every waypoint -----------------------
if __name__ == "__main__":
    for name in LOTS:
        wps = demo(name)
        t = max(sum(w.duration for w in ws) for ws in wps.values())
        pe, re, bad = ik_errors(wps)
        print(f"lot {name}: demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:4])
        assert t * 20 < 3000 and not bad
