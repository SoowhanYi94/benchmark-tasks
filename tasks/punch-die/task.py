"""On the station table: a small press with a solid old die in it, and three more dies, each with one square pocket at a
different place. A loose guard over the ram hides its punch until it's lifted off. The robot must rack the old die, find where the
punch is, slide in the die whose pocket clears it, put the guard back, cycle the press, and move both hands away. Bringing
the punch down on metal, or cycling with the guard off, scores 0 for good. Lots A, B and C put the punch at a different place; PUNCH_LOT picks one (A by default).

Run it:  PUNCH_LOT=A inspect-robots run --embodiment yam_punch -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Rack the old die, change the press to the die that clears its punch, cycle it with its guard on, and move both hands away." --scorer task_score

`yam_punch` (punch_die/embodiment.py) spawns the press, guard and dies and reports each gripper's fingertip midpoint and the ram.
"""

import math
import os

import numpy as np

from punch_die import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
CX, CY = 0.31, 0.0  # the press bed's centre
ALU, DARK, BAY, GUARD_C = (0.80, 0.82, 0.85), (0.20, 0.21, 0.23), (0.85, 0.72, 0.15), (0.45, 0.47, 0.50)

PLACES = {"die_plus": 0.05, "die_mid": 0.0, "die_minus": -0.05}  # each die's pocket, across it, +y toward the left arm
LOTS = {"A": "die_plus", "B": "die_mid", "C": "die_minus"}  # the die that clears the punch
LOT = os.environ.get("PUNCH_LOT", "A").upper()
WANT = LOTS[LOT]
PUNCH_Y = PLACES[WANT]

# a die, in its own frame (origin at the plate's underside, under the pocket): 15 x 16 cm, 5 mm thick, a grip on top at its balance point
# (fingers on a grip are a hinge: a die balanced off it swings nose down as soon as it's lifted)
HX0, HX1, HY, T, POCKET = -0.135, 0.025, 0.075, 0.005, 0.02  # the pocket is 4 cm square, closed on all sides
GRIP = dict(x0=-0.074, x1=-0.044, half_y=0.015, z1=0.05)
DIE_MASS, OLD_MASS, MU = 0.31, 0.335, 0.3  # aluminium, the grips hollow
# the guard, in its own frame (origin at the centre of its underside): 8 x 19 cm, 6 mm thick, a grip at its middle
GUARD = dict(hx=0.04, hy=0.095, t=0.006, grip_x=0.015, grip_y=0.0125, z1=0.03, mass=0.10)


def die_boxes(pocket_y=None):
    b = {"grip": ((GRIP["x0"], -GRIP["half_y"], T), (GRIP["x1"], GRIP["half_y"], GRIP["z1"]))}
    if pocket_y is None:
        b["plate"] = ((HX0, -HY, 0.0), (HX1, HY, T))
        return b
    b["front"] = ((HX0, -HY, 0.0), (-POCKET, HY, T))
    b["back"] = ((POCKET, -HY, 0.0), (HX1, HY, T))
    b["side_a"] = ((-POCKET, -HY, 0.0), (POCKET, pocket_y - POCKET, T))
    b["side_b"] = ((-POCKET, pocket_y + POCKET, 0.0), (POCKET, HY, T))
    return b


GUARD_BOXES = {"plate": ((-GUARD["hx"], -GUARD["hy"], 0.0), (GUARD["hx"], GUARD["hy"], GUARD["t"])),
               "grip": ((-GUARD["grip_x"], -GUARD["grip_y"], GUARD["t"]), (GUARD["grip_x"], GUARD["grip_y"], GUARD["z1"]))}
PARTS = {"die_old": (die_boxes(), OLD_MASS, MU, DARK), **{n: (die_boxes(y), DIE_MASS, MU, ALU) for n, y in PLACES.items()}, "guard": (GUARD_BOXES, GUARD["mass"], 0.6, GUARD_C)}
DIES = tuple(n for n in PARTS if n != "guard")
ROW_X = 0.172  # the three dies wait in a row on the table, grips toward the arm bases
START = {"die_plus": (ROW_X, 0.0), "die_mid": (ROW_X, -0.16), "die_minus": (ROW_X, 0.16)}  # not in the order of their pockets
RACK = (0.285, -0.20)  # the mat's centre
PLATE_X = (HX0 + HX1) / 2  # a die's plate centre, in its own frame
RACK_T = 0.002
GUARD_HOME = (CX, CY, TOP + M.COLUMN["z1"])  # on the columns and ledges, over the ram
GUARD_ASIDE = (0.27, 0.27)
BED_Z = TOP + M.BED["z"]

OBJECTS = [
    Box("press_mech", pos=(CX, CY, TOP), size=(0.01, 0.01, 0.01)),
    Box("rack", pos=(*RACK, TOP + RACK_T / 2), size=(0.17, 0.17, RACK_T), color=BAY),
    Box("guard", pos=(*GUARD_HOME[:2], GUARD_HOME[2] + 0.0005), size=(0.01, 0.01, 0.01), mass=GUARD["mass"]),
    Box("die_old", pos=(CX, CY, BED_Z + 0.0005), size=(0.01, 0.01, 0.01), mass=OLD_MASS),
    *[Box(n, pos=(*START[n], TOP + 0.0005), size=(0.01, 0.01, 0.01), mass=DIE_MASS) for n in PLACES],
]

# -- scoring -----------------------------------------------------------------------------------------------------------
FLAT, SQUARE = math.radians(5), math.radians(5)
HOME_X, HOME_Y, HOME_Z = 0.004, 0.005, 0.003  # a die this near its home on the bed counts as in the press
RACK_XY, RACK_Z = 0.03, 0.01
STROKE, UP = 0.027, 0.003  # ram travel for a full stroke; within this of the top counts as back up
LET_GO, HANDS_CLEAR, STILL_S = 0.05, 0.08, 1.0
LIN, ANG = 0.01, 0.5
GUARD_XY, GUARD_Z = 0.02, 0.004
_trial = {"still": 0.0, "bad": False, "stroked": False}
_TOUCH = 0.0005  # a punch this near metal counts as touching it; the right die keeps 1 mm at the far edge of its play


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears `bad`."""
    _trial.update(still=0.0, bad=False, stroked=False)


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _flat(part):
    return _rot(part.quat_xyzw)[2, 2] >= math.cos(FLAT)


def _grip_pos(name, part):
    x = 0.0 if name == "guard" else (GRIP["x0"] + GRIP["x1"]) / 2
    z = GUARD["z1"] if name == "guard" else GRIP["z1"]
    return np.asarray(part.pos, float) + _rot(part.quat_xyzw) @ np.array((x, 0.0, z - 0.01))


def _away(objects, name, dist):
    return all(np.linalg.norm(np.asarray(objects[a].pos, float) - _grip_pos(name, objects[name])) >= dist for a in ("left_tcp", "right_tcp"))


def guard_on(objects):
    g = objects["guard"]
    return bool(abs(g.pos[0] - GUARD_HOME[0]) <= GUARD_XY and abs(g.pos[1] - GUARD_HOME[1]) <= GUARD_XY and abs(g.pos[2] - GUARD_HOME[2]) <= GUARD_Z and _flat(g) and _away(objects, "guard", LET_GO))


def racked(objects):
    d = objects["die_old"]
    c = np.asarray(d.pos, float) + _rot(d.quat_xyzw) @ np.array((PLATE_X, 0.0, 0.0))
    return bool(abs(c[0] - RACK[0]) <= RACK_XY and abs(c[1] - RACK[1]) <= RACK_XY and abs(d.pos[2] - TOP - RACK_T) <= RACK_Z and _flat(d) and _away(objects, "die_old", LET_GO))


def home(objects, name=None):
    """The die that clears the punch is in the press: on the bed, square, against the rear stop, and let go."""
    name = WANT if name is None else name
    d = objects[name]
    R = _rot(d.quat_xyzw)
    return bool(abs(d.pos[0] - CX) <= HOME_X and abs(d.pos[1] - CY) <= HOME_Y and abs(d.pos[2] - BED_Z) <= HOME_Z and _flat(d) and R[0, 0] >= math.cos(SQUARE) and _away(objects, name, LET_GO))


def punch_hits(name, die, ram_travel):
    """Is the punch, this far down, touching or through metal of this die? Any part of it, at any insertion."""
    tip = TOP + M.PUNCH["tip"] - ram_travel
    if tip >= die.pos[2] + T + _TOUCH or die.pos[2] >= TOP + M.PUNCH["top"] - ram_travel + _TOUCH:
        return False  # the die's plate is wholly below the tip, or wholly above the punch
    c = _rot(die.quat_xyzw).T @ (np.array((CX, CY + PUNCH_Y, tip)) - np.asarray(die.pos, float))
    h = M.PUNCH["half"] + _TOUCH
    return any(c[0] + h > lo[0] and c[0] - h < hi[0] and c[1] + h > lo[1] and c[1] - h < hi[1] for lo, hi in PARTS[name][0].values())


def hands_clear(objects):
    pad = np.array((CX, CY + (M.PAD["y0"] + M.PAD["y1"]) / 2, TOP + M.PAD["z1"]))
    return all(np.linalg.norm(np.asarray(objects[a].pos, float) - pad) >= HANDS_CLEAR for a in ("left_tcp", "right_tcp")) and all(_away(objects, n, HANDS_CLEAR) for n in PARTS)


def _resting(o):
    return np.linalg.norm(o.lin_vel) <= LIN and np.linalg.norm(o.ang_vel) <= ANG


def note_step(objects, dt):
    """Called by the embodiment after every control step: violations, the stroke, and the clock for all at rest with hands clear."""
    travel = objects["ram"].pos[0]
    if any(punch_hits(n, objects[n], travel) for n in DIES) or (travel >= STROKE and not guard_on(objects)):
        _trial["bad"] = True
    if travel >= STROKE and home(objects) and racked(objects) and guard_on(objects):
        _trial["stroked"] = True
    still = travel <= UP and abs(objects["ram"].lin_vel[0]) <= LIN and all(_resting(objects[n]) for n in PARTS) and hands_clear(objects)
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. Bringing the punch down on metal, or a full stroke with the guard off, is
    0 for good, and so is a part off the table. Otherwise 0.1 with the old die let go in the rack; 0.5 with the right die also let
    go in the press; 0.8 once the press has made a full stroke on it with the guard on, and the guard is on, the ram is back up and
    both hands are clear; 1.0 if all that's been at rest for the last 1 s."""
    if _trial["bad"] or any(objects[n].pos[2] < TOP - 0.01 for n in PARTS) or not racked(objects):
        return 0.0
    if not home(objects):
        return 0.1
    if not (_trial["stroked"] and guard_on(objects) and objects["ram"].pos[0] <= UP and hands_clear(objects)):
        return 0.5
    return 1.0 if _trial["still"] >= STILL_S else 0.8


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SEGMENT = 0.03
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.33, TOP + 0.18
GX = (GRIP["x0"] + GRIP["x1"]) / 2
GZ = GRIP["z1"] - 0.008  # the fingertips' height on a grip, above the die's underside
LIFT = 0.085  # the underside's height off the table, carrying
STAGE_X = CX - 0.045  # a die set on the bed here is clear of the rails and the ram


class _Plan:
    def __init__(self):
        self.wps = {a: [] for a in SIDE}
        self.t = {a: 0.0 for a in SIDE}
        self.last = {}
        for a in SIDE:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None):
        p0, f0, y0 = self.last[arm]
        p = p0 if pos is None else tuple(float(v) for v in pos)
        f = f0 if finger is None else finger
        y = y0 if yaw is None else yaw
        # short segments: a long waypoint moves in joint space, so the hand arcs and tilts on the way
        n = 1 if label == "away" else max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / SEGMENT))  # going away, nothing is held and the arc is harmless
        for k in range(1, n + 1):
            q = tuple(float(v) for v in np.asarray(p0) + (np.asarray(p) - np.asarray(p0)) * k / n)
            self.wps[arm].append(Waypoint(dur / n, q, f0 + (f - f0) * k / n, label, y0 + (y - y0) * k / n))
        self.last[arm] = (p, f, y)
        self.t[arm] += dur

    def sync(self):
        end = max(self.t.values())
        for a in SIDE:
            if self.t[a] < end - 1e-9:
                self.go(a, end - self.t[a], label="wait")


def _fingers(width):
    return -FINGER_PER_M * width


READY_W, GRIP_W = _fingers(2 * GRIP["half_y"] + 0.03), _fingers(2 * GRIP["half_y"] - 0.004)


def _park(plan, arm):
    plan.go(arm, 1.4, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


def _at(x, y, under):
    """The fingertips on a die's grip, with the die's centre at (x, y) and its underside at this height."""
    return (x + GX, y, under + GZ)


def _pick(plan, arm, x, y, under, label):
    plan.go(arm, 1.6, _at(x, y, under + 0.05), READY_W, f"{label}: above its grip", 0.0)
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 0.9, _at(x, y, under), label="descend")
    plan.go(arm, 0.6, finger=GRIP_W, label="grip")
    plan.go(arm, 0.3, label="settle")


def _guard_at(x, y, under):
    return (x, y, under + GUARD["z1"] - 0.012)


def _lift_guard_off(plan):
    x, y, z = GUARD_HOME
    r, g = _fingers(2 * GUARD["grip_y"] + 0.03), _fingers(2 * GUARD["grip_y"] - 0.004)
    plan.go("left", 1.6, _guard_at(x, y, z + 0.03), r, "guard: above its grip", 0.0)
    plan.go("left", 0.4, label="settle")
    plan.go("left", 0.8, _guard_at(x, y, z), label="descend")
    plan.go("left", 0.6, finger=g, label="grip")
    plan.go("left", 0.9, _guard_at(x, y, z + 0.03), label="lift it off")
    plan.go("left", 1.6, _guard_at(*GUARD_ASIDE, z + 0.03), label="to one side")
    plan.go("left", 1.2, _guard_at(*GUARD_ASIDE, TOP + 0.001), label="set down")
    plan.go("left", 0.5, finger=r, label="let go")
    plan.go("left", 0.8, _guard_at(*GUARD_ASIDE, TOP + 0.05), label="up")
    _park(plan, "left")
    plan.sync()


def _put_guard_back(plan):
    x, y, z = GUARD_HOME
    r, g = _fingers(2 * GUARD["grip_y"] + 0.03), _fingers(2 * GUARD["grip_y"] - 0.004)
    plan.go("left", 1.6, _guard_at(*GUARD_ASIDE, TOP + 0.05), r, "guard: above its grip", 0.0)
    plan.go("left", 0.4, label="settle")
    plan.go("left", 0.8, _guard_at(*GUARD_ASIDE, TOP), label="descend")
    plan.go("left", 0.6, finger=g, label="grip")
    plan.go("left", 1.2, _guard_at(*GUARD_ASIDE, z + 0.03), label="lift")
    plan.go("left", 1.6, _guard_at(x, y, z + 0.03), label="over the ram")
    plan.go("left", 0.5, label="settle")
    plan.go("left", 0.9, _guard_at(x, y, z + 0.001), label="set it on")
    plan.go("left", 0.5, finger=r, label="let go")
    plan.go("left", 0.8, _guard_at(x, y, z + 0.03), label="up")
    plan.sync()


def _stroke(plan):
    px, py, z = CX, CY + (M.PAD["y0"] + M.PAD["y1"]) / 2, TOP + M.PAD["z1"]
    plan.go("left", 1.2, (px, py, z + 0.05), FINGER_CLOSED, "above the ram's pad", 0.0)
    plan.go("left", 0.4, label="settle")
    plan.go("left", 0.8, (px, py, z + 0.016), label="onto it")
    plan.go("left", 2.2, (px, py, z + 0.016 - M.TRAVEL - 0.03), label="press")  # past the stop: the soft arm's push grows with how far short it is
    plan.go("left", 0.8, label="hold")
    plan.go("left", 1.6, (px, py, z + 0.05), label="let it back up")
    _park(plan, "left")
    plan.sync()


def demo(lot=None):
    lot = LOT if lot is None else lot
    want = os.environ.get("PUNCH_DEMO_DIE", LOTS[lot])  # debugging only: another die, to show a crash
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    plan.sync()
    bed = BED_Z
    _lift_guard_off(plan)  # the punch shows, the same for every lot
    plan.go("left", 1.5, label="look")
    plan.sync()
    if os.environ.get("PUNCH_DEMO_STOP") == "1":
        return plan.wps
    # the old die: straight out along the rails, up, across, onto the rack
    _pick(plan, "right", CX, CY, bed, "old die")
    plan.go("right", 1.6, _at(STAGE_X, CY, bed), label="slide it out")
    plan.go("right", 1.0, _at(STAGE_X, CY, TOP + LIFT), label="lift")
    plan.go("right", 2.0, _at(STAGE_X, RACK[1], TOP + LIFT), label="clear of the press")  # sideways first: its rear edge is close to the columns
    plan.go("right", 0.9, _at(RACK[0] - PLATE_X, RACK[1], TOP + LIFT), label="over the rack")
    plan.go("right", 0.5, label="settle")
    plan.go("right", 0.9, _at(RACK[0] - PLATE_X, RACK[1], TOP + RACK_T + 0.001), label="set down")
    plan.go("right", 0.5, finger=READY_W, label="let go")
    plan.go("right", 0.8, _at(RACK[0] - PLATE_X, RACK[1], TOP + 0.06), label="up")
    _park(plan, "right")
    plan.sync()
    # the die whose pocket is under the punch; the near arm carries it, in front of the row so it passes over the others
    sx, sy = START[want]
    arm = "left" if sy > 0.08 else "right"
    _pick(plan, arm, sx, sy, TOP, want)
    plan.go(arm, 1.0, _at(sx, sy, TOP + LIFT), label="lift")
    plan.go(arm, 1.4, _at(STAGE_X, sy, TOP + LIFT), label="forward")
    plan.go(arm, 1.6, _at(STAGE_X, CY, TOP + LIFT), label="over the bed")
    plan.go(arm, 0.6, label="settle")
    plan.go(arm, 1.0, _at(STAGE_X, CY, bed + 0.001), label="set it on the bed")
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 2.0, _at(CX + 0.012, CY, bed + 0.001), label="slide it home")  # past the stop, a little
    plan.go(arm, 0.5, label="hold")
    plan.go(arm, 0.5, finger=READY_W, label="let go")
    plan.go(arm, 0.8, _at(CX, CY, bed + 0.05), label="up")
    _park(plan, arm)
    plan.sync()
    _put_guard_back(plan)
    _stroke(plan)
    plan.go("left", 2.0, label="still")
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
                bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1)))
    return worst_p, worst_r, bad


# -- self-check (no simulator): every lot's demo fits the run limit and IK reaches every waypoint -----------------------
if __name__ == "__main__":
    for name in LOTS:
        wps = demo(name)
        t = max(sum(w.duration for w in ws) for ws in wps.values())
        pe, re, bad = ik_errors(wps)
        print(f"lot {name}: demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:4])
        assert t * 20 < 3000 and not bad
