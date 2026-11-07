"""A rollover fixture stands on an assembly bench: a cradle on a horizontal axle, turned by a crank. A housing has to go in it recess down,
be retained by the sliding bar, and be turned over and locked, so that a wear pad can be fitted in the recess, now facing up. Turning
without the bar closed loses the housing. Fitting the pad before the bolt is home, or the housing leaving the cradle once turned, scores 0 for good.

Run it:  inspect-robots run --embodiment yam_trunnion -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Lay the housing in the rollover fixture with its orange recess down, slide the blue bar across it, turn the fixture over, push the red bolt home, fit the white wear pad in the recess, and move both hands away." --scorer task_score

`yam_trunnion` (trunnion_fixture/embodiment.py) spawns the fixture and parts and reports each gripper's fingertip midpoint and the fixture's joints.
"""

import math
import os

import numpy as np

from trunnion_fixture import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
AX = 0.29  # the axle's x, on y=0
AXLE = TOP + M.AXLE
ALU, ORANGE, WHITE = (0.80, 0.82, 0.85), (0.90, 0.45, 0.08), (0.93, 0.93, 0.90)

PART = dict(half_x=0.04, half_y=0.035, h=0.03, rec_x=0.03, rec_y=0.02, depth=0.008, mass=0.40)  # 8 x 7 x 3 cm aluminium; origin: centre of the underside
PAD = dict(half_x=0.025, half_y=0.015, h=0.03, mass=0.063)  # 5 x 3 x 3 cm acetal; origin: centre of its underside


def _part_boxes():
    p = PART
    x, y, rx, ry, d = p["half_x"], p["half_y"], p["rec_x"], p["rec_y"], p["depth"]
    return {"n": ((-x, ry, 0.0), (x, y, d), ORANGE), "s": ((-x, -y, 0.0), (x, -ry, d), ORANGE), "e": ((rx, -ry, 0.0), (x, ry, d), ORANGE), "w": ((-x, -ry, 0.0), (-rx, ry, d), ORANGE),
            "floor": ((-x, -y, d), (x, y, d + 0.003), ORANGE), "body": ((-x, -y, d + 0.003), (x, y, p["h"]), ALU)}


PARTS = {"housing": (_part_boxes(), PART["mass"], 0.5),
         "pad": ({"block": ((-PAD["half_x"], -PAD["half_y"], 0.0), (PAD["half_x"], PAD["half_y"], PAD["h"]), WHITE)}, PAD["mass"], 0.5)}
PART_START, PAD_START = (0.17, 0.24), (0.27, 0.26)

OBJECTS = [
    Box(FIXTURE_NAME := "fixture", pos=(AX, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    Box("housing", pos=(*PART_START, TOP + 0.0005), size=(0.01, 0.01, 0.01), mass=PART["mass"]),
    Box("pad", pos=(*PAD_START, TOP + 0.0005), size=(0.01, 0.01, 0.01), mass=PAD["mass"]),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
SEAT_W = (M.LEDGE["w1"] - 0.002, M.BAR["w0"] - PART["h"] + 0.002)  # the underside, in the cradle's axes, between the ledges and the bar
SEAT_U, SEAT_V, SEAT_TILT = 0.012, 0.011, math.radians(10)
BAR_CLOSED, BOLT_HOME, OVER, ROLLING = M.BAR["travel"] - 0.005, 0.012, math.radians(6), math.radians(30)
PAD_XY, PAD_Z, FLAT = 0.007, 0.005, math.radians(10)
LIN, ANG, RATE, STILL_S = 0.01, 0.5, math.radians(3), 1.0
_trial = {"still": 0.0, "bad": False, "why": ""}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears `bad`."""
    _trial.update(still=0.0, bad=False, why="")


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _cradle(objects):
    """The cradle's rotation: its axes in the world."""
    th = objects["fixture"].pos[0]
    c, s = math.cos(th), math.sin(th)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def part_offset(objects):
    """The housing's origin (the centre of its underside) in the cradle's axes from the axle."""
    return _cradle(objects).T @ (np.asarray(objects["housing"].pos, float) - np.array((AX, 0.0, AXLE)))


def part_tilt(objects):
    """The angle between the housing's own up and the cradle's: near 180 degrees means it went in upside down."""
    return math.acos(float(np.clip(_rot(objects["housing"].quat_xyzw)[:, 2] @ _cradle(objects)[:, 2], -1.0, 1.0)))


def part_in(objects):
    o = part_offset(objects)
    return bool(abs(o[0]) <= SEAT_U and abs(o[1]) <= SEAT_V and SEAT_W[0] <= o[2] <= SEAT_W[1] and part_tilt(objects) <= SEAT_TILT)


def _clear_of(objects, part, half_x, half_y, z0, z1):
    """Both fingertip midpoints this far from a part's box, in its own frame: open fingers sit 4.7 cm out, so 6 cm means no finger's on it."""
    R = _rot(part.quat_xyzw)
    for a in ("left_tcp", "right_tcp"):
        d = R.T @ (np.asarray(objects[a].pos, float) - np.asarray(part.pos, float))
        q = d - np.clip(d, (-half_x, -half_y, z0), (half_x, half_y, z1))
        if np.linalg.norm(q) < 0.06:
            return False
    return True


def part_seated(objects):
    """In the cradle, the right way up, and not being held: once the bar's closed the bar holds it, whatever the hands are doing nearby."""
    return part_in(objects) and (bar_closed(objects) or _clear_of(objects, objects["housing"], PART["half_x"], PART["half_y"], 0.0, PART["h"]))


def pad_offset(objects):
    """The pad's origin in the housing's own frame."""
    h = objects["housing"]
    return _rot(h.quat_xyzw).T @ (np.asarray(objects["pad"].pos, float) - np.asarray(h.pos, float))


def pad_in(objects):
    """In the recess, resting on its floor, its own up pointing away from the floor. Judged in the housing's frame."""
    o = pad_offset(objects)
    up = _rot(objects["pad"].quat_xyzw)[:, 2] @ -_rot(objects["housing"].quat_xyzw)[:, 2]
    return bool(abs(o[0]) <= PAD_XY and abs(o[1]) <= PAD_XY and abs(o[2] - PART["depth"]) <= PAD_Z and up >= math.cos(FLAT))


def pad_seated(objects):
    return pad_in(objects) and _clear_of(objects, objects["pad"], PAD["half_x"], PAD["half_y"], 0.0, PAD["h"])


def bar_closed(objects):
    return objects["fixture"].pos[1] >= BAR_CLOSED


def bolt_home(objects):
    return objects["fixture"].pos[2] >= BOLT_HOME


def turned_over(objects):
    return abs(objects["fixture"].pos[0] - math.pi) <= OVER


def _knob(objects):
    th = objects["fixture"].pos[0]
    return np.array((AX - M.CRANK["r"] * math.cos(th), M.CRANK["y"], AXLE + M.CRANK["r"] * math.sin(th)))


def hands_clear(objects):
    spots = (_knob(objects), np.array((AX, M.BOLT["end"] - objects["fixture"].pos[2], TOP + M.BOLT["lug_top"])))
    return (all(np.linalg.norm(np.asarray(objects[a].pos, float) - s) >= 0.06 for a in ("left_tcp", "right_tcp") for s in spots)
            and _clear_of(objects, objects["housing"], PART["half_x"], PART["half_y"], 0.0, PART["h"])
            and _clear_of(objects, objects["pad"], PAD["half_x"], PAD["half_y"], 0.0, PAD["h"]))


def _latch(why):
    if not _trial["bad"]:
        _trial.update(bad=True, why=why)


def note_step(objects, dt):
    """Called by the embodiment after every control step: the latched violations, and the clock for all at rest."""
    f = objects["fixture"]
    if f.pos[0] > ROLLING:
        if not part_in(objects):
            _latch("turned with the housing not in the cradle")
        if not bar_closed(objects):
            _latch("turned with the bar open")
    if pad_in(objects) and not bolt_home(objects):
        _latch("pad fitted before the bolt was home")
    if any(objects[n].pos[2] < TOP - 0.01 for n in PARTS):
        _latch("a part left the table")
    still = (abs(f.lin_vel[0]) < RATE and abs(f.lin_vel[1]) < LIN and abs(f.lin_vel[2]) < LIN and hands_clear(objects)
             and all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in PARTS))
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. 0 for good if the fixture turns past 30 degrees without the housing in it and the bar
    closed, if the pad is in the recess before the bolt is home, or if a part leaves the table. Otherwise a ladder: 0.2 housing in, the right
    way up, let go; 0.4 and the bar closed; 0.6 and turned over within 6 degrees; 0.8 and the bolt home; 0.9 and the pad in the recess, let go;
    1.0 and both hands clear of everything and all at rest for the last 1 s."""
    if _trial["bad"] or not part_seated(objects):
        return 0.0
    if not bar_closed(objects):
        return 0.2
    if not turned_over(objects):
        return 0.4
    if not bolt_home(objects):
        return 0.6
    if not pad_seated(objects):
        return 0.8
    return 1.0 if hands_clear(objects) and _trial["still"] >= STILL_S else 0.9


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.03
KNOB_YAW = math.pi / 2  # the right hand's fingers close along x, across the sleeve's flats


def _fingers(width):
    return -FINGER_PER_M * width


PART_R, PART_G = _fingers(2 * PART["half_y"] + 0.016), _fingers(2 * PART["half_y"] - 0.011)
PAD_R, PAD_G = _fingers(2 * PAD["half_y"] + 0.02), _fingers(2 * PAD["half_y"] - 0.006)
KNOB_R, KNOB_G = _fingers(2 * M.CRANK["half_x"] + 0.03), _fingers(2 * M.CRANK["half_x"] - 0.006)
PART_GZ, PAD_GZ, KNOB_GZ = 0.02, 0.02, 0.003
CARRY = TOP + 0.125  # hand height carrying the housing over the cradle
SIDE_X = 0.17  # parts come round the fixture's near side, clear of its pillars


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
        n = 1 if label == "away" else max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / SEGMENT))
        for k in range(1, n + 1):
            q = tuple(float(v) for v in np.asarray(p0) + (np.asarray(p) - np.asarray(p0)) * k / n)
            self.wps[arm].append(Waypoint(dur / n, q, f0 + (f - f0) * k / n, label, y0 + (y - y0) * k / n))
        self.last[arm] = (p, f, y)
        self.t[arm] += dur

    def until(self, arm, t):
        if t > self.t[arm] + 1e-9:
            self.go(arm, t - self.t[arm], label="wait")
        assert t >= self.t[arm] - 1e-6, (arm, t, self.t[arm])


def _park(plan, arm):
    plan.go(arm, 1.4, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


def knob_at(th, lift=KNOB_GZ):
    """Where the hand holds the sleeve, the crank at this roll."""
    return AX - M.CRANK["r"] * math.cos(th), M.CRANK["y"], AXLE + M.CRANK["r"] * math.sin(th) + lift


def load_part(plan):
    arm = "left"
    x, y = PART_START
    plan.go(arm, 1.6, (x, y, TOP + PART["h"] + 0.04), PART_R, "above the housing", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, TOP + PART_GZ), label="descend")
    plan.go(arm, 0.8, finger=PART_G, label="grip")
    plan.go(arm, 1.0, (x, y, CARRY), label="lift")
    plan.go(arm, 2.4, (SIDE_X, 0.0, CARRY), label="round the fixture")
    plan.go(arm, 1.6, (AX, 0.0, CARRY), label="over the cradle")
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 1.6, (AX, 0.0, AXLE + M.LEDGE["w1"] + 0.002 + PART_GZ), label="lay it in")
    plan.go(arm, 0.5, label="settle")
    plan.go(arm, 0.6, finger=PART_R, label="let go")
    plan.go(arm, 1.0, (AX, 0.0, CARRY), label="up")
    plan.go(arm, 1.4, (SIDE_X, 0.04, CARRY), label="back")
    _park(plan, arm)


def slide_bar(plan):
    """The closed right fingers behind the bar's lug, pushing along +y, aiming past the bar's stop."""
    arm = "right"
    br = M.BAR
    z = AXLE + br["lug_w"] - 0.012
    face = br["park"] - br["half_v"] - br["lug_v"]  # the lug's near face, parked
    plan.go(arm, 1.6, (AX, face - 0.035, z + 0.03), FINGER_CLOSED, "behind the bar's lug", 0.0)
    plan.go(arm, 0.6, (AX, face - 0.035, z), label="down")
    plan.go(arm, 4.0, (AX, face + br["travel"] - 0.015 + 0.015, z), label="slide the bar across")
    plan.go(arm, 0.4, label="hold")
    plan.go(arm, 2.0, (AX, face - 0.02, z), label="back off")
    plan.go(arm, 0.6, (AX, face - 0.02, z + 0.03), label="up")


def roll(plan, to=186.0, dur=10.0):  # aims past 180: the sleeve slips about 4 mm in the fingers on the last, vertical, stretch
    """The right hand takes the sleeve and carries it over the top, staying square: the sleeve turns on its pin."""
    arm = "right"
    plan.go(arm, 1.6, (*knob_at(0.0)[:2], AXLE + 0.08), KNOB_R, "above the knob", KNOB_YAW)
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 1.0, knob_at(0.0), label="down onto the knob")
    plan.go(arm, 0.8, finger=KNOB_G, label="grip the knob")
    plan.go(arm, 0.3, label="settle")
    n = 24
    for i in range(1, n + 1):
        plan.go(arm, dur / n, knob_at(math.radians(to) * i / n), label="over the top")


def release_knob(plan, at=186.0):
    arm = "right"
    plan.go(arm, 0.6, finger=KNOB_R, label="let go of the knob")
    x, y, z = knob_at(math.radians(at))
    plan.go(arm, 1.0, (x, y, z + 0.08), label="up")
    _park(plan, arm)


def push_bolt(plan):
    """The closed left fingers behind the bolt's lug, pushing along -y, aiming past its stop."""
    arm = "left"
    bo = M.BOLT
    z = TOP + bo["lug_top"] - 0.012
    plan.go(arm, 1.6, (AX, bo["end"] + 0.035, z + 0.03), FINGER_CLOSED, "behind the bolt", 0.0)
    plan.go(arm, 0.6, (AX, bo["end"] + 0.035, z), label="down")
    plan.go(arm, 2.0, (AX, bo["end"] - bo["travel"], z), label="push the bolt home")
    plan.go(arm, 0.4, label="hold")
    plan.go(arm, 1.0, (AX, bo["end"] + 0.03, z), label="back off")
    plan.go(arm, 0.6, (AX, bo["end"] + 0.03, z + 0.03), label="up")


def fit_pad(plan):
    arm = "left"
    x, y = PAD_START
    floor = AXLE - M.BAR["w0"] + PART["h"] - PART["depth"]  # the recess's floor, the housing resting on the bar, turned over
    carry = floor + PART["depth"] + 0.016 + PAD_GZ
    plan.go(arm, 1.6, (x, y, TOP + PAD["h"] + 0.04), PAD_R, "above the pad", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, TOP + PAD_GZ), label="descend")
    plan.go(arm, 0.6, finger=PAD_G, label="grip")
    plan.go(arm, 1.0, (x, y, carry), label="lift")
    plan.go(arm, 2.0, (SIDE_X, 0.0, carry), label="round the fixture")
    plan.go(arm, 1.6, (AX, 0.0, carry), label="over the recess")
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 1.4, (AX, 0.0, floor + 0.002 + PAD_GZ), label="into the recess")
    plan.go(arm, 0.5, label="settle")
    plan.go(arm, 0.6, finger=PAD_R, label="let go")
    plan.go(arm, 1.0, (AX, 0.0, carry + 0.01), label="up")
    plan.go(arm, 1.2, (SIDE_X, 0.04, carry + 0.01), label="back")
    _park(plan, arm)


def demo():
    plan = _Plan()
    mode = os.environ.get("TRUNNION_DEMO", "")  # debugging only: arc, nobar, nobolt, empty, nopad, padfirst
    for a in SIDE:
        _park(plan, a)
    if mode != "empty":
        load_part(plan)
    plan.until("right", plan.t["left"])
    if mode != "nobar":
        slide_bar(plan)
    roll(plan)
    plan.until("left", plan.t["right"])
    if mode == "arc":
        plan.go("right", 4.0, label="hold")
        return _finish(plan)
    if mode == "nobolt":  # let go with the bolt withdrawn: the cradle runs on to its far stop
        release_knob(plan)
        return _finish(plan)
    if mode == "padfirst":  # the right hand keeps holding the crank while the pad goes in
        fit_pad(plan)
        push_bolt(plan)
        plan.until("right", plan.t["left"])
        release_knob(plan)
        return _finish(plan)
    push_bolt(plan)
    plan.until("right", plan.t["left"])
    release_knob(plan)
    plan.until("left", plan.t["right"])
    if mode != "nopad":
        fit_pad(plan)
    return _finish(plan)


def _finish(plan):
    plan.go("left", 2.0, label="still")
    end = max(plan.t.values())
    for a in SIDE:
        plan.until(a, end)
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


# -- self-check (no simulator): the demo fits the run limit and IK reaches every waypoint -------------------------------
if __name__ == "__main__":
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad = ik_errors(wps)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:6])
    assert t * 20 < 3000 and not bad
