"""A turbine fitter's job: four identical seal segments have to run round a covered ring groove, in through the gate and round clockwise,
and the gate is then closed with its bar. A step on the groove's floor can only be crossed with a segment lifted by its knob, and neither
arm reaches round the whole ring, so the first two segments are carried by one arm and taken on by the other.

Run it:  inspect-robots run --embodiment yam_seal -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Run all four seal segments in through the gate and clockwise round the ring's groove until all four lie under the cover, lifting a segment over the floor step at the yellow mark where it has to cross it, then lay the orange bar in the gate and move both hands away." --scorer task_score

`yam_seal` (seal_segments/embodiment.py) spawns the ring and parts and reports each gripper's fingertip midpoint.
"""

import math
import os

import numpy as np

from seal_segments import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
CX, CY = 0.27, 0.0  # the ring's centre
R = M.R
GATE, GATE_HALF = 144.0, 40.0  # degrees from +x about the centre: the cover's cut away from 104 to 184
STEP, STEP_HALF = -36.0, 2.0  # the floor's step
SEG_DEG = 66.0
HALF = SEG_DEG / 2
STOP_FACE = GATE + GATE_HALF - 360.0  # -176: the stop's face, where a clockwise run ends
DROP = GATE - 3.0  # 141: a segment's laid in here, clear of the stop and the cover's edge
EXIT_FACE = GATE - GATE_HALF  # 104: a segment clockwise of this is wholly under the cover
BLUE, ORANGE = (0.10, 0.40, 0.85), (0.92, 0.45, 0.08)
ALU = (0.72, 0.74, 0.77)
AL_DENSITY = 2700.0

SEG = dict(r0=0.088, r1=0.112, h=0.014, stem=0.012, stem_top=0.046, knob=0.03, knob_top=0.068, n=7, mass=0.17)
BAR = dict(r0=0.070, r1=0.130, half_deg=34.0, t=0.006, dog_r0=0.086, dog_r1=0.114, dog_deg=25.0, dog_half=2.0, dog_h=0.026, stem_top=0.016, knob=0.03, knob_top=0.040, mass=0.205)


def _seg_boxes():
    """Own frame: origin under the knob at the mean radius on the underside, x radially outward, y along the arc, z up."""
    out = {}
    step = math.radians(SEG_DEG / SEG["n"])
    for i in range(SEG["n"]):
        a0 = math.radians(-HALF) + i * step
        c, size, yaw = M.arc_box(SEG["r0"], SEG["r1"], a0, a0 + step, 0.0, SEG["h"])
        out[f"arc_{i}"] = ((c[0] - R, c[1], c[2]), size, yaw, "body")
    s, k = SEG["stem"], SEG["knob"]
    out["stem"] = ((0.0, 0.0, (SEG["h"] + SEG["stem_top"]) / 2), (s, s, SEG["stem_top"] - SEG["h"]), 0.0, "body")
    out["knob"] = ((0.0, 0.0, (SEG["stem_top"] + SEG["knob_top"]) / 2), (k, k, SEG["knob_top"] - SEG["stem_top"]), 0.0, "knob")
    return out


def _bar_boxes():
    """Own frame: origin on the plate's underside at the mean radius, x radially outward, y along the arc, z up."""
    out = {}
    n = 8
    span = 2 * BAR["half_deg"]
    for i in range(n):
        a0 = math.radians(-BAR["half_deg"] + i * span / n)
        c, size, yaw = M.arc_box(BAR["r0"], BAR["r1"], a0, a0 + math.radians(span / n), 0.0, BAR["t"])
        out[f"plate_{i}"] = ((c[0] - R, c[1], c[2]), size, yaw, "knob")
    for tag, d in (("a", -BAR["dog_deg"]), ("b", BAR["dog_deg"])):
        c, size, yaw = M.arc_box(BAR["dog_r0"], BAR["dog_r1"], math.radians(d - BAR["dog_half"]), math.radians(d + BAR["dog_half"]), -BAR["dog_h"], 0.0)
        out[f"dog_{tag}"] = ((c[0] - R, c[1], c[2]), size, yaw, "knob")
    s, k = SEG["stem"], BAR["knob"]
    out["stem"] = ((0.0, 0.0, (BAR["t"] + BAR["stem_top"]) / 2), (s, s, BAR["stem_top"] - BAR["t"]), 0.0, "knob")
    out["knob"] = ((0.0, 0.0, (BAR["stem_top"] + BAR["knob_top"]) / 2), (k, k, BAR["knob_top"] - BAR["stem_top"]), 0.0, "knob")
    return out


MUS = {"body": M.MU, "knob": 0.8}
PARTS = {**{f"seg_{c}": (_seg_boxes(), SEG["mass"], MUS, {"body": ALU, "knob": BLUE}) for c in "abcd"},
         "bar": (_bar_boxes(), BAR["mass"], MUS, {"body": ALU, "knob": ORANGE})}
SEGMENTS = tuple(f"seg_{c}" for c in "abcd")
# Laid out with the radial direction along y, so a hand with its fingers closing along y picks them up
SEG_START = {"seg_a": (0.28, 0.29), "seg_b": (0.14, 0.29), "seg_c": (0.28, 0.20), "seg_d": (0.14, 0.20)}
BAR_START = (0.20, -0.27)
START_YAW = math.pi / 2

OBJECTS = [
    Box("ring", pos=(CX, CY, TOP), size=(0.01, 0.01, 0.01)),
    *[Box(n, pos=(*SEG_START[n], TOP + 0.0005), size=(0.01, 0.01, 0.01), yaw=START_YAW, mass=SEG["mass"]) for n in SEGMENTS],
    Box("bar", pos=(*BAR_START, TOP + BAR["dog_h"] + 0.0005), size=(0.01, 0.01, 0.01), yaw=START_YAW, mass=BAR["mass"]),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
IN_R, SEAT_Z, FLAT, IN_YAW, PAST_TOL, MIN_APART = 0.007, 0.004, math.radians(5), 8.0, 1.5, SEG_DEG - 8.0
BAR_ANGLE, BAR_R, BAR_Z = 6.0, 0.009, 0.004
LIN, ANG, STILL_S = 0.01, 0.5, 1.0
_trial = {"still": 0.0, "bad": False}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears `bad`."""
    _trial.update(still=0.0, bad=False)


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def ring_frame(part):
    """A part in the ring's frame: its angle (deg), radius and height above the groove's floor, and how far its own radial axis is from the radius through it (deg)."""
    d = np.asarray(part.pos, float) - np.array((CX, CY, TOP + M.FLOOR))
    angle = math.degrees(math.atan2(d[1], d[0]))
    Rm = _rot(part.quat_xyzw)
    own = math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))
    return angle, math.hypot(d[0], d[1]), d[2], _wrap(own - angle)


def _flat(part):
    return _rot(part.quat_xyzw)[2, 2] >= math.cos(FLAT)


def _clear_of(objects, part, lo, hi):
    """Both fingertip midpoints this far from a part's box, in its own frame: open fingers sit 4.7 cm out, so 6 cm means no finger's on it."""
    Rm = _rot(part.quat_xyzw)
    for a in ("left_tcp", "right_tcp"):
        d = Rm.T @ (np.asarray(objects[a].pos, float) - np.asarray(part.pos, float))
        if np.linalg.norm(d - np.clip(d, lo, hi)) < 0.06:
            return False
    return True


_y_half = (SEG["r1"]) * math.sin(math.radians(HALF))
SEG_BOX = (np.array((SEG["r0"] * math.cos(math.radians(HALF)) - R, -_y_half, 0.0)), np.array((SEG["r1"] - R, _y_half, SEG["knob_top"])))
_by = BAR["r1"] * math.sin(math.radians(BAR["half_deg"]))
BAR_BOX = (np.array((BAR["r0"] * math.cos(math.radians(BAR["half_deg"])) - R, -_by, -BAR["dog_h"])), np.array((BAR["r1"] - R, _by, BAR["knob_top"])))


def under_cover(angle):
    """Wholly past the gate: from the gate's clockwise edge round to the stop."""
    u = (EXIT_FACE - HALF - angle) % 360.0  # how far clockwise past the first place it clears the gate
    span = EXIT_FACE - HALF - (STOP_FACE + HALF)
    return u <= span + PAST_TOL or u >= 360.0 - PAST_TOL


def seg_seated(objects, n):
    p = objects[n]
    angle, r, z, skew = ring_frame(p)
    return bool(abs(r - R) <= IN_R and abs(z) <= SEAT_Z and abs(skew) <= IN_YAW and _flat(p) and under_cover(angle) and _clear_of(objects, p, *SEG_BOX))


def segments_in(objects):
    """How many segments lie seated under the cover and let go. They're identical, so it doesn't matter which, and two in one place can't count twice."""
    angles = sorted(ring_frame(objects[n])[0] for n in SEGMENTS if seg_seated(objects, n))
    count, last = 0, None
    for a in angles:
        if last is None or a - last >= MIN_APART:
            count, last = count + 1, a
    return count


def bar_seated(objects):
    p = objects["bar"]
    angle, r, z, skew = ring_frame(p)
    return bool(abs(_wrap(angle - GATE)) <= BAR_ANGLE and abs(r - R) <= BAR_R and abs(z - (M.GROOVE["h"] + M.COVER_T)) <= BAR_Z and abs(skew) <= BAR_ANGLE and _flat(p) and _clear_of(objects, p, *BAR_BOX))


def hands_clear(objects):
    return all(_clear_of(objects, objects[n], *SEG_BOX) for n in SEGMENTS) and _clear_of(objects, objects["bar"], *BAR_BOX)


def note_step(objects, dt):
    """Called by the embodiment after every control step: a part off the table, and the clock for all at rest."""
    if any(objects[n].pos[2] < TOP - 0.01 for n in PARTS):
        _trial["bad"] = True
    still = all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in PARTS) and hands_clear(objects)
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. 0 for good if a part leaves the table. Otherwise 0.2 for each segment seated
    under the cover and let go; 0.9 with all four and the bar seated and let go; 1.0 with all at rest and both hands clear for the last 1 s."""
    if _trial["bad"]:
        return 0.0
    n = segments_in(objects)
    if n < 4 or not bar_seated(objects):
        return round(0.2 * n, 3)
    return 1.0 if hands_clear(objects) and _trial["still"] >= STILL_S else 0.9


def seg_debug(objects, n):
    angle, r, z, skew = ring_frame(objects[n])
    return f"{n[-1]}[{angle:7.1f} r{1000 * (r - R):+5.1f} z{1000 * z:+6.1f} sk{skew:+5.1f}]"


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.40, TOP + 0.18
SEGMENT = 0.03
ARC_STEP = 3.0  # degrees between arc waypoints
RATE = 15.0  # degrees/s along the arc
HANDOVER = 3.0  # where the left arm lets a segment go for the right
LIFT = 0.015
CLEAR_STEP = STEP - STEP_HALF - HALF - 3.0  # -71: its tail's past the step
Z_FLOOR = M.FLOOR
SEG_GZ = SEG["knob_top"] - 0.010
BAR_GZ = BAR["knob_top"] - 0.010


def _fingers(width):
    return -FINGER_PER_M * width


KNOB_R, KNOB_G = _fingers(SEG["knob"] + 0.03), _fingers(SEG["knob"] - 0.004)


def _at(angle, z):
    t = math.radians(angle)
    return CX + R * math.cos(t), CY + R * math.sin(t), TOP + z


def _yaw(arm, angle):
    """Fingers close along the radius: the part's balance point is on the hinge's axis. Each arm's wrist has its own window."""
    return math.radians(angle - 90.0 if arm == "left" else angle + 90.0)


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

    def arc(self, arm, a0, a1, z, label, rate=RATE):
        """Carry along the arc, the wrist turning with it."""
        n = max(1, math.ceil(abs(a1 - a0) / ARC_STEP))
        for i in range(1, n + 1):
            a = a0 + (a1 - a0) * i / n
            self.go(arm, abs(a1 - a0) / n / rate, _at(a, z), label=label, yaw=_yaw(arm, a))


def _park(plan, arm):
    plan.go(arm, 1.6, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


def _pick_seg(plan, name):
    x, y = SEG_START[name]
    plan.go("left", 1.8, (x, y, TOP + SEG["knob_top"] + 0.05), KNOB_R, f"{name}: above it", 0.0)
    plan.go("left", 0.3, label="settle")
    plan.go("left", 0.9, (x, y, TOP + SEG_GZ), label="descend")
    plan.go("left", 0.6, finger=KNOB_G, label="grip")
    plan.go("left", 0.9, (x, y, TOP + SEG_GZ + 0.09), label="lift")


def _into_gate(plan):
    """Lay it on the groove's floor in the middle of the gate."""
    z = M.FLOOR + SEG_GZ
    plan.go("left", 2.4, _at(DROP, z + 0.075), label="over the gate", yaw=_yaw("left", DROP))
    plan.go("left", 0.4, label="settle")
    plan.go("left", 1.4, _at(DROP, z + 0.001), label="down in the gate")
    plan.go("left", 0.3, label="settle")
    return z + 0.001


def _left_carry(plan, name, end, z_top=0.03):
    _pick_seg(plan, name)
    z = _into_gate(plan)
    plan.arc("left", DROP, end, z, "carry round")
    plan.go("left", 0.3, label="settle")
    plan.go("left", 0.6, finger=KNOB_R, label="let go")
    plan.go("left", 0.6, _at(end, z + z_top), label="up")
    x, y, h = _at(end, z + z_top)
    plan.go("left", 1.0, (x - 0.08, y + 0.04, h), label="back")  # the arm's near full stretch out here, so it comes back before it rises
    _park(plan, "left")


def _right_carry(plan, end, lift=True):
    z = M.FLOOR + SEG_GZ + 0.001
    x, y, h = _at(HANDOVER, z + 0.03)
    plan.go("right", 2.0, (x - 0.06, y, h), KNOB_R, "inboard of the knob", _yaw("right", HANDOVER))  # low: the arm's near full stretch out here
    plan.go("right", 0.6, (x, y, h), label="over the knob")
    plan.go("right", 0.3, label="settle")
    plan.go("right", 0.7, _at(HANDOVER, z), label="descend")
    plan.go("right", 0.6, finger=KNOB_G, label="take it")
    up = z + (LIFT if lift else 0.0)
    plan.go("right", 0.6, _at(HANDOVER, up), label="lift over the step")
    plan.arc("right", HANDOVER, CLEAR_STEP, up, "over the step")
    plan.go("right", 0.6, _at(CLEAR_STEP, z), label="down")
    plan.arc("right", CLEAR_STEP, end, z, "carry on")
    plan.go("right", 0.3, label="settle")
    plan.go("right", 0.6, finger=KNOB_R, label="let go")
    plan.go("right", 0.8, _at(end, z + 0.05), label="up")
    _park(plan, "right")


def _lay_bar(plan):
    z = M.GROOVE["h"] + M.COVER_T + M.FLOOR + BAR_GZ
    x, y = BAR_START
    plan.go("right", 1.8, (x, y, TOP + BAR["dog_h"] + BAR["knob_top"] + 0.05), KNOB_R, "bar: above it", 0.0)
    plan.go("right", 0.3, label="settle")
    plan.go("right", 0.9, (x, y, TOP + BAR["dog_h"] + BAR_GZ), label="descend")
    plan.go("right", 0.6, finger=KNOB_G, label="grip")
    plan.go("right", 0.9, (x, y, TOP + BAR["dog_h"] + BAR_GZ + 0.09), label="lift")
    yaw = math.radians(GATE - 90.0)
    plan.go("right", 2.4, _at(GATE, z + 0.05), label="over the gate", yaw=yaw)
    plan.go("right", 0.4, label="settle")
    plan.go("right", 1.6, _at(GATE, z + 0.002), label="down")
    plan.go("right", 0.3, label="settle")
    plan.go("right", 0.6, finger=KNOB_R, label="let go")
    plan.go("right", 0.8, _at(GATE, z + 0.07), label="up")
    _park(plan, "right")


FIRST_END = STOP_FACE + HALF  # -143: nose on the stop
SECOND_END = FIRST_END + SEG_DEG + 0.5
THIRD_END = STEP + STEP_HALF + HALF + 1.0  # 0: nose just short of the step
FOURTH_END = THIRD_END + SEG_DEG + 0.5


def demo():
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    test = os.environ.get("SEAL_TEST", "")
    wrong = os.environ.get("SEAL_WRONG", "")
    if test == "travel":  # first test: one segment 90 degrees round, on the floor, and let go
        _left_carry(plan, "seg_a", DROP - 90.0)
        return _finish(plan)
    if test == "cross":  # one segment handed over and lifted across the step
        _left_carry(plan, "seg_a", HANDOVER)
        plan.until("right", plan.t["left"])
        _right_carry(plan, FIRST_END)
        return _finish(plan)
    _left_carry(plan, "seg_a", HANDOVER)
    plan.until("right", plan.t["left"])
    if wrong == "chain":  # debugging only: one arm, pushing the first segment on with the next instead of handing it over
        _left_carry(plan, "seg_b", THIRD_END + SEG_DEG - 12.0)
        _left_carry(plan, "seg_c", THIRD_END + 2 * SEG_DEG - 24.0)
        return _finish(plan)
    if wrong == "nolift":  # debugging only: the right arm slides it into the step instead of lifting it
        _right_carry(plan, FIRST_END, lift=False)
        return _finish(plan)
    _right_carry(plan, FIRST_END)
    plan.until("left", plan.t["right"])
    _left_carry(plan, "seg_b", HANDOVER)
    plan.until("right", plan.t["left"])
    _right_carry(plan, SECOND_END)
    plan.until("left", plan.t["right"])
    _left_carry(plan, "seg_c", THIRD_END)
    if wrong == "early":  # debugging only: the bar goes on with a segment still out
        plan.until("right", plan.t["left"])
        _lay_bar(plan)
        return _finish(plan)
    _left_carry(plan, "seg_d", FOURTH_END)
    plan.until("right", plan.t["left"])
    _lay_bar(plan)
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
            if pe > 1e-3 or re > math.radians(1.0):
                bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1), round(math.degrees(re), 1)))
    return worst_p, worst_r, bad


def reach_limits():
    """Why two arms: walk each arm along the arc from its own side, with the grip straight down, and see where it stops reaching."""
    out = {}
    for arm, angles, s in (("left", range(100, -121, -5), -90.0), ("right", range(-100, 121, 5), 90.0)):
        q = np.array(READY, float)
        for a in angles:
            q2, pe, re = ik.solve(np.asarray(_at(a, M.FLOOR + SEG_GZ)), ik.matrix_to_quat_xyzw(grip_rotation(math.radians(a + s))), q, base_pos=BASES[arm], iters=120)
            if pe > 1e-3 or re > math.radians(1.0):
                out[arm] = a
                break
            q = q2
    return out


# -- self-check (no simulator): the demo fits the run limit and IK reaches every waypoint -------------------------------
if __name__ == "__main__":
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad = ik_errors(wps)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:6], len(bad))
    print("arms stop reaching at", reach_limits(), "ends", FIRST_END, SECOND_END, THIRD_END, FOURTH_END, "clear step", CLEAR_STEP)
    assert t * 20 < 3000 and not bad
