"""A calibration lab's dead-weight tester stands on the table with a test gauge plumbed to it. The robot loads both slotted brass weights
round the piston's stem, pumps the lever until the piston floats in the green band, spins the weights, and while they spin turns the
gauge's trim knob until the needle meets the red mark, then moves both hands away.

Run it:  inspect-robots run --embodiment yam_deadweight -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Load both weights onto the piston, pump until the orange flag is in the green band, spin the weights, turn the blue knob while they spin until the gauge needle is on the red mark, and move both hands away." --scorer task_score

`yam_deadweight` (deadweight_test/embodiment.py) spawns the tester and weights and reports each gripper's fingertip midpoint and the rig.
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
BRASS, STEEL = (0.78, 0.62, 0.22), (0.30, 0.32, 0.35)
GRAVITY = 9.81

R_W, SLOT, BACK, H_W, DENSITY = 0.035, 0.015, 0.015, 0.016, 8500.0  # 7 cm across, 1.6 cm thick, a 3 cm slot from 1.5 cm behind the centre out through +x; brass


def weight_mass():
    """The slotted plate's own mass: a circle less the slot."""
    area = math.pi * R_W ** 2 - (SLOT * math.sqrt(R_W ** 2 - SLOT ** 2) + R_W ** 2 * math.asin(SLOT / R_W)) - 2 * SLOT * BACK
    return round(area * H_W * DENSITY, 3)


WEIGHT = dict(r=R_W, slot=SLOT, back=BACK, h=H_W, mass=weight_mass())
COM_BACK = 0.0062  # the plate's balance point lies this far behind its centre, away from the slot

GEOM = dict(
    piston_at=(0.30, -0.10), platen_z=0.025, platen_t=0.005, platen_half=0.022, column_half=0.016, stem_half=0.008, stem_h=0.0565,
    bar_half=0.033, bar_width=0.008, bar_t=0.01, piston_mass=0.10, rotor_mass=0.15, spin_damping=0.05,
    top_stop=0.055, oil_k=420.0, oil_c=30.0,
    lever_pivot=(0.15, 0.25), lever_z=0.045, lever_len=0.185, lever_pad=0.028, lever_mass=0.06, lever_up=-18.0, lever_down=4.0, lever_k=0.012, lever_c=0.0012, lever_preload=12.0,
    knob_at=(0.33, 0.10), knob_z=0.05, knob_half=0.026, knob_width=0.008, knob_t=0.02, knob_mass=0.05, knob_limit=110.0, knob_c=0.002,
    gauge_at=(0.43, 0.10), gauge_r=0.055,
    flag_x0=0.385, flag_y=-0.10, flag_gain=2.0, flag_half=0.02, band=(0.021, 0.037),
)
PX, PY = GEOM["piston_at"]
PLATEN = GEOM["platen_z"]
STROKE_RISE = 0.010  # a full lever stroke raises the oil's level this much
RACK_H = 0.03
RACK = {"weight_a": (0.20, -0.20), "weight_b": (0.20, -0.30)}
WEIGHTS = tuple(RACK)
LOAD = GEOM["piston_mass"] + GEOM["rotor_mass"] + 2 * WEIGHT["mass"]  # what the oil carries, kg
FULL_N = LOAD * GRAVITY
_TEST = os.environ.get("DW_TEST", "")  # debugging only

_start = {n: (*RACK[n], TOP + RACK_H + 0.0005) for n in WEIGHTS}
if _TEST == "spin":  # debugging only: start loaded and floating, for the flick alone
    _start = {"weight_a": (PX, PY, TOP + PLATEN + 0.0005), "weight_b": (PX, PY, TOP + PLATEN + H_W + 0.001)}

OBJECTS = [
    Box(RIG := "rig", pos=(0.0, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    *[Box(n, pos=_start[n], size=(0.01, 0.01, 0.01), mass=WEIGHT["mass"]) for n in WEIGHTS],
    *[Box(f"rail_{n}_{s}", pos=(x, y + sign * 0.02, TOP + RACK_H / 2), size=(0.05, 0.01, RACK_H), color=STEEL)
      for n, (x, y) in RACK.items() for s, sign in (("l", 1), ("r", -1))],
]

# -- the hydraulics, the stick and the gauge: pure functions of the sim's real motion ---------------------------------------
PUMP_GAIN = STROKE_RISE / (GEOM["lever_down"] - GEOM["lever_up"])  # m of oil level per degree of downstroke
START_ERROR = -40.0  # the needle's error against the red mark, in degrees, with no stick and the knob untouched
TRIM_GAIN = 0.6  # needle degrees per knob degree
NEEDLE_PER_N = 200.0 / 12.0  # needle degrees per N the oil carries
STICK = 25.0  # the piston's friction, as needle degrees: reads high after rising and low after sinking, until spun
SPIN = 2.0  # rad/s: both weights and the rotor at least this fast, floating, frees the piston
FLOATING = 0.005
STICK_MOVE, KNOB_MOVE = 0.0015, 2.0  # piston motion in m that re-sticks it; knob motion in degrees that counts as setting it
_FREE_S = 0.3

# -- scoring -----------------------------------------------------------------------------------------------------------
OVER_PUMP = 0.05
IN_SLOT = 0.003  # the stem may ride this far past the slot's own play
SEAT_Z, FLAT = 0.004, math.radians(5)
NEEDLE_TOL = 5.0
HAND_CLEAR, LIN, ANG, RIG_RATE, STILL_S = 0.08, 0.01, 0.5, 5.0, 1.0
LEVER_REST = 3.0
_trial = {}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears `bad`."""
    _trial.update(still=0.0, bad=False, oil=FLOAT_H + 0.021 if _TEST == "spin" else 0.0, peak=None, stick=STICK if _TEST != "spin" else 0.0, stuck_h=None, freed=False, spinning=False, knob_ref=0.0, set_spinning=False)


reset_latch()


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _flat(part):
    return _rot(part.quat_xyzw)[2, 2] >= math.cos(FLAT)


def offset_on_stem(objects, name):
    """The stem's centre in the weight's own frame, and the weight's height over the platen."""
    w = objects[name]
    d = _rot(w.quat_xyzw).T @ (np.array((PX, PY, w.pos[2])) - np.asarray(w.pos, float))
    return np.array((d[0], d[1], w.pos[2] - (TOP + PLATEN + objects[RIG].pos[0])))


def _on_stem(objects, name):
    """Round the stem, in its slot, and resting flat on the platen or on the other weight. The weights are identical."""
    o = offset_on_stem(objects, name)
    play = SLOT - GEOM["stem_half"] + IN_SLOT
    in_slot = -play - 0.0 <= o[0] <= R_W and abs(o[1]) <= play
    rest = min(abs(o[2]), abs(o[2] - H_W)) <= SEAT_Z
    return bool(in_slot and rest and _flat(objects[name]))


def _clear_of(objects, part, half, top):
    """Both fingertip midpoints this far from a part's box, in its own frame: open fingers sit 4.7 cm out, so 6 cm means no finger's on it."""
    R = _rot(part.quat_xyzw)
    for a in ("left_tcp", "right_tcp"):
        d = R.T @ (np.asarray(objects[a].pos, float) - np.asarray(part.pos, float))
        q = d - np.clip(d, (-half, -half, 0.0), (half, half, top))
        if np.linalg.norm(q) < 0.06:
            return False
    return True


def weight_on(objects, name):
    return _on_stem(objects, name) and _clear_of(objects, objects[name], R_W, H_W)


def both_on(objects):
    return all(_on_stem(objects, n) for n in WEIGHTS) and abs(abs(offset_on_stem(objects, WEIGHTS[0])[2] - offset_on_stem(objects, WEIGHTS[1])[2]) - H_W) <= SEAT_Z


def in_band(objects):
    return GEOM["band"][0] <= objects[RIG].pos[0] <= GEOM["band"][1]


def carried_n(objects):
    """What the oil carries, from the spring: the real weight on the piston."""
    return GEOM["oil_k"] * (_trial["oil"] - objects[RIG].pos[0])


def needle_error(objects):
    """The needle's error against the red mark, in degrees: + is clockwise past it."""
    return (carried_n(objects) - FULL_N) * NEEDLE_PER_N + START_ERROR + TRIM_GAIN * objects[RIG].pos[2] + _trial["stick"]


def _near_knob_or_lever(objects):
    k = np.array((*GEOM["knob_at"], TOP + GEOM["knob_z"]))
    lx, ly = GEOM["lever_pivot"]
    pad = np.array((lx + GEOM["lever_len"] - GEOM["lever_pad"], ly, TOP + 0.08))
    return any(np.linalg.norm(np.asarray(objects[a].pos, float) - p) < HAND_CLEAR for a in ("left_tcp", "right_tcp") for p in (k, pad))


def hands_clear(objects):
    return not _near_knob_or_lever(objects) and all(_clear_of(objects, objects[n], R_W, H_W) for n in WEIGHTS)


def _rate(objects):
    return min([abs(objects[RIG].ang_vel[2])] + [abs(objects[n].ang_vel[2]) for n in WEIGHTS])


def note_step(objects, dt):
    """Called by the embodiment after every control step: the lever's downstrokes add oil, spinning frees the piston's friction, moving the piston unspun re-sticks it."""
    rig = objects[RIG]
    h, lever, knob = (float(v) for v in rig.pos)
    t = _trial
    # a check valve: only a downstroke adds oil, and only what it travels past its lowest point since it last rose
    if t["peak"] is None or lever < t["peak"] - 1.0:
        t["peak"] = lever
    elif lever > t["peak"]:
        t["oil"] += PUMP_GAIN * (lever - t["peak"])
        t["peak"] = lever
    if t["stuck_h"] is None:
        t["stuck_h"] = h
    if h >= OVER_PUMP or any(objects[n].pos[2] < TOP - 0.01 for n in WEIGHTS):
        t["bad"] = True
    on = both_on(objects)
    t["spinning"] = on and h >= FLOATING and _rate(objects) >= SPIN
    if t["spinning"]:
        t["stick"] *= math.exp(-dt / _FREE_S)
        if abs(t["stick"]) < 0.5:
            t["stick"], t["freed"] = 0.0, True
        t["stuck_h"] = h
    elif abs(h - t["stuck_h"]) >= STICK_MOVE:
        t["stick"] = math.copysign(STICK, h - t["stuck_h"])
        t["stuck_h"], t["freed"] = h, False
    if not on:
        t["freed"] = False
    if abs(knob - t["knob_ref"]) >= KNOB_MOVE:
        t["knob_ref"], t["set_spinning"] = knob, t["spinning"]
    still = (abs(rig.lin_vel[0]) <= LIN and abs(rig.lin_vel[1]) <= RIG_RATE and abs(rig.lin_vel[2]) <= RIG_RATE and hands_clear(objects)
             and all(abs(objects[n].lin_vel[2]) <= LIN and np.linalg.norm(objects[n].ang_vel[:2]) <= ANG for n in WEIGHTS))
    t["still"] = t["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. 0 for good if the piston's pumped to its top stop or a weight leaves the table.
    Otherwise 0.15 for each weight on the stem and let go; 0.5 with both on and the piston in the band; 0.75 with the weights also spun
    since the piston last moved; 0.9 with the needle also on the mark and the knob last set while they spun; 1.0 with the lever up, both
    hands clear and all at rest for the last 1 s."""
    if _trial["bad"]:
        return 0.0
    total = 0.15 * sum(weight_on(objects, n) for n in WEIGHTS)
    if total < 0.3 - 1e-9 or not both_on(objects):
        return round(total, 3)
    if not in_band(objects):
        return 0.3
    if not _trial["freed"]:
        return 0.5
    if abs(needle_error(objects)) > NEEDLE_TOL or not _trial["set_spinning"]:
        return 0.75
    rest = abs(objects[RIG].pos[1] - GEOM["lever_up"]) <= LEVER_REST
    return 1.0 if rest and hands_clear(objects) and _trial["still"] >= STILL_S else 0.9


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.03
FLICK = dict(x=PX + 0.02, y0=PY - 0.07, y1=PY + 0.05, dur=0.3)  # a closed hand sweeps past the bar's end along a straight line


def _fingers(width):
    return -FINGER_PER_M * width


W_R, W_G = _fingers(0.09), _fingers(2 * (R_W - 0.001) - 0.01)
KNOB_R, KNOB_G = _fingers(0.05), _fingers(2 * GEOM["knob_width"] - 0.004)
GRIP_ABOVE = 0.016  # TCP above the weight's underside, so the fingertips stay clear of a weight below
KNOB_TURN = -START_ERROR / TRIM_GAIN
FLOAT_H = 0.0275  # the piston's height after five strokes, read from the flag against the band


class _Plan:
    def __init__(self):
        self.wps = {a: [] for a in SIDE}
        self.t = {a: 0.0 for a in SIDE}
        self.last = {}
        for a in SIDE:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None, one=False):
        p0, f0, y0 = self.last[arm]
        p = p0 if pos is None else tuple(float(v) for v in pos)
        f = f0 if finger is None else finger
        y = y0 if yaw is None else yaw
        n = 1 if one else max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / SEGMENT))
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
    plan.go(arm, 1.4, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0, one=True)


def _load(plan, name, layer):
    """Lift a weight off its rails by its balance point, slide its slot round the stem, set it down."""
    arm = "right"
    x, y = RACK[name]
    gx = x - COM_BACK
    plan.go(arm, 1.6, (gx, y, TOP + 0.11), W_R, f"{name}: above it", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.9, (gx, y, TOP + RACK_H + GRIP_ABOVE), label="descend")
    plan.go(arm, 0.7, finger=W_G, label="grip")
    plan.go(arm, 0.8, (gx, y, TOP + 0.09), label="lift")
    seat = PLATEN + layer * H_W
    z_slide = TOP + seat + 0.008 + GRIP_ABOVE
    x_out = PX - 0.062 - COM_BACK
    plan.go(arm, 1.2, (x_out, PY, TOP + 0.09), label="to the stem")
    plan.go(arm, 0.8, (x_out, PY, z_slide), label="down beside it")
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 1.6, (PX - COM_BACK, PY, z_slide), label="slide on")
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.5, (PX - COM_BACK, PY, TOP + seat + 0.002 + GRIP_ABOVE), label="down")
    plan.go(arm, 0.7, finger=W_R, label="let go")
    plan.go(arm, 1.0, (PX - COM_BACK, PY, TOP + 0.125), label="up")


def _pump(plan, strokes):
    arm = "left"
    lx, ly = GEOM["lever_pivot"]
    x = lx + GEOM["lever_len"] - GEOM["lever_pad"] - 0.004
    plan.go(arm, 1.6, (x, ly, TOP + 0.14), FINGER_CLOSED, "over the lever", 0.0)
    for _ in range(strokes):
        plan.go(arm, 0.9, (x, ly, TOP + 0.028), label="stroke down")
        plan.go(arm, 0.8, (x, ly, TOP + 0.135), label="stroke up")


def _grip_knob(plan):
    arm = "left"
    kx, ky = GEOM["knob_at"]
    z = TOP + GEOM["knob_z"] + 0.008 + 0.004 + 0.013
    plan.go(arm, 1.4, (kx, ky, TOP + 0.12), KNOB_R, "over the knob", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (kx, ky, z), label="onto the wings")
    plan.go(arm, 0.6, finger=KNOB_G, label="grip")


def _turn_knob(plan, deg):
    arm = "left"
    kx, ky = GEOM["knob_at"]
    plan.go(arm, 2.4, yaw=math.radians(deg), label="turn the knob", one=True)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.6, finger=KNOB_R, label="let go")
    plan.go(arm, 0.8, (kx, ky, TOP + 0.12), label="up")
    _park(plan, arm)


def _flick(plan, h):
    arm = "right"
    f = FLICK
    z = TOP + PLATEN + GEOM["stem_h"] + h + GEOM["bar_t"] / 2 + 0.008
    plan.go(arm, 1.2, (f["x"], f["y0"], z + 0.03), FINGER_CLOSED, "beside the bar")
    plan.go(arm, 0.5, (f["x"], f["y0"], z), label="down")
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, f["dur"], (f["x"], f["y1"], z), label="flick", one=True)
    done = plan.t[arm]
    plan.go(arm, 0.4, (f["x"], f["y1"], z + 0.02), label="up", one=True)
    _park(plan, arm)
    return done


def demo():
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    wrong = os.environ.get("DW_WRONG", "")  # debugging only: the wrong ways
    if _TEST == "spin":
        plan.go("right", 1.0, label="wait")
        _flick(plan, FLOAT_H)
        plan.go("right", 25.0, label="watch")
        return _finish(plan)
    if os.environ.get("DW_SWAP") == "1":  # debugging only: the other weight first, since they're identical
        _load(plan, "weight_b", 0)
        _load(plan, "weight_a", 1)
    else:
        _load(plan, "weight_a", 0)
        _load(plan, "weight_b", 1)
    plan.go("right", 1.4, (FLICK["x"], FLICK["y0"], TOP + 0.15), label="stand by")
    plan.until("left", plan.t["right"])
    _pump(plan, 8 if wrong == "overpump" else 5)
    _grip_knob(plan)
    if wrong == "null_first":  # the null made on the stuck needle: spinning then knocks it off the mark
        _turn_knob(plan, (-START_ERROR - STICK) / TRIM_GAIN)
        plan.until("right", plan.t["left"])
        _flick(plan, FLOAT_H)
        return _finish(plan)
    if wrong == "no_spin":
        _turn_knob(plan, (-START_ERROR - STICK) / TRIM_GAIN)
        return _finish(plan)
    plan.until("right", plan.t["left"])
    flicked = _flick(plan, FLOAT_H)
    plan.until("left", flicked + (45.0 if wrong == "null_late" else 0.2))  # null_late: the right turn, but only after the spin has died
    _turn_knob(plan, KNOB_TURN)
    return _finish(plan)


def _finish(plan):
    end = max(plan.t.values()) + 2.0
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
                bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1), round(math.degrees(re), 1)))
    return worst_p, worst_r, bad


# -- self-check (no simulator): the demo fits the run limit and IK reaches every waypoint -------------------------------
if __name__ == "__main__":
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad = ik_errors(wps)
    print(f"weight {WEIGHT['mass']} kg, carries {FULL_N:.2f} N, sink {1000 * FULL_N / GEOM['oil_k']:.1f} mm, floats at {1000 * (5 * STROKE_RISE - FULL_N / GEOM['oil_k']):.1f} mm")
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:6])
    assert t * 20 < 3000 and not bad
