"""Hot-tapping a live pipe: a tapping machine works through an open-top valve. The robot must set the machine on its saddle, open the valve, feed the
boring bar until it cuts the coupon out of the pipe, withdraw it, shut the valve, take the machine back to its stand and fit the blank. Gate and bar
collide, so neither can pass the other. Lifting the machine off with the gate open once the pipe is cut sprays product and is a 0 for good.

Run it:  inspect-robots run --embodiment yam_hot_tap -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Hot-tap the live pipe through the valve to cut out its coupon, return the tapping machine to its yellow stand, leave the valve shut with the blank on its outlet, and move both hands away." --scorer task_score

`yam_hot_tap` (hot_tap/embodiment.py) spawns the line, machine and blank.
"""

import math
import os

import numpy as np

from hot_tap import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Hot-tap the live pipe through the valve to cut out its coupon, return the tapping machine to its yellow stand, leave the valve shut with the blank on its outlet, and move both hands away."
X0 = 0.28  # the bar's axis in world x
SEAT_Y = 0.03  # the machine's middle, seated against the valve
SEAT_Z = M.FLOOR_Z
STAND = (0.19, 0.20)
STAND_Z = 0.002
RACK = (0.24, -0.25)
BLANK_Y = -0.0585  # where the blank's stem sits, between the ribs
RACK_H = 0.055
POST_Y = (M.GATE["post_y0"] + M.GATE["post_y1"]) / 2
STEEL, BRASS = (0.30, 0.32, 0.35), (0.70, 0.55, 0.25)

BLANK = dict(flange_x=0.03, flange_y=0.0125, flange_t=0.008, stem_x=0.015, stem_y=0.004, stem_l=0.05, knob_x=0.012, knob_y=0.008, knob_h=0.03, mass=0.096)
_BLANK_BOXES = {"flange": ((-BLANK["flange_x"], -BLANK["flange_y"], 0.0), (BLANK["flange_x"], BLANK["flange_y"], BLANK["flange_t"])),
                "stem": ((-BLANK["stem_x"], -BLANK["stem_y"], -BLANK["stem_l"]), (BLANK["stem_x"], BLANK["stem_y"], 0.0)),
                "knob": ((-BLANK["knob_x"], -BLANK["knob_y"], BLANK["flange_t"]), (BLANK["knob_x"], BLANK["knob_y"], BLANK["flange_t"] + BLANK["knob_h"]))}
PARTS = {"blank": (_BLANK_BOXES, BLANK["mass"], 0.6, BRASS)}

OBJECTS = [
    Box("line", pos=(X0, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    Box("machine", pos=(*STAND, TOP + STAND_Z + 0.0005), size=(0.01, 0.01, 0.01)),
    Box("stand", pos=(*STAND, TOP + STAND_Z / 2), size=(0.12, 0.24, STAND_Z), color=(0.92, 0.78, 0.12)),
    Box("blank", pos=(*RACK, TOP + RACK_H + 0.0005), size=(0.01, 0.01, 0.01), mass=BLANK["mass"]),
    *[Box(f"rail_{s}", pos=(RACK[0] + sign * 0.024, RACK[1], TOP + RACK_H / 2), size=(0.01, 0.05, RACK_H), color=STEEL) for s, sign in (("a", 1), ("b", -1))],
]

# -- scoring -----------------------------------------------------------------------------------------------------------
GATE_SHUT, GATE_OPEN_SPRAY, FLAT = 0.004, 0.005, math.radians(5)
SEAT_XY, SEAT_DZ, STAND_XY, STAND_DZ, BLANK_XY, BLANK_DZ = 0.012, 0.006, 0.04, 0.006, 0.006, 0.004
CUT_PLUG, CUT_NOSE, CUT_S = 0.005, 0.002, 0.6
LIN, ANG, STILL_S = 0.01, 0.5, 1.0
_trial = {"still": 0.0, "bad": False, "cut": False, "spray": False, "cutting": 0.0}


def reset_latch():
    _trial.update(still=0.0, bad=False, cut=False, spray=False, cutting=0.0)


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _flat(part):
    return _rot(part.quat_xyzw)[2, 2] >= math.cos(FLAT)


def pitch_deg(objects):
    R = _rot(objects["machine"].quat_xyzw)
    return math.degrees(math.atan2(R[2, 1], R[1, 1]))


def roll_deg(objects):
    R = _rot(objects["machine"].quat_xyzw)
    return math.degrees(math.atan2(R[2, 0], R[0, 0]))


def _dseat(objects):
    return np.asarray(objects["machine"].pos, float) - (X0, SEAT_Y, TOP + SEAT_Z)


def seated(objects):
    d = _dseat(objects)
    return bool(abs(d[0]) <= SEAT_XY and abs(d[1]) <= SEAT_XY and abs(d[2]) <= SEAT_DZ and _flat(objects["machine"]))


def on_stand(objects):
    d = np.asarray(objects["machine"].pos, float) - (*STAND, TOP + STAND_Z)
    return bool(abs(d[0]) <= STAND_XY and abs(d[1]) <= STAND_XY and abs(d[2]) <= STAND_DZ and _flat(objects["machine"]))


def nose_y(objects):
    """The bar's nose along the bar axis in world y; the machine is square enough in the saddle to neglect its yaw."""
    return float(objects["machine"].pos[1]) + M.BAR["nose"] - float(objects["feed"].pos[0])


def gate_shut(objects):
    return float(objects["line"].pos[0]) <= GATE_SHUT


def blank_on(objects):
    b = objects["blank"]
    d = np.asarray(b.pos, float) - (X0, BLANK_Y, TOP + M.WALL_TOP)
    return bool(abs(d[0]) <= BLANK_XY and abs(d[1]) <= BLANK_XY and abs(d[2]) <= BLANK_DZ and _flat(b))


def _clear_of(objects, part, lo, hi):
    """Both fingertip midpoints this far from a part's box, in its own frame: open fingers sit 4.7 cm out, so 6 cm means no finger's on it."""
    R = _rot(part.quat_xyzw)
    for a in ("left_tcp", "right_tcp"):
        d = R.T @ (np.asarray(objects[a].pos, float) - np.asarray(part.pos, float))
        q = d - np.clip(d, lo, hi)
        if np.linalg.norm(q) < 0.06:
            return False
    return True


def _machine_clear(objects):
    return _clear_of(objects, objects["machine"], (-M.BODY["x"], -M.BODY["y"], 0.0), (M.BODY["x"], M.BODY["y"], M.BLOCK["z1"]))


def _blank_clear(objects):
    return _clear_of(objects, objects["blank"], (-BLANK["flange_x"], -BLANK["flange_y"], -BLANK["stem_l"]), (BLANK["flange_x"], BLANK["flange_y"], BLANK["flange_t"] + BLANK["knob_h"]))


def _post_at(objects):
    return np.array((X0 - float(objects["line"].pos[0]), POST_Y, TOP + (M.GATE["z1"] + M.GATE["post_z1"]) / 2))


def hands_clear(objects):
    p = _post_at(objects)
    return _machine_clear(objects) and _blank_clear(objects) and all(np.linalg.norm(np.asarray(objects[a].pos, float) - p) >= 0.06 for a in ("left_tcp", "right_tcp"))


def note_step(objects, dt):
    """Called by the embodiment after every control step: the cut when a real bar has pressed the sprung plug and held it, the spray, parts leaving, and the clock for all at rest."""
    plug, nose = float(objects["line"].pos[1]), nose_y(objects)
    face = M.PLUG["face"] - plug
    pressing = seated(objects) and plug >= CUT_PLUG and abs(nose - face) <= CUT_NOSE
    _trial["cutting"] = _trial["cutting"] + dt if pressing and not _trial["cut"] else 0.0
    if _trial["cutting"] >= CUT_S:
        _trial["cut"] = True
    if _trial["cut"] and not seated(objects) and float(objects["line"].pos[0]) > GATE_OPEN_SPRAY:
        _trial["spray"] = True
    if any(objects[n].pos[2] < TOP - 0.01 for n in ("machine", "blank")):
        _trial["bad"] = True
    still = hands_clear(objects) and all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in ("machine", "blank")) and abs(objects["line"].lin_vel[0]) <= LIN
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end; no early end. 0 for good if the line was opened to atmosphere (the machine off with the gate open after the cut) or a part left the table.
    0.1 the machine seated and let go. 0.3 the coupon cut and the machine still seated. 0.6 cut, the gate shut, machine still seated and let go.
    0.8 cut, the gate shut, the machine back on its stand and let go. 0.9 and the blank on the outlet, let go. 1.0 with both hands clear and all at rest 1 s."""
    if _trial["bad"] or _trial["spray"]:
        return 0.0
    if not _trial["cut"]:
        return 0.1 if seated(objects) and _machine_clear(objects) else 0.0
    if seated(objects):
        return 0.6 if gate_shut(objects) and _machine_clear(objects) else 0.3
    if not (gate_shut(objects) and on_stand(objects) and _machine_clear(objects)):
        return 0.3
    if not (blank_on(objects) and _blank_clear(objects)):
        return 0.8
    return 1.0 if hands_clear(objects) and _trial["still"] >= STILL_S else 0.9


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.03
YAW_X = math.pi / 2  # fingers close along world x


def _fingers(width):
    return -FINGER_PER_M * width


GRIP_LOCAL_Z = 0.07  # TCP above the machine's underside, on the grip block
POST_LOCAL_Z = 0.075
CARRY_Z = 0.145
BLOCK_R, BLOCK_G = _fingers(0.03 + 0.03), _fingers(0.03 - 0.004)
POST_R, POST_G = _fingers(0.024 + 0.03), _fingers(0.024 - 0.004)
GPOST_R, GPOST_G = _fingers(0.024 + 0.03), _fingers(0.024 - 0.004)
GPOST_Z = 0.105
KNOB_Z = BLANK["flange_t"] + BLANK["knob_h"] - 0.012
BLANK_R, BLANK_G = _fingers(2 * BLANK["knob_x"] + 0.03), _fingers(2 * BLANK["knob_x"] - 0.004)


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


def _park(plan, arm):
    plan.go(arm, 1.4, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


def _pick_machine(plan, arm, x, y, z0, grip=BLOCK_G):
    gy = y + M.BLOCK["y_at"]
    plan.go(arm, 1.6, (x, gy, TOP + z0 + GRIP_LOCAL_Z + 0.06), BLOCK_R, "machine: above the grip", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 1.0, (x, gy, TOP + z0 + GRIP_LOCAL_Z), label="descend")
    plan.go(arm, 0.7, finger=grip, label="grip")


def _set_machine_on_saddle(plan):
    arm = "left"
    plan.go(arm, 0.8, label="grip firm")
    plan.go(arm, 1.0, (STAND[0], STAND[1] + M.BLOCK["y_at"], TOP + CARRY_Z), label="lift")
    gy = SEAT_Y + M.BLOCK["y_at"]
    plan.go(arm, 2.4, (X0, gy - 0.004, TOP + CARRY_Z), label="carry over the saddle")
    plan.go(arm, 0.5, label="settle over the saddle")
    plan.go(arm, 1.6, (X0, gy - 0.004, TOP + SEAT_Z + GRIP_LOCAL_Z + 0.001), label="lower onto the saddle")
    plan.go(arm, 1.2, (X0, gy - 0.010, TOP + SEAT_Z + GRIP_LOCAL_Z + 0.001), label="slide to the valve")
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.6, finger=BLOCK_R, label="let go")
    plan.go(arm, 1.0, (X0, gy, TOP + SEAT_Z + GRIP_LOCAL_Z + 0.07), label="up")
    _park(plan, arm)


def _gate(plan, frm, to, end):
    arm = "right"
    frm_x, to_x = X0 - frm, X0 - to
    plan.go(arm, 1.8, (frm_x, POST_Y, TOP + GPOST_Z + 0.05), GPOST_R, "gate: above its post", YAW_X)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (frm_x, POST_Y, TOP + GPOST_Z), label="descend")
    plan.go(arm, 0.6, finger=GPOST_G, label="grip")
    plan.go(arm, 2.0, (to_x, POST_Y, TOP + GPOST_Z), label="slide the gate")
    plan.go(arm, 0.4, label="hold")
    plan.go(arm, 0.5, (end, POST_Y, TOP + GPOST_Z), label="back off")
    plan.go(arm, 0.5, finger=GPOST_R, label="let go")
    plan.go(arm, 0.8, (end, POST_Y, TOP + GPOST_Z + 0.06), label="up")
    _park(plan, arm)


def _feed_bar(plan, feed=0.052, past=0.004, retract=True):
    arm = "left"
    y0 = SEAT_Y + M.BAR["post_y"]
    z = TOP + SEAT_Z + POST_LOCAL_Z
    plan.go(arm, 1.8, (X0, y0, z + 0.06), POST_R, "feed: above its post", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (X0, y0, z), label="descend")
    plan.go(arm, 0.6, finger=POST_G, label="grip")
    plan.go(arm, 3.2, (X0, y0 - feed - past, z), label="feed the bar")
    plan.go(arm, 1.0, label="dwell")
    if retract:
        plan.go(arm, 2.4, (X0, y0 + 0.002, z), label="withdraw the bar")
    plan.go(arm, 0.4, label="hold")
    plan.go(arm, 0.5, finger=POST_R, label="let go")
    plan.go(arm, 1.0, (X0, y0 if retract else y0 - feed, z + 0.07), label="up")
    _park(plan, arm)


def _take_machine_off(plan):
    arm = "left"
    gy = SEAT_Y + M.BLOCK["y_at"]
    _pick_machine(plan, arm, X0, SEAT_Y, SEAT_Z)
    plan.go(arm, 1.4, (X0, gy, TOP + CARRY_Z), label="lift off")
    plan.go(arm, 2.4, (STAND[0], STAND[1] + M.BLOCK["y_at"], TOP + CARRY_Z), label="carry to the stand")
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 1.6, (STAND[0], STAND[1] + M.BLOCK["y_at"], TOP + STAND_Z + GRIP_LOCAL_Z + 0.001), label="lower onto the stand")
    plan.go(arm, 0.6, finger=BLOCK_R, label="let go")
    plan.go(arm, 1.0, (STAND[0], STAND[1], TOP + 0.15), label="up")
    _park(plan, arm)


def _fit_blank(plan):
    arm = "right"
    x, y = RACK
    z = TOP + RACK_H + KNOB_Z
    plan.go(arm, 1.8, (x, y, z + 0.06), BLANK_R, "blank: above it", YAW_X)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, z), label="descend")
    plan.go(arm, 0.6, finger=BLANK_G, label="grip")
    zc = TOP + M.WALL_TOP + 0.075 + KNOB_Z
    plan.go(arm, 1.2, (x, y, zc), label="lift out")
    plan.go(arm, 2.6, (X0 - 0.06, BLANK_Y, zc), label="carry beside the gate's post")  # the stem passes below the post's top
    plan.go(arm, 0.5, label="settle")
    plan.go(arm, 1.4, (X0, BLANK_Y, zc), label="over the outlet")
    plan.go(arm, 0.8, label="settle")
    plan.go(arm, 1.6, (X0, BLANK_Y, TOP + M.WALL_TOP + KNOB_Z + 0.001), label="lower it in")
    plan.go(arm, 0.5, finger=BLANK_R, label="let go")
    plan.go(arm, 1.0, (X0, BLANK_Y, TOP + M.WALL_TOP + KNOB_Z + 0.07), label="up")
    _park(plan, arm)


def _first_sim(plan):
    arm = "left"
    y0 = STAND[1] + M.BAR["post_y"]
    z = TOP + STAND_Z + POST_LOCAL_Z
    gy = STAND[1] + M.BLOCK["y_at"]
    lift = TOP + STAND_Z + GRIP_LOCAL_Z + 0.07
    _pick_machine(plan, arm, *STAND, STAND_Z)
    plan.go(arm, 1.4, (STAND[0], gy, lift), label="lift, bar retracted")
    plan.go(arm, 2.5, label="hold retracted")
    plan.go(arm, 1.2, (STAND[0], gy, TOP + STAND_Z + GRIP_LOCAL_Z + 0.001), label="down")
    plan.go(arm, 0.5, finger=BLOCK_R, label="let go")
    plan.go(arm, 0.8, (STAND[0], gy, lift), label="up")
    plan.go(arm, 1.5, (STAND[0], y0, z + 0.06), POST_R, "over the post")
    plan.go(arm, 0.8, (STAND[0], y0, z), label="descend")
    plan.go(arm, 0.6, finger=POST_G, label="grip")
    plan.go(arm, 2.6, (STAND[0], y0 - M.BAR["stroke"] - 0.004, z), label="feed fully")
    plan.go(arm, 0.3, finger=POST_R, label="let go")
    plan.go(arm, 0.8, (STAND[0], y0 - M.BAR["stroke"], z + 0.06), label="up")
    plan.go(arm, 1.0, (STAND[0], gy, TOP + STAND_Z + GRIP_LOCAL_Z + 0.06), BLOCK_R, "over the grip")
    plan.go(arm, 0.8, (STAND[0], gy, TOP + STAND_Z + GRIP_LOCAL_Z), label="descend")
    plan.go(arm, 0.7, finger=BLOCK_G, label="grip")
    plan.go(arm, 1.4, (STAND[0], gy, lift), label="lift, bar fed")
    plan.go(arm, 2.5, label="hold fed")
    plan.go(arm, 1.2, (STAND[0], gy, TOP + STAND_Z + GRIP_LOCAL_Z + 0.001), label="down")
    plan.go(arm, 0.5, finger=BLOCK_R, label="let go")
    plan.go(arm, 1.0, (STAND[0], gy, lift), label="up")
    _park(plan, arm)


def demo():
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    if os.environ.get("HT_FIRST") == "1":  # debugging only: the machine's balance, retracted and fed
        _first_sim(plan)
        return _finish(plan)
    wrong = os.environ.get("HT_DEMO_WRONG", "")  # debugging only: "lift_open" takes the machine off with the gate still open, "shut_on_bar" shuts the gate with the bar still in
    _pick_machine(plan, "left", *STAND, STAND_Z)
    _set_machine_on_saddle(plan)
    plan.until("right", plan.t["left"])
    _gate(plan, 0.0, M.GATE["travel"] + 0.004, X0 - M.GATE["travel"])
    plan.until("left", plan.t["right"])
    if wrong == "shut_on_bar":
        _feed_bar(plan, retract=False)
        plan.until("right", plan.t["left"])
        _gate(plan, M.GATE["travel"], -0.004, X0 - 0.0)
        return _finish(plan)
    _feed_bar(plan)
    plan.until("right", plan.t["left"])
    if wrong == "lift_open":
        _take_machine_off(plan)
        return _finish(plan)
    _gate(plan, M.GATE["travel"], -0.004, X0 + 0.0)
    plan.until("left", plan.t["right"])
    _take_machine_off(plan)
    plan.until("right", plan.t["left"])
    _fit_blank(plan)
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
    for env in ("", "1"):
        if env:
            os.environ["HT_FIRST"] = env
        wps = demo()
        t = max(sum(w.duration for w in ws) for ws in wps.values())
        pe, re, bad = ik_errors(wps)
        print(f"{'first' if env else 'demo'} {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:6], len(bad))
        assert t * 20 < 3000 and not bad
