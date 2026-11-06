"""On the station table: a warped flange on a turntable, a sprung ram over its rim, and a dial pointer beside it. The flange
looks plain. Turning it by its crank handle moves the pointer, which peaks when the high spot is under the ram. The robot must
turn the flange until the pointer peaks, press there once, then move both hands away. A press anywhere else bends good metal,
and that can't be undone. Lots A, B, C and D put the high spot at a different angle; FLANGE_LOT picks one (A by default).

Run it:  FLANGE_LOT=A inspect-robots run --embodiment yam_flange -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Press the warped flange at its high spot, then move both hands away." --scorer task_score

`yam_flange` (flange_press/embodiment.py) spawns the press and reports each gripper's fingertip midpoint and the press's state.
"""

import math
import os

import numpy as np

from flange_press import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
CX, CY = 0.29, -0.02  # the turntable's centre
LOTS = {"A": 60.0, "B": -60.0, "C": 120.0, "D": -120.0}  # the turn, in degrees, that brings the high spot under the ram
LOT = os.environ.get("FLANGE_LOT", "A").upper()
TARGET = math.radians(LOTS[LOT])

OBJECTS = [
    # placeholders: the yam_flange embodiment spawns the press (mech) in their place
    Box("press_frame", pos=(CX, CY, TOP), size=(0.01, 0.01, 0.01)),
    Box("press_mech", pos=(CX, CY, TOP), size=(0.01, 0.01, 0.01)),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
PEAK_WIDTH = math.radians(60)  # the pointer falls from full scale to zero over this much turn either side
WINDOW = math.radians(15)  # the high spot counts as under the ram within this
PRESSED, HOME = 0.02, 0.003  # ram travel for a press; within this of the top counts as released
LET_GO, HANDS_CLEAR, STILL_S, STILL_RATE = 0.06, 0.08, 1.0, math.radians(3)
_trial = {"still": 0.0, "good": False, "bad": False}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears `bad`."""
    _trial.update(still=0.0, good=False, bad=False)


def _off(turn, target=None):
    d = float(turn) - (TARGET if target is None else target)
    return abs(math.atan2(math.sin(d), math.cos(d)))


def reading(turn):
    """The pointer's travel along its scale, m: full scale with the high spot under the ram, 0 once 60 degrees off."""
    return M.DIAL["travel"] * max(0.0, 1.0 - _off(turn) / PEAK_WIDTH)


def handle_pos(turn):
    return np.array((CX + M.CRANK_R * math.sin(turn), CY - M.CRANK_R * math.cos(turn), TOP + M.HANDLE["z1"]))


def _ram_pos():
    return np.array((CX, CY + M.RAM["y"], TOP + M.RAM["z1"]))


def hands_clear(objects):
    """Both fingertip midpoints at least 8 cm from the handle and from the ram's pad."""
    return all(min(np.linalg.norm(np.asarray(objects[a].pos, float) - objects["handle"].pos), np.linalg.norm(np.asarray(objects[a].pos, float) - _ram_pos())) >= HANDS_CLEAR
               for a in ("left_tcp", "right_tcp"))


def _let_go(objects):
    return all(min(np.linalg.norm(np.asarray(objects[a].pos, float) - objects["handle"].pos), np.linalg.norm(np.asarray(objects[a].pos, float) - _ram_pos())) >= LET_GO
               for a in ("left_tcp", "right_tcp"))


def note_step(objects, dt):
    """Called by the embodiment after every control step: the press's outcome, and the clock for all at rest with hands clear."""
    turn, ram = objects["press"].pos[:2]
    if ram >= PRESSED:
        if _off(turn) <= WINDOW:
            _trial["good"] = True
        else:
            _trial["bad"] = True
    still = abs(objects["press"].lin_vel[0]) <= STILL_RATE and ram <= HOME and hands_clear(objects)
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. A press anywhere but the high spot is 0 for good. Otherwise, with the
    high spot under the ram and both hands off the handle and ram: 0.5 for that alone, 0.85 if it was pressed there and the ram
    is back up with hands clear, 1.0 if it's then all been at rest for the last 1 s."""
    turn, ram = objects["press"].pos[:2]
    if _trial["bad"] or _off(turn) > WINDOW or not _let_go(objects):
        return 0.0
    if not (_trial["good"] and ram <= HOME and hands_clear(objects)):
        return 0.5
    return 1.0 if _trial["still"] >= STILL_S else 0.85


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.33, TOP + 0.18
SWEEP = math.radians(150)  # the demo first turns this far one way, to read the pointer as a policy would


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


def _fingers(width):
    return -FINGER_PER_M * width


def _park(plan, arm):
    plan.go(arm, 1.4, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


GZ = TOP + M.HANDLE["z1"] - 0.008


def _crank(plan, frm, to, label):
    """Turn the flange from frm to to in steps, the gripped handle free to swivel, so the wrist doesn't turn."""
    n = max(1, round(abs(to - frm) / math.radians(15)))
    for k in range(1, n + 1):
        a = frm + (to - frm) * k / n
        plan.go("right", 0.45, (CX + M.CRANK_R * math.sin(a), CY - M.CRANK_R * math.cos(a), GZ), label=label)


def demo(lot=None):
    lot = LOT if lot is None else lot
    target = math.radians(LOTS[os.environ.get("FLANGE_DEMO_PRESS", lot)])  # the variable: debugging only, a wrong press
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    plan.sync()
    ready, grip = _fingers(2 * M.HANDLE["half"] + 0.03), _fingers(2 * M.HANDLE["half"] - 0.004)
    h0 = (CX, CY - M.CRANK_R)
    plan.go("right", 1.6, (*h0, GZ + 0.05), ready, "above the handle", 0.0)
    plan.go("right", 0.4, label="settle")
    plan.go("right", 0.9, (*h0, GZ), label="descend")
    plan.go("right", 0.6, finger=grip, label="grip")
    plan.go("right", 0.3, label="settle")
    # a full search, the same for every lot: one way, then the other, then back to the peak
    _crank(plan, 0.0, SWEEP, "turn, watching the pointer")
    _crank(plan, SWEEP, -SWEEP, "turn the other way")
    _crank(plan, -SWEEP, target, "back to the peak")
    plan.go("right", 0.8, label="settle")
    plan.go("right", 0.5, finger=ready, label="let go")
    a = target
    above = (CX + M.CRANK_R * math.sin(a), CY - M.CRANK_R * math.cos(a), GZ + 0.05)
    plan.go("right", 0.8, above, label="up")
    _park(plan, "right")
    plan.sync()
    px, py, z = CX, CY + M.RAM["y"], TOP + M.RAM["z1"]
    plan.go("left", 1.6, (px, py, z + 0.05), FINGER_CLOSED, "above the ram", 0.0)
    plan.go("left", 0.4, label="settle")
    plan.go("left", 0.8, (px, py, z + 0.016), label="onto it")
    plan.go("left", 1.8, (px, py, z + 0.016 - M.RAM["travel"] - 0.03), label="press")  # past the stop: the soft arm's push grows with how far short it is
    plan.go("left", 0.6, label="hold")
    plan.go("left", 1.4, (px, py, z + 0.05), label="let it back up")
    _park(plan, "left")
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
        print(f"lot {name}: demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:3])
        assert t * 20 < 2400 and not bad
