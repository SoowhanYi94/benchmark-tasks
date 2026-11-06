"""On the station table: a rod line runs across the bench, from a handle pad on the right to a blue gate on the left. It
has four joints that all look alike. One has parted, so working the handle no longer moves the gate. Working it opens a red
gap at the parted joint, which is how a technician finds the fault: follow the motion to where it stops. A slotted splice plate
waits on a stand. The robot must find the parted joint and seat the plate across it, so the handle opens the gate again, then
let go. Lots A, B, C and D part a different joint; the environment variable ROD_LOT picks one (A by default).

Run it:  ROD_LOT=A inspect-robots run --embodiment yam_rod -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Restore the drive, work the handle to confirm the gate opens, then move both hands away." --scorer task_score

`yam_rod` (rod_line/embodiment.py) spawns the line and plate and reports each gripper's fingertip midpoint and the bars' travel.
"""

import math
import os

import numpy as np

from rod_line import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
STEEL = (0.30, 0.32, 0.35)
X0 = 0.28
LOTS = {"A": 0, "B": 1, "C": 2, "D": 3}  # which joint has parted, from the handle end
LOT = os.environ.get("ROD_LOT", "A").upper()
PARTED = LOTS[LOT]
PLATE_START = (0.15, 0.0, TOP + 0.03)
STAND = dict(size=(0.05, 0.05, 0.03))

OBJECTS = [
    # placeholders: the yam_rod embodiment spawns the line and the plate (mech) in their place
    Box("line_frame", pos=(X0, 0.0, TOP), size=(0.01, 0.01, 0.01), color=STEEL),
    Box("line_mech", pos=(X0, 0.0, TOP), size=(0.01, 0.01, 0.01), color=STEEL),
    Box("splice_plate", pos=PLATE_START, size=(0.01, 0.01, 0.01), mass=M.PLATE_MASS),
    Box("plate_stand", pos=(PLATE_START[0], PLATE_START[1], TOP + 0.015), size=STAND["size"], color=STEEL),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
HANDLE_MIN, GATE_MIN = 0.035, 0.02  # how far the handle and the gate have to move for the drive to count as restored
HOME = 0.005  # both bars back within this of home
HANDS_AWAY = 0.06  # no fingertip midpoint this near the driven bar or the gate while the drive is proved
HANDS_CLEAR = 0.08
LET_GO = 0.06  # no credit while a fingertip midpoint is this near the plate, as when it's still held
FLAT = math.radians(5)
SEAT_X, SEAT_Y, SEAT_Z = 0.008, 0.008, 0.003  # the plate over a joint: within these, from where it sits on the bars at home
STILL_S = 1.0
_trial = {"still": 0.0, "proof": False}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears the proof."""
    _trial.update(still=0.0, proof=False)


def _rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def plate_on(objects):
    """The joint index the plate is seated across (studs through its slots, flat on the bars, along them), or None. The
    plate may have ridden along with the bars, up to the stroke."""
    o = objects["splice_plate"]
    R = _rot(o.quat_xyzw)
    if R[2, 2] < math.cos(FLAT) or abs(R[1, 1]) < math.cos(math.radians(15)):
        return None
    if abs(o.pos[0] - X0) > SEAT_X or abs(o.pos[2] - (TOP + M.BAR_Z1)) > SEAT_Z:
        return None
    handle, gate = objects["slides"].pos[:2]
    for i, y in enumerate(M.JOINTS):
        d = o.pos[1] - y
        if -M.STROKE - SEAT_Y <= d <= SEAT_Y:
            # across the parted joint, each bar's studs must be through the plate's slots: it can't be wedged outside them
            return i if i != PARTED or (abs(d + handle) <= SEAT_Y and abs(d + gate) <= SEAT_Y) else None
    return None


def _near_driven(objects, arm):
    """A fingertip midpoint within HANDS_AWAY of the driven bar or the gate, which could push the gate open by hand."""
    p = np.asarray(objects[arm].pos, float)
    slid = float(objects["slides"].pos[1])
    lo = np.array((X0 - M.GATE["half_x"], M.JOINTS[PARTED] - slid, TOP + M.BAR_Z0))
    hi = np.array((X0 + M.GATE["half_x"], M.GATE["y"] + M.GATE["half_y"] - slid, TOP + M.GATE["z1"]))
    return bool(np.linalg.norm(p - np.clip(p, lo, hi)) < HANDS_AWAY)


def hands_clear(objects):
    """Both fingertip midpoints at least 8 cm from the line (handle and gate included) and the plate."""
    lo = np.array((X0 - M.GATE["half_x"], M.LINE_LOW - M.PAD["half_y"] - 0.02, TOP))
    hi = np.array((X0 + M.GATE["half_x"], M.GATE["y"] + M.GATE["half_y"], TOP + M.GATE["z1"]))
    for arm in ("left_tcp", "right_tcp"):
        p = np.asarray(objects[arm].pos, float)
        if np.linalg.norm(p - np.clip(p, lo, hi)) < HANDS_CLEAR or np.linalg.norm(p - objects["splice_plate"].pos) < HANDS_CLEAR:
            return False
    return True


def note_step(objects, dt):
    """Called by the embodiment after every control step: the drive's proof, and the clock for parts at rest with hands clear."""
    handle, gate = objects["slides"].pos[:2]
    if (handle >= HANDLE_MIN and gate >= GATE_MIN and plate_on(objects) == PARTED
            and not any(_near_driven(objects, a) for a in ("left_tcp", "right_tcp"))):
        _trial["proof"] = True
    _trial["still"] = _trial["still"] + dt if objects["splice_plate"].resting() and hands_clear(objects) else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. 0 unless the plate is across the parted joint and let go. 0.4 for that alone,
    0.7 if the handle has also opened the gate with no hand near the gate, and 1.0 if then both bars are back home and all
    is at rest with both hands clear for the last 1 s."""
    if plate_on(objects) != PARTED or any(np.linalg.norm(objects[a].pos - objects["splice_plate"].pos) < LET_GO for a in ("left_tcp", "right_tcp")):
        return 0.0
    if not _trial["proof"]:
        return 0.4
    handle, gate = objects["slides"].pos[:2]
    return 1.0 if handle <= HOME and gate <= HOME and _trial["still"] >= STILL_S else 0.7


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
PAD_TIP_PAST_TCP = 0.013
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
PARK = 0.18, 0.32, TOP + 0.18
PUSH_Z = TOP + 0.065 + PAD_TIP_PAST_TCP  # pad tips on the handle pad, clear of the bar
PUSH_FROM = M.PAD["y"] + M.PAD["half_y"] + 0.025  # TCP on the pad's +y side
PUSH_TO = PUSH_FROM - M.STROKE - 0.05  # past the stop: the soft arm's push grows with how far short it is


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


def _yaw(arm):  # fingers close along x, so the pair is thin along the push
    return min((math.pi / 2 + k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def _fingers(width):
    return -FINGER_PER_M * width


def _park(plan, arm):
    plan.go(arm, 1.2, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


def _work_handle(plan, label):
    plan.go("right", 1.4, (X0, PUSH_FROM, PUSH_Z + 0.04), FINGER_CLOSED, f"{label}: over the handle's near side", _yaw("right"))
    plan.go("right", 0.4, label="settle")
    plan.go("right", 0.8, (X0, PUSH_FROM, PUSH_Z), label="down beside it")
    plan.go("right", 1.6, (X0, PUSH_TO, PUSH_Z), label=label)
    plan.go("right", 1.0, label="hold")
    plan.go("right", 1.2, (X0, PUSH_FROM, PUSH_Z), label="let it back")
    plan.go("right", 0.6, (X0, PUSH_FROM, PUSH_Z + 0.05), label="up")
    _park(plan, "right")


def demo(lot=None):
    joint = M.JOINTS[LOTS[os.environ.get("ROD_DEMO_JOINT", LOT if lot is None else lot)]]  # the variable: debugging only, a wrong joint
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    plan.sync()
    _work_handle(plan, "work the handle: the gap shows where it has parted")
    plan.sync()
    arm = "left" if joint > 0 else "right"  # the arm on the joint's side reaches it
    gz = M.PLATE_GRIP["z1"] - 0.006
    ready, grip = _fingers(2 * M.PLATE_GRIP["half"] + 0.03), _fingers(2 * M.PLATE_GRIP["half"] - 0.004)
    sx, sy, sz = PLATE_START
    seat = TOP + M.BAR_Z1
    carry = seat + M.STUD_H + gz + 0.02
    plan.go(arm, 1.4, (sx, sy, sz + gz + 0.05), ready, "plate: above its grip", _yaw(arm))
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.9, (sx, sy, sz + gz), label="descend")
    plan.go(arm, 0.6, finger=grip, label="grip")
    plan.go(arm, 1.0, (sx, sy, carry), label="lift")
    plan.go(arm, 2.8, (X0, joint, carry), label="over the parted joint")
    plan.go(arm, 1.2, label="settle")
    plan.go(arm, 1.4, (X0, joint, seat + gz + 0.002), label="down over the studs")
    plan.go(arm, 0.5, label="settle")
    plan.go(arm, 0.6, finger=ready, label="let go")
    plan.go(arm, 0.8, (X0, joint, carry + 0.02), label="up")
    _park(plan, arm)
    plan.sync()
    _work_handle(plan, "work the handle: the gate should open")
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
                bad.append((arm, w.label[:28], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1)))
    return worst_p, worst_r, bad


# -- self-check (no simulator): every lot's demo fits the run limit and IK reaches every waypoint -----------------------
if __name__ == "__main__":
    for name in LOTS:
        wps = demo(name)
        t = max(sum(w.duration for w in ws) for ws in wps.values())
        pe, re, bad = ik_errors(wps)
        print(f"lot {name}: demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:4])
        assert t * 20 < 2400 and not bad
