"""An assembly line's indexing dial turns steadily on the station table, with two pockets on it. The robot must fit a block in each
pocket and a peg in each block while the dial keeps turning, in either order and with either block and peg, then press the stop pad
and move both hands away. Stopping the dial
early, or holding it back, scores 0 for good.

Run it:  inspect-robots run --embodiment yam_dial -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Fit a block in each blue pocket on the turning dial and a peg in each block, press the red pad to stop the dial, and move both hands away." --scorer task_score

`yam_dial` (dial_build/embodiment.py) spawns the dial and parts and reports each gripper's fingertip midpoint and the dial.
"""

import math
import os

import numpy as np

from dial_build import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
CX, CY = 0.27, 0.0  # the dial's axle
RATE = math.radians(M.RATE)
FIXTURES = (math.radians(30.0), math.radians(210.0))  # each pocket's angle at the start, about the axle from +x
PAD_AT = (0.31, 0.215)  # the stop pad's centre
PAD_AT_REL = (PAD_AT[0] - CX, PAD_AT[1] - CY)
PAD_PRESSED = 0.01
RIM = M.DISC["z1"]  # the disc's top, above the table
STEEL, ALU, ORANGE = (0.30, 0.32, 0.35), (0.80, 0.82, 0.85), (0.90, 0.45, 0.08)

BLOCK = dict(half=0.03, h=0.03, bore=0.015, mass=0.22)  # 6 cm square, 3 cm tall, a 3 cm square bore; aluminium. Origin: centre of its underside
PEG = dict(shaft=0.008, shaft_l=0.025, head=0.02, head_h=0.025, mass=0.13)  # a 1.6 cm square shaft under a 4 cm square head; aluminium. Origin: centre of the head's underside


def _block_boxes():
    h, b, t = BLOCK["half"], BLOCK["bore"], BLOCK["h"]
    return {"n": ((-h, b, 0.0), (h, h, t)), "s": ((-h, -h, 0.0), (h, -b, t)), "e": ((b, -b, 0.0), (h, b, t)), "w": ((-h, -b, 0.0), (-b, b, t))}


_PEG_BOXES = {"shaft": ((-PEG["shaft"], -PEG["shaft"], -PEG["shaft_l"]), (PEG["shaft"], PEG["shaft"], 0.0)),
              "head": ((-PEG["head"], -PEG["head"], 0.0), (PEG["head"], PEG["head"], PEG["head_h"]))}
PARTS = {"block_a": (_block_boxes(), BLOCK["mass"], 0.6, ALU), "block_b": (_block_boxes(), BLOCK["mass"], 0.6, ALU),
         "peg_a": (_PEG_BOXES, PEG["mass"], 0.6, ORANGE), "peg_b": (_PEG_BOXES, PEG["mass"], 0.6, ORANGE)}
BLOCK_START = {"block_a": (0.13, 0.26), "block_b": (0.21, 0.27)}
PEG_START = {"peg_a": (0.13, -0.26), "peg_b": (0.21, -0.27)}
RACK_H = 0.03  # pegs hang by their heads across two rails, this high
BLOCKS, PEGS = tuple(BLOCK_START), tuple(PEG_START)

OBJECTS = [
    Box(DIAL_NAME := "dial", pos=(CX, CY, TOP), size=(0.01, 0.01, 0.01)),
    *[Box(n, pos=(*BLOCK_START[n], TOP + 0.0005), size=(0.01, 0.01, 0.01), mass=BLOCK["mass"]) for n in BLOCK_START],
    *[Box(n, pos=(*PEG_START[n], TOP + RACK_H + 0.0005), size=(0.01, 0.01, 0.01), mass=PEG["mass"]) for n in PEG_START],
    *[Box(f"rail_{n}_{s}", pos=(x, y + sign * 0.015, TOP + RACK_H / 2), size=(0.05, 0.01, RACK_H), color=STEEL)
      for n, (x, y) in PEG_START.items() for s, sign in (("l", 1), ("r", -1))],
]

# -- scoring -----------------------------------------------------------------------------------------------------------
IN_POCKET, SEAT_Z, FLAT, PEG_XY, PEG_Z = 0.011, 0.004, math.radians(5), 0.009, 0.004
HANDS_CLEAR, LIN, ANG, STILL_S = 0.08, 0.01, 0.5, 1.0
TURNING = math.radians(3)  # below this before the pad is pressed, the dial's being held back
HELD_S = 1.0  # slower than TURNING for this long, unstopped, is the dial held back: a part set down jolts it for a moment
_trial = {"still": 0.0, "bad": False, "moving": False, "slow": 0.0}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears `bad`."""
    _trial.update(still=0.0, bad=False, moving=False, slow=0.0)


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def pocket_at(turn, k):
    """The pocket's centre on the disc's top, and its turn from square."""
    th = FIXTURES[k] + turn
    return np.array((CX + M.POCKET["radius"] * math.cos(th), CY + M.POCKET["radius"] * math.sin(th), TOP + RIM)), th - math.pi / 2


def _offset(objects, name, k):
    """A block's offset from pocket k's centre, along the pocket's sides, and in height."""
    c, roll = pocket_at(objects["dial"].pos[0], k)
    d = np.asarray(objects[name].pos, float) - c
    s, co = math.sin(roll), math.cos(roll)
    return np.array((co * d[0] + s * d[1], -s * d[0] + co * d[1], d[2]))


def block_in(objects, k):
    """Whichever block is nearest pocket k: the blocks are identical, and neither belongs to a pocket."""
    return min(BLOCKS, key=lambda n: np.linalg.norm(_offset(objects, n, k)))


def offset_in_pocket(objects, k):
    return _offset(objects, block_in(objects, k), k)


def _flat(part):
    return _rot(part.quat_xyzw)[2, 2] >= math.cos(FLAT)


def _clear_of(objects, part, half, top):
    """Both fingertip midpoints this far from a part's box, in its own frame: open fingers sit 4.7 cm out, so 6 cm means no finger's on it."""
    R = _rot(part.quat_xyzw)
    for a in ("left_tcp", "right_tcp"):
        d = R.T @ (np.asarray(objects[a].pos, float) - np.asarray(part.pos, float))
        q = d - np.clip(d, (-half, -half, 0.0), (half, half, top))
        if np.linalg.norm(q) < 0.06:
            return False
    return True


def block_seated(objects, k):
    b = objects[block_in(objects, k)]
    o = offset_in_pocket(objects, k)
    return bool(abs(o[0]) <= IN_POCKET and abs(o[1]) <= IN_POCKET and abs(o[2]) <= SEAT_Z and _flat(b) and _clear_of(objects, b, BLOCK["half"], BLOCK["h"]))


def peg_seated(objects, k):
    """Whichever peg is nearest the block in pocket k, in that block's own frame: the dial's angle mustn't matter."""
    b = objects[block_in(objects, k)]
    R = _rot(b.quat_xyzw)
    p = min((objects[n] for n in PEGS), key=lambda q: np.linalg.norm(np.asarray(q.pos, float) - np.asarray(b.pos, float)))
    d = R.T @ (np.asarray(p.pos, float) - np.asarray(b.pos, float))
    return bool(block_seated(objects, k) and abs(d[0]) <= PEG_XY and abs(d[1]) <= PEG_XY and abs(d[2] - BLOCK["h"]) <= PEG_Z and _flat(p) and _clear_of(objects, p, PEG["head"], PEG["head_h"]))


def _pad():
    return np.array((*PAD_AT, TOP + M.PAD["z1"]))


def hands_clear(objects):
    return all(np.linalg.norm(np.asarray(objects[a].pos, float) - _pad()) >= HANDS_CLEAR for a in ("left_tcp", "right_tcp")) and all(
        _clear_of(objects, objects[n], BLOCK["half"], BLOCK["h"]) for n in BLOCKS) and all(_clear_of(objects, objects[n], PEG["head"], PEG["head_h"]) for n in PEGS)


def note_step(objects, dt):
    """Called by the embodiment after every control step: the dial held or stopped early, a part off the table, and the clock for all at rest."""
    dial = objects["dial"]
    done = all(peg_seated(objects, k) for k in (0, 1))
    if abs(dial.lin_vel[0]) >= TURNING:
        _trial["moving"] = True
    if dial.pos[1] >= PAD_PRESSED and not done and not dial.pos[2]:
        _trial["bad"] = True  # stopped before both are built
    _trial["slow"] = _trial["slow"] + dt if _trial["moving"] and not dial.pos[2] and dial.pos[1] < PAD_PRESSED and abs(dial.lin_vel[0]) < TURNING else 0.0
    if _trial["slow"] >= HELD_S:
        _trial["bad"] = True  # held back
    if any(objects[n].pos[2] < TOP - 0.01 for n in PARTS):
        _trial["bad"] = True
    still = abs(dial.lin_vel[0]) < TURNING and abs(dial.lin_vel[1]) < LIN and all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in PARTS) and hands_clear(objects)
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. 0 for good if the dial's stopped before both are built, if it's held back,
    or if a part leaves the table. Otherwise 0.2 for each block seated and let go, and a further 0.3 for each peg seated in its seated
    block and let go; 1.0 only with both built, the dial stopped, both hands clear and all at rest for the last 1 s."""
    if _trial["bad"]:
        return 0.0
    total = sum(0.2 * block_seated(objects, k) + 0.3 * peg_seated(objects, k) for k in (0, 1))
    if total >= 1.0 - 1e-9:
        return 1.0 if objects["dial"].pos[2] and hands_clear(objects) and _trial["still"] >= STILL_S else 0.9
    return round(total, 3)


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.03
TRACK_DT = 0.25
LEAD = 0.0  # seconds: how far ahead of the dial the hand aims


def _fingers(width):
    return -FINGER_PER_M * width


BLOCK_R, BLOCK_G = _fingers(2 * BLOCK["half"] + 0.03), _fingers(2 * BLOCK["half"] - 0.004)
PEG_R, PEG_G = _fingers(2 * PEG["head"] + 0.03), _fingers(2 * PEG["head"] - 0.004)
BLOCK_GZ, PEG_GZ = BLOCK["h"] - 0.012, PEG["head_h"] - 0.01


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

    def track(self, arm, k, dur, z0, z1, finger=None, label=""):
        """Follow fixture k for this long, moving from height z0 to z1 above the disc's top, turning the wrist with the dial."""
        n = max(1, round(dur / TRACK_DT))
        for i in range(1, n + 1):
            t = self.t[arm] + dur / n
            c, roll = pocket_at(RATE * (t + LEAD), k)
            z = z0 + (z1 - z0) * i / n
            f = None if finger is None else self.last[arm][1] + (finger - self.last[arm][1]) * i / n
            self.go(arm, dur / n, (c[0], c[1], c[2] + z), f, label, _wrap(roll))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a)) if abs(math.atan2(math.sin(a), math.cos(a))) <= math.pi / 4 else (a + math.pi / 2 * round(-a / (math.pi / 2)))


def _park(plan, arm):
    plan.go(arm, 1.4, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


def t_at(k, deg, after=0.0):
    """The first time after this that fixture k reaches this angle."""
    t = (math.radians(deg) - FIXTURES[k]) % (2 * math.pi) / RATE
    return t + 2 * math.pi / RATE * math.ceil(max(0.0, after - t) * RATE / (2 * math.pi))


def _place_block(plan, name, k, t_mid):
    arm = "left"
    x, y = BLOCK_START[name]
    plan.go(arm, 1.6, (x, y, TOP + BLOCK["h"] + 0.04), BLOCK_R, f"{name}: above it", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, TOP + BLOCK_GZ), label="descend")
    plan.go(arm, 0.6, finger=BLOCK_G, label="grip")
    plan.go(arm, 0.8, (x, y, TOP + 0.11), label="lift")
    # meet the pocket above its path, come down with it, let go, and go up with it
    t_meet = t_mid - 2.6
    c, roll = pocket_at(RATE * t_meet, k)
    plan.go(arm, max(1.2, t_meet - plan.t[arm] - 0.2), (c[0], c[1], c[2] + 0.07), label="over the pocket", yaw=_wrap(roll))
    plan.until(arm, t_meet)
    plan.track(arm, k, 1.0, 0.07, 0.07, label="match it")
    plan.track(arm, k, 1.6, 0.07, BLOCK_GZ + 0.002, label="down with it")
    plan.track(arm, k, 0.6, BLOCK_GZ + 0.002, BLOCK_GZ + 0.002, finger=BLOCK_R, label="let go")
    plan.track(arm, k, 1.0, BLOCK_GZ + 0.002, 0.075, label="up with it")
    _park(plan, arm)


def _place_peg(plan, name, k, t_mid):
    arm = "right"
    x, y = PEG_START[name]
    plan.go(arm, 1.6, (x, y, TOP + RACK_H + PEG["head_h"] + 0.04), PEG_R, f"{name}: above it", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, TOP + RACK_H + PEG_GZ), label="descend")
    plan.go(arm, 0.6, finger=PEG_G, label="grip")
    plan.go(arm, 0.8, (x, y, TOP + RACK_H + PEG_GZ + 0.04), label="lift")
    plan.go(arm, 0.6, (x, y, TOP + 0.13), label="up")
    top = BLOCK["h"]
    t_meet = t_mid - 2.6
    c, roll = pocket_at(RATE * t_meet, k)
    plan.go(arm, max(1.2, t_meet - plan.t[arm] - 0.2), (c[0], c[1], c[2] + top + PEG["shaft_l"] + PEG_GZ + 0.03), label="over the block", yaw=_wrap(roll))
    plan.until(arm, t_meet)
    z_up = top + PEG["shaft_l"] + PEG_GZ + 0.03
    plan.track(arm, k, 1.0, z_up, z_up, label="match it")
    plan.track(arm, k, 1.6, z_up, top + PEG_GZ + 0.002, label="down with it")
    plan.track(arm, k, 0.6, top + PEG_GZ + 0.002, top + PEG_GZ + 0.002, finger=PEG_R, label="let go")
    plan.track(arm, k, 1.0, top + PEG_GZ + 0.002, z_up, label="up with it")
    _park(plan, arm)


def demo():
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    # the left arm works as a fixture passes through 90 degrees, the right arm as it passes through 270
    swap = os.environ.get("DIAL_DEMO_SWAP") == "1"  # debugging only: the other block and peg first, since neither belongs to a pocket
    ba, bb, pa, pb = ("block_b", "block_a", "peg_b", "peg_a") if swap else ("block_a", "block_b", "peg_a", "peg_b")
    _place_block(plan, ba, 0, t_at(0, 90.0))
    if os.environ.get("DIAL_DEMO_EARLY") == "1":  # debugging only: press the stop pad with the job unfinished
        _press_pad(plan)
        return _finish(plan)
    _place_peg(plan, pa, 0, t_at(0, 270.0))
    _place_block(plan, bb, 1, t_at(1, 90.0))
    _place_peg(plan, pb, 1, t_at(1, 270.0, after=t_at(1, 90.0)))
    plan.until("left", plan.t["right"])
    _press_pad(plan)
    return _finish(plan)


def _press_pad(plan):
    arm = "left"
    px, py, z = *PAD_AT, TOP + M.PAD["z1"]
    plan.go(arm, 1.4, (px, py, z + 0.05), FINGER_CLOSED, "above the stop pad", 0.0)
    plan.go(arm, 0.8, (px, py, z + 0.016), label="onto it")
    plan.go(arm, 1.4, (px, py, z + 0.016 - M.PAD["travel"] - 0.02), label="press")
    plan.go(arm, 0.5, label="hold")
    plan.go(arm, 1.0, (px, py, z + 0.05), label="let it up")
    _park(plan, arm)


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
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:4])
    for k, n in enumerate(("A", "B")):
        print(n, "block at", round(t_at(k, 90.0), 1), "peg at", round(t_at(k, 270.0, after=t_at(k, 90.0)), 1))
    assert t * 20 < 3000 and not bad
