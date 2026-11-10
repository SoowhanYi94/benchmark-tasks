"""A cast-iron V-belt pulley is seized on a gearmotor's output shaft, its back face 19 mm from the bearing housing. Two pry bars lie
on wooden rests in front of the machine. The robot must break the pulley loose with the bars, without touching it by hand while it
is still seized, then take it off the shaft and stand it in the V-block.

Run it:  inspect-robots run --embodiment yam_pulley -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Use the pry bars to break the pulley loose on the motor shaft without touching it by hand, then take it off the shaft and stand it in the V-block." --scorer task_score

`yam_pulley` (pulley_pry/embodiment.py) spawns the machine, pulley and bars and reports each gripper's fingertip midpoint.
"""

import math
import os

import numpy as np

from pulley_pry import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
P, B = M.PULLEY, M.BAR
XP = M.XH + M.GAP  # the pulley's back face at the start
ZS0 = M.ZS
DRAG = float(os.environ.get("PULLEY_DRAG", "60"))  # N: the seized band's grip on the bore, as a capped axial drag
DRAG_DAMPING = 2.0e6  # N s/m: below the cap the drag is viscous; 20 N creeps 0.6 mm a minute
SEAT = M.SEIZED  # m: the seized band. Once the pulley has come this far it is loose: the drag is switched off for good
RACK_X = 0.15
RACK = {"l": (RACK_X, 0.11, 0.25), "r": (RACK_X, -0.11, -0.25)}
RACK_TIP = 0.075  # a bar's tip end lies this far from the centre line on its rests
VBLOCK = (0.30, -0.19)
V_Z = M.vblock_rest_z()

SEATED = os.environ.get("PULLEY_START") == "seated"  # debugging only: both bars start in the gap, tips against the bearing cap


def _bar_start(sg):
    if SEATED:
        return (M.XH + B["t"] / 2, sg * (M.BOSS + 0.0005), TOP + M.ZS)
    return (RACK_X, sg * RACK_TIP, TOP + 0.04 + B["h"] / 2 + 0.002)


OBJECTS = [
    Box("machine", pos=(0.0, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    Box("pulley", pos=(XP, 0.0, TOP + M.ZS), size=(0.01, 0.01, 0.01), mass=P["mass"]),
    Box("bar_l", pos=_bar_start(1), size=(0.01, 0.01, 0.01), mass=B["mass"]),
    Box("bar_r", pos=_bar_start(-1), size=(0.01, 0.01, 0.01), mass=B["mass"]),
]
SPAWNED = ("machine", "pulley", "bar_l", "bar_r")
PARTS = ("pulley", "bar_l", "bar_r")

# -- scoring -----------------------------------------------------------------------------------------------------------
RUNGS = ((0.003, 0.25), (0.006, 0.5), (SEAT, 0.75))
TOUCH = 0.05  # a hand this close to the pulley's box while it's seized is a hand on it
CLEAR, LIN, ANG, STILL_S = 0.06, 0.01, 0.5, 1.0
IN_V = dict(x=0.025, y=0.012, z=0.008, axis=math.radians(10))
_trial = {"loose": False, "bad": False, "why": "", "still": 0.0}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears `bad`."""
    _trial.update(loose=False, bad=False, why="", still=0.0)


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def travel(objects):
    """How far the pulley's back face has come off along the shaft."""
    return float(objects["pulley"].pos[0] - XP)


def cock(objects):
    """The pulley axis's yaw and pitch from the shaft's, in degrees."""
    a = _rot(objects["pulley"].quat_xyzw)[:, 0]
    return math.degrees(math.atan2(a[1], a[0])), math.degrees(math.atan2(a[2], math.hypot(a[0], a[1])))


def bar_turn(objects, name):
    a = _rot(objects[name].quat_xyzw)[:, 1]
    return math.degrees(math.atan2(-a[0], a[1]))


def bar_tilt(objects, name):
    a = _rot(objects[name].quat_xyzw)[:, 1]
    return math.degrees(math.asin(max(-1.0, min(1.0, a[2]))))


def _near(objects, part, lo, hi, d):
    """Either fingertip midpoint within d of a part's box, in the part's own frame."""
    R = _rot(part.quat_xyzw)
    for a in ("left_tcp", "right_tcp"):
        q = R.T @ (np.asarray(objects[a].pos, float) - np.asarray(part.pos, float))
        if np.linalg.norm(q - np.clip(q, lo, hi)) < d:
            return True
    return False


PULLEY_BOX = ((0.0, -P["r_o"], -P["r_o"]), (P["hub_l"], P["r_o"], P["r_o"]))


def hands_on_pulley(objects, d=TOUCH):
    return _near(objects, objects["pulley"], *PULLEY_BOX, d)


def in_vblock(objects):
    p = objects["pulley"]
    R = _rot(p.quat_xyzw)
    c = np.asarray(p.pos, float) + R @ np.array((P["rim_w"] / 2, 0.0, 0.0))  # the rim's middle
    return bool(abs(c[0] - VBLOCK[0]) <= IN_V["x"] and abs(c[1] - VBLOCK[1]) <= IN_V["y"] and abs(c[2] - (TOP + V_Z)) <= IN_V["z"]
                and abs(R[0, 0]) >= math.cos(IN_V["axis"]))


def note_step(objects, dt):
    """Called after every control step: the seized band cleared, a hand on the pulley while seized, a part off the table, the clock for all at rest."""
    if not _trial["loose"] and hands_on_pulley(objects):
        _trial["bad"], _trial["why"] = True, _trial["why"] or "hand on the seized pulley"
    if travel(objects) >= SEAT:
        _trial["loose"] = True
    if any(objects[n].pos[2] < TOP - 0.01 for n in PARTS):
        _trial["bad"], _trial["why"] = True, _trial["why"] or "off the table"
    still = all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in PARTS)
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run. 0 for good after a hand on the pulley while it's seized, or a part off the table. Otherwise by how far it
    has come along the shaft: 0.25 at 3 mm, 0.5 at 6 mm, 0.75 once loose (8 mm), and still 0.75 anywhere off the shaft; 1.0 standing
    in the V-block, both hands 6 cm from it, everything at rest for the last 1 s."""
    if _trial["bad"]:
        return 0.0
    if _trial["loose"] and in_vblock(objects) and not hands_on_pulley(objects, CLEAR) and _trial["still"] >= STILL_S:
        return 1.0
    if _trial["loose"]:
        return 0.75
    s = travel(objects)
    return max([0.0] + [v for d, v in RUNGS if s >= d])


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
BAR_OF = {"left": "bar_l", "right": "bar_r"}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.02
GRIP_AT = 0.21  # along a bar from its tip end
BAR_X = M.XH + B["t"] / 2 + 0.0015  # the bar's centre line, its back against the housing face
THETA = float(os.environ.get("PULLEY_THETA", "0.85"))  # rad: how far the demo swings each handle (aimed past what the pulley needs)


def _fingers(width):
    return -FINGER_PER_M * width


SLEEVE = B["t"] + 0.006
BAR_R, BAR_G = _fingers(SLEEVE + 0.03), _fingers(SLEEVE - 0.006)
HUB_R, HUB_G = _fingers(2 * P["r_hub"] + 0.03), _fingers(2 * P["r_hub"] - 0.02)
HUB_GRIP = P["hub_l"] - 0.008  # along the pulley axis from its back face: 16 mm clear of the rim's front face


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


def _arc(arm, th, tip_y=M.BOSS):
    """The grip point on a bar whose tip end is at tip_y, back on the housing face, swung th about its fulcrum corner."""
    sg = SIDE[arm]
    fx, fy = M.XH, sg * (M.W - M.BEVEL)
    gx, gy = BAR_X - fx, sg * (tip_y + GRIP_AT) - fy
    a = sg * th
    return fx + gx * math.cos(a) - gy * math.sin(a), fy + gx * math.sin(a) + gy * math.cos(a)


def _bar_yaw(arm, th=0.0):
    return -SIDE[arm] * math.pi / 2 + SIDE[arm] * th  # fingers close across the bar (along x when th = 0); this way round the wrist has room to follow the swing


def _fetch_bar(plan, arm):
    sg = SIDE[arm]
    x, y, z = RACK_X, sg * (RACK_TIP + GRIP_AT), TOP + 0.04 + B["h"] / 2
    yw = _bar_yaw(arm)
    plan.go(arm, 1.6, (x, y, z + 0.06), BAR_R, f"{arm} bar: above it", yw)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, z + 0.001), label="descend")
    plan.go(arm, 0.6, finger=BAR_G, label="grip")
    plan.go(arm, 0.8, (x, y, TOP + M.ZS + 0.035), label="lift")
    plan.go(arm, 1.6, (BAR_X, y, TOP + M.ZS + 0.035), label="over the rails")
    plan.go(arm, 0.8, (BAR_X, y, TOP + M.ZS + 0.002), label="onto the rails")
    tx, ty = _arc(arm, 0.0, M.BOSS - 0.004)
    plan.go(arm, 1.4, (tx - 0.0005, ty, TOP + M.ZS + 0.002), label="tip into the gap")  # aimed 4 mm past the bearing cap
    plan.go(arm, 0.4, label="seated")


def _lever(plan, arm, dur, th0, th1, label):
    n = max(2, math.ceil(abs(th1 - th0) / 0.025))
    for k in range(1, n + 1):
        th = th0 + (th1 - th0) * k / n
        x, y = _arc(arm, th, M.BOSS - 0.004)
        plan.go(arm, dur / n, (x - 0.0005, y, TOP + M.ZS + 0.002), label=label, yaw=_bar_yaw(arm, th))


def _return_bar(plan, arm):
    sg = SIDE[arm]
    x, y = _arc(arm, 0.0, M.BOSS)
    plan.go(arm, 1.0, (x + 0.002, sg * (RACK_TIP + GRIP_AT), TOP + M.ZS + 0.004), label="bar out of the gap")
    plan.go(arm, 0.6, (x + 0.002, sg * (RACK_TIP + GRIP_AT), TOP + M.ZS + 0.035), label="lift it")
    plan.go(arm, 1.6, (RACK_X, sg * (RACK_TIP + GRIP_AT), TOP + M.ZS + 0.035), label="back over the rests")
    plan.go(arm, 0.8, (RACK_X, sg * (RACK_TIP + GRIP_AT), TOP + 0.04 + B["h"] / 2 + 0.003), label="onto the rests")
    plan.go(arm, 0.5, finger=BAR_R, label="let go")
    plan.go(arm, 0.6, (RACK_X, sg * (RACK_TIP + GRIP_AT), TOP + 0.04 + 0.08), label="up")


RIM_W = 0.050  # the fingers' opening over the rim: pads 12 mm clear of each face, the back pad 7 mm clear of the housing face
RIM_R, RIM_G = _fingers(RIM_W), _fingers(P["rim_w"] - 0.012)
RIM_Z = P["r_o"]  # TCP above the axis: the pad tips 32 mm above it, clear of the bearing cap (28 mm) and the hub (20 mm)
LOOSE_AT = SEAT + 0.002  # where the levered pulley's back face is taken to be: the levers are aimed past the 8 mm band


def _take_pulley(plan, arm, start):
    """Pinch the pulley across its back face and rim front face at the top (pads 40 mm wide: the hub is too short to pinch), slide it off,
    stand it in the V-block."""
    x0 = XP + start + P["rim_w"] / 2
    z = TOP + M.ZS
    yw = _bar_yaw(arm)  # fingers close along x
    plan.go(arm, 1.8, (x0, 0.0, z + RIM_Z + 0.035), RIM_R, "above the rim", yw)
    plan.go(arm, 0.8, label="settle")
    plan.go(arm, 1.2, (x0, 0.0, z + RIM_Z), label="down over the rim")
    plan.go(arm, 1.0, label="settle")
    plan.go(arm, 0.8, finger=RIM_G, label="pinch")
    # the back face must pass the shaft end (55 mm from its start); this is at the edge of the arm's reach at the pinch height
    x1 = M.SHAFT_END + 0.008 + P["rim_w"] / 2 + 0.005  # back face 8 mm past the shaft end, aimed 5 mm past: the hand lags at this reach
    plan.go(arm, 4.0, (x1, 0.0, z + RIM_Z + 0.002), label="slide it off")
    plan.go(arm, 1.2, label="clear of the end")
    plan.go(arm, 1.0, (x1, -0.035, z + RIM_Z + 0.004), label="sideways off the shaft line")  # no move back along x until the bore is clear of the shaft
    plan.go(arm, 1.4, (0.32, -0.10, z + RIM_Z + 0.05), label="lift")
    vx = VBLOCK[0]  # the pinch is on the rim's middle
    zc = TOP + V_Z + RIM_Z
    plan.go(arm, 1.6, (vx, VBLOCK[1], zc + 0.04), label="over the V-block")
    plan.go(arm, 0.6, label="settle")
    plan.go(arm, 1.4, (vx, VBLOCK[1], zc + 0.003), label="into the V")
    plan.go(arm, 0.6, finger=RIM_R, label="let go")
    plan.go(arm, 1.0, (vx, VBLOCK[1], zc + 0.075), label="up")


def demo():
    plan = _Plan()
    mode = os.environ.get("PULLEY_DEMO", "")  # debugging only
    for a in SIDE:
        _park(plan, a)
    if mode == "park":
        plan.go("left", 30.0, label="still")
        return _finish(plan)
    if mode == "grab":  # the shortcut: pinch the seized hub and pull it straight off by hand, aimed 10 cm past
        x0, z = XP + HUB_GRIP, TOP + M.ZS
        plan.go("right", 1.8, (x0, 0.0, z + 0.07), HUB_R, "above the hub", 0.0)
        plan.go("right", 1.0, (x0, 0.0, z + 0.002), label="down to the hub")
        plan.go("right", 0.6, finger=HUB_G, label="grip")
        plan.go("right", 4.0, (x0 + 0.10, 0.0, z + 0.002), label="pull")
        plan.go("right", 6.0, label="hold the pull")
        return _finish(plan)
    for a in SIDE:
        _fetch_bar(plan, a)
    t = max(plan.t.values())
    for a in SIDE:
        plan.until(a, t)
    levers = ("left",) if mode == "one" else tuple(SIDE)  # one: only the left bar is levered
    for a in levers:
        if mode == "seq":  # the shortcut: lever the bars one after the other, not together
            plan.until(a, max(plan.t.values()))
        _lever(plan, a, 7.0, 0.0, THETA, "lever")
        plan.go(a, 1.0, label="hold")
        _lever(plan, a, 2.0, THETA, 0.0, "swing back")
    t = max(plan.t.values())
    for a in SIDE:
        plan.until(a, t)
    for a in SIDE:
        _return_bar(plan, a)
    _park(plan, "left")
    if mode == "one":
        plan.go("right", 0.5, label="wait")
    _take_pulley(plan, "right", LOOSE_AT)
    _park(plan, "right")
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
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:6], len(bad))
    print("lever ratio at the grip", round((M.BOSS - 0.004 + GRIP_AT - M.W + M.BEVEL) / (M.W - M.BEVEL - M.BOSS - 0.003), 2), "V rest z", round(V_Z, 4))
    assert t * 20 < 3000 and not bad
