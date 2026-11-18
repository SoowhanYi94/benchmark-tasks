"""A worm gear reducer (Motovario NMRV 030, 1.2 kg) lies in a rollover fixture with its output face J down. Its output flange has to be fitted
to that face: retain the gearbox with the sliding bar, turn the fixture over and lock it with the bolt, seat the flange on the face's 55 mm
spigot (its four pre-fitted M6 socket head screws drop into the tapped holes only when the flange is turned to line up), and run the four screws
down with a stubby screwdriver. A scorer-applied 10 N lift then checks that the screws really hold the flange.

Run it:  inspect-robots run --embodiment yam_trunnion -E task=task.py -E object_jitter=0 --policy yam_scripted -P hold_s=10 --no-prompt \\
           --instruction "Secure the gearbox in the rollover fixture, turn it over and lock it, then fit the output flange on its spigot and screw it down." --scorer task_score

`yam_trunnion` (trunnion_fixture/embodiment.py) spawns the fixture and parts, applies the screw rule and the lift test that this file
decides, and reports each gripper's fingertip midpoint and the fixture's joints.
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
AX, Y0 = 0.24, -0.03  # the axle's x; the fixture's centre line in y
AXLE = TOP + M.AXLE
SCREWS = [f"screw{i}" for i in range(4)]
PARTS = ["gearbox", "flange", *SCREWS, "driver"]
GB_START = (AX + M.GB_UV[0], Y0 + M.GB_UV[1], AXLE + 0.0003)
FL_START = (M.POST["x"], M.POST["y"], TOP + M.POST["h"] - M.FL["recess"] + 0.0003)
DRV_START = (M.HOLSTER["x"], M.HOLSTER["y"], TOP + M.HOLSTER["h"] - M.DRV["blade"] + 0.0003)  # the handle resting on the holster's top
FL_YAW = math.radians(35)  # the flange sits on its post turned 35 deg from where its screws line up with the gearbox's holes


def _rz(v, a):
    return (v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a), v[2])


START = {"gearbox": GB_START, "flange": FL_START, "driver": DRV_START,
         **{s: tuple(np.add(FL_START, _rz(M.screw_offset(i), FL_YAW))) for i, s in enumerate(SCREWS)}}
START_Q = {n: (0.0, 0.0, math.sin(FL_YAW / 2), math.cos(FL_YAW / 2)) for n in ("flange", *SCREWS)}  # x, y, z, w
MASSES = {"gearbox": M.GB["mass"], "flange": M.FL["mass"], "driver": M.DRV["mass"], **{s: M.SC["mass"] for s in SCREWS}}

OBJECTS = [Box("fixture", pos=(AX, Y0, TOP), size=(0.01, 0.01, 0.01))] + [
    Box(n, pos=START[n], size=(0.01, 0.01, 0.01), mass=MASSES[n]) for n in PARTS]

# -- the screw rule (the embodiment applies the switches; this file decides) -----------------------------------------------------------------
PITCH = 0.001  # m per turn: the real M6 coarse pitch; a screw is run down 9 mm, 9 turns, to its seat
BIT_OFF, BIT_FLOOR, BIT_TILT = 0.0015, 0.0015, math.radians(8)  # the bit in the socket: off the axis, above the socket's floor, off square
CATCH_OFF, CATCH_DEPTH = 0.002, 0.0005  # the tip in a tapped hole's mouth: off its axis along each side of the drawn 8 mm square hole (as far as it
# lets the screw's chamfered point, 4 mm across at face J, sit), and below the face (it starts 1 mm in)
SEAT_TRAVEL = -(M.SC["length"] - M.FL["plate"])  # the tip's depth below face J with the head on the flange: -10 mm
OVERRUN = 0.0005  # the thread's target runs 0.5 mm past the seat, so a seated head clamps the flange
SEAT_GAP = 0.0005  # a head counts as seated within 0.5 mm of the flange's top
# the thread's resistance to the driver (this task's own estimates; Coulomb-like, saturating at this rate of turn)
T_RUN, T_SNUG, T_RATE = 0.15, 0.5, 0.3  # N m running down (thread and retaining washer); N m once the head is seated; rad/s
SNUG_TURN = math.radians(15)  # a seated screw is snug once the driver has turned it on this far against T_SNUG
LIFT_N, LIFT_RAMP, LIFT_HOLD, LIFT_RISE, LIFT_CUT = 10.0, 0.5, 1.0, 0.0005, 0.001  # the scorer's lift test on the flange


def _fresh_screws():
    return [{"hole": None, "target": 0.0, "turn": 0.0, "last": None, "unlocked": False, "on": False, "snug": 0.0} for _ in SCREWS]


_trial = {"still": 0.0, "bad": False, "why": "", "screws": _fresh_screws(), "lift": None, "lift_t": 0.0, "lift_ref": 0.0, "lift_max": 0.0,
          "lift_ok": False}
_actions = []  # switches for the embodiment: ("glue", i, on), ("thread", i, hole, on, target), ("target", i, hole, target)


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears `bad`."""
    _trial.update(still=0.0, bad=False, why="", screws=_fresh_screws(), lift=None, lift_t=0.0, lift_ref=0.0, lift_max=0.0, lift_ok=False)
    _actions.clear()


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _local(objects, name, p):
    o = objects[name]
    return _rot(o.quat_xyzw).T @ (np.asarray(p, float) - np.asarray(o.pos, float))


def _world(objects, name, p):
    o = objects[name]
    return np.asarray(o.pos, float) + _rot(o.quat_xyzw) @ np.asarray(p, float)


def _cradle(objects):
    th = objects["fixture"].pos[0]
    c, s = math.cos(th), math.sin(th)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _axle():
    return np.array((AX, Y0, AXLE))


# -- the gearbox, bar, turn and bolt --------------------------------------------------------------------------------------------------------
GB_U, GB_V, GB_W, GB_TILT = 0.005, 0.005, (-0.001, 0.0055), math.radians(10)
BAR_CLOSED, BOLT_HOME, OVER, ROLLING = M.BAR["travel"] - 0.005, 0.012, math.radians(6), math.radians(30)
LIN, ANG, RATE, STILL_S = 0.01, 0.5, math.radians(3), 1.0
RUN_STEPS = 9600


def gb_offset(objects):
    """The gearbox's output axis point in the cradle's axes, from where it lies at the start."""
    o = _cradle(objects).T @ (np.asarray(objects["gearbox"].pos, float) - _axle())
    return o - np.array((M.GB_UV[0], M.GB_UV[1], 0.0))


def gb_tilt(objects):
    return math.acos(float(np.clip(_rot(objects["gearbox"].quat_xyzw)[:, 2] @ _cradle(objects)[:, 2], -1.0, 1.0)))


def gb_in(objects):
    o = gb_offset(objects)
    return bool(abs(o[0]) <= GB_U and abs(o[1]) <= GB_V and GB_W[0] <= o[2] <= GB_W[1] and gb_tilt(objects) <= GB_TILT)


def bar_closed(objects):
    return objects["fixture"].pos[1] >= BAR_CLOSED


def bolt_home(objects):
    return objects["fixture"].pos[2] >= BOLT_HOME


def turned_over(objects):
    return abs(objects["fixture"].pos[0] - math.pi) <= OVER


# -- the flange --------------------------------------------------------------------------------------------------------------------------
FL_FACE, FL_CENTRE = 0.0005, 0.001


def flange_gaps(objects):
    """Each corner of the flange's housing face above face J, in the gearbox's frame (m); and its centre's distance from the output axis."""
    H = M.FL["half"]
    gaps = []
    for cx, cy in ((H, H), (-H, H), (-H, -H), (H, -H)):
        p = _local(objects, "gearbox", _world(objects, "flange", (cx, cy, 0.0)))
        gaps.append(-p[2] - M.GB["face"])
    c = _local(objects, "gearbox", objects["flange"].pos)
    return gaps, float(math.hypot(c[0], c[1]))


def flange_on(objects):
    """Seated on the J face's spigot: every corner of its face within 0.5 mm of face J, its centre within 1 mm of the output axis."""
    gaps, off = flange_gaps(objects)
    return bool(all(-FL_FACE <= g <= FL_FACE for g in gaps) and off <= FL_CENTRE)


def head_gap(objects, i):
    """Screw i's head above the flange's top (m), in the flange's frame."""
    return float(_local(objects, "flange", objects[SCREWS[i]].pos)[2] - M.FL["plate"])


def head_down(objects, i):
    s = _trial["screws"][i]
    return bool(s["on"] and head_gap(objects, i) <= SEAT_GAP)


def screw_seated(objects, i):
    """Down: head within 0.5 mm of the flange, turned on SNUG_TURN against T_SNUG once there, and never turned with the bolt out."""
    s = _trial["screws"][i]
    return bool(head_down(objects, i) and not s["unlocked"] and s["snug"] >= SNUG_TURN)


def _tip(objects, i):
    return _world(objects, SCREWS[i], (0.0, 0.0, -M.SC["length"]))


def _hole_of(objects, i):
    """(hole, the tip's offset from its axis along the hole's sides (the larger of the two), depth below face J) for the tapped hole nearest
    screw i's tip, in the gearbox's frame. The holes are drawn square, sides along the gearbox's axes."""
    p = _local(objects, "gearbox", _tip(objects, i))
    d = [math.hypot(p[0] - hx, p[1] - hy) for hx, hy in M.HOLES]
    j = int(np.argmin(d))
    return j, max(abs(p[0] - M.HOLES[j][0]), abs(p[1] - M.HOLES[j][1])), p[2] + M.GB["face"]


def bit_parts(objects, i):
    """The driver's bit against screw i's socket: (off the axis, above the socket's floor, off square in rad)."""
    R = _rot(objects[SCREWS[i]].quat_xyzw)
    floor = _world(objects, SCREWS[i], (0.0, 0.0, M.SC["head_h"] - M.SC["socket_d"]))
    d = np.asarray(objects["driver"].pos, float) - floor  # the driver's origin is its bit's tip
    along = float(d @ R[:, 2])
    off = float(np.linalg.norm(d - along * R[:, 2]))
    sq = float(np.clip(_rot(objects["driver"].quat_xyzw)[:, 2] @ R[:, 2], -1.0, 1.0))
    return off, along, math.acos(sq)


def bit_in(objects, i):
    """The driver's bit in screw i's socket: its tip within 1.5 mm of the screw's axis and of the socket's floor, square within 8 deg."""
    off, along, tilt = bit_parts(objects, i)
    return bool(off <= BIT_OFF and abs(along) <= BIT_FLOOR and tilt <= BIT_TILT)


def _driver_angle(objects, i):
    """The driver's turn about screw i's axis, measured against the gearbox (rad)."""
    a = _rot(objects["driver"].quat_xyzw)[:, 0]
    g = _rot(objects["gearbox"].quat_xyzw)
    ax = _rot(objects[SCREWS[i]].quat_xyzw)[:, 2]
    a = a - (a @ ax) * ax
    e1 = g[:, 0] - (g[:, 0] @ ax) * ax
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(ax, e1)
    return math.atan2(float(a @ e2), float(a @ e1))


def screw_step(objects):
    """The declared screw rule, once per control step. While the bit sits in a screw's socket, the driver's turn about the screw's axis is the
    screw's turn (clockwise from the head tightens). A screw whose tip is in a tapped hole catches its thread on the first tightening turn: it
    leaves its retaining washer (glue off) and from then on runs PITCH along the hole per turn, down to its seat. Turns made while the bolt is
    not home mark that screw as not credited."""
    locked = bolt_home(objects)
    for i, s in enumerate(_trial["screws"]):
        if not bit_in(objects, i):
            s["last"] = None
            continue
        a = _driver_angle(objects, i)
        if s["last"] is None:
            s["last"] = a
            continue
        tighten = -((a - s["last"] + math.pi) % (2 * math.pi) - math.pi)
        s["last"] = a
        if abs(tighten) < 1e-4:
            continue
        if tighten > 0 and not locked:
            s["unlocked"] = True
        if not s["on"]:
            j, off, depth = _hole_of(objects, i)
            if tighten > 0 and off <= CATCH_OFF and depth >= CATCH_DEPTH:
                s.update(on=True, hole=j, target=-depth, turn=0.0)
                _actions.append(("glue", i, False))
                _actions.append(("thread", i, j, True, -depth))
            continue
        s["turn"] += tighten
        if head_down(objects, i) and tighten > 0:
            s["snug"] += tighten
        s["target"] = float(np.clip(s["target"] - PITCH * tighten / (2 * math.pi), SEAT_TRAVEL - OVERRUN, 0.0))
        _actions.append(("target", i, s["hole"], s["target"]))


def driver_torque(objects):
    """The thread's resistance on the driver (world torque, N m) while its bit is in a caught screw's socket: it opposes the driver's turn
    about the screw's axis, T_RUN while the screw runs down and T_SNUG once its head is down, saturating above T_RATE."""
    for i, s in enumerate(_trial["screws"]):
        if s["on"] and bit_in(objects, i):
            ax = _rot(objects[SCREWS[i]].quat_xyzw)[:, 2]
            w = float(np.asarray(objects["driver"].ang_vel, float) @ ax)
            T = T_SNUG if head_down(objects, i) else T_RUN
            return -T * float(np.clip(w / T_RATE, -1.0, 1.0)) * ax
    return np.zeros(3)


def take_actions():
    out = list(_actions)
    _actions.clear()
    return out


# -- hands, rest, the lift test, score --------------------------------------------------------------------------------------------------------
def _clear_of_box(objects, name, lo, hi):
    """Both fingertip midpoints 6 cm or more from a part's box, in its own frame: open fingers sit 4.7 cm out, so no finger is on it."""
    for a in ("left_tcp", "right_tcp"):
        d = _local(objects, name, objects[a].pos)
        if np.linalg.norm(d - np.clip(d, lo, hi)) < 0.06:
            return False
    return True


def _knob(objects):
    th = objects["fixture"].pos[0]
    return np.array((AX - M.CRANK["r"] * math.cos(th), Y0 + M.CRANK["y"], AXLE + M.CRANK["r"] * math.sin(th)))


def _bolt_lug(objects):
    return np.array((AX, Y0 + M.BOLT["end"] - objects["fixture"].pos[2], TOP + M.BOLT["lug_top"]))


def flange_free(objects):
    H = M.FL["half"]
    return _clear_of_box(objects, "flange", (-H, -H, 0.0), (H, H, M.FL["top"]))


def hands_clear(objects):
    spots = (_knob(objects), _bolt_lug(objects))
    d = M.DRV
    return (all(np.linalg.norm(np.asarray(objects[a].pos, float) - s) >= 0.06 for a in ("left_tcp", "right_tcp") for s in spots)
            and flange_free(objects)
            and _clear_of_box(objects, "driver", (-d["handle_r"], -d["handle_r"], 0.0), (d["handle_r"], d["handle_r"], d["blade"] + d["handle"])))


def _latch(why):
    if not _trial["bad"]:
        _trial.update(bad=True, why=why)


def lift_force():
    """The upward force (N, world) the embodiment puts on the flange now: the scorer's lift test."""
    if _trial["lift"] == "ramp":
        return LIFT_N * min(1.0, _trial["lift_t"] / LIFT_RAMP)
    return LIFT_N if _trial["lift"] == "hold" else 0.0


def _rise(objects):
    return max(flange_gaps(objects)[0])


def note_step(objects, dt):
    """Called by the embodiment after every control step: the latched zero, the screw rule, the rest clock and the lift test."""
    f = objects["fixture"]
    if f.pos[0] > ROLLING and not (gb_in(objects) and bar_closed(objects)):
        _latch("turned with the bar not closed over the gearbox")
    if any(objects[n].pos[2] < TOP - 0.01 for n in ("gearbox", "flange")):
        _latch("the gearbox or the flange left the table")
    screw_step(objects)
    parts = ("gearbox", "flange", "driver")
    still = (abs(f.lin_vel[0]) < RATE and abs(f.lin_vel[1]) < LIN and abs(f.lin_vel[2]) < LIN and hands_clear(objects)
             and all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in parts))
    _trial["still"] = _trial["still"] + dt if still else 0.0
    ready = all(screw_seated(objects, i) for i in range(4)) and flange_on(objects)
    t = _trial
    if t["lift"] is None:
        if t["lift_ok"] and (not ready or abs(_rise(objects) - t["lift_ref"]) > 0.0003):
            t["lift_ok"] = False  # the flange moved or a screw came loose afterwards: test again once still
        if ready and not t["lift_ok"] and t["still"] >= STILL_S:
            t.update(lift="ramp", lift_t=0.0, lift_ref=_rise(objects), lift_max=0.0)
        return
    t["lift_t"] += dt
    t["lift_max"] = max(t["lift_max"], _rise(objects) - t["lift_ref"])
    if t["lift_max"] > LIFT_CUT:  # it is coming off: stop pulling, failed
        t.update(lift=None, lift_ok=False)
        t["still"] = 0.0
    elif t["lift"] == "ramp" and t["lift_t"] >= LIFT_RAMP:
        t.update(lift="hold", lift_t=0.0)
    elif t["lift"] == "hold" and t["lift_t"] >= LIFT_HOLD:
        t.update(lift=None, lift_ok=t["lift_max"] <= LIFT_RISE)
        t["still"] = 0.0


def score(objects):
    """Read at the end of the run; there's no early end. 0 for good if the fixture turns past 30 degrees without the gearbox in it under the
    closed bar, or if the gearbox or flange leaves the table. Otherwise: 0.25 gearbox in the cradle, bar closed; 0.5 also turned over and the
    bolt home; 0.75 also the flange seated on the J face's spigot and let go; 0.9 also two diagonally opposite screws down (seated, then
    turned on 15 deg against 0.5 N m); 1.0 all four down, the 10 N lift test passed, both hands clear of the flange, driver, knob and bolt, and all at rest for the last 1 s."""
    if _trial["bad"] or not (gb_in(objects) and bar_closed(objects)):
        return 0.0
    if not (turned_over(objects) and bolt_home(objects)):
        return 0.25
    if not (flange_on(objects) and flange_free(objects)):
        return 0.5
    seated = [screw_seated(objects, i) for i in range(4)]
    if not ((seated[0] and seated[2]) or (seated[1] and seated[3])):
        return 0.75
    if all(seated) and _trial["lift_ok"] and hands_clear(objects) and _trial["still"] >= STILL_S:
        return 1.0
    return 0.9


# -- scripted demo -----------------------------------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT, YAW_STEP = 0.03, math.radians(30)
KNOB_YAW = math.pi / 2


def _fingers(width):
    return -FINGER_PER_M * width


KNOB_R, KNOB_G = _fingers(2 * M.CRANK["half_x"] + 0.03), _fingers(2 * M.CRANK["half_x"] - 0.006)
KNOB_GZ = 0.003
NECK = 2 * M.FL["neck_r"]
FL_R, FL_G = _fingers(NECK + 0.016), _fingers(NECK - 0.008)
HANDLE = 2 * M.DRV["handle_r"]
DRV_R, DRV_G, DRV_LOOSE = _fingers(HANDLE + 0.016), _fingers(HANDLE - 0.020), _fingers(HANDLE + 0.005)  # 10 mm a side: the full 50 N squeeze
FL_TCP = 0.031  # the hand's TCP above the flange's housing face when holding the neck (its pads stop just above the plate)
DRV_TCP = M.DRV["blade"] + 0.0246  # the hand's TCP above the bit's tip when holding the handle (its pads at the handle's bottom)
STROKE_HI, STROKE_LO = math.radians(150), math.radians(-60)  # the right wrist's turning stroke: clockwise from above tightens
STROKES = int(os.environ.get("TRUNNION_STROKES", "18"))  # 18 x 210 deg = 10.5 turns: 9 to the seat, then on against T_SNUG


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
        n = max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / SEGMENT), math.ceil(abs(y - y0) / YAW_STEP))
        if label == "away" and abs(y - y0) < 1e-9:
            n = 1
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


def knob_at(th, lift=KNOB_GZ):
    return AX - M.CRANK["r"] * math.cos(th), Y0 + M.CRANK["y"], AXLE + M.CRANK["r"] * math.sin(th) + lift


DEMO_ROLL = math.radians(float(os.environ.get("TRUNNION_ROLL", "183.2")))  # where the bolted cradle comes to rest (seen in a dev run)
DEMO_GB = (0.0001, -0.0013, 0.004)  # where the turned gearbox lies in the cradle's axes: on the bar, in its play (seen in a dev run)


def face_frame():
    """Face J's centre and the seated flange's rotation, once the cradle is turned over and bolted (world): what the camera shows."""
    c, s = math.cos(DEMO_ROLL), math.sin(DEMO_ROLL)
    Rc = np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))
    o = _axle() + Rc @ np.array((M.GB_UV[0] + DEMO_GB[0], M.GB_UV[1] + DEMO_GB[1], DEMO_GB[2]))
    return o + Rc @ np.array((0.0, 0.0, -M.GB["face"])), Rc @ np.array(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))


def output_axis_turned():
    """Where the output axis meets face J once the cradle is turned over (world)."""
    return face_frame()[0]


def screw_point(i, z):
    """A point on screw i's axis, z above the seated flange's housing face (world)."""
    c, Rf = face_frame()
    return c + Rf @ np.array((M.HOLES[i][0], M.HOLES[i][1], z))


def slide_bar(plan):
    """The closed right fingers behind the bar's lug, pushing along +y, aiming past the bar's stop."""
    arm, br = "right", M.BAR
    z = AXLE + br["lug_w"] - 0.012
    face = Y0 + br["park"] - br["half_v"] - br["lug_v"]
    plan.go(arm, 1.6, (AX, face - 0.035, z + 0.03), FINGER_CLOSED, "behind the bar's lug", 0.0)
    plan.go(arm, 0.6, (AX, face - 0.035, z), label="down")
    plan.go(arm, 4.0, (AX, face + br["travel"] + 0.004, z), label="slide the bar across")
    plan.go(arm, 0.4, label="hold")
    plan.go(arm, 1.6, (AX, face - 0.02, z), label="back off")
    plan.go(arm, 0.6, (AX, face - 0.02, z + 0.03), label="up")


ROLL_AIM = float(os.environ.get("TRUNNION_AIM", "181.0"))  # degrees: the hand ends here; the cradle came to rest 1 deg past its aim in a dev run


def roll(plan, to=ROLL_AIM, dur=12.0):
    """The right hand takes the sleeve and carries it over the top, staying square: the sleeve turns on its pin."""
    arm = "right"
    plan.go(arm, 1.6, (*knob_at(0.0)[:2], AXLE + 0.08), KNOB_R, "above the knob", KNOB_YAW)
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 1.0, knob_at(0.0), label="down onto the knob")
    plan.go(arm, 0.8, finger=KNOB_G, label="grip the knob")
    plan.go(arm, 0.3, label="settle")
    n = 30
    for i in range(1, n + 1):
        plan.go(arm, dur / n, knob_at(math.radians(to) * i / n), label="over the top")


def release_knob(plan, at=ROLL_AIM):
    arm = "right"
    plan.go(arm, 0.6, finger=KNOB_R, label="let go of the knob")
    x, y, z = knob_at(math.radians(at))
    plan.go(arm, 1.0, (x, y, z + 0.08), label="up")
    _park(plan, arm)


def push_bolt(plan):
    """The closed left fingers behind the bolt's lug, pushing along -y, aiming past its stop."""
    arm, bo = "left", M.BOLT
    z = TOP + bo["lug_top"] - 0.012
    y = Y0 + bo["end"]
    plan.go(arm, 1.6, (AX, y + 0.035, z + 0.03), FINGER_CLOSED, "behind the bolt", 0.0)
    plan.go(arm, 0.6, (AX, y + 0.035, z), label="down")
    plan.go(arm, 2.0, (AX, y - bo["travel"] - 0.003, z), label="push the bolt home")
    plan.go(arm, 0.4, label="hold")
    plan.go(arm, 1.0, (AX, y + 0.03, z), label="back off")
    plan.go(arm, 0.6, (AX, y + 0.03, z + 0.03), label="up")


def fit_flange(plan, target=None, turn=True):
    """The left hand takes the flange by its neck from the kitting post (fingers between the screw heads), turns it 35 deg on the way so
    its screws line up with the tapped holes, sets it on the spigot a few mm up, lets go, and presses its neck."""
    arm = "left"
    x, y, z = FL_START
    t = output_axis_turned() if target is None else np.asarray(target, float)
    carry = TOP + 0.155
    y1 = 0.0 if turn else FL_YAW
    plan.go(arm, 1.6, (x, y, z + FL_TCP + 0.05), FL_R, "above the flange", FL_YAW)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, z + FL_TCP), label="down")
    plan.go(arm, 0.8, finger=FL_G, label="grip the neck")
    plan.go(arm, 1.0, (x, y, carry), label="lift")
    plan.go(arm, 1.2, label="turn it to line up its screws", yaw=y1)  # turned over the post: its corners would catch the pillar on the way
    plan.go(arm, 2.0, (0.155, t[1], carry), label="in front of the pillar")
    plan.go(arm, 1.5, (t[0], t[1], carry), label="over the spigot")
    plan.go(arm, 0.6, label="settle")
    plan.go(arm, 1.4, (t[0], t[1], t[2] + FL_TCP + 0.003), label="onto the spigot")
    plan.go(arm, 0.5, label="settle")
    plan.go(arm, 0.6, finger=FL_R, label="let go")
    plan.go(arm, 0.8, (t[0], t[1], t[2] + M.FL["top"] + 0.0236 + 0.008), label="open hand up, clear of the neck")
    plan.go(arm, 0.5, finger=FINGER_CLOSED, label="close above the neck")
    plan.go(arm, 0.8, (t[0], t[1], t[2] + M.FL["top"] + 0.0236 - 0.004), label="press the neck")
    plan.go(arm, 0.6, label="hold")
    plan.go(arm, 0.8, (t[0], t[1], carry), label="up")
    plan.go(arm, 1.5, (0.155, t[1], carry), label="back in front of the pillar", yaw=0.0)  # a straight way to the park crosses the bolt's lug
    _park(plan, arm)


def screw_xy(i):
    """Screw i over its hole once the flange is on (world x, y): the flange sits square, so its holes are the gearbox's, mirrored in x."""
    p = screw_point(i, M.SCREW_START_Z)
    return p[0], p[1]


def take_driver(plan):
    arm = "right"
    x, y, z = DRV_START
    plan.go(arm, 1.8, (x, y, z + DRV_TCP + 0.05), DRV_R, "above the driver", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, z + DRV_TCP), label="down")
    plan.go(arm, 0.6, finger=DRV_G, label="grip the handle")
    plan.go(arm, 1.0, (x, y, z + DRV_TCP + 0.045), label="out of the holster")


def _arc(plan, arm, a0, a1, z, dur):
    """Move the hand along a circle round the output axis (r 33.5 mm, over the heads, clear of the flange's neck), from angle a0 to a1."""
    c = output_axis_turned()
    n = max(1, math.ceil(abs(a1 - a0) / math.radians(30)))
    for k in range(1, n + 1):
        a = a0 + (a1 - a0) * k / n
        plan.go(arm, dur / n, (c[0] + ARC_R * math.cos(a), c[1] + ARC_R * math.sin(a), z), label="round the neck")


ARC_R = 0.0335


DRV_IN_HAND = (0.0017, 0.0001)  # the bit's tip off the hand's axis at yaw 0 (m, world x, y), as the handle sits in the fingers (seen in a dev run)


def _bit_go(plan, i, z0, z1, y0, y1, dur, label, finger=None):
    """Move the hand so that the bit's tip runs from z0 to z1 over screw i's axis (heights over the flange's face) while the hand turns from
    yaw y0 to y1: the tip stays on the axis because the hand orbits it by the tip's offset in the fingers."""
    n = max(1, math.ceil(abs(y1 - y0) / math.radians(15)))
    for k in range(1, n + 1):
        yw = y0 + (y1 - y0) * k / n
        c, s_ = math.cos(yw), math.sin(yw)
        off = np.array((c * DRV_IN_HAND[0] - s_ * DRV_IN_HAND[1], s_ * DRV_IN_HAND[0] + c * DRV_IN_HAND[1], 0.0))
        p = screw_point(i, z0 + (z1 - z0) * k / n) + np.array((0.0, 0.0, DRV_TCP + 0.0005)) - off
        plan.go("right", dur / n, p, finger if k == n else None, label, yw)


def drive_screws(plan, order=(0, 2, 1, 3), strokes=STROKES, short=None):
    arm = "right"
    zf0 = M.SCREW_START_Z + M.SC["head_h"] - M.SC["socket_d"]  # a socket's floor before its screw runs down, over the flange's face
    zseat = M.FL["plate"] + M.SC["head_h"] - M.SC["socket_d"]
    zh = M.SCREW_START_Z + M.SC["head_h"] + 0.003  # the bit 3 mm over an unscrewed head
    hover = max(screw_point(i, zh)[2] for i in range(4)) + DRV_TCP
    c = output_axis_turned()
    ang = [math.atan2(M.HOLES[i][1], M.HOLES[i][0]) for i in range(4)]
    prev = None
    for i in order:
        if prev is None:
            plan.go(arm, 2.0, (c[0] + ARC_R * math.cos(ang[i] - 0.5), c[1] + ARC_R * math.sin(ang[i] - 0.5), hover), label="to the flange", yaw=STROKE_HI)
            _arc(plan, arm, ang[i] - 0.5, ang[i], hover, 0.6)
        else:
            a1 = ang[i] if abs(ang[i] - ang[prev]) <= math.pi + 1e-6 else ang[i] - 2 * math.pi * np.sign(ang[i] - ang[prev])
            _arc(plan, arm, ang[prev], a1, hover, 0.5 * abs(a1 - ang[prev]) / (math.pi / 2) + 0.4)
        _bit_go(plan, i, zh, zh, STROKE_HI, STROKE_HI, 0.5, f"over screw {i}")
        plan.go(arm, 0.3, label="settle")
        zf = zf0
        _bit_go(plan, i, zh, zf, STROKE_HI, STROKE_HI, 1.2, "bit into the socket")
        plan.go(arm, 0.3, label="settle")
        sweep = STROKE_HI - STROKE_LO
        n_strokes = strokes // 2 if i == short else strokes
        for k in range(n_strokes):  # the hand keeps hold of the driver throughout: turn, lift the bit clear, turn back in the air, set it in again
            z1 = max(zf - PITCH * sweep / (2 * math.pi), zseat)
            _bit_go(plan, i, zf, z1, STROKE_HI, STROKE_LO, 1.5, f"turn {k + 1}")
            zf = z1
            if k < n_strokes - 1:
                _bit_go(plan, i, zf, zf + 0.004, STROKE_LO, STROKE_LO, 0.4, "bit out")
                _bit_go(plan, i, zf + 0.004, zf + 0.004, STROKE_LO, STROKE_HI, 1.0, "wind back")
                _bit_go(plan, i, zf + 0.004, zf, STROKE_HI, STROKE_HI, 0.6, "bit in")
                plan.go(arm, 0.2, label="settle")
        _bit_go(plan, i, zf, zh, STROKE_LO, STROKE_LO, 0.6, "bit out")
        p = plan.last[arm][0]
        plan.go(arm, 0.3, (p[0], p[1], hover), label="up")
        plan.go(arm, 0.8, label="wind back", yaw=STROKE_HI)
        prev = i
    p = plan.last[arm][0]
    plan.go(arm, 1.0, (p[0], p[1], hover + 0.02), label="up")


def stow_driver(plan):
    arm = "right"
    x, y, z = DRV_START
    x, y = x - DRV_IN_HAND[0], y - DRV_IN_HAND[1]  # the bit, not the hand, over the bore
    plan.go(arm, 3.5, (x, y, z + DRV_TCP + 0.045), label="over the holster", yaw=0.0)
    plan.go(arm, 0.5, label="settle")
    plan.go(arm, 1.2, (x, y, z + DRV_TCP + 0.002), label="into the holster")
    plan.go(arm, 0.5, finger=DRV_R, label="let go")
    plan.go(arm, 0.8, (x, y, z + DRV_TCP + 0.06), label="up")
    _park(plan, arm)


def demo():
    plan = _Plan()
    mode = os.environ.get("TRUNNION_DEMO", "")  # wrong ways: noturn (flange on the up face, no turn), nobolt (turned, not bolted), noscrews, noalign (flange set down unturned),
    # halfscrew (the last screw, 3, gets half its strokes: left about half run)
    for a in SIDE:
        _park(plan, a)
    slide_bar(plan)
    if mode == "noturn":  # the flange is set on face K (up at the start) and the screws are driven there
        plan.until("left", plan.t["right"])
        _park(plan, "right")
        k = np.array((AX + M.GB_UV[0], Y0 + M.GB_UV[1], AXLE + M.GB["face"]))
        fit_flange(plan, target=(k[0], k[1], k[2] + M.BAR["w1"] - M.GB["face"]))
        return _finish(plan)
    roll(plan)
    plan.until("left", plan.t["right"])
    if mode == "nobolt":  # the knob is let go with the bolt withdrawn: the cradle runs on to its far stop; the rest goes ahead
        release_knob(plan)
    else:
        push_bolt(plan)
        plan.until("right", plan.t["left"])
        release_knob(plan)
    plan.until("left", plan.t["right"])
    fit_flange(plan, turn=mode != "noalign")
    if mode in ("noscrews", "noalign"):
        return _finish(plan)
    plan.until("right", plan.t["left"] - 4.0)
    take_driver(plan)
    plan.until("right", plan.t["left"])
    drive_screws(plan, short=3 if mode == "halfscrew" else None)
    stow_driver(plan)
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


if __name__ == "__main__":
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad = ik_errors(wps)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:8])
    assert t * 20 < RUN_STEPS and not bad
