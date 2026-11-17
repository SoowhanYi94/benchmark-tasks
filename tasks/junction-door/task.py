"""A standard small sheet-steel junction box at its real catalogue size (200 x 150 x 80 mm) lies on its back on the
bench for wiring, screwed down, its hinged door standing open at 90 degrees. A red 22 mm pilot lamp is mounted in the door near the hinge; its black 5 mm cord
(30 cm) leaves the back of the lamp and hangs across the box and over its front edge, its connector plug lying on the
bench. Inside the box, on the back plate, are the panel socket for the plug and a spring cable clip. The robot must
plug the lamp in, clip the cord and shut the door until its catch clicks. After that the door is tested: it is
opened to its stop (120 degrees) and shut again; the plug must stay in and the cord must not get trapped.

Instruction: "Wire the door lamp to the socket inside the junction box, fixing its cord in the cable clip, and latch
the door shut."

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score --instruction "<INSTRUCTION>"
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bodykit  # noqa: E402  (hooks yam_arena: multi-shape bodies, hinge, sliders, step/reset callbacks)
import cablekit  # noqa: E402,F401  (hooks yam_arena: the cord)
import numpy as np  # noqa: E402
from bodykit import Body, BoxShape, CylShape  # noqa: E402
from cablekit import Plug, cable  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"

INSTRUCTION = ("Wire the door lamp to the socket inside the junction box, fixing its cord in the cable clip, and latch "
               "the door shut.")

DEBUG = bool(os.environ.get("JD_DEBUG"))
DEV = os.environ.get("JD_DEV", "")  # dev-only mechanism checks (dev/README in STATUS); unset for real trials

# -- colours ---------------------------------------------------------------------------------------------------
STEEL = (0.80, 0.81, 0.80)  # RAL 7035 light grey powder coat
DARK = (0.12, 0.12, 0.13)
CHROME = (0.70, 0.71, 0.74)
RED = (0.95, 0.08, 0.06)
GREEN = (0.10, 0.60, 0.25)
CORD = (0.08, 0.08, 0.08)
BLUE = (0.15, 0.35, 0.85)
NATURAL = (0.93, 0.91, 0.84)  # natural (unpigmented) PA66: the cable clip

# -- geometry (world frame: x forward, y left, z up, m) -----------------------------------------------------------
X0, X1 = 0.17, 0.37  # box outside, along the hinge (200 mm)
Y0, Y1 = -0.10, 0.05  # box outside across (150 mm); the hinge is on the +y (left) edge
H = 0.080  # box depth (outside), back plate on the bench
T = 0.002  # wall and back-plate thickness in the sim (real: 1.2 mm sheet steel)
XC, YC = (X0 + X1) / 2, (Y0 + Y1) / 2
HINGE = (XC, Y1, TOP + H)  # the hinge line runs along x through this point (outer top edge of the hinge-side wall)
OPEN0 = 90.0  # starting door angle (deg from shut)
STOP = 120.0  # the door's opening stop (deg)

# door (body frame = shut door, origin on the hinge line; the plate runs toward -y)
DOOR_W, DOOR_D, DOOR_T = 0.200, 0.150, 0.002
DOOR_MASS = 0.28  # 1 mm steel door 0.24 kg + lamp 0.04 kg
LAMP_LX, LAMP_LY = -0.080, -0.045  # lamp centre in the door frame (80 mm toward the robot from the door's middle,
#                                    45 mm in from the hinge line)
LAMP_BACK = 0.032  # how far the LED lamp body reaches behind (inside) the door
# catch: spring bolt in a housing on the front wall, keeper hook hanging from the door's free edge
CATCH_X = XC

# socket (panel connector, entry along +x) and spring clip on the back plate
SOCK_X = 0.330  # opening face (x), facing the robot
SOCK_Y = -0.035
SOCK_Z = TOP + 0.013  # cavity centre height
CAV_W, CAV_H, CAV_D = 0.019, 0.015, 0.016  # cavity: width (y) x height (z) x depth (x)
PL, PW, PH = 0.044, 0.015, 0.012  # plug body with its strain-relief boot: length (along the cord) x width x height
SEATED_MIN = 0.012  # m: seated when the plug's tip is at least 12 mm inside (16 mm is home)
# cable clip: HellermannTyton RB5-PA66-NA (151-14059), an adhesive snap-in clip; maker's dimensions L 19.0, W 19.0,
# W2 9.7, H 6.0 mm, cable up to 5.0 mm. Our own (not published): base 1 mm with its adhesive pad, two arms 2.1 mm
# thick (W2 9.7 = 2 x 2.1 + a 5.5 mm channel), one arm's flex modelled as a sprung jaw with the snap lip.
CLIP_L, CLIP_W, CLIP_W2, CLIP_H = 0.019, 0.019, 0.0097, 0.006
CLIP_XC = 0.268  # centre along x: unchanged from the earlier 16 mm clip, so the cord's slack across the hinge stays
CLIP_X0, CLIP_X1 = CLIP_XC - CLIP_L / 2, CLIP_XC + CLIP_L / 2  # the clip's length along x (the cord runs along x)
CLIP_Y = SOCK_Y + 0.0015  # where the cord naturally runs from the plug toward the hinge
CLIP_BASE = 0.001  # base with its adhesive pad (ours)
CLIP_FLOOR = TOP + T + CLIP_BASE  # channel floor
CLIP_TOP = TOP + T + CLIP_H  # arm tops: a seated 5 mm cord lies flush with them
CLIP_GAP = 0.0055  # between the arms (ours)
CLIP_ARM = (CLIP_W2 - CLIP_GAP) / 2  # 2.1 mm
LIP_Z = 0.0039  # the lip's edge above the floor (ours): above the seated cord's centre (2.5 mm)
LIP_IN = 0.00145  # how far the lip's edge reaches into the channel (ours): entry 4.05 mm for a 5 mm cord
SOCK_TOP = TOP + 0.023
CORD_R = 0.0025
CORD_LEN = 0.30

INNER = (X0 + T, X1 - T, Y0 + T, Y1 - T)  # inside of the walls


def _q_x(deg: float) -> tuple[float, float, float, float]:
    a = math.radians(deg) / 2
    return (math.sin(a), 0.0, 0.0, math.cos(a))


def door_to_world(p, open_deg: float) -> np.ndarray:
    """A door-frame point (shut-door frame) at door angle ``open_deg`` -> world."""
    a = math.radians(open_deg)
    lx, ly, lz = p
    # opening turns the door about +x by -open_deg: -y goes to +z at 90 deg
    y = math.cos(a) * ly + math.sin(a) * lz
    z = -math.sin(a) * ly + math.cos(a) * lz
    return np.array([HINGE[0] + lx, HINGE[1] + y, HINGE[2] + z])


LAMP_REAR_LOCAL = (LAMP_LX, LAMP_LY, -LAMP_BACK)
CORD_START = tuple(door_to_world(LAMP_REAR_LOCAL, OPEN0).tolist())


def _box(c, s, color, **kw) -> BoxShape:
    return BoxShape(tuple(float(v) for v in c), tuple(float(v) for v in s), color, **kw)


def _local(p) -> tuple:
    """World point -> junction-box body frame (origin XC, YC, TOP)."""
    return (p[0] - XC, p[1] - YC, p[2] - TOP)


def _chamfer(p1, p2, width, normal, t=0.0015) -> BoxShape:
    """A lead-in plate whose face (toward ``normal``) lies on the line p1-p2 (world points)."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    off = -n * t / 2
    return bodykit.plate(_local(np.asarray(p1) + off), _local(np.asarray(p2) + off), width, t, tuple(n), DARK)


def _jbox() -> Body:
    hh = H - T
    shapes = [
        _box((0, 0, T / 2), (X1 - X0, Y1 - Y0, T), STEEL),  # back plate
        _box((0, Y0 - YC + T / 2, T + hh / 2), (X1 - X0, T, hh), STEEL),  # front wall (-y)
        _box((0, Y1 - YC - T / 2, T + hh / 2), (X1 - X0, T, hh), STEEL),  # hinge-side wall (+y)
        _box((X0 - XC + T / 2, 0, T + hh / 2), (T, Y1 - Y0 - 2 * T, hh), STEEL),  # end walls
        _box((X1 - XC - T / 2, 0, T + hh / 2), (T, Y1 - Y0 - 2 * T, hh), STEEL),
    ]
    # hinge knuckles (visual only) on the outer top edge of the hinge-side wall
    for dx in (-0.075, 0.075):
        shapes.append(CylShape((dx, Y1 - YC, H), 0.004, 0.030, CHROME, collide=False, axis="X"))
    # catch housing on the front wall's inner face
    shapes.append(_box(_local((CATCH_X, Y0 + 0.006, TOP + 0.066)), (0.024, 0.008, 0.024), DARK))
    # panel socket: a block with a cavity opening toward -x (the robot); a sprung contact in the roof presses the plug
    # down onto the cavity floor
    sx, sy, sz = SOCK_X, SOCK_Y, SOCK_Z
    blk_len, back = CAV_D + 0.006, 0.006
    y_lo, y_hi = sy - CAV_W / 2, sy + CAV_W / 2
    z_lo, z_hi = sz - CAV_H / 2, sz + CAV_H / 2
    lead = 0.004  # 45 deg lead-in chamfer round the mouth
    wx, wl = sx + lead + (CAV_D - lead) / 2, CAV_D - lead
    shapes += [
        _box(_local((sx + CAV_D + back / 2, sy, (TOP + T + SOCK_TOP) / 2)), (back, 0.029, SOCK_TOP - TOP - T), DARK),
        _box(_local((wx, sy, (TOP + T + z_lo) / 2)), (wl, 0.029, z_lo - TOP - T), DARK),
        _box(_local((wx, sy, (z_hi + SOCK_TOP) / 2)), (wl, 0.029, SOCK_TOP - z_hi), DARK),
        _box(_local((wx, (sy - 0.0145 + y_lo) / 2, sz)), (wl, y_lo - (sy - 0.0145), CAV_H), DARK),
        _box(_local((wx, (sy + 0.0145 + y_hi) / 2, sz)), (wl, sy + 0.0145 - y_hi, CAV_H), DARK),
        _chamfer((sx, sy, z_lo - lead), (sx + lead, sy, z_lo), 0.029, (-1, 0, 1)),
        _chamfer((sx, sy, z_hi + lead), (sx + lead, sy, z_hi), 0.029, (-1, 0, -1)),
        _chamfer((sx, y_lo - lead, sz), (sx + lead, y_lo, sz), CAV_H + 2 * lead, (-1, 1, 0)),
        _chamfer((sx, y_hi + lead, sz), (sx + lead, y_hi, sz), CAV_H + 2 * lead, (-1, -1, 0)),
        _box(_local((sx - 0.0005, sy, SOCK_TOP + 0.0005)), (0.002, 0.020, 0.001), GREEN, collide=False),  # marking
    ]
    _ = blk_len
    # cable clip (RB5): 19 x 19 base on its adhesive pad, the fixed -y arm (the sprung +y arm is its own body)
    cx, cl = (CLIP_X0 + CLIP_X1) / 2, CLIP_X1 - CLIP_X0
    shapes += [
        _box(_local((cx, CLIP_Y, (TOP + T + CLIP_FLOOR) / 2)), (cl, CLIP_W, CLIP_BASE), NATURAL),
        _box(_local((cx, CLIP_Y - (CLIP_GAP + CLIP_ARM) / 2, (CLIP_FLOOR + CLIP_TOP) / 2)),
             (cl, CLIP_ARM, CLIP_TOP - CLIP_FLOOR), NATURAL),
    ]
    return Body.build("jbox", pos=(XC, YC, TOP), mass=2.0, shapes=shapes, kinematic=True, friction=(0.5, 0.45))


def _door() -> Body:
    lx, ly = LAMP_LX, LAMP_LY
    shapes = [
        _box((0, -DOOR_D / 2, DOOR_T / 2), (DOOR_W, DOOR_D, DOOR_T), STEEL),
        # pilot lamp: chrome bezel and red lens outside, black body and contact block inside
        CylShape((lx, ly, DOOR_T + 0.005), 0.015, 0.010, CHROME),
        CylShape((lx, ly, DOOR_T + 0.013), 0.0115, 0.006, RED),
        CylShape((lx, ly, -0.011), 0.0140, 0.022, DARK),
        _box((lx, ly, -0.027), (0.030, 0.030, 0.010), (0.85, 0.85, 0.82)),
        # keeper: a plate hanging from the free edge with a hook foot that the catch bolt snaps over
        _box((0, (Y0 + 0.0205) - Y1, -0.00825), (0.016, 0.003, 0.0165), DARK),
        _box((0, (Y0 + 0.01775) - Y1, -0.0155), (0.016, 0.0085, 0.003), DARK),
    ]
    com = (-0.006, -0.0707, -0.002)
    return Body.build("door", pos=HINGE, mass=DOOR_MASS, shapes=shapes, start_xyzw=_q_x(-OPEN0), com=com,
                      friction=(0.5, 0.45), damping=(0.0, 0.0), solver_iterations=32)


def _diamond(c, length, side, color) -> BoxShape:
    """A square bar along x turned 45 deg about x: a ramp on each face."""
    return BoxShape(tuple(c), (length, side, side), color, quat_xyzw=_q_x(45.0))


def _bolt() -> Body:
    p = (CATCH_X, Y0 + 0.011, TOP + 0.070)  # body origin: middle of the base block
    shapes = [
        _box((0, 0, 0), (0.016, 0.010, 0.004), CHROME),  # flat underside: holds the keeper's hook
        bodykit.plate((0, -0.001, 0.008), (0, 0.005, 0.002), 0.016, 0.0015, (0, 1, 1), CHROME),  # 45 deg lead-in ramp
    ]
    return Body.build("catch_bolt", pos=p, mass=0.004, shapes=shapes, ignore=("jbox",), solver_iterations=32)


def _pad() -> Body:
    """The socket's contact spring: a pad in the cavity roof whose underside sits 12.5 mm above the floor (the plug is
    12 mm tall), sprung downward, with a 20 deg lead-in."""
    z_lo = SOCK_Z - CAV_H / 2
    p = (SOCK_X + 0.0130, SOCK_Y, z_lo + 0.0125 + 0.0010)  # body origin: middle of the pad, its underside at +12.5 mm
    shapes = [
        _box((0, 0, 0), (0.006, 0.012, 0.002), CHROME),
        bodykit.plate((-0.0099, 0, 0.0035), (-0.003, 0, -0.001), 0.012, 0.0015, (-0.34, 0, -0.94), CHROME),
    ]
    return Body.build("socket_pad", pos=p, mass=0.002, shapes=shapes, ignore=("jbox",), friction=(0.3, 0.3),
                      solver_iterations=32)


def _jaw() -> Body:
    cx = (CLIP_X0 + CLIP_X1) / 2
    hz = CLIP_TOP - CLIP_FLOOR - 0.0005  # the arm stands 0.5 mm clear of the floor (it flexes, in the sim it slides)
    p = (cx, CLIP_Y + CLIP_GAP / 2, CLIP_TOP - hz / 2)  # body origin: on the arm's inner face, mid-height
    side = 0.0015  # lip: a square bar turned 45 deg; its inner edge LIP_IN into the channel at LIP_Z above the floor
    lip_y = -LIP_IN + side / math.sqrt(2)
    shapes = [
        _box((0, CLIP_ARM / 2, 0), (CLIP_X1 - CLIP_X0 - 0.001, CLIP_ARM, hz), NATURAL),
        # a cord pressed down spreads the arm and snaps under the lip; pulled up, it has to spread the arm again
        _diamond((0, lip_y, CLIP_FLOOR + LIP_Z - p[2]), CLIP_X1 - CLIP_X0 - 0.001, side, NATURAL),
    ]
    return Body.build("clip_jaw", pos=p, mass=0.002, shapes=shapes, ignore=("jbox",), friction=(0.15, 0.15), slick=True,
                      solver_iterations=32)


JBOX, DOOR, BOLT, PAD, JAW = _jbox(), _door(), _bolt(), _pad(), _jaw()
# Friction hinge (0.35 N m): a stiff spring drive capped at the friction torque, whose anchor the task file drags
# along whenever the door is pushed past it (Coulomb friction: it sticks below 0.35 N m and slips at 0.35 N m).
FRICTION = float(os.environ.get("JD_FRICTION", 0.35))  # N m
FRICTION_K = 0.2  # N m per degree
HINGE_J = bodykit.hinge("door_hinge", "jbox", "door", anchor=HINGE, axis="X", limits_deg=(-STOP, 0.0),
                        spring=FRICTION_K, damping=0.005, target_deg=-OPEN0, max_torque=FRICTION)
BOLT_J = bodykit.slider("bolt_slide", "jbox", "catch_bolt", anchor=BOLT.pos, axis="Y", limits=(-0.006, 0.0),
                        stiffness=300.0, damping=1.0, target=0.0015)
PAD_J = bodykit.slider("pad_slide", "jbox", "socket_pad", anchor=PAD.pos, axis="Z", limits=(0.0, 0.0035),
                       stiffness=400.0, damping=1.0, target=-0.001)
JAW_J = bodykit.slider("jaw_slide", "jbox", "clip_jaw", anchor=JAW.pos, axis="Y", limits=(0.0, 0.003),
                       stiffness=600.0, damping=1.0, target=-0.001)
_SHORT = DEV == "catch"  # dev only: a 9 cm cord without its plug that falls inside the box, out of the catch's way
CORD_SPEC = cable("cord", start=CORD_START, yaw=-math.pi / 2, length=0.09 if _SHORT else CORD_LEN, radius=CORD_R,
                  segment=0.015, anchor_start=True, anchor_body="door", cable_mass=0.012, bend_stiffness=0.03,
                  bend_damping=0.003, end_plug=None if _SHORT else Plug(size=(PL, PW, PH), mass=0.012, color=DARK),
                  color=CORD)
OBJECTS = [JBOX, DOOR, BOLT, PAD, JAW, CORD_SPEC]

# ================================================================================================================
# Scoring. Everything below is pure Python (tests drive it with synthetic states).
# ================================================================================================================
SHUT_DEG = 1.5  # bolt out and door this close: the hook is under the bolt (2 mm play = 0.86 deg; above ~1 deg the hook
#                would hold the bolt back)
NEAR_SHUT_DEG = 8.0  # "closing": any cord across the rim now is trapped in the seam
BOLT_OUT = -0.0008  # bolt slider position (m) above which the bolt is out (over the hook)
TEST_OPEN_DEG = 115.0  # the door test must swing the door at least this far
KEEP_OUT = (X0 - 0.03, X1 + 0.03, Y0 - 0.03, Y1 + 0.14, TOP + 0.26)  # hands-clear zone (x0, x1, y0, y1, z top)
STILL_S = 1.0
TEST_T = {"unlatch": 0.4, "open": 2.4, "hold": 3.4, "close": 5.4, "settle": 7.0}  # test phase timeline (s)


def door_angle(door_state) -> float:
    """Door opening angle (deg, 0 = shut) from the door body's orientation."""
    r = bodykit.rotation(door_state.quat_xyzw)
    return float(math.degrees(math.atan2(-r[2, 1], r[1, 1])))


def plug_depth(cord_state) -> float:
    """How far the plug's tip is inside the socket (m), or -1 when the plug is not in line with the cavity."""
    p, d = cord_state.end_plug, cord_state.end_dir
    if p is None or d is None or not np.isfinite(p).all():
        return -1.0
    if abs(p[1] - SOCK_Y) > 0.004 or abs(p[2] - SOCK_Z) > 0.004 or float(d[0]) < math.cos(math.radians(15)):
        return -1.0
    return float(p[0] + PL / 2 - SOCK_X)


def _dense(points: np.ndarray, step: float = 0.004) -> np.ndarray:
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return np.array(out)


def cord_points(cord_state) -> np.ndarray:
    """The cord's centre line, lamp end to plug, sampled every ~4 mm."""
    return _dense(cord_state.polyline())


def in_clip(points: np.ndarray) -> bool:
    """The cord lies in the clip's channel along its whole length (between the walls, below the top)."""
    m = (points[:, 0] >= CLIP_X0 + 0.002) & (points[:, 0] <= CLIP_X1 - 0.002)
    if m.sum() < 2:
        return False
    q = points[m]
    return bool((np.abs(q[:, 1] - CLIP_Y) <= 0.003).all() and (q[:, 2] <= CLIP_FLOOR + LIP_Z).all())


def outside_box(points: np.ndarray) -> bool:
    """Some of the cord is outside the box's walls (with the door shut it can only get there through the seam)."""
    x0, x1, y0, y1 = INNER
    m = (points[:, 0] < x0 + CORD_R) | (points[:, 0] > x1 - CORD_R) | (points[:, 1] < y0 + CORD_R) | (
        points[:, 1] > y1 - CORD_R) | (points[:, 2] > TOP + H - CORD_R)
    return bool(m.any())


def hands_clear(gripper_points) -> bool:
    """No part of either gripper (gripper body, both finger tips, TCP) is over or near the box."""
    x0, x1, y0, y1, ztop = KEEP_OUT
    for p in gripper_points:
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and p[2] <= ztop:
            return False
    return True


class Tracker:
    """The trial's history and the door test. ``update`` takes one control step's measurements."""

    def __init__(self):
        self.t = 0.0
        self.ever_seated = False
        self.unseated = False  # plug below 12 mm at some moment after it was first seated (robot phase)
        self.shut_before_plug = False  # door within 8 deg of shut before the plug was ever seated
        self.pinched = False  # cord across the rim while the door was within 8 deg of shut (robot phase)
        self.still_since = None
        self.phase = "robot"  # robot -> test -> done (or failed)
        self.t_test = None
        self.test_max_open = 0.0
        self.test_unseated = False
        self.test_pinched = False
        self.test_clip_lost = False
        self.test_hands = False
        self.opened_ok = False
        self.final_ok_since = None
        self.last = {}

    # -- the full-score state of the robot's work -----------------------------------------------------------------
    @staticmethod
    def work_done(m) -> bool:
        return m["depth"] >= SEATED_MIN and m["clipped"] and m["door"] < SHUT_DEG and m["bolt"] > BOLT_OUT

    def update(self, m: dict) -> dict:
        """m: t, door (deg), bolt (m), depth (m), clipped, outside (cord outside the walls), hands (clear), still.
        Returns the door-test command for this step: None (leave the door alone) or a dict."""
        self.t, self.last = m["t"], m
        seated = m["depth"] >= SEATED_MIN
        cmd = None
        if self.phase == "robot":
            if seated:
                self.ever_seated = True
            elif self.ever_seated:
                self.unseated = True
            if m["door"] < NEAR_SHUT_DEG and not self.ever_seated:
                self.shut_before_plug = True
            if m["door"] < NEAR_SHUT_DEG and m["outside"]:
                self.pinched = True
            ok = self.work_done(m) and m["hands"] and m["still"] and not self.violated()
            self.still_since = (self.still_since if self.still_since is not None else self.t) if ok else None
            if ok and self.t - self.still_since >= STILL_S:
                self.phase, self.t_test = "test", self.t
        elif self.phase == "test":
            dt = self.t - self.t_test
            self.test_max_open = max(self.test_max_open, m["door"])
            if not seated:
                self.test_unseated = True
            if not m["clipped"]:
                self.test_clip_lost = True
            if not m["hands"]:
                self.test_hands = True
            if m["door"] < NEAR_SHUT_DEG and m["outside"]:  # trapped in the seam: zero, not a failed test
                self.test_pinched = self.pinched = True
            if dt <= TEST_T["hold"] and self.test_max_open >= TEST_OPEN_DEG and not (
                    self.test_unseated or self.test_clip_lost or self.test_hands):
                self.opened_ok = True
            cmd = self._test_command(dt)
            if dt >= TEST_T["settle"]:
                self.phase = "done"
        if self.phase == "done":
            ok = self.passed() and self.work_done(m) and m["hands"] and m["still"]
            self.final_ok_since = (self.final_ok_since if self.final_ok_since is not None else self.t) if ok else None
        return cmd

    @staticmethod
    def _test_command(dt: float) -> dict:
        """The tester's hand: unlatch, swing the door open to its stop in 2 s, hold 1 s, swing it shut in 2 s and
        push until it latches, let go. ``target`` is where the hand leads the door (deg)."""
        u, o, h, c = TEST_T["unlatch"], TEST_T["open"], TEST_T["hold"], TEST_T["close"]
        if dt < u:
            return {"bolt": "retract", "hinge": "friction"}
        if dt < h:
            return {"bolt": "retract", "hinge": "lead", "target": (STOP + 5.0) * min(1.0, (dt - u) / (o - u))}
        if dt < c:
            return {"bolt": "free", "hinge": "lead", "target": (STOP + 5.0) * (1.0 - min(1.0, (dt - h) / (c - h))) - 25.0}
        return {"bolt": "free", "hinge": "friction"}

    def violated(self) -> bool:
        return self.unseated or self.shut_before_plug or self.pinched

    def passed(self) -> bool:
        return self.opened_ok and not (self.test_unseated or self.test_clip_lost or self.test_pinched or self.test_hands)

    def final_still(self) -> bool:
        """After the door test: re-latched with the plug seated, the cord clipped, hands clear and everything still,
        held for the last STILL_S (1 s) without a break."""
        return self.final_ok_since is not None and self.t - self.final_ok_since >= STILL_S - 1e-9

    def score(self) -> float:
        """0.2 plug seated; 0.4 and the cord in the clip; 0.6 and the door latched (the door test starts once this has
        held still for 1 s); 0.8 the test swung the door to its stop with the plug seated and the cord clipped; 1.0 it
        shut and latched it again with nothing trapped, all still for 1 s after the test (0.8 until then). Every rung needs both hands clear of the box.
        0 if the cord was ever across the rim with the door within 8 deg of shut, or the door came that close before
        the plug was first seated; at most 0.2 once the plug has come out after being seated and before the door test
        began. A plug pulled out during the door test fails the test (0.6, or 0.8 after the door reached 115 deg)."""
        m = self.last
        if not m or self.shut_before_plug or self.pinched:
            return 0.0
        if self.phase != "robot":  # the robot's work was complete and held still for 1 s: the door test decides
            if not self.opened_ok:
                return 0.6
            if self.phase == "done" and self.final_still():  # re-latched, nothing trapped, all still for 1 s
                return 1.0
            return 0.8
        if not m["hands"]:
            return 0.0
        s = 0.0
        if m["depth"] >= SEATED_MIN:
            s = 0.2
            if m["clipped"]:
                s = 0.4
                if m["door"] < SHUT_DEG and m["bolt"] > BOLT_OUT:
                    s = 0.6
        if self.unseated:
            s = min(s, 0.2)
        return s

    def success(self) -> bool:
        return self.phase == "done" and self.score() == 1.0


# -- glue to the simulator ---------------------------------------------------------------------------------------
_TR = Tracker()
_FR = {"anchor": OPEN0, "door": OPEN0}
_ENV: dict = {"env": None, "step": 0, "t": 0.0}
CONTROL_DT = 0.05


@bodykit.on_reset
def _reset() -> None:
    global _TR
    _TR = Tracker()
    _ENV["step"], _ENV["t"] = 0, 0.0
    _FR.update(anchor=OPEN0, door=OPEN0)


def gripper_points(env) -> list[np.ndarray]:
    """Gripper body, both finger tips and the TCP of both arms (world, m)."""
    if env is None:
        return []
    base = env.unwrapped
    origin = bodykit.as_numpy(base.scene.env_origins[0])
    out = []
    for key in ("left_robot", "right_robot"):
        art = base.scene[key]
        names = list(art.body_names)
        pos = bodykit.as_numpy(art.data.body_pos_w)[0] - origin
        g, a, b = pos[names.index("gripper")], pos[names.index("tip_left")], pos[names.index("tip_right")]
        mid = (a + b) / 2
        ax = (mid - g) / (np.linalg.norm(mid - g) + 1e-12)
        out += [g, a, b, g + ax * 0.134]
    return out


def measure(objects, env=None, t: float = 0.0) -> dict:
    cord = objects["cord"]
    pts = cord_points(cord)
    door = objects["door"]
    still = (float(np.linalg.norm(door.ang_vel)) < 0.05 and float(np.linalg.norm(cord.lin_vel)) < 0.01)
    bolt = float(objects["catch_bolt"].pos[1] - BOLT.pos[1])
    return {"t": t, "door": door_angle(door), "bolt": bolt, "depth": plug_depth(cord), "clipped": in_clip(pts),
            "outside": outside_box(pts), "hands": hands_clear(gripper_points(env)) if env is not None else True,
            "still": still}


def _apply(cmd) -> None:
    if cmd is None:
        return
    b = cmd["bolt"]
    if b == "retract":
        bodykit.set_slider_drive("bolt_slide", target=-0.008)
    else:
        bodykit.set_slider_drive("bolt_slide", target=BOLT_J.target)
    h = cmd["hinge"]
    if h == "lead":  # a hand leading the door with at most 0.8 N m (about 6 N at the free edge)
        bodykit.set_hinge_drive("door_hinge", target_deg=-cmd["target"], spring=0.05, damping=0.01, max_torque=0.8)
    else:  # back to the friction hinge, anchored where the door is now
        _FR["anchor"] = _FR["door"]
        bodykit.set_hinge_drive("door_hinge", target_deg=-_FR["anchor"], spring=FRICTION_K, damping=HINGE_J.damping,
                                max_torque=FRICTION)


def friction_anchor(anchor: float, door: float) -> float:
    """Coulomb friction as a capped spring: drag the spring's anchor (deg) so it is never more than the slip
    angle (friction torque / spring rate) from the door."""
    slip = FRICTION / FRICTION_K
    if door > anchor + slip:
        return door - slip
    if door < anchor - slip:
        return door + slip
    return anchor


def _friction(door: float) -> None:
    _FR["door"] = door
    a = friction_anchor(_FR["anchor"], door)
    if a != _FR["anchor"]:
        _FR["anchor"] = a
        bodykit.set_hinge_drive("door_hinge", target_deg=-a)


@bodykit.on_step
def _step(env) -> None:
    from yam_arena.scene import object_states

    _ENV["env"] = env
    _ENV["step"] += 1
    _ENV["t"] = _ENV["step"] * CONTROL_DT
    st = object_states(env, [o.name for o in OBJECTS])
    m = measure(st, env, _ENV["t"])
    if LIVE:
        servo(_ENV["t"], st["cord"])
    if DEV:
        import dev_hooks  # dev/dev_hooks.py: no-robot mechanism checks

        dev_hooks.step(env, st, m, _TR, gripper_points)
        dev_hooks.tie(env, st, _TR)
    cmd = _TR.update(m)
    _FR["door"] = m["door"]
    _apply(cmd)
    if (cmd is None or cmd["hinge"] == "friction") and not getattr(_TR, "dev_hand", False):
        _friction(m["door"])
    if DEBUG and (_ENV["step"] % 10 == 0 or cmd is not None and _ENV["step"] % 2 == 0):
        print(f"[jd t={m['t']:6.2f}] pad={(st['socket_pad'].pos[2] - PAD.pos[2]) * 1000:.1f} jaw={(st['clip_jaw'].pos[1] - JAW.pos[1]) * 1000:.1f} door={m['door']:6.1f} bolt={m['bolt'] * 1000:5.1f}mm "
              f"depth={m['depth'] * 1000:6.1f}mm clip={m['clipped']} out={m['outside']} hands={m['hands']} "
              f"still={m['still']} phase={_TR.phase} viol={_TR.unseated, _TR.shut_before_plug, _TR.pinched} "
              f"test={_TR.test_max_open:.0f},{_TR.opened_ok},{_TR.test_unseated},{_TR.test_clip_lost},"
              f"{_TR.test_pinched} score={_TR.score()}", flush=True)


if DEV:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "dev"))


def score(objects) -> float:
    return _TR.score()


def success(objects) -> bool:
    return _TR.success()


def distance(objects) -> float:
    return max(0.0, SEATED_MIN - plug_depth(objects["cord"]))


# -- scripted demo (yam_scripted plays it open loop; the layout is fixed, so the plug's resting place is known) ---------
DEMO = os.environ.get("JD_DEMO", "correct")  # tests only: correct | loose (cord left unclipped) | unlatched (door not pressed home) | shut_first | none
PLUG_START = (0.190, -0.272, TOP + 0.014)  # TCP over the plug lying flat on the bench (sim run "scene5"), tip toward -y
PINCH_BACK = 0.014  # the plug is gripped this far behind its centre (8 mm from its rear end)
TIP_PAST_TCP = PINCH_BACK + PL / 2  # tip ahead of the TCP once the plug points +x
FINGER_SHUT = 0.0
UP = TOP + 0.11  # carry height: clears the walls with the plug hanging under the fingers
PARK = {"left": (0.10, 0.22, TOP + 0.15), "right": (0.10, -0.22, TOP + 0.15)}


def _w(d, p, f, label, yaw=0.0):
    from yam_arena.demo import Waypoint

    return Waypoint(float(d), tuple(float(v) for v in p), float(f), label, float(yaw))


def _door_push(a_deg: float, s: float, push: float = 0.004) -> tuple:
    """TCP for the shut fingers' lower -y corner to touch the door's outer face ``s`` from the hinge at door angle
    ``a_deg``, ``push`` m into it (the door gives way under the push, against its friction)."""
    a = math.radians(a_deg)
    u = np.array([-math.cos(a), math.sin(a)])  # along the door, hinge to free edge (y, z)
    n = np.array([math.sin(a), math.cos(a)])  # outer face normal
    c = np.array(HINGE[1:]) + s * u + (DOOR_T - push) * n
    return (0.290, c[0] + 0.0064, c[1] + 0.0128)


# Closed loop where it matters (the demo may read the simulator's state; the policy and the embodiment run in one
# process and the policy reads each waypoint's position when it plays it): while lining the plug up and pushing it
# in, the task file nudges those waypoints by the plug's measured error from where it should be.
LIVE: list = []


def _live(arm, wps, label, kind, follow=0):
    i = next(k for k, w in enumerate(wps) if w.label == label)
    start = sum(w.duration for w in wps[:i])
    LIVE.append({"arm": arm, "kind": kind, "t0": start, "t1": start + wps[i].duration,
                 "wps": [(w, np.array(w.pos), w.yaw) for w in wps[i:i + 1 + follow]], "delta": np.zeros(3), "dyaw": 0.0})


def _plug_target(kind) -> np.ndarray:
    tip = SOCK_X - 0.004 if kind == "align" else SOCK_X + 0.014
    return np.array([tip - PL / 2, SOCK_Y, SOCK_Z - CAV_H / 2 + PH / 2 + 0.0004])  # just clear of the floor


def servo(t: float, cord_state) -> None:
    """Nudge the live waypoints (and the ones after them) toward the plug's target pose."""
    p, d = cord_state.end_plug, cord_state.end_dir
    if p is None or d is None:
        return
    for e in LIVE:
        if not (e["t0"] < t <= e["t1"]):
            continue
        err = _plug_target(e["kind"]) - p
        if e["kind"] == "push":
            err[0] = 0.0
        e["delta"] += np.clip(0.25 * err, -0.0015, 0.0015)
        e["dyaw"] += float(np.clip(-0.25 * math.atan2(d[1], d[0]), -0.01, 0.01))
        for w, base, yaw in e["wps"]:
            w.pos = tuple((base + e["delta"]).tolist())
            w.yaw = yaw + e["dyaw"]


def demo_plan() -> dict:
    LIVE.clear()
    from yam_arena.constants import FINGER_OPEN

    px, py, pz = PLUG_START
    g = py + PINCH_BACK  # grip 14 mm behind the plug's centre (toward the cord)
    in_x = SOCK_X + 0.026 - TIP_PAST_TCP  # aim 10 mm past home: the arm is soft, the plug stops at the cavity's end
    pre_x = SOCK_X - 0.022 - TIP_PAST_TCP
    zs = SOCK_Z + 0.008  # TCP with the plug centred on the socket cavity
    right, left = [], []
    if DEMO == "none":
        return {}
    if DEMO != "shut_first":
        right += [
            _w(1.8, (px, g, TOP + 0.10), -0.030, "over the plug", -math.pi / 2),
            _w(1.2, (px, g, pz), -0.030, "down to the plug", -math.pi / 2),
            _w(0.6, (px, g, pz), FINGER_SHUT, "grip the plug", -math.pi / 2),
            _w(1.0, (px, g + 0.06, TOP + 0.06), FINGER_SHUT, "lift toward the box", -1.2),
            _w(1.0, (px, Y0 - 0.01, UP), FINGER_SHUT, "up over the front wall", -0.8),
            _w(2.0, (pre_x, SOCK_Y, UP), FINGER_SHUT, "carry, turn the plug to point at the socket", 0.0),
            _w(0.6, (pre_x, SOCK_Y, UP), FINGER_SHUT, "settle", 0.0),
            _w(1.5, (pre_x - 0.01, SOCK_Y, TOP + 0.030), FINGER_SHUT, "down in front of the socket"),
            _w(1.0, (pre_x - 0.01, SOCK_Y, TOP + 0.015), FINGER_SHUT, "set the plug flat on the back plate"),
            _w(0.8, (pre_x, SOCK_Y, zs), FINGER_SHUT, "up to the socket's height"),
            _w(1.5, (pre_x, SOCK_Y, zs), FINGER_SHUT, "line the plug up with the socket"),
            _w(2.0, (in_x, SOCK_Y, zs), FINGER_SHUT, "push the plug in"),
            _w(0.4, (in_x, SOCK_Y, zs), FINGER_SHUT, "hold"),
            _w(0.6, (in_x, SOCK_Y, zs), -0.012, "let go"),
            _w(1.0, (in_x - 0.01, SOCK_Y, TOP + 0.06), -0.012, "up"),
        ]
        _live("right", right, "line the plug up with the socket", "align", follow=5)
        _live("right", right, "push the plug in", "push", follow=3)
        cx = 0.2445  # grip the cord 5 mm beside the clip, on the hinge-loop side
        if DEMO != "loose":
            right += [
                _w(1.2, (cx, CLIP_Y - 0.0015, TOP + 0.06), -0.018, "over the cord beside the clip"),
                _w(1.0, (cx, CLIP_Y - 0.0015, TOP + 0.016), -0.018, "down around the cord"),
                _w(0.6, (cx, CLIP_Y - 0.0015, TOP + 0.016), FINGER_SHUT, "take the cord"),
            ]
            right += [_w(0.5, (cx, CLIP_Y, TOP + 0.016), -0.012, "let go"),
                      _w(0.8, (cx, CLIP_Y, TOP + 0.04), -0.012, "up")]
            mid = (CLIP_X0 + CLIP_X1) / 2
            zp = CLIP_TOP + 0.001 + 0.0128 - 0.016  # aim 17 mm below the seated cord top: the arm is soft (~0.4 N/mm)
            q = math.pi / 2  # fingers across the cord: the 13 mm-thick pair fits between the clip's ends
            right += [
                _w(0.8, (mid, CLIP_Y, TOP + 0.04), FINGER_SHUT, "over the clip", q),
                _w(1.0, (mid, CLIP_Y, zp), FINGER_SHUT, "press the cord into the clip", q),
                _w(0.5, (mid, CLIP_Y, zp), FINGER_SHUT, "hold", q),
                _w(0.5, (mid, CLIP_Y, TOP + 0.03), FINGER_SHUT, "up", q),
                _w(0.8, (mid, CLIP_Y, zp), FINGER_SHUT, "press again", q),
                _w(0.5, (mid, CLIP_Y, zp), FINGER_SHUT, "hold", q),
                _w(0.8, (mid, CLIP_Y, TOP + 0.06), FINGER_SHUT, "up", q),
            ]
        right += [_w(1.5, (0.20, -0.17, TOP + 0.14), FINGER_SHUT, "out of the way")]
    # left arm: swing the door shut with the shut fingers' side, sweeping down the outside of the door
    t_right = sum(w.duration for w in right)
    sweep = [(90, 0.035), (75, 0.040), (60, 0.048), (45, 0.060), (32, 0.080), (20, 0.110), (14, 0.120)]
    left += [_w(max(t_right, 0.5), PARK["left"], FINGER_SHUT, "wait"),
             _w(1.5, (0.290, 0.085, TOP + 0.13), FINGER_SHUT, "beside the door")]
    left += [_w(1.2 if i else 1.0, _door_push(a, s), FINGER_SHUT, f"push the door to {a} deg") for i, (a, s) in
             enumerate(sweep)]
    last = _door_push(*sweep[-1])
    left += [_w(1.0, (last[0], last[1] + 0.03, TOP + 0.13), FINGER_SHUT, "off the door"),
             _w(1.5, PARK["left"], FINGER_SHUT, "park")]
    # right arm: press the door's free edge down until the catch clicks
    t_left = sum(w.duration for w in left)
    zd = TOP + H + DOOR_T + 0.0128 - 0.014  # aim 14 mm below the shut door's top: a firm press (the arm is soft)
    right += [_w(max(t_left - sum(w.duration for w in right) - 2.5, 0.5), (0.20, -0.17, TOP + 0.14), FINGER_SHUT, "wait"),
              _w(1.2, (CATCH_X, Y0 + 0.022, TOP + 0.15), FINGER_SHUT, "over the door's free edge"),
              _w(2.0, (CATCH_X, Y0 + 0.022, zd), FINGER_SHUT, "press the door shut until the catch clicks"),
              _w(1.0, (CATCH_X, Y0 + 0.022, zd), FINGER_SHUT, "hold"),
              _w(1.0, (CATCH_X, Y0 + 0.022, TOP + 0.15), FINGER_SHUT, "up"),
              _w(1.5, PARK["right"], FINGER_SHUT, "park")]
    if DEMO == "unlatched":  # the door is swung nearly shut but never pressed until the catch clicks
        right = right[:-6] + [_w(1.5, PARK["right"], FINGER_SHUT, "park")]
    if DEMO == "shut_first":  # the door swung shut before the cord is plugged in or clipped
        right = right[-6:]
    return {"left": left, "right": right}


def demo():
    return demo_plan()
