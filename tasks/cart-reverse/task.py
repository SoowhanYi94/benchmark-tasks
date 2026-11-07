"""A shop tow cart stands in a marked bay with a chock behind one wheel. The robot must take the chock away, back the cart rear-first by its drawbar
along an S into the green bay across the aisle, and chock it there. The scripted demo passes the bar between its hands; the scorer doesn't require that. Pushing a hinged fixed-axle cart in
reverse is unstable: the cart turns away from any error, so the hand has to steer it.

Run it:  inspect-robots run --embodiment yam_cart -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "First take the chock away from behind the cart's wheel, then back the cart rear-first into the green bay using only the orange drawbar handle, chock a wheel on its open side, and move both hands away. Don't jackknife or tip the cart, skid its wheels, or touch its body while it moves." --scorer task_score

`yam_cart` (cart_reverse/embodiment.py) spawns the cart and chock and reports each gripper's fingertip midpoint and the cart.
"""

import math
import os

import numpy as np

from cart_reverse import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
A_AT = (0.22, 0.03)  # the axle's middle, in bay A at the start
B_AT = (0.34, 0.31)  # and where bay B wants it
HEADING0 = -math.pi / 2  # the cart's +x (axle toward hitch) points along -y in both bays
CART_QUAT = (0.0, 0.0, math.sin(HEADING0 / 2), math.cos(HEADING0 / 2))
WB, BAR = M.WB, M.BAR
WR = M.WHEEL["r"]
YELLOW, GREEN, GREY = (0.90, 0.45, 0.08), (0.10, 0.60, 0.25), (0.45, 0.47, 0.50)  # this value renders yellow under the station lights
CHOCK = dict(half=0.02, h=0.03, post=0.0125, post_h=0.07, mass=0.19)  # a 4 cm square block 3 cm tall under a 2.5 cm square post to 7 cm; aluminium. Origin: centre of its underside
PARTS = {"chock": ({"foot": ((-CHOCK["half"], -CHOCK["half"], 0.0), (CHOCK["half"], CHOCK["half"], CHOCK["h"])),
                    "post": ((-CHOCK["post"], -CHOCK["post"], CHOCK["h"]), (CHOCK["post"], CHOCK["post"], CHOCK["post_h"]))}, CHOCK["mass"], 0.6, YELLOW)}
# behind the wheel nearer the robot, 2 mm from it: it blocks the way the cart must back
CHOCK_START = (A_AT[0] - M.WHEEL["y"], A_AT[1] + WR + CHOCK["half"] + 0.002)
RAIL = dict(h=0.015, w=0.01, half_gap=0.115)
B_STOP_Y = B_AT[1] + WR + 0.005
PAINT = 0.0004


def _bays():
    """Each bay's painted outline, a hair proud of the table, and bay B's low back stop: the cart stands on open table, nothing walls the aisle."""
    g, w, h = RAIL["half_gap"], RAIL["w"], RAIL["h"]
    sy = B_STOP_Y
    out = [Box("b_stop", pos=(B_AT[0], sy + w / 2, TOP + h / 2), size=(2 * g + 2 * w, w, h), color=GREEN)]
    ln = 0.11
    out += [Box(f"b_line_{s}", pos=(B_AT[0] + k * (g + 0.01), sy - ln / 2, TOP + PAINT / 2), size=(0.02, ln, PAINT), color=GREEN) for s, k in (("l", 1), ("r", -1))]
    out += [Box(f"a_line_{s}", pos=(A_AT[0] + k * (g + 0.01), A_AT[1] + 0.02 - ln / 2, TOP + PAINT / 2), size=(0.02, ln, PAINT), color=GREY) for s, k in (("l", 1), ("r", -1))]
    return out


OBJECTS = [
    Box("cart", pos=(*A_AT, TOP + 0.0005), size=(0.01, 0.01, 0.01), mass=0.5),
    Box("chock", pos=(*CHOCK_START, TOP + 0.0005), size=(0.01, 0.01, 0.01), mass=CHOCK["mass"]),
    *_bays(),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
BAY_X, BAY_Y, BAY_YAW = 0.025, 0.03, math.radians(10)
TOWARD_BAY, HALFWAY_Y, SLIP_DEAD = 0.06, 0.17, 0.0002
JACKKNIFE, TILT, LIFTED, SLIP = math.radians(70), math.radians(6), 0.012, 0.03
CHOCK_NEAR, CHOCK_FAR, CHOCK_SIDE, CHOCK_ZFLAT = WR + CHOCK["half"] - 0.004, WR + CHOCK["half"] + 0.015, 0.02, math.radians(8)
AT_HANDLE, NEAR_PART, SHIFT, TURN = 0.015, 0.06, 0.005, math.radians(3)
CHOCK_LIFT, CHOCK_CLEAR, AT_WHEEL, CHOCK_SHIFT, CHOCK_TURN = 0.01, 0.08, 0.065, 0.02, math.radians(8)
TURNED_ROUND = math.radians(75)
LIN, ANG, STILL_S = 0.01, 0.5, 1.0
_trial = {"still": 0.0, "bad": False, "why": "", "slip": 0.0, "last": None, "chock_off": False, "spell": None, "body_shift": 0.0, "body_turn": 0.0, "chock_shift": 0.0, "chock_turn": 0.0}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears `bad`."""
    _trial.update(still=0.0, bad=False, why="", slip=0.0, last=None, chock_off=False, spell=None, body_shift=0.0, body_turn=0.0, chock_shift=0.0, chock_turn=0.0)


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def heading(objects):
    """The way the cart's nose points, about the vertical."""
    R = _rot(objects["cart"].quat_xyzw)
    return math.atan2(R[1, 0], R[0, 0])


def tilt(objects):
    return math.acos(float(np.clip(_rot(objects["cart"].quat_xyzw)[2, 2], -1.0, 1.0)))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def handle_at(objects):
    """The drawbar handle's centre, level with the hitch, in the world."""
    th, h = heading(objects) + objects["hitch"].pos[0], np.asarray(objects["cart"].pos, float)
    c, s = math.cos(heading(objects)), math.sin(heading(objects))
    return np.array((h[0] + WB * c + BAR * math.cos(th), h[1] + WB * s + BAR * math.sin(th), h[2] + M.HITCH_Z))


def _hand_off_box(objects, centre, yaw, half_x, half_y, z0, z1, gap):
    """Both fingertip midpoints at least `gap` from a box, judged in that box's own frame. An open finger sits 4.7 cm out, so 6 cm means no finger's on it."""
    c, s = math.cos(yaw), math.sin(yaw)
    for a in ("left_tcp", "right_tcp"):
        v = np.asarray(objects[a].pos, float) - centre
        d = np.array((c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]))
        q = d - np.clip(d, (-half_x, -half_y, z0), (half_x, half_y, z1))
        if np.linalg.norm(q) < gap:
            return False
    return True


def bar_let_go(objects):
    h = handle_at(objects)
    return _hand_off_box(objects, h, heading(objects) + objects["hitch"].pos[0], M.HANDLE["half_x"], M.HANDLE["half_y"], M.HANDLE["z0"] - M.HITCH_Z, M.HANDLE["z1"] - M.HITCH_Z, 0.06)


def chock_let_go(objects):
    p = objects["chock"]
    return _hand_off_box(objects, np.asarray(p.pos, float), math.atan2(_rot(p.quat_xyzw)[1, 0], _rot(p.quat_xyzw)[0, 0]), CHOCK["half"], CHOCK["half"], 0.0, CHOCK["post_h"], 0.06)


def _box_distance(v, lo, hi):
    return float(np.linalg.norm(v - np.clip(v, lo, hi)))


def _at_handle(objects, p):
    """A fingertip midpoint on the handle: within 1.5 cm of its box, in its own frame."""
    th = heading(objects) + objects["hitch"].pos[0]
    c, s = math.cos(th), math.sin(th)
    v = p - handle_at(objects)
    d = np.array((c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]))
    h = M.HANDLE
    return _box_distance(d, (-h["half_x"], -h["half_y"], h["z0"] - M.HITCH_Z), (h["half_x"], h["half_y"], h["z1"] - M.HITCH_Z)) <= AT_HANDLE


def _near_cart(objects, p):
    """A fingertip midpoint close enough that a finger could be on some part other than the handle: an open finger sits 4.7 cm from the midpoint,
    so 6 cm of the body and nose, the axle and wheels, or the rod, each judged in its own frame. A hand on the handle doesn't count."""
    if _at_handle(objects, p):
        return False
    d, w = M.DECK, M.WHEEL
    v = _rot(objects["cart"].quat_xyzw).T @ (p - np.asarray(objects["cart"].pos, float))
    if _box_distance(v, (d["x0"], -d["half"], 0.0), (WB + 0.012, d["half"], M.HITCH_Z + 0.008 if v[0] > d["x1"] else M.TOTE["z1"])) < NEAR_PART:
        return True
    if _box_distance(v, (-w["r"], -w["y"] - w["r"], 0.0), (w["r"], w["y"] + w["r"], 2 * w["r"])) < NEAR_PART:
        return True
    th = heading(objects) + objects["hitch"].pos[0]
    c, s = math.cos(th), math.sin(th)
    u = p - np.asarray(objects["cart"].pos, float) - np.array((WB * math.cos(heading(objects)), WB * math.sin(heading(objects)), M.HITCH_Z))
    r = np.array((c * u[0] + s * u[1], -s * u[0] + c * u[1], u[2]))
    return _box_distance(r, (0.0, -0.006, -0.006), (BAR - M.HANDLE["half_x"], 0.006, 0.006)) < NEAR_PART


def hand_near_cart(objects):
    return any(_near_cart(objects, np.asarray(objects[a].pos, float)) for a in ("left_tcp", "right_tcp"))


def _hand_on_body(objects):
    """Near a part other than the handle, and not simply holding the chock against a wheel: letting go of the chock is judged by its own rung."""
    chocking = _chock_at_wheel(objects)
    return any(_near_cart(objects, p) and not (chocking and _at_chock(objects, p)) for p in (np.asarray(objects[n].pos, float) for n in ("left_tcp", "right_tcp")))


def _chock_at_wheel(objects):
    """The chock on the table, its centre within 6.5 cm of a wheel's centre: close enough to be being set against it or taken from it."""
    p, c = np.asarray(objects["chock"].pos, float), np.asarray(objects["cart"].pos, float)
    R = _rot(objects["cart"].quat_xyzw)
    return bool(p[2] <= TOP + CHOCK_LIFT and any(np.linalg.norm((p - (c + k * M.WHEEL["y"] * R[:, 1]))[:2]) <= AT_WHEEL for k in (1, -1)))


def _at_chock(objects, p):
    """A fingertip midpoint on the chock: within 1.5 cm of its box, in its own frame."""
    c = objects["chock"]
    v = _rot(c.quat_xyzw).T @ (p - np.asarray(c.pos, float))
    return _box_distance(v, (-CHOCK["half"], -CHOCK["half"], 0.0), (CHOCK["half"], CHOCK["half"], CHOCK["post_h"])) <= AT_HANDLE


def _note_near(objects, a, h):
    """Add up how far the cart goes while a hand's near a part other than the handle. Each spell counts its net travel, so solver jitter cancels but a
    push made in steps with retreats between them doesn't. A hand on a chock at a wheel is chocking: the cart may roll on the chock, up to its own
    smaller allowance, and that isn't a hand on the cart."""
    hands = [np.asarray(objects[n].pos, float) for n in ("left_tcp", "right_tcp")]
    chocking = _chock_at_wheel(objects)
    near = [_near_cart(objects, p) for p in hands]
    on_chock = [chocking and _at_chock(objects, p) for p in hands]
    kind = "body" if any(n and not c for n, c in zip(near, on_chock)) else "chock" if any(n and c for n, c in zip(near, on_chock)) else None
    if _trial["spell"] is not None and _trial["spell"][0] != kind:
        k, a0, h0 = _trial["spell"]
        _trial[k + "_shift"] += float(np.linalg.norm((a - a0)[:2]))
        _trial[k + "_turn"] += abs(_wrap(h - h0))
        _trial["spell"] = None
    if kind is not None and _trial["spell"] is None:
        _trial["spell"] = (kind, a.copy(), h)
    shift, turn = {"body": 0.0, "chock": 0.0}, {"body": 0.0, "chock": 0.0}
    if _trial["spell"] is not None:
        k, a0, h0 = _trial["spell"]
        shift[k], turn[k] = float(np.linalg.norm((a - a0)[:2])), abs(_wrap(h - h0))
    if _trial["body_shift"] + shift["body"] >= SHIFT or _trial["body_turn"] + turn["body"] >= TURN:
        _latch("hand on the cart")
    if _trial["chock_shift"] + shift["chock"] >= CHOCK_SHIFT or _trial["chock_turn"] + turn["chock"] >= CHOCK_TURN:
        _latch("moved the cart with the chock")


def parked(objects):
    """Rear-first in bay B: the axle's middle within 2.5 cm across and 3 cm along, nose within 10 degrees of straight out."""
    a = np.asarray(objects["cart"].pos, float)
    return bool(abs(a[0] - B_AT[0]) <= BAY_X and abs(a[1] - B_AT[1]) <= BAY_Y and abs(_wrap(heading(objects) - HEADING0)) <= BAY_YAW and bar_let_go(objects) and not _hand_on_body(objects))


def chocked(objects):
    """The chock stands on the table against either wheel, on the side toward the hitch, judged in the cart's own frame."""
    p = objects["chock"]
    R = _rot(objects["cart"].quat_xyzw)
    v = R.T @ (np.asarray(p.pos, float) - np.asarray(objects["cart"].pos, float))
    return bool(CHOCK_NEAR <= v[0] <= CHOCK_FAR and abs(abs(v[1]) - M.WHEEL["y"]) <= CHOCK_SIDE and abs(p.pos[2] - TOP) <= 0.005 and _rot(p.quat_xyzw)[2, 2] >= math.cos(CHOCK_ZFLAT) and chock_let_go(objects))


def chock_taken(objects):
    """The chock's off: lifted 1 cm, or its centre 8 cm from both wheels' centres. Pushed along by the cart, it's neither."""
    p, c = np.asarray(objects["chock"].pos, float), np.asarray(objects["cart"].pos, float)
    R = _rot(objects["cart"].quat_xyzw)
    return bool(p[2] >= TOP + CHOCK_LIFT or all(np.linalg.norm((p - (c + k * M.WHEEL["y"] * R[:, 1]))[:2]) >= CHOCK_CLEAR for k in (1, -1)))


def _latch(why):
    if not _trial["bad"]:
        _trial["bad"], _trial["why"] = True, why


def note_step(objects, dt):
    """Called by the embodiment after every control step: the things that end a trial for good, the sideways skid, and the clock for all at rest."""
    cart, chock = objects["cart"], objects["chock"]
    a = np.asarray(cart.pos, float)
    v = np.asarray(cart.lin_vel, float)
    # the cart's frame is under the axle's middle, so its own step sideways, along the axle, is the skid; under the deadband is solver noise
    R = _rot(cart.quat_xyzw)
    if _trial["last"] is not None:
        side = abs(float(R[:, 1] @ (a - _trial["last"])))
        if side > SLIP_DEAD:
            _trial["slip"] += side
    _trial["last"] = a.copy()
    if _trial["slip"] > SLIP:
        _latch("skid")
    if abs(objects["hitch"].pos[0]) >= JACKKNIFE:
        _latch("jackknife")
    if tilt(objects) >= TILT or a[2] > TOP + LIFTED:
        _latch("tipped or lifted")
    if a[2] < TOP - 0.01 or chock.pos[2] < TOP - 0.01:
        _latch("off the table")
    if abs(_wrap(heading(objects) - HEADING0)) >= TURNED_ROUND:
        _latch("turned round")  # with the nose within this of -y, a forward roll always goes away from the bay, so progress toward it can only be reversing
    # no speed gate: a slow crawl with a hand on the nose would slip under one
    _note_near(objects, a, heading(objects))
    if chock_taken(objects):
        _trial["chock_off"] = True
    if np.linalg.norm(a[:2] - np.array(A_AT)) >= 0.015 and not _trial["chock_off"]:
        _latch("pushed against the chock")
    still = np.linalg.norm(v) <= LIN and np.linalg.norm(cart.ang_vel) <= ANG and np.linalg.norm(chock.lin_vel) <= LIN and np.linalg.norm(chock.ang_vel) <= ANG and bar_let_go(objects) and chock_let_go(objects) and not hand_near_cart(objects)
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. 0 for good on a jackknife, a tip or lift, a skid, a nose turned 75 degrees from -y, a hand on any part but the handle while
    the cart shifts, the cart moved far with the chock, a cart moved before the chock's off, or anything off the table. Otherwise 0.2 with the axle 6 cm toward the bay, 0.4 past halfway,
    0.7 parked rear-first in bay B and let go, 0.9 also chocked and let go, 1.0 also all at rest for the last 1 s."""
    if _trial["bad"]:
        return 0.0
    a = np.asarray(objects["cart"].pos, float)
    if parked(objects):
        return (1.0 if _trial["still"] >= STILL_S else 0.9) if chocked(objects) else 0.7
    if a[1] >= HALFWAY_Y:
        return 0.4
    return 0.2 if a[1] >= A_AT[1] + TOWARD_BAY else 0.0


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.03
HAND_YAW = {"right": math.pi / 2, "left": -math.pi / 2}  # fingers close across the bar, which stays along -y
GRIP_Z = TOP + 0.06  # the handle's middle, 5 mm proud so the arm's droop can't press the bar on its rest stop
HOLD_Z = GRIP_Z + 0.004
HAND_Y_HANDOVER = -0.03
PACE = 0.8  # seconds per cm in reverse


def _fingers(width):
    return -FINGER_PER_M * width


HANDLE_G = _fingers(2 * M.HANDLE["half_y"] - 0.004)
CHOCK_G, CHOCK_R = _fingers(2 * CHOCK["post"] - 0.004), _fingers(2 * CHOCK["post"] + 0.03)
CHOCK_GZ = 0.055
CHOCK_SPOT = (0.09, -0.01)  # where the right hand leaves the chock for the left: out of the cart's way, in reach of both


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


def _smooth(s):
    s = min(1.0, max(0.0, s))
    return s * s * (3 - 2 * s)


def _axle_x(y):
    """The axle's planned lane across the aisle: the S from bay A's line to bay B's."""
    y0, y1 = A_AT[1] + 0.005, B_AT[1] - 0.04
    return A_AT[0] + (B_AT[0] - A_AT[0]) * _smooth((y - y0) / (y1 - y0))


def hand_for_axle(y):
    """Where the hand has to be for the axle to run along the lane: a hinged fixed-axle cart follows the line from axle to hitch, so the hitch goes
    ahead of the lane's tangent by the wheelbase, and the bar is kept along -y with the handle a bar's length further."""
    d = 1e-4
    slope = (_axle_x(y + d) - _axle_x(y - d)) / (2 * d)
    n = math.hypot(slope, 1.0)
    hx, hy = _axle_x(y) - WB * slope / n, y - WB / n
    return hx, hy - BAR


def _take_chock_off(plan):
    arm = "right"
    x, y = CHOCK_START
    plan.go(arm, 1.6, (x, y, TOP + 0.12), CHOCK_R, "above the chock", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, TOP + CHOCK_GZ), label="descend")
    plan.go(arm, 0.6, finger=CHOCK_G, label="grip")
    plan.go(arm, 0.8, (x, y, TOP + 0.12), label="lift")
    sx, sy = CHOCK_SPOT
    plan.go(arm, 2.0, (sx, sy, TOP + 0.12), label="over the spot")
    plan.go(arm, 1.0, (sx, sy, TOP + CHOCK_GZ + 0.002), label="down")
    plan.go(arm, 0.6, finger=CHOCK_R, label="let go")
    plan.go(arm, 0.8, (sx, sy, TOP + 0.12), label="up")


def _grip_bar(plan, arm, x, y):
    yaw = HAND_YAW[arm]
    plan.go(arm, 1.8, (x, y, TOP + 0.11), _fingers(0.06), "above the handle", yaw)
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 0.8, (x, y, GRIP_Z), label="descend")
    plan.go(arm, 0.6, finger=HANDLE_G, label="grip")
    plan.go(arm, 0.4, (x, y, HOLD_Z), label="take it")


def _let_go_bar(plan, arm):
    x, y, _ = plan.last[arm][0]
    plan.go(arm, 0.3, (x, y, GRIP_Z), label="ease")
    plan.go(arm, 0.6, finger=_fingers(0.06), label="let go")
    plan.go(arm, 0.8, (x, y, TOP + 0.13), label="up")


def _reverse(plan, arm, y0, y1):
    """Back the axle from y0 to y1 along the lane, in short steps."""
    n = max(1, round(abs(y1 - y0) / 0.01))
    for k in range(1, n + 1):
        y = y0 + (y1 - y0) * k / n
        x, h = hand_for_axle(y)
        plan.go(arm, PACE * abs(y1 - y0) * 100 / n, (x, h, HOLD_Z), label="reverse")


def _chock_it(plan):
    arm = "left"
    sx, sy = CHOCK_SPOT
    plan.go(arm, 1.6, (sx, sy, TOP + 0.12), CHOCK_R, "above the chock", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (sx, sy, TOP + CHOCK_GZ), label="descend")
    plan.go(arm, 0.6, finger=CHOCK_G, label="grip")
    plan.go(arm, 0.8, (sx, sy, TOP + 0.12), label="lift")
    x, y = B_AT[0] - M.WHEEL["y"], B_AT[1] - WR - CHOCK["half"] - 0.006
    plan.go(arm, 2.0, (x, y - 0.03, TOP + 0.12), label="over the wheel")
    plan.go(arm, 0.9, (x, y - 0.03, TOP + CHOCK_GZ + 0.002), label="down")
    plan.go(arm, 0.9, (x, y, TOP + CHOCK_GZ + 0.002), label="slide in")
    plan.go(arm, 0.6, finger=CHOCK_R, label="let go")
    plan.go(arm, 0.8, (x, y, TOP + 0.12), label="up")


def _finish(plan):
    for a in SIDE:
        _park(plan, a)
    plan.go("left", 2.0, label="still")
    end = max(plan.t.values())
    for a in SIDE:
        plan.until(a, end)
    return plan.wps


def _nose_for_axle(y):
    """The nose's place and the cart's heading for the axle at y on the lane."""
    x, h = hand_for_axle(y)
    d = 1e-4
    slope = (_axle_x(y + d) - _axle_x(y - d)) / (2 * d)
    return x, h + BAR, math.atan2(-1.0, -slope) - HEADING0


def _nose_run(plan):
    """Debugging only: the shortcut the scorer must refuse. Grip the nose rigidly, steer the cart along the lane by turning the wrist with it."""
    z = TOP + 0.045
    y_change = next(float(y) for y in np.arange(A_AT[1], B_AT[1], 0.002) if _nose_for_axle(float(y))[1] >= HAND_Y_HANDOVER + 0.03)
    for arm, y0, y1 in (("right", A_AT[1], y_change), ("left", y_change, B_AT[1] + 0.01)):
        x, h, turn = _nose_for_axle(y0)
        plan.until(arm, plan.t["right" if arm == "left" else "left"])
        plan.go(arm, 1.8, (x, h, TOP + 0.11), _fingers(0.06), "above the nose", HAND_YAW[arm] + turn)
        plan.go(arm, 0.8, (x, h, z), label="descend")
        plan.go(arm, 0.6, finger=_fingers(2 * 0.012 - 0.004), label="grip the nose")
        n = max(1, round(abs(y1 - y0) / 0.01))
        for k in range(1, n + 1):
            x, h, turn = _nose_for_axle(y0 + (y1 - y0) * k / n)
            plan.go(arm, PACE, (x, h, z), label="reverse", yaw=HAND_YAW[arm] + turn)
        plan.go(arm, 0.6, finger=_fingers(0.06), label="let go")
        plan.go(arm, 0.8, (x, h, TOP + 0.13), label="up")
        if arm == "right":
            _park(plan, arm)
    _chock_it(plan)
    return _finish(plan)


def demo():
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    mode = os.environ.get("CART_TEST", "")  # debugging only: straight, offset (the first test), line, nochock, leave, nose, pull (wrong-way runs)
    if mode != "nochock":
        _take_chock_off(plan)
    x0, h0 = hand_for_axle(A_AT[1])
    if mode == "nose":  # the shortcut: a rigid grip on the nose holds the heading, so nothing's unstable
        return _nose_run(plan)
    _grip_bar(plan, "right", x0, h0)
    plan.go("right", 0.5, label="settle")
    if mode == "straight":  # the riskiest-first test: reverse dead straight, 20 cm
        plan.go("right", 16.0, (x0, h0 + 0.20, HOLD_Z), label="reverse")
        return _finish(plan)
    if mode == "offset":  # the same with the hitch 1 cm off line first: does the error grow?
        plan.go("right", 2.0, (x0 - 0.01, h0, HOLD_Z), label="offset")
        plan.go("right", 16.0, (x0 - 0.01, h0 + 0.20, HOLD_Z), label="reverse")
        return _finish(plan)
    if mode == "pull":  # drag it 7 cm the wrong way, away from the bay
        plan.go("right", 5.0, (x0, h0 - 0.07, HOLD_Z), label="pull")
        _let_go_bar(plan, "right")
        return _finish(plan)
    if mode == "line":  # the hand simply walks the straight line to its end point, with no steering
        xe, he = hand_for_axle(B_AT[1] + 0.01)
        plan.go("right", 8.0, (x0 + (xe - x0) * 0.5, h0 + (he - h0) * 0.5, HOLD_Z), label="reverse")
        _let_go_bar(plan, "right")
        _park(plan, "right")
        plan.until("left", plan.t["right"])
        _grip_bar(plan, "left", x0 + (xe - x0) * 0.5, h0 + (he - h0) * 0.5)
        plan.go("left", 8.0, (xe, he, HOLD_Z), label="reverse")
        _let_go_bar(plan, "left")
        return _finish(plan)
    y_hand = next(float(y) for y in np.arange(A_AT[1], B_AT[1], 0.002) if hand_for_axle(float(y))[1] >= HAND_Y_HANDOVER)
    _reverse(plan, "right", A_AT[1], y_hand)
    _let_go_bar(plan, "right")
    _park(plan, "right")
    plan.until("left", plan.t["right"])
    x, h = hand_for_axle(y_hand)
    _grip_bar(plan, "left", x, h)
    _reverse(plan, "left", y_hand, B_AT[1] + 0.01)
    _let_go_bar(plan, "left")
    if mode != "leave":
        _chock_it(plan)
    return _finish(plan)


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
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", len(bad), bad[:8])
    assert t * 20 < 3000 and not bad
