"""On the station table: an engine-valve fixture, a cast-iron cylinder-head block with a valve stem standing up through
a valve spring, and a fixed steel rocker arm over it. At the rocker's end a vertical adjusting screw runs down through
the rocker; its tip is 8 mm above the valve stem (the valve lash, far too wide). A T-bar with two pins tops the screw;
a loosened locknut with four spoked pins sits on the screw 7 mm above the rocker. A feeler gauge (an aluminium handle
with a 2 mm steel blade) lies on the table to the left. The robot must set the lash to the blade's 2 mm: slide the
blade into the gap between the screw's tip and the stem, turn the screw down until it rests on the blade, turn the
locknut down onto the rocker, then pull the blade out and take both hands away.

The screw moves 2 mm per turn (clockwise from above takes it down). It turns only by its T-bar and holds where it is
left; the loose locknut turns with it. Once the locknut is seated on the rocker the screw is locked and won't turn
until the nut is loosened.

Instruction: "Set the valve lash with the feeler gauge: slide its blade under the adjusting screw, turn the screw
down onto it, tighten the locknut onto the rocker, then pull the blade out and move both hands away."

Needs the `yam_valve` embodiment from this folder (`uv pip install -e . --no-deps`):
    inspect-robots run --embodiment yam_valve -E task=task.py --instruction "..." --scorer task_score ...
"""

import math

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

from valve_lash import fixture as F

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Set the valve lash with the feeler gauge: slide its blade under the adjusting screw, turn the "
               "screw down onto it, tighten the locknut onto the rocker, then pull the blade out and move both "
               "hands away.")

# -- layout (the fixture frame is world-aligned, its origin on the table under the screw's axis) -------------------
X0, Y0 = 0.26, 0.0
GAUGE_START = (0.22, 0.27)  # gauge frame origin (under the handle's centre); the blade points to -y

OBJECTS = [
    # placeholders: the yam_valve embodiment spawns the fixture's frame, its screw and nut, and the gauge in their place
    Box("head_frame", pos=(X0, Y0, TOP), size=(0.01, 0.01, 0.01), color=F.COLORS["cast"]),
    Box("head_mech", pos=(X0, Y0, TOP), size=(0.01, 0.01, 0.01), color=F.COLORS["steel"]),
    Box("gauge", pos=(*GAUGE_START, TOP), size=F.HANDLE, mass=F.GAUGE_MASS, color=F.COLORS["alu"]),
]

# -- scoring -------------------------------------------------------------------------------------------------------
BLADE_T = F.BLADE[2]
GAP_LO, GAP_HI = BLADE_T - 0.0003, BLADE_T + 0.0003  # the lash: 2.0 mm +- 0.3 mm
BLADE_CLEAR = 0.015  # no part of the blade within 1.5 cm of the screw's axis between the stem and the rocker
HANDS_CLEAR = 0.12  # each fingertip centre at least 12 cm (horizontally) from the screw's axis
AXIS_SLIP, TILT = 0.001, math.radians(2)  # the screw and nut stay on their axis (else the mechanism was forced)


def yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _tilt(q):
    x, y, z, w = q
    return math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def gap(objects):
    """The lash (m): the screw tip's height above the valve stem's top."""
    return F.GAP0 + objects["screw"].pos[2] - TOP


def nut_gap(objects):
    """The locknut's underside above the rocker's top (m)."""
    return F.NUT_LIFT0 + objects["nut"].pos[2] - TOP


def intact(objects):
    """Screw and nut still on their axis and upright."""
    for name in ("screw", "nut"):
        s = objects[name]
        if math.hypot(s.pos[0] - X0, s.pos[1] - Y0) > AXIS_SLIP or _tilt(s.quat_xyzw) > TILT:
            return False
    return True


def lash_ok(objects):
    return bool(GAP_LO <= gap(objects) <= GAP_HI)


SEATED_BAND = (-0.0005, 0.0002)  # the nut's underside within -0.5..+0.2 mm of the rocker's top


def seated(objects):
    return bool(SEATED_BAND[0] <= nut_gap(objects) <= SEATED_BAND[1])


def _blade_distance(objects):
    """Least distance (m) from the blade to the screw's axis between the stem's top and the rocker's underside."""
    g = objects["gauge_root"]
    r = _rot(g.quat_xyzw)
    lo, hi = F.blade_box()
    best = math.inf
    for z in np.linspace(F.STEM_TOP, F.ROCKER_BOTTOM, 13):
        p = r.T @ (np.array((X0, Y0, TOP + z)) - g.pos)
        best = min(best, float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi)))))
    return best


def _blade_bottom_at_axis(objects):
    """World height of the blade's underside where it crosses the screw's axis (debug)."""
    g = objects["gauge_root"]
    r = _rot(g.quat_xyzw)
    p = r.T @ (np.array((X0, Y0, 0.0)) - g.pos)  # the axis line in the gauge frame: p + s * (r.T @ z)
    d = r.T @ np.array((0.0, 0.0, 1.0))
    s = -p[2] / d[2]  # where it meets the blade's underside plane (gauge z = 0)
    return float(s)


RESTING = (-0.0005, 0.0003)  # m: "on" a surface, allowing for contact give under load


def blade_touched(objects):
    """The gauge in use: the blade across the axis, its underside on the stem's top, and the screw's tip down on the
    blade's top (each within -0.5..+0.3 mm). The yam_valve embodiment latches this as `blade_used` for the trial."""
    if not blade_in_gap(objects):
        return False
    bottom = _blade_bottom_at_axis(objects) - TOP
    tip = F.STEM_TOP + gap(objects)
    on_stem = RESTING[0] <= bottom - F.STEM_TOP <= RESTING[1]
    on_blade = RESTING[0] <= tip - (bottom + BLADE_T) <= RESTING[1]
    return bool(on_stem and on_blade)


def blade_used(objects):
    return bool(objects["blade_used"].pos[0] > 0.5)


def blade_in_gap(objects):
    return _blade_distance(objects) == 0.0


def blade_clear(objects):
    return _blade_distance(objects) >= BLADE_CLEAR


def hands_clear(objects):
    return all(math.hypot(objects[a].pos[0] - X0, objects[a].pos[1] - Y0) >= HANDS_CLEAR for a in ("left_tcp", "right_tcp"))


def success(objects):
    return blade_used(objects) and intact(objects) and lash_ok(objects) and seated(objects) and blade_clear(objects) and hands_clear(objects)


def score(objects):
    """0.4 lash within 2.0 +- 0.3 mm; +0.3 with the locknut seated on the rocker; +0.3 with the blade out of the gap
    and both hands 12 cm clear of the screw (1.0). Nothing unless the screw was set on the blade at some point in
    the trial (blade_used), or if the screw or nut has been forced off its axis."""
    if not (blade_used(objects) and intact(objects)):
        return 0.0
    ok = lash_ok(objects)
    s = ok and seated(objects)
    return 0.4 * ok + 0.3 * s + 0.3 * (s and blade_clear(objects) and hands_clear(objects))


# -- scripted demo -------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
ARC_SPEED = math.radians(45)  # rad/s while turning a handle
SCREW_OVER = math.radians(45)  # turn on past the blade: the blade stops the screw, the pin slips in the fingers; it covers slips on the way
BACKOFF = math.radians(55)  # then turn the screw back this much (0.31 mm; turning the nut drags it back down)
NUT_OVER = math.radians(25)  # turn on past seating (the rocker stops the nut)
HOVER = 0.02  # above a pin before descending onto it
GRIP_DZ = 0.005  # TCP this far below a pin's top (the pads end just above the T-bar or spoke)
BLADE_LIFT = 0.0045  # commanded blade underside above the stem while sliding in (the held gauge sags about 2 mm)


class _Plan:
    def __init__(self):
        self.wps = {"left": [], "right": []}
        self.t = {"left": 0.0, "right": 0.0}
        self.last = {}
        for a in self.wps:
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
        for a in self.wps:
            if self.t[a] < end - 1e-9:
                self.go(a, end - self.t[a], label="wait")


def _yaw(arm, base):
    return min((base + k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def _fingers(width, extra):
    return -FINGER_PER_M * (width + extra)


def _w(x, y, z):
    return (X0 + x, Y0 + y, TOP + z)


def _gauge_tcp(origin_xy, origin_z):
    """TCP for the grip on the handle (centred, 4 mm under its top) with the gauge's origin at origin_xy, origin_z."""
    return (origin_xy[0], origin_xy[1], origin_z + F.HANDLE[2] - 0.004)


def _insert(plan):
    """Left arm: pick the gauge by its handle, lift it to the gap's height and slide the blade in along -y until its
    tip is 8 mm past the axis, then let the blade down onto the stem."""
    yaw = _yaw("left", math.pi / 2)  # fingers close along x, across the handle's 18 mm
    grip, ready = _fingers(F.HANDLE[0], -0.004), _fingers(F.HANDLE[0], 0.04)
    sx, sy = GAUGE_START
    home_y = Y0 - 0.008 + (-F.BLADE_Y[0])  # gauge origin y with the blade's tip 8 mm past the axis
    out_y = Y0 + 0.02 + (-F.BLADE_Y[0])  # blade tip 2 cm short of the axis
    z_in = TOP + F.STEM_TOP + BLADE_LIFT
    plan.go("left", 1.4, _gauge_tcp((sx, sy), TOP + 0.05), ready, "gauge: above it", yaw)
    plan.go("left", 0.3, label="settle")
    plan.go("left", 1.0, _gauge_tcp((sx, sy), TOP), label="descend")
    plan.go("left", 0.6, finger=grip, label="grip the handle")
    plan.go("left", 1.0, _gauge_tcp((sx, sy), z_in + 0.02), label="lift")
    plan.go("left", 1.6, _gauge_tcp((X0, out_y), z_in + 0.02), label="carry to the gap")
    plan.go("left", 0.8, _gauge_tcp((X0, out_y), z_in), label="down to the gap's height")
    plan.go("left", 0.4, label="settle")
    plan.go("left", 1.6, _gauge_tcp((X0, home_y), z_in), label="slide the blade in")
    plan.go("left", 0.6, _gauge_tcp((X0, home_y), TOP + F.STEM_TOP + 0.0022), label="blade down on the stem")
    return home_y


def _turn(plan, pin_xy, pin_top, total, grips, yaw, label, sign=1, ready=FINGER_OPEN):
    """Right arm: turn a handle clockwise (from above; ``sign=-1``: anticlockwise) by ``total`` rad, at most ``per`` rad per grip, starting each
    grip at the pin at angle ``grips[0]``. ``pin_xy(phi)`` gives the pin (fixture frame) at angle phi,
    ``pin_top(turned)`` its top's height after ``turned`` rad."""
    start_phi, per = grips
    grip = _fingers(2 * F.PIN_R, -0.005)  # opened wide, closing centres a pin that is a little off
    turned = 0.0
    while turned < total - 1e-6:
        arc = min(per, total - turned)

        def at(phi, done):
            x, y = pin_xy(phi)
            return _w(x, y, pin_top(done) - GRIP_DZ)

        p = at(start_phi, turned)
        if turned > 0:  # round by the near (-y) side, clear of the screw's top and of the other hand
            v = _w(0.0, pin_xy(-math.pi / 2)[1] - 0.025, pin_top(turned) + HOVER)
            plan.go("right", 0.8, v, label=f"{label}: round the near side")
        plan.go("right", 1.4, (p[0], p[1], p[2] + HOVER), ready, f"{label}: above the pin", yaw)
        plan.go("right", 0.3, label="settle")
        plan.go("right", 0.8, p, label="descend onto the pin")
        plan.go("right", 0.5, finger=grip, label="grip")
        n = max(2, math.ceil(arc / math.radians(15)))
        for k in range(1, n + 1):
            d = arc * k / n
            plan.go("right", (arc / n) / ARC_SPEED, at(start_phi - sign * d, turned + d),
                    label=f"{label}: {math.degrees(turned + d):.0f} deg")
        turned += arc
        end = plan.last["right"][0]
        plan.go("right", 0.4, finger=ready, label="release")
        plan.go("right", 0.6, (end[0], end[1], end[2] + HOVER), label="up")


def _screw(plan):
    total = 2 * math.pi * (F.GAP0 - BLADE_T) / F.PITCH + SCREW_OVER
    stop = 2 * math.pi * (F.GAP0 - BLADE_T) / F.PITCH  # where the blade stops it

    def top(done):
        return F.SCREW_PIN_TOP0 - F.PITCH * min(done, stop) / (2 * math.pi)

    _turn(plan, lambda phi: (F.TBAR_R * math.cos(phi), F.TBAR_R * math.sin(phi)), top, total,
          (0.0, math.pi), _yaw("right", 0.0), "screw")
    # then back it off a little, anticlockwise, so the blade is just held rather than pinched
    _turn(plan, lambda phi: (F.TBAR_R * math.cos(phi), F.TBAR_R * math.sin(phi)), lambda done: top(stop),
          BACKOFF, (math.pi, BACKOFF), _yaw("right", 0.0), "back off", sign=-1)


def _nut(plan):
    ride = F.GAP0 - BLADE_T - F.PITCH * BACKOFF / (2 * math.pi)  # the nut rode with the screw (and turned with it)
    lift = F.NUT_LIFT0 - ride
    total = 2 * math.pi * lift / F.PITCH + NUT_OVER
    stop = 2 * math.pi * lift / F.PITCH

    def top(done):
        return F.NUT_PIN_TOP0 - ride - F.PITCH * min(done, stop) / (2 * math.pi)

    _turn(plan, lambda phi: (F.SPOKE_R * math.cos(phi), F.SPOKE_R * math.sin(phi)), top, total,
          (F.NUT_PHASE0 + BACKOFF, math.pi / 2), _yaw("right", math.pi / 2), "locknut",
          ready=_fingers(2 * F.PIN_R, 0.035))  # not too wide: the inner finger stays clear of the T-bar


def demo():
    plan = _Plan()
    plan.go("right", 2.0, _w(0.0, -0.10, 0.16), FINGER_OPEN, "wait clear of the gauge", _yaw("right", 0.0))
    home_y = _insert(plan)
    plan.sync()
    _screw(plan)
    _nut(plan)
    plan.go("right", 1.0, (0.16, -0.30, TOP + 0.20), FINGER_OPEN, "out of the way")
    plan.sync()
    sx, sy = GAUGE_START
    z_in = TOP + F.STEM_TOP + BLADE_LIFT
    plan.go("left", 0.5, _gauge_tcp((X0, home_y), z_in), label="lift the blade off the stem")
    plan.go("left", 1.6, _gauge_tcp((X0, home_y + 0.07), z_in), label="pull the blade out")
    plan.go("left", 1.4, _gauge_tcp((sx, sy), TOP + 0.02), label="carry it back")
    plan.go("left", 0.8, _gauge_tcp((sx, sy), TOP + 0.001), label="set it down")
    plan.go("left", 0.5, finger=_fingers(F.HANDLE[0], 0.04), label="let go")
    plan.go("left", 0.8, _gauge_tcp((sx, sy), TOP + 0.06), label="up")
    plan.go("left", 1.0, (0.16, 0.30, TOP + 0.20), FINGER_OPEN, "out of the way")
    plan.go("left", 2.0, label="hold still")
    plan.sync()
    return plan.wps
