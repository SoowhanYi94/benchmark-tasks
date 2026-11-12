"""On the station table: a small lathe headstock with a 70 mm 3-jaw chuck, its spindle running along y (away from the
right arm into the headstock) and the chuck's face toward the right arm. The jaws are open. A round aluminium bar
(20 mm x 120 mm) lies along y in a V-block on the right; the chuck key (a T-handle with a 10 mm square drive) stands in
its holder, an orange block on the left. The socket for the key is on top of the chuck: the key goes in vertically and
turning it anticlockwise, seen from above, closes all three jaws together. The instruction:

Chuck the bar in the lathe with 50 mm sticking out of the jaws, tighten it with the chuck key, and leave the lathe ready to start with the key put away in its holder.

Tight means the bar holds: once the key is out
of the chuck and every hand is 6 cm or more from the bar, the judge pulls the bar straight out of the chuck with
30 N (ramped over 0.5 s, then held 1 s); it must slip less than 1 mm. Leaving the key in or on the chuck, its jaws or
the bar caps the score at 0.1.

Needs the `yam_chuck` embodiment from this folder (`uv pip install -e . --no-deps`):
    inspect-robots run --embodiment yam_chuck -E task=task.py --instruction "..." --scorer task_score ...
"""

import math

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

from chuck_key import lathe as L

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- layout: the lathe frame (chuck_key.lathe) sits at (X0, Y0) on the table, unrotated ---------------------------------
X0, Y0 = 0.30, -0.04  # world x, y of the spindle axis in the plane of the jaws' front faces


def to_world(x, y, z):
    return (X0 + x, Y0 + y, TOP + z)


AXIS_Z = TOP + L.AXIS_Z  # spindle axis (world z)
STICK_OUT, STICK_TOL = 0.050, 0.003  # bar out of the jaws' front faces
SOCKET = to_world(0.0, L.SOCKET_Y, L.SOCKET_Z)
HOLDER = to_world(*L.HOLDER_XY, 0.0)[:2]
KEY_HOME_Z = TOP + L.KEY_HOME_Z
BAR_START = to_world(*L.VBLOCK_XY, L.BAR_START_Z)
KEY_START = (*HOLDER, KEY_HOME_Z)
KEY_START_YAW = -math.pi / 2  # T-bar along y

OBJECTS = [
    # placeholders: the yam_chuck embodiment spawns the lathe (an articulation), the bar and the key in their place
    Box("lathe", pos=(X0, Y0, TOP), size=(0.01, 0.01, 0.01), color=(0.3, 0.34, 0.36)),
    Box("bar", pos=BAR_START, size=(0.01, 0.01, 0.01), mass=L.BAR_MASS),
    Box("key", pos=KEY_START, size=(0.01, 0.01, 0.01), mass=L.KEY_MASS, yaw=KEY_START_YAW),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
PULL_N, PULL_RAMP_S, PULL_SLIP, PULL_HOLD_S = 30.0, 0.5, 0.001, 1.0  # the judge's pull test
HANDS_OFF = 0.06  # every fingertip centre at least this far from the bar's surface
KEY_CLEAR = 0.07  # ... and from the key's T-bar centre (the T-bar is 4.5 cm each side) for the key to count as put away
CAP = 0.1  # the most a trial scores with the key in or on the chuck
JAW_MID_Y = Y0 + L.JAW_DEPTH / 2


def _axis(q, local):
    x, y, z, w = q
    m = np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                  (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                  (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))))
    return m @ np.asarray(local, float)


def bar_axis(objects):
    """The bar's axis direction, pointing into the chuck (+y)."""
    d = _axis(objects["bar"].quat_xyzw, (0.0, 1.0, 0.0))
    return d if d[1] >= 0 else -d


def bar_tilt(objects):
    """Angle between the bar and the spindle (rad)."""
    return math.acos(min(1.0, abs(float(bar_axis(objects)[1]))))


def stick_out(objects):
    """How far the bar's outer end is out of the jaws' front faces, along the spindle (m)."""
    end = objects["bar"].pos - bar_axis(objects) * L.BAR_LEN / 2
    return Y0 - float(end[1])


def bar_offset(objects):
    """Where the bar's axis crosses the jaws' mid-plane, relative to the spindle axis: (x, z) in m."""
    d = bar_axis(objects)
    if d[1] < 1e-6:
        return np.array((1.0, 1.0))
    p = objects["bar"].pos + d * (JAW_MID_Y - objects["bar"].pos[1]) / d[1]
    return np.array((p[0] - X0, p[2] - AXIS_Z))


def inserted(objects):
    """In the open jaws: within 3 degrees of the spindle, its axis within 11 mm of the spindle's at the jaws, and
    50 +/- 3 mm out of the jaws."""
    return bool(bar_tilt(objects) <= math.radians(3) and np.linalg.norm(bar_offset(objects)) <= 0.011
                and abs(stick_out(objects) - STICK_OUT) <= STICK_TOL)


def jaw_gaps(objects):
    """Each jaw face's distance from the bar's surface at the jaws' mid-plane (m; negative: pressed in)."""
    c = bar_offset(objects)
    r = objects["jaws"].pos  # each face's radius from the spindle axis
    return np.array([r[k] - float(L.jaw_dir(k) @ c) - L.BAR_R for k in range(3)])


def clamped(objects):
    """Inserted, all three jaws on it (each face within 0.5 mm of its surface), centred within 0.5 mm and within
    1 degree of the spindle."""
    return bool(inserted(objects) and np.all(np.abs(jaw_gaps(objects)) <= 0.0005)
                and np.linalg.norm(bar_offset(objects)) <= 0.0005 and bar_tilt(objects) <= math.radians(1))


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def hands_off(objects):
    """Every fingertip centre at least HANDS_OFF from the bar's surface."""
    c, d = objects["bar"].pos, bar_axis(objects)
    a, b = c - d * L.BAR_LEN / 2, c + d * L.BAR_LEN / 2
    return all(_seg_dist(np.asarray(objects[t].pos, float), a, b) - L.BAR_R >= HANDS_OFF for t in ("left_tcp", "right_tcp"))


def pull_ready(objects):
    """When the judge pulls (the yam_chuck embodiment applies the pull while this holds): all three jaws on the bar,
    every hand clear of it, and the key out of the chuck (so the pull tests the finished job, not a half-tightened
    one)."""
    return clamped(objects) and hands_off(objects) and not key_on_chuck(objects)


def locked(objects):
    """The pull has held 30 N for 1 s with the bar slipping less than 1 mm."""
    force, slip, held = objects["pull_test"].pos
    return bool(force >= PULL_N - 1e-6 and slip < PULL_SLIP and held >= PULL_HOLD_S - 1e-6)


def key_tilt(objects):
    return math.acos(min(1.0, abs(float(_axis(objects["key"].quat_xyzw, (0.0, 0.0, 1.0))[2]))))


def key_centre(objects):
    """Midway up the key's shaft."""
    k = objects["key"]
    return np.asarray(k.pos, float) + _axis(k.quat_xyzw, (0.0, 0.0, L.SHAFT_LEN / 2))


CAP_R = L.CHUCK_R + 0.01  # the cap's region: within 1 cm of the chuck's radius round the spindle axis ...
CAP_Y = (Y0 - STICK_OUT - 0.01, Y0 + L.BODY_Y[1] + 0.01)  # ... from 1 cm past the bar's end to 1 cm behind the chuck


def key_points(objects):
    """The key's drive tip, shoulder, shaft middle, T-bar centre and T-bar ends (world)."""
    k = objects["key"]
    local = ((0, 0, -L.DRIVE_LEN), (0, 0, 0), (0, 0, L.SHAFT_LEN / 2), (0, 0, L.TBAR_Z),
             (L.TBAR_HALF, 0, L.TBAR_Z), (-L.TBAR_HALF, 0, L.TBAR_Z))
    return [np.asarray(k.pos, float) + _axis(k.quat_xyzw, p) for p in local]


def key_on_chuck(objects):
    """The key in or on the chuck, its jaws or the bar sticking out of them: any of key_points within CAP_R of the
    spindle axis, between CAP_Y. (A key in the socket has its shoulder 3.5 cm from the axis.)"""
    return any(math.hypot(p[0] - X0, p[2] - AXIS_Z) <= CAP_R and CAP_Y[0] <= p[1] <= CAP_Y[1]
               for p in key_points(objects))


def key_home(objects):
    """Standing in its holder: upright within 10 degrees, its axis within 3 mm of the hole's centre, down on the
    hole's floor (key frame within 3 mm of its home height), and every fingertip centre KEY_CLEAR from its T-bar."""
    k = objects["key"]
    tbar = np.asarray(k.pos, float) + _axis(k.quat_xyzw, (0.0, 0.0, L.TBAR_Z))
    clear = all(np.linalg.norm(np.asarray(objects[t].pos, float) - tbar) >= KEY_CLEAR for t in ("left_tcp", "right_tcp"))
    return bool(key_tilt(objects) <= math.radians(10) and math.hypot(k.pos[0] - HOLDER[0], k.pos[1] - HOLDER[1]) <= 0.003
                and abs(k.pos[2] - KEY_HOME_Z) <= 0.003 and clear)


def score(objects):
    """0.2 bar in the jaws at 50 +/- 3 mm stick-out; +0.2 all three jaws on it, centred; +0.3 it holds the judge's
    pull with hands clear; +0.3 the key back in its holder (counted once the bar holds). Then the cap: key in or on
    the chuck (key_on_chuck), at most 0.1."""
    p = inserted(objects)
    c = p and clamped(objects)
    lk = c and hands_off(objects) and locked(objects)
    s = 0.2 * p + 0.2 * c + 0.3 * lk + 0.3 * (lk and key_home(objects))
    return min(s, CAP) if key_on_chuck(objects) else s


def success(objects):
    return score(objects) >= 1.0 - 1e-9


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
GRIP_R = 0.030  # the left hand holds the key's T-bar 3 cm from its axis
TBAR_SEATED = TOP + L.KEY_SEATED_Z + L.TBAR_Z  # T-bar axis height with the key in the socket
TBAR_HOME = KEY_HOME_Z + L.TBAR_Z
CARRY_Z = TBAR_SEATED + 0.020  # T-bar height carrying the key over the chuck (drive 8 mm above the socket)
TURN = math.radians(750)  # hand travel in four sweeps; the key ends ~60 degrees behind it (KEY_LAG per regrip), so
# the pinion turns ~680: ~610 to close the jaws on the bar, ~70 more to clamp it
ARC_SPEED = math.radians(45)  # rad/s while turning
KEY_LAG = math.radians(20)  # the key ends each turn about 20 degrees behind the hand (socket play, arm lag; measured)
BAR_GRIP = 0.022  # the right hand holds the bar this far from its outer end
BAR_TRIM = (0.002, 0.0025, 0.0027)  # the held bar runs about 2 mm short in x, 3 mm low and 2.7 mm short of
# its stick-out (measured: it settles at 52.7 mm), so aim past it
REST_DROP = (L.R_OPEN - L.BAR_R) / math.sin(math.radians(30))  # the bar rests this far below the axis in the open jaws


class _Plan:
    def __init__(self):
        self.wps = {a: [] for a in ("left", "right")}
        self.t = {a: 0.0 for a in self.wps}
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


def _fingers(width):
    return -FINGER_PER_M * width


def _bar(plan):
    """Right arm: grip the bar near its outer end (fingers along x), slide it into the open jaws along the spindle to
    50 mm stick-out, lower it onto the two lower jaws and let go."""
    yaw = math.pi / 2  # fingers close along x
    grip, ready = _fingers(2 * L.BAR_R - 0.010), _fingers(2 * L.BAR_R + 0.04)
    bx, by, bz = BAR_START
    gy = by - L.BAR_LEN / 2 + BAR_GRIP
    end_y = Y0 - STICK_OUT  # the bar's outer end when it is in
    pre_y = Y0 - 0.010 - L.BAR_LEN  # the bar's outer end, 1 cm short of the jaws
    plan.go("right", 1.4, (bx, gy, bz + 0.06), ready, "bar: above it", yaw)
    plan.go("right", 1.0, (bx, gy, bz), label="descend")
    plan.go("right", 0.6, finger=grip, label="grip")
    plan.go("right", 0.8, (bx, gy, bz + 0.035), label="lift")
    ax, az = X0 + BAR_TRIM[0], AXIS_Z + BAR_TRIM[1]
    plan.go("right", 1.6, (ax, pre_y + BAR_GRIP, az), label="line it up with the spindle")
    plan.go("right", 0.4, label="settle")
    ey = end_y + BAR_GRIP + BAR_TRIM[2]
    plan.go("right", 2.4, (ax, ey, az), label="slide it into the jaws")
    plan.go("right", 0.6, (ax, ey, az - REST_DROP + 0.002), label="onto the lower jaws")
    plan.go("right", 0.5, finger=ready, label="let go")
    plan.go("right", 0.8, (X0, ey, AXIS_Z + 0.08), label="up")
    plan.go("right", 1.0, (0.16, -0.30, TOP + 0.20), FINGER_OPEN, "out of the way")


def _tcp(axis_xy, alpha, z):
    """The left hand's point on the T-bar, GRIP_R out from the key's axis along the rod's direction alpha."""
    return (axis_xy[0] + GRIP_R * math.cos(alpha), axis_xy[1] + GRIP_R * math.sin(alpha), z)


def _yaw(alpha):
    """Fingers across the rod: yaw = alpha - pi (the left arm's branch for alpha in 15..285 degrees)."""
    return alpha - math.pi


def _fetch_key(plan):
    """Left arm: grip the key's T-bar in the holder and lift it clear."""
    grip, ready = _fingers(2 * L.TBAR_R - 0.005), _fingers(2 * L.TBAR_R + 0.040)
    a = math.pi / 2  # the T-bar's +y end
    plan.go("left", 1.4, _tcp(HOLDER, a, TBAR_HOME + 0.06), ready, "key: above the holder", _yaw(a))
    plan.go("left", 1.0, _tcp(HOLDER, a, TBAR_HOME), label="descend")
    plan.go("left", 0.6, finger=grip, label="grip")
    plan.go("left", 1.2, _tcp(HOLDER, a, CARRY_Z), label="lift it out")
    return grip, ready


def _turn_key(plan, grip, ready):
    """Carry the key round the headstock to the socket, set it in, turn it TURN in regrips, take it out and stand
    it back in the holder."""
    sock = SOCKET[:2]  # the socket's lead-in takes up the hand's few mm of error
    a = math.pi / 2
    plan.go("left", 1.4, _tcp((HOLDER[0], sock[1]), a, CARRY_Z), label="carry: in front of the headstock")
    plan.go("left", 1.4, _tcp(sock, a, CARRY_Z), label="over the socket")
    plan.go("left", 0.4, label="settle")
    plan.go("left", 1.5, _tcp(sock, a, TBAR_SEATED + 0.001), label="key into the socket")
    sweeps, turned, start = [], 0.0, a
    while turned < TURN - 1e-9:
        arc = min(math.radians(240) - start, TURN - turned)
        sweeps.append((start, arc))
        turned += arc
        start = start + arc - KEY_LAG - math.pi  # regrip the other end of the T-bar, where the key really is
    for i, (s, arc) in enumerate(sweeps):
        if i:
            m = s + math.pi / 2  # wind the wrist back through the middle, not the short way round past joint 6's limit
            plan.go("left", 0.6, _tcp(sock, m, TBAR_SEATED + 0.025), ready, "regrip: wind back", _yaw(m))
            plan.go("left", 0.6, _tcp(sock, s, TBAR_SEATED + 0.025), label="above the other end", yaw=_yaw(s))
            plan.go("left", 0.8, _tcp(sock, s, TBAR_SEATED), label="descend")
            plan.go("left", 0.5, finger=grip, label="grip")
        n = max(2, math.ceil(arc / math.radians(10)))
        for k in range(1, n + 1):
            al = s + arc * k / n
            plan.go("left", (arc / n) / ARC_SPEED, _tcp(sock, al, TBAR_SEATED), label=f"turn {math.degrees(al):.0f}", yaw=_yaw(al))
        if i < len(sweeps) - 1:
            end = plan.last["left"][0]
            plan.go("left", 0.4, finger=ready, label="release")
            plan.go("left", 0.6, (end[0], end[1], TBAR_SEATED + 0.025), label="up")
    end_a = sweeps[-1][0] + sweeps[-1][1]
    plan.go("left", 0.4, label="settle")
    plan.go("left", 1.6, _tcp(sock, end_a, CARRY_Z), label="pull the key out")
    plan.go("left", 1.6, _tcp((HOLDER[0], sock[1]), a, CARRY_Z), label="carry it back", yaw=_yaw(a))
    plan.go("left", 1.4, _tcp(HOLDER, a, CARRY_Z), label="over the holder")
    plan.go("left", 0.4, label="settle")
    plan.go("left", 2.0, _tcp(HOLDER, a, TBAR_HOME + 0.015), label="lower it into the holder")
    plan.go("left", 0.4, _tcp(HOLDER, a, TBAR_HOME + 0.006), label="just above the floor")
    plan.go("left", 0.5, finger=ready, label="let go")
    plan.go("left", 0.8, _tcp(HOLDER, a, TBAR_HOME + 0.07), label="up")
    plan.go("left", 1.0, (0.16, 0.30, TOP + 0.20), FINGER_OPEN, "out of the way")
    plan.go("left", 3.0, label="wait for the judge")


def demo():
    plan = _Plan()
    _bar(plan)
    grip, ready = _fetch_key(plan)
    plan.sync()  # the right hand is clear of the chuck before the key goes in
    _turn_key(plan, grip, ready)
    plan.sync()
    return plan.wps
