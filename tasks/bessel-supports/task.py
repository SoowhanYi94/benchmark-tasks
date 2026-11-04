"""On the station table: a granite surface plate in the middle of the bench, with a 30 cm precision aluminium
straightedge lying on a cloth on the bench in front of it, along y, and a support block standing on each side. The
straightedge is unmarked. A gauge bar or straightedge supported at its Bessel points, 0.2203 of its length in from
each end, keeps the length of its neutral axis most nearly unchanged as it sags. The robot must stand the two
supports on the plate and lay the straightedge across them, level and along y, with each support within 1 cm of one
of its Bessel points (6.6 cm in from an end), and let go of it.

Run it:  inspect-robots run --embodiment yam_bessel -E task=tasks/bessel-supports/task.py \\
           --instruction "Support the 30 cm straightedge at its Bessel points: stand the two support blocks on the surface plate, lay the straightedge across them, level, with each support 22 percent of its length in from its end, and let go of it." --scorer task_score ...
"""

import math

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

PLATE = dict(center=(0.29, 0.0), size=(0.16, 0.40, 0.02))
PLATE_Z = TOP + PLATE["size"][2]
BAR = (0.025, 0.30, 0.025)  # x, y, z: an aluminium straightedge, 0.5 kg
BAR_L = BAR[1]
BESSEL = 0.2203  # Bessel points: this fraction of the length in from each end
SUPPORT = (0.03, 0.03, 0.04)  # x, y, z: a support block
SUPPORT_START = {"support_a": (0.18, 0.24), "support_b": (0.18, -0.24)}  # one on each side
CLOTH = dict(center=(0.15, 0.0), size=(0.06, 0.34, 0.003))
BAR_START = (0.15, 0.0)
BESSEL_TOL = 0.01  # each support's centre within 1 cm of a Bessel point (6.61 cm in from an end)


OBJECTS = [
    Box("surface_plate", pos=(*PLATE["center"], TOP + PLATE["size"][2] / 2), size=PLATE["size"], color=(0.20, 0.20, 0.22)),
    Box("bar_cloth", pos=(*CLOTH["center"], TOP + CLOTH["size"][2] / 2), size=CLOTH["size"], color=(0.55, 0.20, 0.20)),
    Box("straightedge", pos=(*BAR_START, TOP + CLOTH["size"][2] + BAR[2] / 2), size=BAR, mass=0.5, color=(0.80, 0.82, 0.85)),
    *[Box(n, pos=(*xy, TOP + SUPPORT[2] / 2), size=SUPPORT, mass=0.25, color=(0.25, 0.40, 0.70)) for n, xy in SUPPORT_START.items()],
]


# -- scoring --------------------------------------------------------------------------------------------------------
SEAT_TOL = 0.0015  # the bar's underside within 1.5 mm of each support's top
HANDS_OFF = 0.03  # each gripper's TCP more than 3 cm from the bar: let go of it
LEVEL, ALONG_Y, FACE_FLAT, UPRIGHT = (math.radians(d) for d in (2, 5, 5, 5))
SUPPORT_Y = BAR_L / 2 - BESSEL * BAR_L  # each Bessel point this far from the bar's middle (8.39 cm)


def _rot(quat_xyzw):
    x, y, z, w = quat_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _tilt(quat_xyzw):
    return math.acos(max(-1.0, min(1.0, _rot(quat_xyzw)[2, 2])))


def _support_up(name, objects):
    """Standing upright (within 5 degrees) on the plate, still."""
    o = objects[name]
    (cx, cy), (sx, sy, _) = PLATE["center"], PLATE["size"]
    return bool(abs(o.pos[0] - cx) <= sx / 2 and abs(o.pos[1] - cy) <= sy / 2 and o.low[2] <= PLATE_Z + 0.004
                and _tilt(o.quat_xyzw) <= UPRIGHT and o.resting())


def bar_aligned(b):
    """Its length level within 2 degrees and along y within 5 degrees (either way round), a face flat within 5."""
    r = _rot(b.quat_xyzw)
    a = r[:, 1]
    return bool(math.asin(min(1.0, abs(a[2]))) <= LEVEL and math.atan2(abs(a[0]), abs(a[1])) <= ALONG_Y
                and max(abs(r[2, 0]), abs(r[2, 2])) >= math.cos(FACE_FLAT))


def along_bar(name, objects):
    """(Distance of a support's centre from the bar's middle along the bar, across it.)"""
    b = objects["straightedge"]
    a = _rot(b.quat_xyzw)[:2, 1]
    a = a / np.linalg.norm(a)
    d = np.asarray(objects[name].pos[:2], float) - np.asarray(b.pos[:2], float)
    return float(d @ a), float(abs(d[0] * a[1] - d[1] * a[0]))


def hands_off(objects):
    """Both grippers' TCPs more than HANDS_OFF from the bar's surface."""
    b = objects["straightedge"]
    r = _rot(b.quat_xyzw)
    for arm in ("left_tcp", "right_tcp"):
        p = r.T @ (np.asarray(objects[arm].pos, float) - np.asarray(b.pos, float))
        if np.linalg.norm(np.maximum(np.abs(p) - np.array(BAR) / 2, 0.0)) <= HANDS_OFF:
            return False
    return True


def bar_on_supports(objects):
    """Supported: both blocks upright on the plate, the bar aligned, lying across both of them (one either side of
    its middle, under it) with its underside within 1.5 mm of each block's top, still, and both hands off it."""
    b = objects["straightedge"]
    if not all(_support_up(n, objects) for n in SUPPORT_START) or not b.resting() or not bar_aligned(b):
        return False
    a_z = _rot(b.quat_xyzw)[2, 1]
    s = []
    for n in SUPPORT_START:
        along, across = along_bar(n, objects)
        underside = b.pos[2] + along * a_z - BAR[2] / 2
        if not (abs(along) < BAR_L / 2 and across <= (SUPPORT[0] + BAR[0]) / 2
                and abs(underside - objects[n].high[2]) <= SEAT_TOL):
            return False
        s.append(along)
    return bool(min(s) < 0 < max(s) and hands_off(objects))


def supports_at_bessel(objects):
    """How many supports have their centre within BESSEL_TOL, along the bar, of a Bessel point (one per point)."""
    s = sorted(along_bar(n, objects)[0] for n in SUPPORT_START)
    return sum(abs(v - p) <= BESSEL_TOL for v, p in zip(s, (-SUPPORT_Y, SUPPORT_Y)))


def success(objects):
    return bar_on_supports(objects) and supports_at_bessel(objects) == 2


def score(objects):
    """0 unless supported; 0.5 supported; + 0.25 for each support within 1 cm of a Bessel point."""
    if not bar_on_supports(objects):
        return 0.0
    return float(0.5 + 0.25 * supports_at_bessel(objects))


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
PX = PLATE["center"][0]
GRIP_Y = 0.12  # each arm holds the bar this far from its middle: outside the supports (2 cm clear), the two hands well
# apart (8 cm apart, they collide)
TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
BAR_TRIM_X = 0.0  # carried by both arms (held 12 cm either side of its middle), the straightedge comes down on target


class _Plan:
    """Waypoints for both arms in phases: within a phase each arm moves independently, then both wait for the slower."""

    def __init__(self):
        self.wps = {a: [] for a in SIDE}
        self.t = {a: 0.0 for a in SIDE}
        self.last = {}
        for a in SIDE:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None):
        p, f, y = self.last[arm]
        p = p if pos is None else pos
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


def _yaw(arm, base):
    return min((base + k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def demo():
    plan = _Plan()
    # 1. the supports, one with each arm: the left one to +8.4 cm, the right one to -8.4 cm
    g = SUPPORT[2] - 0.01
    grip, ready = -FINGER_PER_M * (SUPPORT[1] - 0.004), -FINGER_PER_M * (SUPPORT[1] + 0.02)
    for arm, name, y in (("left", "support_a", SUPPORT_Y), ("right", "support_b", -SUPPORT_Y)):
        (sx, sy) = SUPPORT_START[name]
        qx, qy = PX + TRIM[arm][0], y + TRIM[arm][1]
        yaw = _yaw(arm, 0.0)
        z_up = PLATE_Z + SUPPORT[2] + 0.04
        plan.go(arm, 1.3, (sx, sy, z_up), ready, f"{name}: above it", yaw)
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 1.0, (sx, sy, TOP + g), label="descend")
        plan.go(arm, 0.6, finger=grip, label="grip")
        plan.go(arm, 0.9, (sx, sy, z_up), label="lift")
        plan.go(arm, 1.5, (qx, qy, z_up), label="over its place")
        plan.go(arm, 1.0, (qx, qy, PLATE_Z + g + 0.002), label="stand it on the plate")
        plan.go(arm, 0.6, finger=ready, label="release")
        plan.go(arm, 0.8, (qx, qy, z_up), label="up")
        plan.go(arm, 1.0, (0.18, SIDE[arm] * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
        plan.sync()
    # 2. both arms lay the bar across them
    bx, by = BAR_START
    gb = 0.015
    bgrip, bready = -FINGER_PER_M * (BAR[0] - 0.004), -FINGER_PER_M * (BAR[0] + 0.02)
    z_up = PLATE_Z + SUPPORT[2] + 0.03 + gb
    for a, s in SIDE.items():
        plan.go(a, 1.5, (bx, by + s * GRIP_Y, z_up), bready, "straightedge: above it", _yaw(a, math.pi / 2))
    plan.sync()
    for a, s in SIDE.items():
        plan.go(a, 1.0, (bx, by + s * GRIP_Y, TOP + CLOTH["size"][2] + gb), label="descend")
    plan.sync()
    for a in SIDE:
        plan.go(a, 0.6, finger=bgrip, label="grip")
    plan.sync()
    for a, s in SIDE.items():
        plan.go(a, 1.2, (bx, by + s * GRIP_Y, z_up), label="lift together")
        plan.go(a, 2.0, (PX + BAR_TRIM_X, s * GRIP_Y, z_up), label="carry over the supports")
        plan.go(a, 1.2, (PX + BAR_TRIM_X, s * GRIP_Y, PLATE_Z + SUPPORT[2] + gb + 0.002), label="lower onto them")
    plan.sync()
    for a, s in SIDE.items():
        plan.go(a, 0.8, finger=bready, label="release")
        plan.go(a, 0.8, (PX, s * GRIP_Y, z_up), label="up")
        plan.go(a, 1.0, (0.18, s * 0.32, TOP + 0.18), FINGER_OPEN, "back")
    plan.sync()
    return plan.wps


# -- demo-plan check (pure Python): the plan fits the run limit and IK reaches every point of it ----------------------
def check_demo():
    """(Steps the demo takes, worst IK position error in mm, worst rotation error in degrees): every waypoint of both
    arms' plans solved with ik.solve, each warm-started from the one before, as the scripted policy does. Waypoints
    still at the arm's start (READY) pose are skipped: there the player only turns the gripper to point down."""
    from yam_arena.demo import grip_rotation

    steps, mm, deg = 0, 0.0, 0.0
    for arm, wps in demo().items():
        q = np.array(READY, float)
        start, _ = ik.tcp_pose(q, np.array(BASES[arm]))
        for w in wps:
            q, pe, re = ik.solve(w.pos, ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
            if not np.allclose(w.pos, start):
                mm, deg = max(mm, pe * 1000), max(deg, math.degrees(re))
        steps = max(steps, round(sum(w.duration for w in wps) * 20))
    return steps, mm, deg


if __name__ == "__main__":
    steps, mm, deg = check_demo()
    assert steps < 2400 and mm < 1.0 and deg < 1.0, (steps, mm, deg)
    print(f"ok: demo {steps} steps of 2,400, worst IK error {mm:.3f} mm, {deg:.3f} deg")
