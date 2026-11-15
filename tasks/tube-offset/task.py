"""A 3/8 in tube hand bender: a tube lies on a low-friction bed, marked with two yellow bands. The robot must bend it into an
offset: feed it along the bed until a band sits at the former's red mark with the part ahead of it through the guide, sweep the handle
about the former, take the springback into account, turn the kinked tube round on the bed, feed the other band in and bend it the same
way, so the two ends come out parallel, then lay the finished tube on the painted template and move both hands away.

Run it:  inspect-robots run --embodiment yam_tube_offset -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Bend the tube on the hand bender at both yellow bands into a 45 degree offset with parallel ends, lay it on the green template, and move both hands away." --scorer task_score

`yam_tube_offset` (tube_offset/embodiment.py) spawns the bender and the tube, writes the tube's joint limits every step from `ratchet`,
and reports each gripper's fingertip midpoint, the tube's three segments and the handle.
"""

import math
import os

import numpy as np

from tube_offset import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
BED_TOP = TOP + M.BED_H
BENDER, TUBE = "bender", "tube"
L_M, L_N, L_T = M.TUBE["m"], M.TUBE["n"], M.TUBE["t"]
START = (0.26, 0.40, math.pi / 2)  # the grip block's centre and the tube's heading: nose toward +x
OFFSET_LANE_SHIFT = (-0.06, 0.03)
BENT = math.radians(45.0)
TARGET_OFFSET = L_M * math.sin(BENT)  # 9.9 cm


def _template():
    """The finished tube, block at its middle, lying with its long ends along y: where the lanes are painted."""
    bx, by = 0.2476 + OFFSET_LANE_SHIFT[0], 0.19 + OFFSET_LANE_SHIFT[1]
    h = L_M / 2 * math.sin(BENT)
    v = L_M / 2 * math.cos(BENT)
    return (bx, by), (bx + h, by - v - L_T / 2), (bx - h, by + v + L_N / 2)


GRIP_AT, LANE_A, LANE_B = _template()  # lane centres, m in table xy
LANE_HALF = (0.02, 0.0625)  # painted half width and half length
LANES = tuple((c[0] - LANE_HALF[0], c[1] - LANE_HALF[1], c[0] + LANE_HALF[0], c[1] + LANE_HALF[1]) for c in (LANE_A, LANE_B))

OBJECTS = [
    Box(BENDER, pos=(0.0, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    Box(TUBE, pos=(START[0], START[1], BED_TOP + 0.0005), size=(0.01, 0.01, 0.01), yaw=START[2]),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
BAND_X, BAND_Y, HELD_YAW = 0.012, 0.012, math.radians(10)  # a band's over the former within these, its held side in line within this
SPRINGBACK = math.radians(3.0)  # the top of the 1 to 3 degrees Swagelok's hand tube bender manual (MS-13-43, p.19) says to allow
BEND_TOL, RUIN = math.radians(6.0), math.radians(55.0)
PARALLEL, OFFSET_TOL = math.radians(5.0), 0.01
ON_LATERAL, ON_ALONG, ON_YAW = 0.015, 0.025, math.radians(8.0)
LIN, ANG, STILL_S = 0.01, 0.5, 1.0
BENT_ANY = math.radians(10.0)
TOUCH = 0.003  # the shoe's within this of a segment's outer face: it's pressing it
MOVING = math.radians(2.0)  # the handle's advancing faster than this, in radians per second
STEP = math.radians(2.0)  # how far a hinge may advance in one step, while it's being pressed
RECOIL_RATE = math.radians(40.0)
_trial = {"still": 0.0, "bad": False, "why": "", "peak": [0.0, 0.0], "sign": [0, 0], "hi": [0.0, 0.0], "done": [False, False], "on": [False, False], "phi": None}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears a violation or un-bends a band."""
    _trial.update(still=0.0, bad=False, why="", peak=[0.0, 0.0], sign=[0, 0], hi=[0.0, 0.0], done=[False, False], on=[False, False], phi=None)


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _heading(part):
    """The way a segment's own +y points, as a turn from world +y, in radians."""
    R = _rot(part.quat_xyzw)
    return math.atan2(-R[0, 1], R[1, 1])


def _local(part, p):
    return _rot(part.quat_xyzw).T @ (np.asarray(p, float) - np.asarray(part.pos, float))


def band_at(objects, k):
    """Band k's hinge, in table xy. Read from the middle segment: the hinge sits on its end."""
    m = objects["tube_mid"]
    return (_rot(m.quat_xyzw) @ np.array((0.0, (-1 if k == 0 else 1) * L_M / 2, 0.0)) + np.asarray(m.pos, float))[:2]


def shoe_corners(phi):
    """The shoe's four corners in table xy, with the handle at this sweep."""
    rho, h = M.SHOE["rho"], M.SHOE["half"]
    co, si = math.cos(phi), math.sin(phi)
    return [(M.PIVOT[0] + co * (rho + a) - si * b, M.PIVOT[1] + si * (rho + a) + co * b) for a in (-h, h) for b in (-h, h)]


def shoe_gap(objects, k, side):
    """How far the shoe is from the outer face of what swings about band k, measured from positions, or None if it isn't beside it. side: -1 if the band's own
    end is the held one, so the middle segment and the far end swing; +1 if the middle is held, so the band's own end swings. The shoe turns anticlockwise,
    so it can only push from the clockwise side, seen looking out from the hinge along what swings."""
    if side < 0:
        links, s = ("tube_mid", "tube_tail" if k == 0 else "tube_nose"), (1 if k == 0 else -1)
    else:
        links, s = ("tube_nose" if k == 0 else "tube_tail",), (-1 if k == 0 else 1)
    corners = shoe_corners(float(objects["handle"].pos[0]))
    best = None
    for n in links:
        part = objects[n]
        pts = np.array([_local(part, (x, y, part.pos[2])) for x, y in corners])
        lo, hi = BOXES[n]
        if s * pts[:, 0].mean() <= 0.0 or pts[:, 1].max() < lo[1] or pts[:, 1].min() > hi[1]:
            continue
        g = float(np.min(s * pts[:, 0])) - M.TUBE["w"] / 2
        best = g if best is None else min(best, g)
    return best


def _band_ready(objects, k):
    """A band can bend only with its hinge over the former's mark, and the part on the -y side lying in line, as the guide holds it."""
    j = band_at(objects, k)
    if abs(j[0] - M.BAND_AT[0]) > BAND_X or abs(j[1] - M.BAND_AT[1]) > BAND_Y:
        return False, 0
    end = objects["tube_nose" if k == 0 else "tube_tail"]
    far = -(L_M + L_N) / 2 if k == 0 else (L_M + L_T) / 2  # a segment's origin is the middle's: its own middle lies this far along it
    child_low = (_rot(end.quat_xyzw) @ np.array((0.0, far, 0.0)))[1] + end.pos[1] < j[1]
    held = end if child_low else objects["tube_mid"]
    yaw = _heading(held)
    return abs(math.atan2(math.sin(2 * yaw), math.cos(2 * yaw))) / 2 <= HELD_YAW, (-1 if child_low else 1)


def ratchet(objects, dt):
    """The tube's plastic bend, written as the two hinges' limits for this step: (lo, hi) per band, in radians. A band is locked where it's set unless
    it's over the former, the handle's advancing and its shoe is really pressing what swings: a hand can't bend it, nor can it swing it against a parked
    handle. Bending lifts its peak for good; once the shoe comes off, it recoils by the springback to peak minus 3 degrees. A band that's reached 48
    degrees locks for good when the shoe comes off. There's no stop for the handle to push against at a good angle: past one the tube's ruined."""
    q = objects["tube_q"].pos
    phi = float(objects["handle"].pos[0])
    advancing = _trial["phi"] is not None and (phi - _trial["phi"]) / dt > MOVING
    _trial["phi"] = phi
    out = []
    for k in (0, 1):
        ready, side = _band_ready(objects, k)
        far = np.linalg.norm(band_at(objects, k) - np.array(M.BAND_AT)) > 2.5 * BAND_X
        on = (_trial["on"][k] or (ready and side == (_trial["sign"][k] or side))) and not far and not _trial["done"][k]
        _trial["on"][k] = on
        if on and side == 0:  # kept on through a bend, past the first test's tolerance: which side it's on doesn't change
            side = _trial["sign"][k]
        gap = shoe_gap(objects, k, side) if on and side else None
        pressing = gap is not None and gap <= TOUCH
        if pressing and _trial["sign"][k] == 0:
            _trial["sign"][k] = side
        s = _trial["sign"][k] or 1
        if pressing and advancing:
            _trial["peak"][k] = max(_trial["peak"][k], s * float(q[k]), 0.0)
        set_ = _trial["peak"][k] - SPRINGBACK if _trial["peak"][k] > SPRINGBACK else 0.0
        if pressing and advancing:
            hi = max(set_, s * float(q[k]) + STEP)
        elif pressing:
            hi = max(set_, _trial["peak"][k])  # the shoe's parked on it, holding it
        else:
            hi = set_
            if _trial["peak"][k] >= BENT + SPRINGBACK - 1e-9:
                _trial["done"][k] = True
        # a limit moves back slowly, so a recoil doesn't kick; forward at once, as the shoe leads
        _trial["hi"][k] = max(hi, _trial["hi"][k] - RECOIL_RATE * dt)
        lo_, hi_ = set_, _trial["hi"][k]
        out.append((lo_, hi_) if s > 0 else (-hi_, -lo_))
    return out


# -- what the finished tube reads ------------------------------------------------------------------------------------
def bends(objects):
    """The two bends, signed."""
    q = objects["tube_q"].pos
    return float(q[0]), float(q[1])


def bend_good(a):
    return BENT - BEND_TOL <= abs(a) <= BENT + BEND_TOL


def _boxes():
    w, h, hm = M.TUBE["w"] / 2, M.TUBE["h"], L_M / 2
    return {"tube_mid": ((-w, -hm, 0.0), (w, hm, h + M.GRIP["h"])), "tube_nose": ((-w, -hm - L_N, 0.0), (w, -hm, h)), "tube_tail": ((-w, hm, 0.0), (w, hm + L_T, h))}


BOXES = _boxes()


def let_go(objects):
    """Both fingertip midpoints at least 6 cm from every segment's box, in its own frame: open fingers sit 4.7 cm out, so 6 cm means no finger's on it."""
    for name, (lo, hi) in BOXES.items():
        part = objects[name]
        for a in ("left_tcp", "right_tcp"):
            d = _local(part, objects[a].pos)
            if np.linalg.norm(d - np.clip(d, lo, hi)) < 0.06:
                return False
    return True


def offset_of(objects):
    """How far apart the two ends' centre lines lie, along the middle segment's own frame; the tape measure's reading on real pipe."""
    a, b = bends(objects)
    return L_M * math.sin((abs(a) + abs(b)) / 2)


def parallel(objects):
    a, b = bends(objects)
    return a * b > 0 and abs(a - b) <= PARALLEL


def _on_lane(part, length, c, far):
    """A segment's middle within a lane, and lying along it, either way round: the finished tube reads the same turned end for end."""
    mid = _rot(part.quat_xyzw) @ np.array((0.0, far * (L_M / 2 + length / 2), 0.0)) + np.asarray(part.pos, float)
    d = mid[:2] - np.array(c)
    yaw = _heading(part)
    off = abs(math.atan2(math.sin(2 * yaw), math.cos(2 * yaw))) / 2
    return abs(d[0]) <= ON_LATERAL and abs(d[1]) <= ON_ALONG and off <= ON_YAW


def on_template(objects):
    n, t = objects["tube_nose"], objects["tube_tail"]
    return (_on_lane(n, L_N, LANE_B, -1) and _on_lane(t, L_T, LANE_A, 1)) or (_on_lane(n, L_N, LANE_A, -1) and _on_lane(t, L_T, LANE_B, 1))


def on_bed(objects):
    """Every corner of every segment over the bed: an end can't overhang."""
    for n, (lo, hi) in BOXES.items():
        part = objects[n]
        R = _rot(part.quat_xyzw)
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                c = R @ np.array((x, y, 0.0)) + np.asarray(part.pos, float)
                if c[2] < BED_TOP - 0.01 or not (M.BED["x0"] <= c[0] <= M.BED["x1"] and M.BED["y0"] <= c[1] <= M.BED["y1"]):
                    return False
    return True


def note_step(objects, dt):
    """Called by the embodiment after every control step: a bend past saving or two same-way bends ruin the tube for good, as does leaving the bed;
    and the clock for all at rest."""
    a, b = bends(objects)
    if max(_trial["peak"]) - SPRINGBACK > RUIN or max(abs(a), abs(b)) > RUIN:
        _trial.update(bad=True, why=_trial["why"] or "overbent")
    if _trial["sign"][0] * _trial["sign"][1] != 0 and min(_trial["peak"]) >= BENT_ANY and _trial["sign"][0] != _trial["sign"][1]:
        _trial.update(bad=True, why=_trial["why"] or "same way")  # same-way bends make a U, not an offset
    if not on_bed(objects):
        _trial.update(bad=True, why=_trial["why"] or "off the bed")
    still = all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in BOXES) and let_go(objects)
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. 0 for good if a bend passes 55 degrees, if the bends go the same way, or if the tube leaves
    the bed. Otherwise, with both hands off it: 0.25 for one bend at 45 +/- 6 degrees; 0.5 for both; 0.75 with the ends parallel within 5 degrees and
    the offset 9.9 +/- 1 cm; 0.9 lying on the template; 1.0 with it at rest for the last 1 s."""
    if _trial["bad"] or not let_go(objects):
        return 0.0
    a, b = bends(objects)
    n = bend_good(a) + bend_good(b)
    if n < 2:
        return 0.25 if n else 0.0
    if not (parallel(objects) and abs(offset_of(objects) - TARGET_OFFSET) <= OFFSET_TOL):
        return 0.5
    if not on_template(objects):
        return 0.75
    return 1.0 if _trial["still"] >= STILL_S else 0.9


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.025
TURN_STEP = math.radians(8.0)


def _fingers(width):
    return -FINGER_PER_M * width


GRIP_Z = BED_TOP + M.TUBE["h"] + M.GRIP["h"] - 0.012 + 0.002  # the hand on the grip block; the arm droops a little under load
OVER_Z = GRIP_Z + 0.06
BLOCK_R, BLOCK_G = _fingers(0.08), _fingers(M.TUBE["w"] - 0.004)
KNOB_Z = BED_TOP + M.KNOB["z1"] - 0.016
KNOB_R, KNOB_G = _fingers(2 * M.KNOB["half"] + 0.035), _fingers(2 * M.KNOB["half"] - 0.005)
SWEEP = math.radians(float(os.environ.get("TUBE_SWEEP", "119.0")))  # where the sweep ends: short of the handle's stop, by eye on the tube


SWEEP_2 = math.radians(float(os.environ.get("TUBE_SWEEP_2", "115.0")))  # a little less: the second bend's shoe meets the tube further on


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
        n = 1 if label == "away" else max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / SEGMENT), math.ceil(abs(y - y0) / TURN_STEP))
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


def _grip_block(plan, x, y, yaw, label):
    arm = "left"
    plan.go(arm, 1.6, (x, y, OVER_Z), BLOCK_R, f"{label}: over the block", yaw)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, (x, y, GRIP_Z), label="descend")
    plan.go(arm, 0.6, finger=BLOCK_G, label="grip")


def _release_block(plan):
    arm = "left"
    p = plan.last[arm][0]
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 0.6, finger=BLOCK_R, label="let go")
    plan.go(arm, 0.8, (p[0], p[1], OVER_Z + 0.02), label="up")
    _park(plan, arm)


def _sweep(plan, to=SWEEP):
    """The right arm takes the sleeve knob and carries it round the former, wrist steady, and back."""
    arm = "right"
    r, (px, py) = M.HANDLE["r"], M.PIVOT
    at = lambda phi: (px + r * math.cos(phi), py + r * math.sin(phi), KNOB_Z)  # noqa: E731
    plan.go(arm, 1.6, (*at(0.0)[:2], KNOB_Z + 0.035), KNOB_R, "over the knob", 0.0)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, at(0.0), label="descend")
    plan.go(arm, 0.6, finger=KNOB_G, label="grip")
    steps = 12
    for i in range(1, steps + 1):
        plan.go(arm, 7.0 / steps, at(to * i / steps), label="sweep")
    plan.go(arm, 0.6, label="hold")
    for i in range(1, steps + 1):
        plan.go(arm, 6.0 / steps, at(to * (1 - i / steps)), label="back")
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.6, finger=KNOB_R, label="let go")
    plan.go(arm, 0.8, (*at(0.0)[:2], KNOB_Z + 0.035), label="up")
    _park(plan, arm)


# Where a set bend leaves the grip block and the way its segment lies, seen on the bed: the second differs, as the kink changes how the tube settles.
BENT_FIRST = (0.2329, -0.0446, math.radians(50.4))
BENT_SECOND = (0.2386, -0.0478, math.radians(47.4))


def _turn_test(plan):
    """The first test: lift the locked tube by its block, hold 10 s, set it down, then drag it. Debugging only."""
    x, y, yaw = START
    _grip_block(plan, x, y, 0.0, "test")
    plan.go("left", 1.5, (x, y, GRIP_Z + 0.06), label="lift")
    plan.go("left", 10.0, label="hold")
    plan.go("left", 1.5, (x, y, GRIP_Z), label="lower")
    plan.go("left", 4.0, (x, y - 0.1, GRIP_Z), label="drag")
    plan.go("left", 4.0, (x, y - 0.1, GRIP_Z), label="turn", yaw=-math.pi / 2)
    _release_block(plan)


def _parked_shortcut(plan):
    """Debugging only: the handle swept round empty and left parked, then the left arm alone tries to swing the tube round at the mark."""
    arm = "right"
    r, (px, py) = M.HANDLE["r"], M.PIVOT
    at = lambda phi: (px + r * math.cos(phi), py + r * math.sin(phi), KNOB_Z)  # noqa: E731
    plan.go(arm, 1.6, (*at(0.0)[:2], KNOB_Z + 0.035), KNOB_R, "over the knob", 0.0)
    plan.go(arm, 0.8, at(0.0), label="descend")
    plan.go(arm, 0.6, finger=KNOB_G, label="grip")
    for i in range(1, 13):
        plan.go(arm, 7.0 / 12, at(SWEEP * i / 12), label="sweep empty")
    plan.go(arm, 0.6, finger=KNOB_R, label="let go")
    plan.go(arm, 0.8, (*at(SWEEP)[:2], KNOB_Z + 0.035), label="up")
    _park(plan, arm)
    a = "left"
    plan.until(a, plan.t[arm])
    bx, by = M.BAND_AT
    _grip_block(plan, START[0], START[1], START[2] - math.pi / 2, "first")
    plan.go(a, 4.0, (bx, 0.27, GRIP_Z), label="turn it", yaw=-math.pi / 2)
    plan.go(a, 5.0, (bx, by + L_M / 2, GRIP_Z), label="feed")
    for i in range(1, 11):  # swing the block round the hinge to 50 degrees, by hand
        th = math.radians(5.0 * i)
        plan.go(a, 0.6, (bx - L_M / 2 * math.sin(th), by + L_M / 2 * math.cos(th), GRIP_Z), label="swing by hand", yaw=th - math.pi / 2)
    plan.go(a, 1.0, label="hold")
    _release_block(plan)


def demo():
    plan = _Plan()
    for arm in SIDE:
        _park(plan, arm)
    if os.environ.get("TUBE_DEMO_PARKED") == "1":
        _parked_shortcut(plan)
        return _finish(plan)
    if os.environ.get("TUBE_TEST") == "lift":
        _turn_test(plan)
        return _finish(plan)
    a = "left"
    bx, by = M.BAND_AT
    # feed: nose through the gap and into the guide, the first band at the mark
    _grip_block(plan, START[0], START[1], START[2] - math.pi / 2, "first")
    plan.go(a, 4.0, (bx, 0.27, GRIP_Z), label="turn it", yaw=-math.pi / 2)
    plan.go(a, 0.3, label="settle")
    feed = float(os.environ.get("TUBE_FEED", "0.0"))  # debugging only: stop short of the mark
    plan.go(a, 5.0, (bx, by + L_M / 2 + feed, GRIP_Z), label="feed")
    _release_block(plan)
    plan.until("right", plan.t[a])
    over = os.environ.get("TUBE_DEMO_PAST_STOP") == "1"  # debugging only: push the handle into its stop
    _sweep(plan, math.radians(M.HANDLE["stop"]) if over else SWEEP)
    plan.until(a, plan.t["right"])
    if os.environ.get("TUBE_DEMO_ONE") == "1":  # debugging only: one bend, then walk away
        return _finish(plan)
    # pull the kinked tube clear of the guide, turn it round, feed the tail in
    x, y, th = BENT_FIRST
    _grip_block(plan, x, y, th - math.pi / 2, "second")
    plan.go(a, 3.0, (x, y + 0.13, GRIP_Z), label="pull clear")
    plan.go(a, 2.0, (0.25, 0.25, GRIP_Z), label="to the turn")
    # the wrist runs out of turn: a first 105 degrees, a fresh grip the other way round, and the last 30
    plan.go(a, 5.0, (0.25, 0.25, GRIP_Z), label="turn it", yaw=math.pi / 3)
    plan.go(a, 0.3, label="settle")
    plan.go(a, 0.6, finger=BLOCK_R, label="let go")
    plan.go(a, 0.7, (0.25, 0.25, OVER_Z), label="up")
    plan.go(a, 2.5, (0.25, 0.25, OVER_Z), label="wrist round", yaw=-2 * math.pi / 3)
    plan.go(a, 0.7, (0.25, 0.25, GRIP_Z), label="descend")
    plan.go(a, 0.6, finger=BLOCK_G, label="grip")
    plan.go(a, 3.0, (bx, 0.25, GRIP_Z), label="turn it onto the line", yaw=-math.pi / 2)
    plan.go(a, 0.3, label="settle")
    plan.go(a, 5.0, (bx, by + L_M / 2, GRIP_Z), label="feed")
    _release_block(plan)
    plan.until("right", plan.t[a])
    _sweep(plan, SWEEP_2)
    plan.until(a, plan.t["right"])
    # draw the finished tube straight out and across to the template
    x, y, th = BENT_SECOND
    _grip_block(plan, x, y, th - math.pi / 2, "out")
    plan.go(a, 3.0, (x, y + 0.14, GRIP_Z), label="draw out")
    plan.go(a, 3.0, (GRIP_AT[0], GRIP_AT[1], GRIP_Z), label="onto the template", yaw=BENT - math.pi / 2)
    _release_block(plan)
    return _finish(plan)


def _finish(plan):
    plan.go("left", 2.5, label="still")
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
            if pe > 1e-3 or re > math.radians(1.0):
                bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(math.degrees(w.yaw)), round(pe * 1000, 1), round(math.degrees(re), 1)))
    return worst_p, worst_r, bad


# -- self-check (no simulator): the demo fits the run limit and IK reaches every waypoint -------------------------------
if __name__ == "__main__":
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad = ik_errors(wps)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:12], len(bad))
    print("offset", round(TARGET_OFFSET, 4), "lanes", LANES, "sweeps", round(math.degrees(SWEEP), 1), round(math.degrees(SWEEP_2), 1))
    assert t * 20 < 3000 and not bad
