"""Manual TIG (GTAW) welding of an L-shaped butt seam on two clamped steel plates. One arm carries the torch along the joint
at a steady arc length; the other dabs the filler rod into the front of the pool and withdraws it, clear of the tungsten.
Then both tools go back on their rests.

Run it:  inspect-robots run --embodiment yam_tig -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Weld the joint between the white and yellow marks with the TIG torch, dabbing the yellow filler rod into the front of the pool to lay one sound bead, then put both tools back on their rests and move both hands away." --scorer task_score

`yam_tig` (tig_seam/embodiment.py) spawns the bench, torch and rod, reports each gripper's fingertip midpoint, and shows
the bead pieces that note_step switches on.

The weld is a geometric event, scored per 5 mm segment of the 200 mm joint (heat, pool flow and distortion aren't simulated):
- Arc: the torch is in a hand, and its tungsten tip is 2-5 mm above the plates and within 4 mm of the joint line.
- Heat: while the arc is on, each segment whose centre is within 8 mm of the tip gains its arc time. It's molten (the
  pool) while under the arc with 0.5 s or more; fused at 3 s; burnt through at 9 s.
- Filler: a dab is the rod tip, in the other hand, coming down to the plates (within 3.5 mm) after being lifted 6 mm or
  more. If it lands 3-10 mm ahead of the tungsten along the torch's lean and within 4 mm of the line, each molten segment
  within 6 mm of the rod tip gets one dab of filler.
- Contamination, latched: the tungsten tip within 0.5 mm of the plates, or the rod within 0.5 mm of the tungsten.
  From then on there's no arc, and the score is capped at 0.5.
"""

import math
import os

import numpy as np

from tig_seam import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
PLATE_Z = TOP + M.PT  # the plates' top, world
SEGS = M.seam_segments()
N = len(SEGS)
SEG_XY = np.array([p for p, _, _ in SEGS])

OBJECTS = [
    Box("bench", pos=(0.0, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    Box("torch", pos=(M.TORCH_REST["x"], M.TORCH_REST["y"], TOP + M.TORCH_REST_Z), size=(0.01, 0.01, 0.01), mass=M.TORCH_TOTAL,
        yaw=M.TORCH_REST["lean"]),
    Box("rod", pos=(M.ROD_REST["x"], M.ROD_REST["y"], TOP + M.ROD_REST_Z), size=(0.01, 0.01, 0.01), mass=M.ROD_TOTAL,
        yaw=M.ROD_REST["heading"]),
]
TOOLS = ("torch", "rod")

# -- the weld's rules: typical manual-TIG values for 3/32 in (2.4 mm) tungsten on mild steel ---------------------------
ARC = (0.002, 0.005)  # arc length window: tip height above the plates
TOUCH = 0.0005  # the tip this close to the plates is a dip
LAT = 0.004  # tip and rod tip within this of the joint line
R_HEAT = 0.008  # the arc heats segment centres within this of the tip (plan)
H_MELT, H_FUSE, H_BURN = 0.5, 3.0, 9.0  # s of arc: pool, fused, burnt through (at 16 mm of footprint: 1.8-5.3 mm/s of travel)
ZONE = (0.003, 0.010)  # the rod tip this far ahead of the tungsten, along the lean: the front of the pool
ROD_DOWN, ROD_UP = 0.0035, 0.006  # rod tip centre height above the plates: down in a dab; lifted out between dabs
DAB_R = 0.006  # a dab feeds molten segments within this of the rod tip
ROD_KEEP = (M.TUNGSTEN_D + M.ROD_D) / 2 + 0.0005  # rod and tungsten axes closer than this: they touched
HELD = 0.025  # a gripper's fingertip midpoint within this of a tool's grip centre holds it
CAP = 0.5
HANDS_CLEAR, LIN, ANG, STILL_S = 0.06, 0.01, 0.5, 1.0
REST_XY, REST_Z, REST_TILT, REST_TURN = 0.012, 0.006, math.radians(10.0), math.radians(20.0)


def _fresh():
    return {"heat": np.zeros(N), "filler": np.zeros(N, int), "burnt": np.zeros(N, bool), "molten": np.zeros(N, bool), "near": np.zeros(N, bool),
            "dirty": False, "bad": False, "armed": True, "arc": False, "still": 0.0, "dabs": 0, "dabs_cold": 0}


_trial = _fresh()


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears a latch."""
    _trial.clear()
    _trial.update(_fresh())


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _world(part, p):
    return np.asarray(part.pos, float) + _rot(part.quat_xyzw) @ np.asarray(p, float)


def tungsten(objects):
    """The tungsten's tip and the top of its exposed length, world."""
    t = objects["torch"]
    return _world(t, M.TIP), _world(t, M.TIP + (M.CUP["a0"] + 0.004) * M.UP)


def lean(objects):
    """The torch's lean (push direction) in plan, a unit vector."""
    v = _rot(objects["torch"].quat_xyzw)[:2, 0]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array((1.0, 0.0))


def rod_line(objects):
    """The rod's exposed centreline: from the sleeve's underside to the tip, world."""
    r = objects["rod"]
    return _world(r, M.ROD_EXIT), _world(r, M.ROD_TIP)


def _seg_dist(p0, p1, q0, q1):
    """Closest distance between segments p0-p1 and q0-q1 (sampled finely: both are short and straight)."""
    a = np.linspace(0.0, 1.0, 41)[:, None]
    P = p0 + a * (p1 - p0)
    Q = q0 + a * (q1 - q0)
    return float(np.min(np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)))


def holder(objects, name):
    """Which hand ('left_tcp' or 'right_tcp') holds a tool: its fingertip midpoint within HELD of the grip's centre."""
    p = np.asarray(objects[name].pos, float)
    best = min(("left_tcp", "right_tcp"), key=lambda a: np.linalg.norm(np.asarray(objects[a].pos, float) - p))
    return best if np.linalg.norm(np.asarray(objects[best].pos, float) - p) <= HELD else None


def _box_dist(objects, part, lo, hi, hand):
    R = _rot(objects[part].quat_xyzw)
    d = R.T @ (np.asarray(objects[hand].pos, float) - np.asarray(objects[part].pos, float))
    return float(np.linalg.norm(d - np.clip(d, lo, hi)))


def _point_seg(p, a, b):
    d = b - a
    t = min(max(float(np.dot(p - a, d) / np.dot(d, d)), 0.0), 1.0)
    return float(np.linalg.norm(p - (a + t * d)))


def let_go(objects, hand):
    """A hand this far (HANDS_CLEAR) from every part of both tools: open fingers sit 4.7 cm out, so 6 cm means no finger's on them."""
    lo, hi = M.TORCH_HULL
    if _box_dist(objects, "torch", lo, hi, hand) < HANDS_CLEAR or _box_dist(objects, "rod", *M.SLEEVE_BOX, hand) < HANDS_CLEAR:
        return False
    return _point_seg(np.asarray(objects[hand].pos, float), *rod_line(objects)) >= HANDS_CLEAR


def hands_clear(objects):
    return let_go(objects, "left_tcp") and let_go(objects, "right_tcp")


def _turn(R):
    return math.atan2(R[1, 0], R[0, 0])


def _rested(part, x, y, z, heading):
    R = _rot(part.quat_xyzw)
    p = np.asarray(part.pos, float)
    dturn = abs(math.atan2(math.sin(_turn(R) - heading), math.cos(_turn(R) - heading)))
    return bool(math.hypot(p[0] - x, p[1] - y) <= REST_XY and abs(p[2] - z) <= REST_Z and R[2, 2] >= math.cos(REST_TILT) and dturn <= REST_TURN)


def tools_rested(objects):
    return _rested(objects["torch"], M.TORCH_REST["x"], M.TORCH_REST["y"], TOP + M.TORCH_REST_Z, M.TORCH_REST["lean"]) and _rested(
        objects["rod"], M.ROD_REST["x"], M.ROD_REST["y"], TOP + M.ROD_REST_Z, M.ROD_REST["heading"])


def weld_state(objects):
    """This step's geometry: tip height, distance from the line, arc on, rod tip down / ahead / off the line, and who holds what."""
    tip, top = tungsten(objects)
    u = lean(objects)
    r0, rtip = rod_line(objects)
    th, rh = holder(objects, "torch"), holder(objects, "rod")
    return dict(tip=tip, top=top, h=float(tip[2] - PLATE_Z), lat=M.dist_to_seam(tip), u=u, rtip=rtip, r0=r0, rh=float(rtip[2] - PLATE_Z),
                ahead=float(np.dot(rtip[:2] - tip[:2], u)), rlat=M.dist_to_seam(rtip), gap=_seg_dist(tip, top, r0, rtip), torch_hand=th, rod_hand=rh)


def note_step(objects, dt):
    """Called by the embodiment after every control step: heat, pool, dabs, faults, contamination, and the clock for all at rest."""
    tr = _trial
    if any(objects[n].pos[2] < TOP - 0.01 for n in TOOLS):
        tr["bad"] = True
    w = weld_state(objects)
    over = M.SLAB["x"][0] <= w["tip"][0] <= M.SLAB["x"][1] and M.SLAB["y"][0] <= w["tip"][1] <= M.SLAB["y"][1]
    if (over and w["h"] <= TOUCH) or w["gap"] <= ROD_KEEP:
        tr["dirty"] = True
    arc = bool(not tr["dirty"] and w["torch_hand"] is not None and ARC[0] <= w["h"] <= ARC[1] and w["lat"] <= LAT)
    near = np.linalg.norm(SEG_XY - w["tip"][:2], axis=1) <= R_HEAT
    if arc:
        tr["heat"][near] += dt
        tr["burnt"] |= tr["heat"] >= H_BURN
    tr["arc"] = arc
    tr["near"] = near & arc
    tr["molten"] = near & arc & (tr["heat"] >= H_MELT) & ~tr["burnt"]
    if w["rh"] <= ROD_DOWN:
        if tr["armed"]:
            two_hands = w["rod_hand"] is not None and w["torch_hand"] is not None and w["rod_hand"] != w["torch_hand"]
            in_zone = ZONE[0] <= w["ahead"] <= ZONE[1] and w["rlat"] <= LAT
            if two_hands and in_zone and arc:
                feed = tr["molten"] & (np.linalg.norm(SEG_XY - w["rtip"][:2], axis=1) <= DAB_R)
                tr["filler"][feed] += 1
                tr["dabs"] += 1
                tr["dabs_cold"] += int(not feed.any())
        tr["armed"] = False
    elif w["rh"] >= ROD_UP:
        tr["armed"] = True
    still = all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in TOOLS) and hands_clear(objects)
    tr["still"] = tr["still"] + dt if still else 0.0


def good_segments():
    tr = _trial
    return (tr["filler"] >= 1) & (tr["heat"] >= H_FUSE) & (tr["heat"] < H_BURN) & ~tr["burnt"]


def faults():
    """Per segment, for the record: 'good', 'undercut' (fused, no filler), 'cold' (filler, not fused), 'burnt', or '' (untouched)."""
    tr = _trial
    out = []
    for i in range(N):
        if tr["burnt"][i]:
            out.append("burnt")
        elif tr["filler"][i] and tr["heat"][i] >= H_FUSE:
            out.append("good")
        elif tr["filler"][i]:
            out.append("cold")
        elif tr["heat"][i] >= H_FUSE:
            out.append("undercut")
        else:
            out.append("")
    return out


def shown():
    """The bead pieces and torch effects to show now (prim names)."""
    tr = _trial
    on = set()
    for i in range(N):
        if tr["burnt"][i]:
            on.add(f"hole_s{i}")
            continue
        if tr["filler"][i]:
            if tr["heat"][i] >= H_FUSE or tr["near"][i]:
                on.add(f"bead_hi_{i}" if tr["filler"][i] >= 2 else f"bead_lo_{i}")
            else:
                on.add(f"cold_{i}")
        elif tr["heat"][i] >= H_FUSE and not tr["near"][i]:
            on.add(f"cut_{i}")
        if tr["molten"][i]:
            on.add(f"pool_{i}")
    if tr["arc"]:
        on.add("fx_arc")
    if tr["dirty"]:
        on.add("fx_soot")
    return on


def score(objects):
    """Read at the end of the run; there's no early end. 0 for good if a tool falls off the table. Otherwise 0.9 x the fraction
    of the 40 segments with a sound bead (filler, fused, not burnt through), capped at 0.5 after contamination or any burn-through;
    1.0 only with all 40 sound, nothing capped, both tools back on their rests, both hands clear and all at rest for the last 1 s."""
    tr = _trial
    if tr["bad"]:
        return 0.0
    good = good_segments()
    s = 0.9 * float(good.sum()) / N
    capped = tr["dirty"] or bool(tr["burnt"].any())
    if good.all() and not capped and tools_rested(objects) and hands_clear(objects) and tr["still"] >= STILL_S:
        return 1.0
    return round(min(s, CAP) if capped else s, 3)


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Open loop from the known start. The tool geometry and the joint's line are visible; the plan reads no simulator state.
# TIG_DEMO: debugging variants only. "laid": the rod is laid in the joint and let go, then the torch runs alone.
# "torch": the torch runs alone, the rod left on its rest. "fast": travel at twice the speed. "dip": the tungsten dips into the plate mid-seam.
VARIANT = os.environ.get("TIG_DEMO", "")
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.03
DT = 0.05
SPEED = 0.003 * (2.0 if VARIANT == "fast" else 1.0)  # travel, m/s: typical 100-250 mm/min for this work
DAB_T = 1.25  # one dab per 1.25 s: about 3.75 mm apart at 3 mm/s, so each segment is fed
ARC_AIM = 0.0035 + float(os.environ.get("TIG_LIFT", "0"))  # TIG_LIFT: debugging only, to measure the arm's give in free air
AHEAD = 0.0055  # where the rod tip settles, ahead of the tungsten; coming in at 45 degrees it first reaches the scorer's 3.5 mm about 6.5 mm ahead, the middle of its 3-10 mm
# The rod hand's give, measured from logged runs (r4f-r7): on the first leg the tip came down 2 mm to the left and 3 mm
# low, and settled where aimed along the travel; on the second, 4 mm to the right, 1.7 mm low and 4-5 mm further ahead.
# Each dab is aimed the other way by as much, so the tip itself comes down on the line, about 2.5 mm up (0.9 mm clear).
AHEAD_A, AHEAD_B = AHEAD, AHEAD - 0.005  # aimed ahead on each leg
ROD_SIDE_A, ROD_SIDE_B = -0.002, 0.004  # aimed to the left of the travel, on each leg
ROD_LOW_B = 0.0023  # aimed lower on the second leg
ROD_AIM = 0.0055 + float(os.environ.get("TIG_ROD_LIFT", "0"))  # rod tip centre aimed this high above the plates' top on the first leg (TIG_ROD_LIFT: debugging only)
ROD_OUT = 0.008, 0.008  # between dabs the rod tip rises this much and moves this much further ahead: out at 45 degrees, clear of the cup's rim
ROD_WAIT = 0.008, 0.010, 1.0  # before the first dab the rod waits this much higher and further ahead
# still, clear of the tungsten while the torch comes down to strike; it closes in over the last 1 s


def _wait(clock, u):
    """The rod's extra offset while it waits: full until 1 s before the dab clock reaches 0, then easing to none."""
    k = min(max(-clock / ROD_WAIT[2], 0.0), 1.0)
    k = k * k * (3 - 2 * k)
    return k * (np.array((0.0, 0.0, ROD_WAIT[0])) + ROD_WAIT[1] * np.asarray(u, float))
HEAD_A, HEAD_B = math.radians(60.0), math.radians(120.0)  # the rod's heading on each leg
GRIP_DZ, SLEEVE_DZ = 0.003, 0.0025  # TCP above each grip's centre when holding
# Calibration from a logged free-air run (the tip 10 mm up): along the seam (s, mm), how far the tungsten tip sat below its
# target and to the side of it (+y on the first leg, +x on the second), both mm. The plan aims the other way by as much.
CAL = ((0, 6.3, 2.7), (35, 6.3, 2.6), (45, 6.1, 2.8), (55, 5.8, 2.7), (65, 5.7, 2.8), (75, 5.6, 2.6), (85, 5.4, 2.7), (95, 5.2, 2.7),
       (105, 4.9, 2.6), (115, 4.9, 2.7), (120, 4.8, 0.0), (125, 4.7, -2.4), (135, 4.75, -2.2), (145, 5.0, -2.2), (155, 5.25, -2.15),
       (165, 5.3, -2.0), (175, 5.4, -1.7), (185, 5.6, -1.7), (195, 5.8, -1.4), (200, 5.9, -1.4))


def _fingers(width):
    return -FINGER_PER_M * width


TORCH_OPEN, TORCH_GRIP = _fingers(M.GRIP["x"] + 0.03), _fingers(M.GRIP["x"] - 0.004)
ROD_OPEN, ROD_GRIP = _fingers(M.SLEEVE + 0.03), _fingers(M.SLEEVE - 0.004)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def torch_tcp(tip, lean_a):
    """The torch hand's TCP and yaw for the tungsten tip at ``tip`` (world) with the lean at heading ``lean_a``."""
    g = np.asarray(tip, float) - _rz(lean_a) @ M.TIP
    return g + (0.0, 0.0, GRIP_DZ), lean_a - 1.5 * math.pi


def rod_tcp(tip, heading):
    s = np.asarray(tip, float) - _rz(heading) @ M.ROD_TIP
    return s + (0.0, 0.0, SLEEVE_DZ), heading - 0.5 * math.pi  # fingers close along the heading: no pitch hinge


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
        n = 1 if label == "down" else max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / SEGMENT), math.ceil(abs(y - y0) / math.radians(15)))
        for k in range(1, n + 1):
            q = tuple(float(v) for v in np.asarray(p0) + (np.asarray(p) - np.asarray(p0)) * k / n)
            self.wps[arm].append(Waypoint(dur / n, q, f0 + (f - f0) * k / n, label, y0 + (y - y0) * k / n))
        self.last[arm] = (p, f, y)
        self.t[arm] += dur

    def step(self, arm, pos, yaw, label):
        """One control step's target: the weld is planned step by step so the hands move steadily."""
        p0, f0, _ = self.last[arm]
        self.wps[arm].append(Waypoint(DT, tuple(float(v) for v in pos), f0, label, float(yaw)))
        self.last[arm] = (tuple(float(v) for v in pos), f0, float(yaw))
        self.t[arm] += DT

    def until(self, arm, t):
        if t > self.t[arm] + 1e-9:
            self.go(arm, t - self.t[arm], label="wait")


def _park(plan, arm, dur=1.4):
    plan.go(arm, dur, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), None, "away")


def _cal(s):
    """(sag, side) in m at s along the seam."""
    xs, zs, ys = zip(*CAL)
    return float(np.interp(1000 * s, xs, zs)) / 1000, float(np.interp(1000 * s, xs, ys)) / 1000


def _pick_torch(plan):
    a, lean0 = "left", M.TORCH_REST["lean"]
    tip0 = np.array((M.TORCH_REST["x"], M.TORCH_REST["y"], TOP + M.TORCH_REST_Z)) + _rz(lean0) @ M.TIP
    p, yaw = torch_tcp(tip0, lean0)
    plan.go(a, 1.6, p + (0, 0, 0.04), TORCH_OPEN, "torch: above it", yaw)
    plan.go(a, 0.3, label="settle")
    plan.go(a, 0.8, p, label="descend")
    plan.go(a, 0.6, finger=TORCH_GRIP, label="grip")
    plan.go(a, 0.8, p + (0, 0, 0.035), label="lift")


def _pick_rod(plan):
    a, h0 = "right", M.ROD_REST["heading"]
    tip0 = np.array((M.ROD_REST["x"], M.ROD_REST["y"], TOP + M.ROD_REST_Z)) + _rz(h0) @ M.ROD_TIP
    p, yaw = rod_tcp(tip0, h0)
    plan.go(a, 1.6, p + (0, 0, 0.04), ROD_OPEN, "rod: above it", yaw)
    plan.go(a, 0.3, label="settle")
    plan.go(a, 0.8, p, label="descend")
    plan.go(a, 0.6, finger=ROD_GRIP, label="grip")
    plan.go(a, 0.8, p + (0, 0, 0.035), label="lift")


START = -0.002  # the arc is struck and the pool formed 2 mm behind the start mark (the arm leaves the tip ~2 mm forward)


def _weld_track():
    """The torch's path, step by step: (s along the seam, lean heading, tip height above the plates, phase)."""
    out = []
    lean_a, lean_b = math.pi, 1.5 * math.pi
    la = math.hypot(M.SEAM[1][0] - M.SEAM[0][0], M.SEAM[1][1] - M.SEAM[0][1])
    s = START
    n = round(1.5 / DT)
    for k in range(1, n + 1):  # come down to arc length at the start mark
        out.append((s, lean_a, 0.015 + (ARC_AIM - 0.015) * k / n, "strike"))
    out += [(s, lean_a, ARC_AIM, "pool")] * round(1.2 / DT)
    dip_at = 0.06 if VARIANT == "dip" else None
    while s < la - 1e-9:
        s = min(la, s + SPEED * DT)
        z = ARC_AIM
        if dip_at is not None and abs(s - dip_at) < 0.002:
            z = -0.004
        out.append((s, lean_a, z, "leg 1"))
    n = round(1.5 / DT)
    for k in range(1, n + 1):  # turn the torch about its tip at the corner
        out.append((la, lean_a + (lean_b - lean_a) * k / n, ARC_AIM, "corner"))
    while s < M.SEAM_LEN - 1e-9:
        s = min(M.SEAM_LEN, s + SPEED * DT)
        out.append((s, lean_b, ARC_AIM, "leg 2"))
    out += [(M.SEAM_LEN, lean_b, ARC_AIM, "crater")] * round(1.0 / DT)
    n = round(0.8 / DT)
    for k in range(1, n + 1):
        out.append((M.SEAM_LEN, lean_b, ARC_AIM + 0.025 * k / n, "arc off"))
    return out


def _tip_at(s, lean_a, z, cal=True):
    """The tungsten tip's target; with cal, offset for the arm's measured give, so the tip itself lands on the line."""
    p, _ = M.seam_point(s)
    if not cal:
        return np.array((p[0], p[1], PLATE_Z + z))
    sag, side = _cal(s)
    n = np.array((0.0, 1.0)) if s < 0.12 else np.array((1.0, 0.0))
    p = p - side * n
    return np.array((p[0], p[1], PLATE_Z + z + sag))


def _u(lean_a):
    return np.array((math.cos(lean_a), math.sin(lean_a), 0.0))


def _rod_target(tip, lean_a, heading, phase):
    """The rod tip's target for a dab phase in [0, 1): down (in the pool's front) for the middle of the cycle, else out."""
    u = np.array((math.cos(lean_a), math.sin(lean_a), 0.0))
    leg2 = heading > 0.5 * (HEAD_A + HEAD_B)
    side = np.array((-u[1], u[0], 0.0))  # to the left of the travel
    down = np.array((tip[0], tip[1], PLATE_Z + ROD_AIM - (ROD_LOW_B if leg2 else 0.0))) + (AHEAD_B if leg2 else AHEAD_A) * u
    down = down + (ROD_SIDE_B if leg2 else ROD_SIDE_A) * side
    out = down + (0.0, 0.0, ROD_OUT[0]) + ROD_OUT[1] * u  # up and forward out of the pool, away from the tungsten, and back down
    if phase < 0.25:
        k = phase / 0.25
    elif phase < 0.5:
        k = 1.0
    elif phase < 0.75:
        k = 1.0 - (phase - 0.5) / 0.25
    else:
        k = 0.0
    k = k * k * (3 - 2 * k)
    return out + k * (down - out)


def demo():
    plan = _Plan()
    for arm in SIDE:  # down from the ready pose first, so the wrist turns low
        p0 = plan.last[arm][0]
        plan.go(arm, 1.0, (p0[0], p0[1], TOP + 0.15), label="down")
    _pick_torch(plan)
    if VARIANT != "torch":  # "torch": the rod stays on its rest
        _pick_rod(plan)
    track = _weld_track()
    s0, la0, z0, _ = track[0]
    tip0 = _tip_at(s0, la0, z0)
    p, yaw = torch_tcp(tip0, la0)
    plan.go("left", 2.4, p + (-0.03, 0, 0.0), label="to the start", yaw=yaw)
    plan.go("left", 0.8, p, label="over the start")
    rod = VARIANT not in ("torch", "laid")
    if VARIANT == "laid":
        _lay_rod(plan)
    elif rod:
        r0 = _rod_target(_tip_at(START, la0, ARC_AIM, cal=False), la0, HEAD_A, 0.9) + _wait(-9.0, _u(la0))
        rp, ryaw = rod_tcp(r0 + (-0.02, 0, 0.03), HEAD_A)  # 2 cm ahead of the start, so the rod comes in clear of the tungsten
        plan.go("right", 2.4, rp, label="rod: to the start", yaw=ryaw)
        plan.go("right", 0.8, rod_tcp(r0, HEAD_A)[0], label="rod: ready")
    else:
        _park(plan, "right")
    t0 = max(plan.t.values())
    for a in SIDE:
        plan.until(a, t0)
    clock = -1.9  # dab clock (s): the first dab comes once the pool has formed, before the torch moves off the start
    heading = HEAD_A
    first = True  # until the first dab
    for s, lean_a, z, phase in track:
        tip = _tip_at(s, lean_a, z)
        p, yaw = torch_tcp(tip, lean_a)
        plan.step("left", p, yaw, phase)
        if not rod:
            plan.step("right", plan.last["right"][0], plan.last["right"][2], "idle")
            continue
        if phase == "corner":
            k = (lean_a - math.pi) / (0.5 * math.pi)
            heading = HEAD_A + (HEAD_B - HEAD_A) * k * k * (3 - 2 * k)
            ph = 0.9
            clock = -0.3
        elif phase in ("arc off",):
            ph = 0.9
        else:
            clock += DT
            ph = (clock % DAB_T) / DAB_T if clock >= 0 else 0.9
            first = first and clock < 0
        rt = _rod_target(_tip_at(s, lean_a, ARC_AIM, cal=False), lean_a, heading, ph) + _wait(clock if first else 0.0, _u(lean_a))
        if phase == "arc off":
            rt = rt + (0, 0, 0.02 * (z - ARC_AIM) / 0.025)
        rp, ryaw = rod_tcp(rt, heading)
        plan.step("right", rp, ryaw, f"rod {phase}")
    _return_torch(plan)
    if rod:
        _return_rod(plan)
    end = max(plan.t.values()) + 2.0
    for a in SIDE:
        plan.until(a, end)
    return plan.wps


def _lay_rod(plan):
    """Debugging variant: the rod laid along the first leg's joint and let go (cold wire), then the torch runs over it."""
    a = "right"
    tip = _tip_at(0.06, math.pi, 0.0)
    tip[2] = PLATE_Z + 0.0025
    rp, ryaw = rod_tcp(tip + (0, 0, 0.03), HEAD_A)
    plan.go(a, 2.4, rp, label="rod: over the joint", yaw=ryaw)
    plan.go(a, 1.0, rod_tcp(tip, HEAD_A)[0], label="rod: lay it")
    plan.go(a, 0.6, finger=ROD_OPEN, label="rod: let go")
    plan.go(a, 0.8, np.array(plan.last[a][0]) + (0, 0, 0.05), label="up")
    _park(plan, a)


def _return_torch(plan):
    a, lean0 = "left", M.TORCH_REST["lean"]
    g = np.array((M.TORCH_REST["x"], M.TORCH_REST["y"], TOP + M.TORCH_REST_Z))
    p, yaw = torch_tcp(g + _rz(lean0) @ M.TIP, lean0)
    plan.go(a, 3.0, p + (0, 0, 0.04), label="torch: over its rest", yaw=yaw)
    plan.go(a, 1.0, p + (0, 0, 0.001), label="torch: down")
    plan.go(a, 0.6, finger=TORCH_OPEN, label="torch: let go")
    plan.go(a, 0.8, p + (0, 0, 0.05), label="up")
    _park(plan, a)


def _return_rod(plan):
    a, h0 = "right", M.ROD_REST["heading"]
    s = np.array((M.ROD_REST["x"], M.ROD_REST["y"], TOP + M.ROD_REST_Z))
    p, yaw = rod_tcp(s + _rz(h0) @ M.ROD_TIP, h0)
    plan.go(a, 1.0, np.array(plan.last[a][0]) + (0, 0, 0.03), label="rod: up")
    plan.go(a, 3.0, p + (0, 0, 0.04), label="rod: over its rest", yaw=yaw)
    plan.go(a, 1.0, p + (0, 0, 0.001), label="rod: down")
    plan.go(a, 0.6, finger=ROD_OPEN, label="rod: let go")
    plan.go(a, 0.8, p + (0, 0, 0.05), label="up")
    _park(plan, a)


def ik_errors(wps):
    worst_p = worst_r = 0.0
    bad = []
    for arm, ws in wps.items():
        q = np.array(READY, float)
        for w in ws:
            q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
            worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
            if pe > 1e-3 or re > math.radians(1.0):
                bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1), round(math.degrees(re), 2)))
    return worst_p, worst_r, bad


# -- self-check (no simulator): the demo fits the run limit, IK reaches every waypoint, the torch's grip is balanced ------
if __name__ == "__main__":
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad = ik_errors(wps)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg, {len(bad)} bad", bad[:6])
    print("torch COM from under the grip (mm):", np.round(1000 * M.TORCH_COM[:2], 2), "TCP heights above table (cm):",
          round(100 * (M.TORCH_REST_Z + GRIP_DZ), 1), round(100 * (M.PT + ARC_AIM - M.TIP[2] + GRIP_DZ), 1), round(100 * (M.PT - M.ROD_TIP[2] + SLEEVE_DZ), 1))
    assert t * 20 < 3000 and not bad and np.linalg.norm(M.TORCH_COM[:2]) < 0.001
