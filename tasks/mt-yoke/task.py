"""Magnetic-particle (MT) examination of a butt weld with an AC yoke and dry powder, continuous method, in two directions
about 90 degrees apart; then each crack found is marked with a peg and both tools go back on their rests.

Run it:  inspect-robots run --embodiment yam_mt -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Examine the weld between the white marks for cracks with the magnetic yoke and the powder bulb, mark each crack you find with a blue peg, then put both tools back on their rests and move both hands away." --scorer task_score

`yam_mt` (mt_yoke/embodiment.py) spawns the bench, yoke, bulb and pegs, reports each gripper's fingertip midpoint and pads,
the bulb's squeeze, and shows the lamp, puff, powder and indications that note_step switches on.

The magnetic field and the indications are geometric events (no magnetics is simulated):
- Field on: a hand holds the yoke's handle closed (the trigger), and both pole faces are on the plate (within 2.5 mm of
  its top, inside its edges, the yoke tilted under 5 degrees). The yoke's lamp lights.
- Powder flows while the bulb's plunger is squeezed 4 mm or more with its nozzle 10-60 mm above the plate; it lands on
  the weld cells within 35 mm of the nozzle (plan) and shows there. It counts only if, at that step, the field is on and
  the bulb is in the other hand.
- A cell is examined when live powder lands on it inside the yoke's zone: between the poles (within 40 mm of their
  midpoint along the pole line) and within 25 mm of the pole line. It counts for direction A when the pole line is
  within 20 degrees of square to the weld, B within 20 degrees of along it.
- A crack shows (a red line) when live powder lands on its midpoint inside the zone and the crack is at 45 degrees or
  more to the pole line (the field must cross it).
- Smearing, latched: a pole face slides more than 10 mm across the plate while the field is on.
"""

import math
import os

import numpy as np

from mt_yoke import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
PLATE_Z = TOP + M.PZ
CELLS = M.CELLS
N = len(CELLS)
PEGS = ("peg0", "peg1", "peg2")
TOOLS = ("yoke", "bulb")

OBJECTS = [
    Box("bench", pos=(0.0, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    Box("yoke", pos=(M.YOKE_REST["x"], M.YOKE_REST["y"], TOP + M.YOKE_REST["z"]), size=(0.01, 0.01, 0.01), mass=M.YOKE_TOTAL, yaw=M.YOKE_REST["heading"]),
    Box("bulb", pos=(M.BULB_REST["x"], M.BULB_REST["y"], TOP + M.BULB_REST_Z), size=(0.01, 0.01, 0.01), mass=M.BULB_TOTAL, yaw=M.BULB_REST["heading"]),
] + [Box(n, pos=(x, y, TOP + M.TRAY["t"]), size=(0.01, 0.01, 0.01), mass=M.PEG_MASS) for n, (x, y) in zip(PEGS, M.PEG_HOME)]

# -- the examination's rules ------------------------------------------------------------------------------------------
GATE = 0.0025  # pole faces this close to the plate's top: on it
YOKE_TILT = math.radians(5.0)
EDGE = 0.010  # pole centres this far inside the plate's edges
HELD = 0.025  # a fingertip midpoint within this of a tool's grip centre
CLOSED = M.HANDLE["x"] + 0.006  # the pads closer than this: closed on the handle (the trigger)
NOZZLE_Z = (0.010, 0.060)  # nozzle height over the plate for powder to land
R_POWDER = 0.035
ZONE_T, ZONE_N = 0.040, 0.025  # along the pole line from the poles' midpoint; across it
DIR_TOL = math.radians(20.0)
CRACK_ANGLE = math.radians(45.0)
SMEAR = 0.010
MARK_R = 0.010  # a peg's base centre this close to a crack's midpoint marks it
PEG_TILT, PEG_Z = math.radians(15.0), 0.003
HAND_OFF_PEG = 0.03
HANDS_CLEAR, LIN, ANG, STILL_S = 0.06, 0.01, 0.5, 1.0
REST_XY, REST_Z, REST_TILT, REST_TURN = 0.012, 0.006, math.radians(10.0), math.radians(20.0)
SCORE_ONE, SCORE_TWO = 0.4, 0.7


def _fresh():
    return {"A": np.zeros(N, bool), "B": np.zeros(N, bool), "dust": np.zeros(N, bool), "found": {c: False for c in M.CRACKS},
            "field": False, "anchor": None, "smear": False, "bad": False, "puff": False, "still": 0.0, "live_steps": 0, "dry_steps": 0,
            "early": {n: False for n in PEGS}}


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


def _p(objects, name):
    return np.asarray(objects[name].pos, float)


def pad_gap(objects, hand):
    s = objects[hand]
    return float(np.linalg.norm(np.asarray(s.high, float) - np.asarray(s.low, float)))


def _nearest(objects, p):
    best = min(("left_tcp", "right_tcp"), key=lambda a: np.linalg.norm(_p(objects, a) - p))
    return best if np.linalg.norm(_p(objects, best) - p) <= HELD else None


def yoke_hand(objects):
    """The hand holding the yoke's handle closed (the trigger squeezed), or None."""
    h = _nearest(objects, _world(objects["yoke"], M.HANDLE_C))
    return h if h is not None and pad_gap(objects, h) <= CLOSED else None


def bulb_hand(objects):
    return _nearest(objects, _p(objects, "bulb"))


def feet(objects):
    y = objects["yoke"]
    return np.array([_world(y, f) for f in M.FEET])


def squeeze(objects):
    return float(objects["squeeze"].pos[0])


def nozzle(objects):
    return _world(objects["bulb"], M.NOZZLE)


def field_on(objects):
    f = feet(objects)
    R = _rot(objects["yoke"].quat_xyzw)
    (x0, x1), (y0, y1) = M.PLATE["x"], M.PLATE["y"]
    on = all(x0 + EDGE <= p[0] <= x1 - EDGE and y0 + EDGE <= p[1] <= y1 - EDGE and -0.001 <= p[2] - PLATE_Z <= GATE for p in f)
    return bool(on and R[2, 2] >= math.cos(YOKE_TILT) and yoke_hand(objects) is not None)


def zone(objects, pts):
    """Which plan points lie in the field's zone between the poles."""
    f = feet(objects)[:, :2]
    mid = f.mean(axis=0)
    u = (f[1] - f[0]) / max(np.linalg.norm(f[1] - f[0]), 1e-9)
    d = np.asarray(pts, float).reshape(-1, 2) - mid
    t = d @ u
    n = np.abs(d @ np.array((-u[1], u[0])))
    return (np.abs(t) <= ZONE_T) & (n <= ZONE_N)


def pole_line(objects):
    f = feet(objects)[:, :2]
    u = f[1] - f[0]
    return u / max(np.linalg.norm(u), 1e-9)


def direction(objects):
    """'A' (pole line square to the weld, which runs along x), 'B' (along it) or None."""
    u = pole_line(objects)
    if abs(u[0]) <= math.sin(DIR_TOL):
        return "A"
    if abs(u[1]) <= math.sin(DIR_TOL):
        return "B"
    return None


def powder(objects):
    """This step's powder: (flowing, the weld cells it lands on)."""
    nz = nozzle(objects)
    flowing = squeeze(objects) >= M.SQUEEZE_ON and NOZZLE_Z[0] <= nz[2] - PLATE_Z <= NOZZLE_Z[1]
    return flowing, np.linalg.norm(CELLS - nz[:2], axis=1) <= R_POWDER


def _turn(R):
    return math.atan2(R[1, 0], R[0, 0])


def _rested(part, x, y, z, heading):
    R = _rot(part.quat_xyzw)
    p = np.asarray(part.pos, float)
    dturn = abs(math.atan2(math.sin(_turn(R) - heading), math.cos(_turn(R) - heading)))
    return bool(math.hypot(p[0] - x, p[1] - y) <= REST_XY and abs(p[2] - z) <= REST_Z and R[2, 2] >= math.cos(REST_TILT) and dturn <= REST_TURN)


def tools_rested(objects):
    return _rested(objects["yoke"], M.YOKE_REST["x"], M.YOKE_REST["y"], TOP + M.YOKE_REST["z"], M.YOKE_REST["heading"]) and _rested(
        objects["bulb"], M.BULB_REST["x"], M.BULB_REST["y"], TOP + M.BULB_REST_Z, M.BULB_REST["heading"])


def _box_dist(objects, part, lo, hi, hand):
    R = _rot(objects[part].quat_xyzw)
    d = R.T @ (_p(objects, hand) - _p(objects, part))
    return float(np.linalg.norm(d - np.clip(d, lo, hi)))


def let_go(objects, hand):
    """A hand 6 cm or more from every part of both tools and from every peg's head."""
    pegs = min(np.linalg.norm(_p(objects, hand) - _world(objects[n], M.PEG_HEAD_C)) for n in PEGS)
    return (_box_dist(objects, "yoke", *M.YOKE_HULL, hand) >= HANDS_CLEAR and _box_dist(objects, "bulb", *M.BULB_HULL, hand) >= HANDS_CLEAR
            and pegs >= HANDS_CLEAR)


def hands_clear(objects):
    return let_go(objects, "left_tcp") and let_go(objects, "right_tcp")


def _in_tray(p):
    return M.TRAY["x"][0] <= p[0] <= M.TRAY["x"][1] and M.TRAY["y"][0] <= p[1] <= M.TRAY["y"][1] and p[2] >= TOP - 0.005


def peg_marks(objects, name):
    """The crack a peg marks (its base within 10 mm of the midpoint of a crack that had shown before the peg first stood
    there, standing, let go), else None."""
    if _trial["early"][name]:
        return None
    s = objects[name]
    p = np.asarray(s.pos, float)
    if _rot(s.quat_xyzw)[2, 2] < math.cos(PEG_TILT) or abs(p[2] - PLATE_Z) > PEG_Z:
        return None
    head = _world(s, M.PEG_HEAD_C)
    if min(np.linalg.norm(_p(objects, h) - head) for h in ("left_tcp", "right_tcp")) < HAND_OFF_PEG:
        return None
    for c, d in M.CRACKS.items():
        if _trial["found"][c] and np.linalg.norm(p[:2] - d["mid"]) <= MARK_R:
            return c
    return None


def marking(objects):
    """(cracks marked, any stray peg: off the tray and not marking a crack that has shown)."""
    marked, stray = set(), False
    for n in PEGS:
        c = peg_marks(objects, n)
        if c is not None:
            marked.add(c)
        elif not _in_tray(_p(objects, n)):
            stray = True
    return marked, stray


def note_step(objects, dt):
    """Called by the embodiment after every control step: the field, powder, coverage, indications, smearing, and the clock."""
    tr = _trial
    if any(objects[n].pos[2] < TOP - 0.01 for n in TOOLS):
        tr["bad"] = True
    live = field_on(objects)
    f = feet(objects)[:, :2]
    if live:
        if tr["anchor"] is None:
            tr["anchor"] = f.copy()
        elif np.max(np.linalg.norm(f - tr["anchor"], axis=1)) > SMEAR:
            tr["smear"] = True
    else:
        tr["anchor"] = None
    tr["field"] = live
    flowing, hit = powder(objects)
    tr["puff"] = flowing
    if flowing:
        tr["dust"] |= hit
        yh, bh = yoke_hand(objects), bulb_hand(objects)
        if live and bh is not None and bh != yh:
            tr["live_steps"] += 1
            z = zone(objects, CELLS)
            d = direction(objects)
            if d is not None:
                tr[d] |= hit & z
            u = pole_line(objects)
            nz = nozzle(objects)[:2]
            for c, cr in M.CRACKS.items():
                cross = math.acos(min(1.0, abs(float(np.dot(u, cr["dir"])))))
                if cross >= CRACK_ANGLE and np.linalg.norm(cr["mid"] - nz) <= R_POWDER and zone(objects, cr["mid"])[0]:
                    tr["found"][c] = True
        else:
            tr["dry_steps"] += 1
    for n in PEGS:  # a peg set down on a crack before that crack showed is latched as placed blind
        p = _p(objects, n)
        if abs(p[2] - PLATE_Z) <= PEG_Z:
            for c, d in M.CRACKS.items():
                if not tr["found"][c] and np.linalg.norm(p[:2] - d["mid"]) <= MARK_R:
                    tr["early"][n] = True
    parts = TOOLS + PEGS
    still = all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in parts) and hands_clear(objects)
    tr["still"] = tr["still"] + dt if still else 0.0


def shown():
    """The prims to show now: powder patches, indications, the yoke's lamp, the bulb's puff."""
    tr = _trial
    on = {f"dust_{i}" for i in range(N) if tr["dust"][i]}
    on |= {f"ind_{c}" for c, v in tr["found"].items() if v}
    if tr["field"]:
        on.add("fx_lamp")
    if tr["puff"]:
        on.add("fx_puff")
    return on


def score(objects):
    """Read at the end of the run. 0 if a tool falls off the table, the poles smeared, either direction leaves a weld cell
    unexamined, a peg is stray, or no crack is marked. Otherwise 0.4 with one crack marked, 0.7 with both, 1.0 with both
    and both tools back on their rests, both hands clear and all still for the last 1 s."""
    tr = _trial
    if tr["bad"] or tr["smear"] or not (tr["A"].all() and tr["B"].all()):
        return 0.0
    marked, stray = marking(objects)
    if stray or not marked:
        return 0.0
    if len(marked) == 1:
        return SCORE_ONE
    if tools_rested(objects) and hands_clear(objects) and tr["still"] >= STILL_S:
        return 1.0
    return SCORE_TWO


def report(objects):
    tr = _trial
    marked, stray = marking(objects)
    return (f"A {int(tr['A'].sum())}/{N} B {int(tr['B'].sum())}/{N} found {[c for c, v in tr['found'].items() if v]} marked {sorted(marked)} stray {int(stray)} "
            f"smear {int(tr['smear'])} live {tr['live_steps']} dry {tr['dry_steps']} rested {int(tools_rested(objects))} clear {int(hands_clear(objects))} still {tr['still']:.2f}")


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Open loop from the known start: the tools, rests, plate and weld are visible; the plan reads no simulator state, and
# it doesn't know where the cracks are: it examines the whole weld and marks where the red lines show (fixed here).
# MT_DEMO: shortcut variants, for checking they score 0. "dust_first": the weld is dusted with the yoke on its rest, then
# the yoke steps both passes alone. "one_dir": only the first direction. "wrong_peg": the second peg 30 mm off its crack.
# "no_peg": no pegs.
VARIANT = os.environ.get("MT_DEMO", "")
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.03
GRIP_DZ = 0.003  # TCP above each grip's centre when holding
PRESS = float(os.environ.get("MT_PRESS", "0.012"))  # the pole faces aimed this far into the plate, so the hand bears down
NOZZLE_AIM = float(os.environ.get("MT_NOZZLE", "0.025"))  # nozzle aimed this high over the plate
YOKE_SAG = float(os.environ.get("MT_YSAG", "0.012"))  # aimed this much higher in the air for the arm's give (calibrated)
BULB_SAG = float(os.environ.get("MT_BSAG", "0.0"))
PASS_A = (0.180, 0.220, 0.260, 0.300)  # pole line's x for each placement, square to the weld
PASS_B = (0.185, 0.235, 0.285)  # poles' midpoint's x for each placement, along the weld
HEAD_YA, HEAD_YB = -0.5 * math.pi, -math.pi  # yoke headings: the handle at the +y end, then at the far (+x) end
HEAD_BA, HEAD_BB = 0.25 * math.pi, 0.5 * math.pi  # bulb headings: the nozzle coming in from -x -y, then from -y
TIP_A = np.array((0.0, 0.012))  # in direction A the nozzle aims 12 mm toward the handle's pole: the wand passes 27 mm clear of the far pole
WRONG = np.array((0.245, M.WELD_Y))
# where the yoke landed against where it was aimed, from logged runs (r2): aim the other way by as much
STAGE_A, STAGE_B = 0.040, 0.080  # the nozzle's staging point, back along its line (A: less, for the right arm's reach)
LIFT_B = 0.014  # lifted less in direction B: the far placements are near the left arm's reach
AIM_A, AIM_B = np.array((0.003, -0.009)), np.array((0.0, -0.002))
AIM_REST = np.array((0.006, -0.008))  # the yoke set back on its rest (r4)


def _fingers(width):
    return -FINGER_PER_M * width


YOKE_OPEN, YOKE_GRIP = _fingers(M.HANDLE["x"] + 0.030), _fingers(M.HANDLE["x"] - 0.008)
BULB_OPEN, BULB_HOLD, BULB_SQZ = _fingers(M.BULB_GRIP_W + 0.030), _fingers(M.BULB_GRIP_W - 0.0025), _fingers(M.BULB_GRIP_W - M.SQUEEZE_TRAVEL - 0.0015)
PEG_OPEN, PEG_GRIP = _fingers(M.PEG_HEAD["w"] + 0.024), _fingers(M.PEG_HEAD["w"] - 0.004)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def yoke_tcp(xy, heading, z):
    """The yoke hand's TCP and yaw with the yoke's frame (its pole faces) at (xy, z), turned to ``heading``."""
    o = np.array((xy[0], xy[1], z))
    return o + _rz(heading) @ M.HANDLE_C + (0.0, 0.0, GRIP_DZ), heading + 0.5 * math.pi


def bulb_tcp(tip, heading):
    o = np.asarray(tip, float) - _rz(heading) @ M.NOZZLE
    return o + (0.0, 0.0, GRIP_DZ), heading - 0.5 * math.pi


def peg_tcp(base):
    return np.asarray(base, float) + M.PEG_HEAD_C + (0.0, 0.0, GRIP_DZ), 0.0


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

    def until(self, arm, t):
        if t > self.t[arm] + 1e-9:
            self.go(arm, t - self.t[arm], label="wait")

    def sync(self):
        t = max(self.t.values())
        for a in SIDE:
            self.until(a, t)


def _park(plan, arm, dur=1.4):
    plan.go(arm, dur, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), None, "away")


def _rest_yoke_tcp():
    return yoke_tcp((M.YOKE_REST["x"], M.YOKE_REST["y"]), M.YOKE_REST["heading"], TOP + M.YOKE_REST["z"])


def _rest_bulb_tcp():
    o = np.array((M.BULB_REST["x"], M.BULB_REST["y"], TOP + M.BULB_REST_Z))
    return bulb_tcp(o + _rz(M.BULB_REST["heading"]) @ M.NOZZLE, M.BULB_REST["heading"])


def _pick_yoke(plan):
    p, yaw = _rest_yoke_tcp()
    plan.go("left", 1.6, p + (0, 0, 0.04), YOKE_OPEN, "yoke: above it", yaw)
    plan.go("left", 0.3, label="settle")
    plan.go("left", 0.8, p, label="descend")
    plan.go("left", 0.6, finger=YOKE_GRIP, label="grip")
    plan.go("left", 0.8, p + (0, 0, 0.03 + YOKE_SAG), label="lift")


def _pick_bulb(plan):
    p, yaw = _rest_bulb_tcp()
    plan.go("right", 1.6, p + (0, 0, 0.04), BULB_OPEN, "bulb: above it", yaw)
    plan.go("right", 0.3, label="settle")
    plan.go("right", 0.8, p, label="descend")
    plan.go("right", 0.6, finger=BULB_HOLD, label="grip")
    plan.go("right", 0.8, p + (0, 0, 0.04 + BULB_SAG), label="lift")


def _place_yoke(plan, xy, heading, first=False):
    """Carry the yoke 15 mm over its place (30 mm on the first), then set both poles down, bearing down PRESS."""
    hi = 0.03 if first else (0.020 if heading == HEAD_YA else LIFT_B)
    xy = np.asarray(xy, float) + (AIM_A if heading == HEAD_YA else AIM_B)
    p, yaw = yoke_tcp(xy, heading, PLATE_Z + hi + YOKE_SAG)
    plan.go("left", 2.0 if first else 1.2, p, label="yoke: over", yaw=yaw)
    p, _ = yoke_tcp(xy, heading, PLATE_Z - PRESS)
    plan.go("left", 0.8, p, label="yoke: down")
    plan.go("left", 0.5, label="yoke: on")


def _lift_yoke(plan, xy, heading, hi=0.020):
    p, yaw = yoke_tcp(xy, heading, PLATE_Z + hi + YOKE_SAG)
    plan.go("left", 0.6, p, label="yoke: lift", yaw=yaw)


def _nozzle_at(xy, heading, back=0.0, up=0.0):
    u = np.array((math.cos(heading), math.sin(heading), 0.0))
    return np.array((xy[0], xy[1], PLATE_Z + NOZZLE_AIM + BULB_SAG + up)) - back * u


def _stage(xy, heading):
    return bulb_tcp(_nozzle_at(xy, heading, STAGE_A if heading == HEAD_BA else STAGE_B, 0.005), heading)


def _puff(plan, xy, heading, first=False):
    """Stage the nozzle 80 mm back along its own line, slide it straight in between the poles (so the wand never sweeps
    across a pole), squeeze and hold 0.6 s, let go, and slide back out."""
    p, yaw = _stage(xy, heading)
    plan.go("right", 2.0 if first else 1.2, p, label="bulb: staged", yaw=yaw)
    plan.go("right", 0.9, bulb_tcp(_nozzle_at(xy, heading), heading)[0], label="bulb: in")
    plan.go("right", 0.3, finger=BULB_SQZ, label="squeeze")
    plan.go("right", 0.6, label="squeezed")
    plan.go("right", 0.3, finger=BULB_HOLD, label="release")
    plan.go("right", 0.7, p, label="bulb: out")


def _pass(plan, placements, yheading, bheading, bulb=True):
    which_a = yheading == HEAD_YA
    if bulb and which_a:  # the nozzle is staged on its line before the yoke comes down, so it never crosses a pole
        plan.sync()
        p, yaw = _stage(np.asarray((placements[0], M.WELD_Y)) + TIP_A, bheading)
        plan.go("right", 2.4, p, label="bulb: staged first", yaw=yaw)
        plan.sync()  # the yoke waits until it's there
    for k, c in enumerate(placements):
        xy = o = (c, M.WELD_Y)
        _place_yoke(plan, o, yheading, first=k == 0)
        if bulb:
            plan.sync()
            _puff(plan, np.asarray(xy) + (TIP_A if which_a else 0.0), bheading, first=k == 0)
        else:
            plan.go("left", 2.0, label="yoke: on, no powder")
        plan.sync()
        _lift_yoke(plan, o, yheading, 0.020 if which_a else LIFT_B)


def _return_yoke(plan):
    p, yaw = _rest_yoke_tcp()
    p = p + (*AIM_REST, 0.0)
    plan.go("left", 0.8, np.array(plan.last["left"][0]) + (-0.04, 0, 0.02), label="yoke: up")
    plan.go("left", 2.4, p + (0, 0, 0.03 + YOKE_SAG), label="yoke: over its rest", yaw=yaw)
    plan.go("left", 1.0, p + (0, 0, 0.001), label="yoke: down")
    plan.go("left", 0.6, finger=YOKE_OPEN, label="yoke: let go")
    plan.go("left", 0.8, p + (0, 0, 0.06), label="up")
    _park(plan, "left")


def _return_bulb(plan):
    p, yaw = _rest_bulb_tcp()
    plan.go("right", 0.8, np.array(plan.last["right"][0]) + (0, 0, 0.03), label="bulb: up")
    plan.go("right", 2.4, p + (0, 0, 0.03 + BULB_SAG), label="bulb: over its rest", yaw=yaw)
    plan.go("right", 1.0, p + (0, 0, 0.002), label="bulb: down")
    plan.go("right", 0.6, finger=BULB_OPEN, label="bulb: let go")
    plan.go("right", 0.8, p + (0, 0, 0.06), label="up")


def _bulb_wait(plan, heading):
    """The bulb waits back, clear of the yoke while it turns."""
    p, yaw = bulb_tcp(_nozzle_at((0.12, -0.10), heading, 0.0, 0.03), heading)
    plan.go("right", 1.6, p, label="bulb: wait", yaw=yaw)


def _peg(plan, k, target):
    home = np.array((*M.PEG_HOME[k], TOP + M.TRAY["t"]))
    p, yaw = peg_tcp(home)
    plan.go("right", 1.6, p + (0, 0, 0.04), PEG_OPEN, f"peg{k}: above it", yaw)
    plan.go("right", 0.7, p, label="peg: descend")
    plan.go("right", 0.5, finger=PEG_GRIP, label="peg: grip")
    plan.go("right", 0.7, p + (0, 0, 0.05), label="peg: lift")
    q, _ = peg_tcp((target[0], target[1], PLATE_Z + 0.0015))
    plan.go("right", 1.8, q + (0, 0, 0.05), label="peg: over the crack")
    plan.go("right", 0.9, q, label="peg: down")
    plan.go("right", 0.5, finger=PEG_OPEN, label="peg: let go")
    plan.go("right", 0.7, q + (0, 0, 0.05), label="up")


def demo():
    plan = _Plan()
    for arm in SIDE:  # down from the ready pose first, so the wrist turns low
        p0 = plan.last[arm][0]
        plan.go(arm, 1.0, (p0[0], p0[1], TOP + 0.15), label="down")
    _pick_yoke(plan)
    if VARIANT == "dust_first":  # one arm at a time: dust the whole weld with the yoke still on its rest
        _pick_bulb(plan)
        for k, c in enumerate((0.180, 0.220, 0.260, 0.300)):
            _puff(plan, (c, M.WELD_Y), HEAD_BB, first=k == 0)
        _return_bulb(plan)
        plan.sync()
        _pass(plan, PASS_A, HEAD_YA, HEAD_BA, bulb=False)
        _pass(plan, PASS_B, HEAD_YB, HEAD_BB, bulb=False)
    else:
        _pick_bulb(plan)
        plan.sync()
        _pass(plan, PASS_A, HEAD_YA, HEAD_BA)
        if VARIANT != "one_dir":
            _bulb_wait(plan, HEAD_BB)
            _lift_yoke(plan, (PASS_A[-1], M.WELD_Y), HEAD_YA, 0.03)
            plan.sync()
            _pass(plan, PASS_B, HEAD_YB, HEAD_BB)
        _return_bulb(plan)
    _return_yoke(plan)
    plan.sync()
    if VARIANT == "one_dir":
        _peg(plan, 0, M.CRACKS["long"]["mid"])
    elif VARIANT != "no_peg":
        _peg(plan, 0, M.CRACKS["long"]["mid"])
        _peg(plan, 1, WRONG if VARIANT == "wrong_peg" else M.CRACKS["trans"]["mid"])
    _park(plan, "right")
    end = max(plan.t.values()) + 2.0
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
                bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1), round(math.degrees(re), 2)))
    return worst_p, worst_r, bad


def hand_gap(wps):
    """The closest the two TCPs come in plan, on a 0.05 s grid (needs perfect tracking)."""
    from yam_arena.demo import ArmPlan

    plans = {a: ArmPlan(wps[a], BASES[a]) for a in wps}
    T = min(p.duration for p in plans.values())
    best = (9.0, 0.0)
    for k in range(int(T / 0.05)):
        a, b = (np.asarray(plans[s].target(k * 0.05)[0], float) for s in ("left", "right"))
        d = float(np.linalg.norm((a - b)[:2]))
        best = min(best, (d, k * 0.05))
    return best


# -- self-check (no simulator): the demo fits the run limit, IK reaches every waypoint, the hands keep apart -----------
if __name__ == "__main__":
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad = ik_errors(wps)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg, {len(bad)} bad", bad[:8])
    print("closest hands in plan (m, at s):", hand_gap(wps), "yoke COM (mm):", np.round(1000 * M.YOKE_COM, 1))
    assert t * 20 < 3600 and not bad
