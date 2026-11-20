"""A 45 degree offset in 1/8 in soft copper tube on a Swagelok MS-HTB-2 hand tube bender clamped in a vise.

Instruction: "Use the hand bender to bend a 45 degree offset at the tube's two marks, turn the roll support back to 0, and let go."

The tube starts latched in the bender die with mark A at the roll support's 45 degree mark and the roll dies resting
on it. The scripted demo pulls the short handle round to bend A (allowing for springback), swings the handle up and
turns the roll support back to 0; eases the latch; takes the bent leg where it rests and holds it level; slides the
tube along the groove with both hands until mark B is at the 45 degree mark; lets the latch back down onto the tube;
turns the tube half a turn about its straight part by the bent leg, in steps, re-gripping (the latch holds it between
grips); pushes the latch firmly; bends B with the same handle swing (now the opposite way in the tube's own frame);
swings the handle up, turns the roll support back to 0 and lets go; the finished tube stays latched in the bender.

Run it:  inspect-robots run --embodiment yam_tube_offset -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           -P hold_s=10 --instruction "$(python -c "import json;print(json.load(open('task.json'))['instruction'])")" --scorer task_score

`yam_tube_offset` (tube_offset/embodiment.py) spawns the bender and the tube (tube_offset/mech.py), calls `on_step` after
every control step (it writes the tube's plastic hinges and the bender's friction and detents), and reports each
gripper's fingertip midpoint and every link of the bender and the tube.
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
BENDER, TUBE = "bender", "tube"
ARTS = (BENDER, TUBE)

# -- the start: the tube latched in the die, mark A's bend centred 5.6 mm past the tangent point (the 45 deg mark) ------
WRAP_45 = M.R_BEND * math.radians(45.0)  # 11.2 mm of the tube wraps the die in a 45 deg bend
T_DIR = M.ALPHA + math.pi / 2  # the tube's +x at the start: along the free (roll-die) side
X_T0 = M.MARK_GAP / 2 - WRAP_45 / 2  # the tube's own x at the die's tangent point
_t0 = M.polar(M.R_BEND, M.ALPHA)
TUBE_START = (_t0[0] - X_T0 * math.cos(T_DIR), _t0[1] - X_T0 * math.sin(T_DIR), T_DIR)

HINGES = [j for j, *_ in M.tube_joints()]
ZONE_A = [j for j in HINGES if j.startswith("ha")]
ZONE_B = [j for j in HINGES if j.startswith("hb")]


# -- the tube's shape on paper (CPU): forward kinematics from its hinge angles -------------------------------------------
def tube_fk(qs, origin=(0.0, 0.0), yaw=0.0):
    """Each link's frame (x, y, yaw) in the table plane, from the hinge angles {joint: rad}, with the root (mid) at origin, yaw."""
    out = {"mid": (origin[0], origin[1], yaw)}
    for j, parent, child, x in M.tube_joints():
        px, py, pa = out[parent]
        jx, jy = px + x * math.cos(pa), py + x * math.sin(pa)  # the hinge point, in the parent's frame
        a = pa + qs.get(j, 0.0)
        out[child] = (jx - x * math.cos(a), jy - x * math.sin(a), a)
    return out


def ideal_q(a=math.radians(45.0), b=math.radians(45.0), n=5):
    """Hinge angles for two bends, each spread evenly over the n hinges nearest the middle of its wrap."""
    q = {}
    for zone, ang in ((ZONE_A, a), (ZONE_B, b)):
        xs = [abs(x) for j, _, _, x in M.tube_joints() if j in zone]
        mid = M.MARK_GAP / 2
        near = sorted(range(len(zone)), key=lambda i: abs(xs[i] - mid))[:n]
        for i in near:
            q[zone[i]] = ang / n
    return q


def leg_centre(frames, n):
    x0, x1 = next((a, b) for m, a, b in M.tube_links() if m == n)
    fx, fy, fa = frames[n]
    c = (x0 + x1) / 2
    return fx + c * math.cos(fa), fy + c * math.sin(fa), fa


# -- the template: two green lanes painted on the table where the finished offset's straight ends lie -------------------
OFFSET = 0.050  # O; the marks lie L = O x 1.414 apart (Swagelok MS-13-43 p.11)
LANE_AT = (0.27, -0.255)  # where the finished tube's middle lies on the template; its ends run along x
LANE_LEN, LANE_W = 0.08, 0.016
LAID_YAW = -math.radians(45.0)  # the middle's yaw when the ends lie along x


def _lanes():
    f = tube_fk(ideal_q(), LANE_AT, LAID_YAW)
    return leg_centre(f, "leg_a")[:2], leg_centre(f, "leg_b")[:2]


LANE_A, LANE_B = _lanes()
LANES = tuple((c[0] - LANE_LEN / 2, c[1] - LANE_W / 2, c[0] + LANE_LEN / 2, c[1] + LANE_W / 2) for c in (LANE_A, LANE_B))

OBJECTS = [
    Box(BENDER, pos=(0.0, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    Box(TUBE, pos=(TUBE_START[0], TUBE_START[1], TOP + M.Z_T), size=(0.01, 0.01, 0.01), yaw=TUBE_START[2]),
]

# -- the tube's plastic hinges and the bender's friction and detents (written every control step) ----------------------
SUPPORT_HOLD, SUPPORT_K = 0.20, 40.0  # roll support pivot friction (estimate; stiff stick: it springs back at most 0.3 deg when let go): more than the link's detent, so the handle swings up first
LINK_HOLD, LINK_K, CLEVIS_SNAP, CLEVIS_HOLD = 0.08, 1.6, math.radians(-12.0), 0.05  # the link's toggle as a detent at the clevis; lifted, it stays put
LATCH_HOLD, LATCH_K = 0.05, 1.0  # the tube latch seated (below 2 deg): its spring washer clamps the tube (estimate)
LATCH_FIRM, FIRM_AT = float(os.environ.get("TUBE_LATCH_FIRM", "0.12")), math.radians(2.2)  # "push the latch firmly" (step 8): pressed onto the tube for 1 s it clamps harder, 4 N at the pad (estimate)
LATCH_FRICTION = 0.03  # off its seat the latch stays where it is put (spring washer friction on the latch screw, estimate)
LINK_G = float(os.environ.get("TUBE_LINK_G", "0")) or None
ARMATURE = float(os.environ.get("TUBE_ARMATURE", "1e-4"))
DEV = os.environ.get("TUBE_DEV", "")
DEBUG = os.environ.get("TUBE_DEBUG", "") == "1" or bool(DEV)

_trial = {}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears a ruin."""
    _trial.clear()
    _trial.update(still=0.0, bad=False, why="", anchors=None, support=0.0, clevis=0.0, latch=0.0, firm=START_FIRM, press=0.0, sup_prev=0.0, tool=[0.0, 0.0], hand=[0.0, 0.0], next=0.0, peak=[0.0, 0.0], dev={})


def build(stage, name, path, origin):
    if name == BENDER:
        return M.build_bender(stage, path, origin, ())
    return M.build_tube(stage, path, link_g=LINK_G, armature=ARMATURE)


def on_step(emb, s, dt):
    q = s["q"]
    tube, bender = emb.art(TUBE), emb.art(BENDER)
    qs = np.array([q[f"tube/{j}"] for j in HINGES])
    if _trial["anchors"] is None:
        _trial["anchors"] = qs.copy()
    sk = M.M_P / M.K_E
    new = np.minimum(np.maximum(_trial["anchors"], qs - sk), qs + sk)
    note_plastic(new - _trial["anchors"], q["bender/support_turn"] - _trial["sup_prev"], q["bender/clevis_turn"])
    _trial["anchors"], _trial["sup_prev"] = new, q["bender/support_turn"]
    M.drives(tube, emb.store, {j: (float(a), M.M_P, M.K_E, 0.0005) for j, a in zip(HINGES, _trial["anchors"])})
    sup, clev, lat = q["bender/support_turn"], q["bender/clevis_turn"], q["bender/latch_turn"]
    _trial["support"] = M.slip(_trial["support"], sup, SUPPORT_HOLD, SUPPORT_K)
    _trial["clevis"] = M.slip(_trial["clevis"], clev, CLEVIS_HOLD, 2.0)
    _trial["latch"] = M.slip(_trial["latch"], lat, LATCH_FRICTION, 2.0)
    _trial["press"] = _trial["press"] + dt if lat < -FIRM_AT else 0.0
    _trial["firm"] = (_trial["press"] >= 1.0) or (_trial["firm"] and lat < M.LATCH["seat"])  # pressed down 1 s: firm until lifted
    spec = {"support_turn": (_trial["support"], SUPPORT_HOLD, SUPPORT_K, 0.01),
            "clevis_turn": ((0.05, LINK_HOLD, LINK_K, 0.01) if clev > CLEVIS_SNAP else (_trial["clevis"], CLEVIS_HOLD, 2.0, 0.01)),
            "latch_turn": ((-0.10, LATCH_FIRM if _trial["firm"] else LATCH_HOLD, LATCH_K, 0.002) if lat < M.LATCH["seat"] else (_trial["latch"], LATCH_FRICTION, 2.0, 0.002))}
    M.drives(bender, emb.store, spec)
    if DEV:
        _dev(emb, s, dt)
    note_step(s, dt)
    if DEBUG and s["t"] >= _trial["next"]:
        _trial["next"] = s["t"] + 0.25
        _print(s)


# -- dev hands (spikes only: TUBE_DEV) --------------------------------------------------------------------------------
def grip_world(s, r=None):
    """The handle grip point now, from the handle's pose: the force a hand puts in acts here."""
    h = s["bender/handle"]
    local = np.array((*M.grip_point(0.0, 0.0, r), M.Z_H))
    return _rot(h.quat_xyzw) @ local + np.asarray(h.pos, float)


def _dev(emb, s, dt):
    """S1: a dev hand pulls the handle grip round (tangential about the die axis): a slow force ramp to 0.8 M_p first,
    then a capped PD to the target angle, then lets go (springback), then lifts the handle up (CW about the clevis)."""
    t = s["t"]
    q = s["q"]
    d = _trial["dev"]
    r = 0.21
    g = grip_world(s, r)
    rel = g[:2] - np.array(M.C)
    tang = np.array((-rel[1], rel[0])) / max(np.linalg.norm(rel), 1e-6)
    phi = q["bender/support_turn"]
    T1, T2, T3 = 2.0, 8.0, 30.0
    target = float(os.environ.get("DEV_TARGET", "47"))
    fmax = float(os.environ.get("DEV_FMAX", str(2.5 * M.M_P / r)))
    if DEV == "s1":
        if t < T1:
            f = 0.0
        elif t < T2:
            f = 0.8 * M.M_P / r * (t - T1) / (T2 - T1)
        elif t < T3:
            goal = math.radians(target) * min(1.0, (t - T2) / 15.0)
            f = float(np.clip(40.0 * (goal - phi) + 0.8 * M.M_P / r, 0.0, fmax))
        elif t < T3 + 4.0:
            f = 0.0
        elif t < T3 + 8.0:
            f = -1.5 * min(1.0, (t - T3 - 4.0) / 0.5)  # pull back: lifts the handle about the clevis
        else:
            f = 0.0
        d["f"] = f
        M.push(emb.art(BENDER), "handle", (f * tang[0], f * tang[1], 0.0), tuple(np.asarray(g) + emb.origin))
    elif DEV == "s3":  # two dev pinches 3 cm apart on the free leg, a 5 N couple, no handle
        pass


# -- readings ------------------------------------------------------------------------------------------------------------
def _rot(qx):
    x, y, z, w = qx
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


LINKS = [n for n, *_ in M.tube_links()]
DL = 2 * M.ZONE / M.ZONE_LINKS
BENT_MIN = math.radians(1.0)  # a hinge counts as carrying the bend from 1 degree


def bend(s, zone):
    """One mark's bend: (angle, hinges carrying 1 degree or more of it, radius in m or None). Angle: the zone's hinges
    summed, signed. Radius: the bent length over the angle, the bent length being the hinge spacing times the effective
    number of hinges carrying the bend, (sum of angles)^2 / (sum of squared angles): n for n equal hinges, under 2 for a kink."""
    a = np.array([s["q"][f"tube/{j}"] for j in zone])
    tot = float(a.sum())
    if abs(tot) < math.radians(5.0):
        return tot, 0, None
    n = int(np.count_nonzero(np.sign(a) * np.sign(tot) * np.abs(a) >= BENT_MIN))
    n_eff = tot * tot / float(np.sum(a * a))
    return tot, n, n_eff * DL / abs(tot)


BENT, BEND_TOL, RUIN = math.radians(45.0), math.radians(4.0), math.radians(55.0)
RADIUS = (0.010, 0.018)
PARALLEL, OFFSET_TOL = math.radians(3.0), 0.004
ON_ACROSS, ON_ALONG, ON_YAW, ON_Z = 0.010, 0.020, math.radians(6.0), 0.003
LIN, ANG, STILL_S, LET_GO = 0.01, 0.5, 1.0, 0.06
SUPPORT_HOME = math.radians(5.0)  # the roll support back at its 0 mark, where a trial starts
SAME_WAY_MIN = math.radians(10.0)


TOOL_TURN, HAND_SHARE, HAND_MIN = math.radians(0.02), 0.25, math.radians(5.0)
IDX_A = [HINGES.index(j) for j in ZONE_A]
IDX_B = [HINGES.index(j) for j in ZONE_B]


def note_plastic(d_anchor, d_support, clevis):
    """Who made each step's plastic bending: the bender (its roll dies down on the tube, the clevis past the link's snap,
    and the roll support turning forward) or anything else (a hand levering the tube)."""
    tool = clevis > CLEVIS_SNAP and d_support > TOOL_TURN
    for k, idx in enumerate((IDX_A, IDX_B)):
        amt = float(np.sum(np.abs(np.asarray(d_anchor)[idx])))
        _trial["tool" if tool else "hand"][k] += amt


def counted(b, k):
    """A bend counts if it is spread round the die (3+ hinges at 1 degree, radius 10-18 mm) and the bender made it: at
    most a quarter of its plastic bending (or 5 degrees) was made with the roll support still or its roll dies up."""
    a, n, r = b
    return n >= 3 and r is not None and RADIUS[0] <= r <= RADIUS[1] and _trial["hand"][k] <= max(HAND_MIN, HAND_SHARE * abs(a))


MARK_TOL = 0.0035  # a bend's place: its vertex point within 3.5 mm of its mark, along the tube (this task's own value)
LEGS = (("mid", "leg_a"), ("mid", "leg_b"))


def bend_place(q, k):
    """Where bend k (0: mark A, 1: mark B) sits from its mark, in m along the tube (+: toward that end, -: toward the
    middle), or None without a bend. The bend's vertex is where its two straight lengths' centre lines meet (two
    straightedges); its place is the point of the tube's centre line nearest the vertex, the bend's middle."""
    qs = {j: float(q[f"tube/{j}"]) for j in HINGES}
    f = tube_fk(qs)
    (x0, y0, a0), (x1, y1, a1) = f[LEGS[k][0]], f[LEGS[k][1]]
    d0, d1 = np.array((math.cos(a0), math.sin(a0))), np.array((math.cos(a1), math.sin(a1)))
    cr = float(d0[0] * d1[1] - d0[1] * d1[0])
    if abs(cr) < math.sin(math.radians(5.0)):
        return None
    w = np.array((x1 - x0, y1 - y0))
    t = float(w[0] * d1[1] - w[1] * d1[0]) / cr
    vx = np.array((x0, y0)) + t * d0
    sgn = 1.0 if k == 0 else -1.0
    xs = sgn * (M.MARK_GAP / 2 + np.arange(-M.ZONE, M.ZONE + 1e-9, 0.0001))
    dist = [np.linalg.norm(tube_point(f, float(x))[0] - vx) for x in xs]
    return float(sgn * xs[int(np.argmin(dist))] - M.MARK_GAP / 2)


def placed(s, k):
    p = bend_place(s["q"], k)
    return p is not None and abs(p) <= MARK_TOL + 1e-9


def bend_good(b, k, s=None):
    """A correct bend: counted, 45 +/- 4 degrees, and at its mark (when a reading s is given)."""
    return counted(b, k) and BENT - BEND_TOL <= abs(b[0]) <= BENT + BEND_TOL and (s is None or placed(s, k))


def _axis(s, n):
    st = s[f"tube/{n}"]
    R = _rot(st.quat_xyzw)
    return np.asarray(st.pos, float), R[:, 0]


def leg_mid(s, n):
    """A straight leg's middle (its own x centre), in the table frame, and its direction."""
    x0, x1 = next((a, b) for m, a, b in M.tube_links() if m == n)
    p, ax = _axis(s, n)
    R = _rot(s[f"tube/{n}"].quat_xyzw)
    return p + R @ np.array(((x0 + x1) / 2, 0.0, 0.0)), ax


def offset_of(s):
    """How far apart the two straight ends' centre lines lie (a tape across them), and the angle between them."""
    pa, da = leg_mid(s, "leg_a")
    pb, db = leg_mid(s, "leg_b")
    d = da[:2] / np.linalg.norm(da[:2])
    e = db[:2] / np.linalg.norm(db[:2])
    ang = math.acos(min(1.0, abs(float(d @ e))))
    u = (d + np.sign(float(d @ e)) * e)
    u = u / np.linalg.norm(u)
    w = (pb - pa)[:2]
    return abs(float(w[0] * u[1] - w[1] * u[0])), ang


def let_go(s):
    """Both fingertip midpoints at least 6 cm from every link of the tube (open fingers reach 4.7 cm from the midpoint)."""
    for n, x0, x1 in M.tube_links():
        st = s[f"tube/{n}"]
        R = _rot(st.quat_xyzw)
        for a in ("left_tcp", "right_tcp"):
            d = R.T @ (np.asarray(s[a].pos, float) - np.asarray(st.pos, float))
            c = np.array((min(max(d[0], x0), x1), 0.0, 0.0))
            if np.linalg.norm(d - c) < LET_GO:
                return False
    return True


def _on_lane(s, n, c):
    p, ax = leg_mid(s, n)
    yaw = math.atan2(ax[1], ax[0])
    off = abs(math.atan2(math.sin(2 * yaw), math.cos(2 * yaw))) / 2  # either way round, from the lanes' x
    return abs(p[0] - c[0]) <= ON_ALONG and abs(p[1] - c[1]) <= ON_ACROSS and off <= ON_YAW and p[2] - TOP <= M.TUBE_R + ON_Z


def on_template(s):
    return (_on_lane(s, "leg_a", LANE_A) and _on_lane(s, "leg_b", LANE_B)) or (_on_lane(s, "leg_a", LANE_B) and _on_lane(s, "leg_b", LANE_A))


def note_step(s, dt):
    """After every control step: a bend past 55 degrees, or two bends the same way (a U, not an offset), ruins the tube
    for good; and the clock for the tube let go and at rest."""
    ba, bb = bend(s, ZONE_A), bend(s, ZONE_B)
    _trial["peak"] = [max(_trial["peak"][0], abs(ba[0])), max(_trial["peak"][1], abs(bb[0]))]
    if max(_trial["peak"]) > RUIN:
        _trial.update(bad=True, why=_trial["why"] or "overbent")
    if min(abs(ba[0]), abs(bb[0])) >= SAME_WAY_MIN and ba[0] * bb[0] < 0:
        _trial.update(bad=True, why=_trial["why"] or "same way")  # opposite hinge signs at the two marks make a U
    # at rest: every link moved under 1 cm/s and turned under 0.5 rad/s, averaged over the control step (the solver's
    # contact chatter in a clamped tube reads about 2 cm/s instantaneously while nothing moves: d76H)
    now = {n: (np.asarray(s[f"tube/{n}"].pos, float), _rot(s[f"tube/{n}"].quat_xyzw)) for n in LINKS}
    prev = _trial.get("prev")
    still = prev is not None and let_go(s) and all(
        np.linalg.norm(now[n][0] - prev[n][0]) <= LIN * dt
        and math.acos(max(-1.0, min(1.0, (np.trace(now[n][1] @ prev[n][1].T) - 1) / 2))) <= ANG * dt for n in LINKS)
    _trial["prev"] = now
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(s):
    """Read at the end of the run. 0 for good if a bend passed 55 degrees or the bends went the same way. 0 unless the
    tube is let go. Counted bends (spread round the die: 3+ hinges, none over half, radius 10-18 mm): 0.25 for one at
    45 +/- 4 degrees; 0.5 for both, an offset; each bend must also sit at its mark (its vertex point within 3.5 mm of it along the tube)
    or it is not a correct bend; 0.75 with the ends parallel within 3 degrees and 5.0 +/- 0.4 cm apart; 0.9 with
    the roll support also back at its 0 mark within 5 degrees; 1.0 with the tube also at rest for the last 1 s."""
    if not _trial or _trial["bad"] or not let_go(s):
        return 0.0
    ba, bb = bend(s, ZONE_A), bend(s, ZONE_B)
    n = bend_good(ba, 0, s) + bend_good(bb, 1, s)
    if n < 2:
        return 0.25 if n else 0.0
    off, ang = offset_of(s)
    if not (ang <= PARALLEL and abs(off - OFFSET) <= OFFSET_TOL):
        return 0.5
    if abs(s["q"]["bender/support_turn"]) > SUPPORT_HOME:
        return 0.75
    return 1.0 if _trial["still"] >= STILL_S else 0.9


def _print(s):
    q = s["q"]
    d = math.degrees
    ba, bb = bend(s, ZONE_A), bend(s, ZONE_B)
    fa = lambda b: f"{d(b[0]):6.2f}/{b[1]}/{'-' if b[2] is None else round(b[2] * 1000, 1)}"  # noqa: E731
    off, ang = offset_of(s)
    mid = s["tube/mid"].pos
    pl = [bend_place(q, k) for k in (0, 1)]
    print(f"PLACE A {'-' if pl[0] is None else round(pl[0] * 1000, 2)} B {'-' if pl[1] is None else round(pl[1] * 1000, 2)} mm", flush=True)
    print(f"TUBE t={s['t']:6.2f} sup {d(q['bender/support_turn']):6.2f} clev {d(q['bender/clevis_turn']):6.2f} latch {d(q['bender/latch_turn']):5.1f} "
          f"A {fa(ba)} B {fa(bb)} off {off * 1000:5.1f} par {d(ang):5.2f} mid {np.round(1000 * (np.asarray(mid) - (0, 0, TOP)), 1).tolist()} "
          f"vmax {max(np.linalg.norm(s[f'tube/{n}'].lin_vel) for n in LINKS):.3f} L {np.round(1000 * (s['left_tcp'].pos - (0, 0, TOP)), 1).tolist()} "
          f"R {np.round(1000 * (s['right_tcp'].pos - (0, 0, TOP)), 1).tolist()} fing {1000 * q['left/finger']:.1f} {1000 * q['right/finger']:.1f} legz {1000 * (s['tube/leg_a'].pos[2] - TOP):.1f} {1000 * (s['tube/leg_b'].pos[2] - TOP):.1f} tilt {math.degrees(math.acos(min(1.0, _rot(s['tube/mid'].quat_xyzw)[2, 2]))):.1f} firm {int(_trial['firm'])} crankA {np.round(1000 * (s['tube/leg_a'].pos + _rot(s['tube/leg_a'].quat_xyzw)[:, 0] * CRANK_X - (0, 0, TOP)), 1).tolist()} f {_trial['dev'].get('f', 0):.2f} bad {_trial['why']} go {let_go(s)} "
          f"still {_trial['still']:.2f} tool {[round(d(v), 1) for v in _trial['tool']]} hand {[round(d(v), 1) for v in _trial['hand']]} score {score(s):.2f}", flush=True)
    if abs(s["t"] % 2.0) < 0.13:
        print("HINGES A", [round(d(q[f"tube/{j}"]), 2) for j in ZONE_A], "B", [round(d(q[f"tube/{j}"]), 2) for j in ZONE_B], flush=True)


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Open loop. It knows the fixed start, the bender's geometry and where an ideal bend leaves the tube (all visible); it is
# not told the hinges' anchors or the scorer's state. Numbers marked "measured" come from the dev runs' logs.
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.02
TURN_STEP = math.radians(8.0)
ENV = lambda k, d: float(os.environ.get(k, d))  # noqa: E731
GRIP_R = 0.21  # where the left hand holds the short handle's cushion grip, from the die axis
SWEEP_A = math.radians(ENV("TUBE_SWEEP_A", "50.0"))  # how far the roll support is turned for each bend: 45 + springback + arm lag
SWEEP_B = math.radians(ENV("TUBE_SWEEP_B", "56.0"))
LIFT_CMD = math.radians(-50.0)  # the short handle is swung up about the clevis into its stop (-45 deg)
RESET_R = 0.16  # the hand holds the grip's inner end to turn the lifted handle back (0.21 is out of reach there)
HANDLE_Z = TOP + M.Z_H + 0.002
TUBE_Z = TOP + M.Z_T + float(os.environ.get("TUBE_PINCH_DZ", "0.0055"))  # TCP for a pinch: the tube well up the pads (10 mm squeezed out at the tips, d7)
LIFT = float(os.environ.get("TUBE_LIFT", "0.085"))  # how high the tube is carried above the groove


def _fingers(width):
    return -FINGER_PER_M * width


GRIP_OPEN, GRIP_SHUT = _fingers(0.05), _fingers(M.HANDLE["d"] - 0.008)
PINCH_OPEN, PINCH_SHUT = _fingers(0.030), _fingers(M.TUBE_D - float(os.environ.get("TUBE_SQUEEZE", "0.0030")))
FIST = _fingers(0.0)


def uvw(u, v):
    return np.array(M.uv(u, v)[:2])


U_DIR = np.array((math.cos(M.LONG_YAW), math.sin(M.LONG_YAW)))
V_DIR = np.array((math.cos(M.ALPHA), math.sin(M.ALPHA)))
T0 = uvw(0.0, 0.0)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class _Plan:
    def __init__(self):
        self.wps = {a: [] for a in SIDE}
        self.t = {a: 0.0 for a in SIDE}
        self.last = {}
        for a in SIDE:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None, n=None):
        p0, f0, y0 = self.last[arm]
        p = p0 if pos is None else tuple(float(v) for v in pos)
        f = f0 if finger is None else finger
        y = y0 if yaw is None else yaw
        if n is None:
            n = 1 if label == "away" else max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / SEGMENT), math.ceil(abs(y - y0) / TURN_STEP))
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
    plan.go(arm, dur, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


def _finish(plan):
    end = max(plan.t.values()) + 3.0
    for a in SIDE:
        plan.until(a, end)
    return plan.wps


# the left hand on the short handle
def _hyaw(phi, clevis):
    """The left hand's yaw on the handle: fingers closing across it, in the left wrist's window."""
    return _wrap(M.ALPHA + phi + clevis - math.pi)


def _hpt(phi, clevis, dz=0.0, r=None):
    g = M.grip_point(phi, clevis, GRIP_R if r is None else r)
    return (g[0], g[1], HANDLE_Z + dz)


def _take_handle(plan, phi, clevis, label, r=None):
    a = "left"
    hover = 0.05 if r is None else 0.025  # near the far reach, come in lower
    plan.go(a, 1.8, _hpt(phi, clevis, hover, r), GRIP_OPEN, f"{label}: over the grip", _hyaw(phi, clevis))
    plan.go(a, 0.3, label="settle")
    plan.go(a, 0.8, _hpt(phi, clevis, 0.0, r), label="descend")
    plan.go(a, 0.6, finger=GRIP_SHUT, label="grip")


def _leave_handle(plan, phi, clevis, r=None):
    a = "left"
    plan.go(a, 0.3, label="settle")
    plan.go(a, 0.5, finger=GRIP_OPEN, label="open")
    plan.go(a, 0.8, _hpt(phi, clevis, 0.05 if r is None else 0.025, r), label="up")


def _turn_support(plan, phi0, phi1, dur, label, clevis=0.0, r=None):
    n = max(6, int(abs(phi1 - phi0) / math.radians(2.0)))
    for i in range(1, n + 1):
        p = phi0 + (phi1 - phi0) * i / n
        plan.go("left", dur / n, _hpt(p, clevis, 0.0, r), label=label, yaw=_hyaw(p, clevis), n=1)


def _swing_clevis(plan, phi, c0, c1, dur, label, r=None):
    n = max(6, int(abs(c1 - c0) / math.radians(3.0)))
    for i in range(1, n + 1):
        c = c0 + (c1 - c0) * i / n
        plan.go("left", dur / n, _hpt(phi, c, 0.0, r), label=label, yaw=_hyaw(phi, c), n=1)


# the right hand on the latch and the tube
FINGER_LOW = 0.015  # shut fingertips below the TCP (measured: a press stalled on the latch tail at 15 mm)
LATCH_V = M.TUBE_R + 0.0001  # the latch plate's inner face, from the tube's centre line (v)


def _latch_pt(du, dz, th):
    """A point of the latch (du along u from its pivot, dz up from the tube's centre line, at closed) with it open th:
    it turns about the hinge axis (along v), its pad side rising."""
    return du * math.cos(th) + dz * math.sin(th), -du * math.sin(th) + dz * math.cos(th)


def _latch_push(plan, du, dz, dv, th0, th1, label, off=float(os.environ.get("TUBE_LATCH_OFF", "-0.008")), a="left", back=0.030):
    """The right hand, fingers shut and closing along u, pushes a point of the latch's post sideways (along u) so the
    keeper turns from th0 to th1 about its hinge; the hand follows the point, its pads centred on the point's height."""
    yaw = _wrap(M.LONG_YAW + (math.pi / 2 if a == "right" else -math.pi / 2))  # closing axis along u, in the wrist's window
    sgn = 1.0 if th1 > th0 else -1.0

    def at(th, back=0.0, lift=0.0):
        u, z = _latch_pt(du, dz, th)
        u2, _ = _latch_pt(du, dz, th + sgn * 0.01)
        h = 1.0 if u2 > u else -1.0  # which way the point moves along u
        xy = uvw(M.LATCH["pivot"] + u - h * (off + back), LATCH_V + dv)
        return (xy[0], xy[1], TOP + M.Z_T + z + 0.008 + lift)

    hover = 0.04 if a == "right" else 0.012  # the left hand is near its reach here: come in low
    if a == "left":  # first straight over the post, high, then down: keeps the left arm out of the right arm's space
        hi = at(th0, 0.012, 0.0)
        plan.go(a, 1.5, (0.20, 0.12, TOP + 0.20), FIST, f"{label}: high", yaw)
        plan.go(a, 1.5, (hi[0], hi[1], TOP + 0.115), label=f"{label}: above", n=4)
    plan.go(a, 1.8, at(th0, back, hover), FIST, f"{label}: over", yaw, n=max(1, int(np.linalg.norm(np.subtract(at(th0, back, hover), plan.last[a][0])) / SEGMENT)))
    plan.go(a, 0.8, at(th0, back), label="descend")
    n = 12
    for k in range(0, n + 1):
        plan.go(a, (0.8 if k == 0 else 3.0 / n), at(th0 + (th1 - th0) * k / n), label=label, n=1)
    plan.go(a, 0.3, label="settle")
    plan.go(a, 0.6, at(th1, 0.012, hover), label="away")


POST = (sum(M.LATCH["post"][:2]) / 2 - M.LATCH["pivot"], M.LATCH["post"][4] - 0.002, sum(M.LATCH["post"][2:4]) / 2)  # push point: (du, dz, dv)


def _latch_to(plan, th0, th1, label, a="right"):
    """Push the latch's post from th0 to th1 (it stays where it is put, by its spring washer; below 2 deg it seats and
    clamps the tube)."""
    _latch_push(plan, *POST, th0, th1, label, a=a)


START_FIRM = os.environ.get("TUBE_START_FIRM", "0") == "1"  # the latch pushed firmly at the start (manual step 8)
EASE_OFF = float(os.environ.get("TUBE_EASE_OFF", "-0.008"))
OPEN_OFF = float(os.environ.get("TUBE_OPEN_OFF", "-0.020"))  # the fist aims 2 cm past the post to open the firm latch (d60-61L: aiming 8 mm past, it stalled at 0.3 deg)
LOOSE, FIRM = M.LATCH["loose"], -math.radians(8.0)  # "just enough to hold the tube in place" (MS-13-43 step 4), "firmly" (step 8)
EASED = math.radians(float(os.environ.get("TUBE_EASED", "20.0")))  # where the eased latch ends up (measured, d34L: the push carries it past 8 deg)
OUT = 0.006  # the tube slides 6 mm out from under the groove's upper lip before it lifts
T_DIR = np.array((math.cos(TUBE_START[2]), math.sin(TUBE_START[2])))  # along the groove, toward the roll dies' side


def _tube_frames(stage):
    """Where the tube lies (link frames, table xy, planar) at each stage, from an ideal bend. A half-turn roll about the
    straight part's axis mirrors the shape in the plane: its hinge angles change sign in this planar picture."""
    qa = {j: v for j, v in ideal_q().items() if j in ZONE_A}
    qb = {j: v for j, v in ideal_q().items() if j in ZONE_B}
    a_origin = np.array(TUBE_START[:2])
    s_origin = a_origin + M.MARK_GAP * T_DIR  # slid along the groove by the mark spacing: mark B at the 45 deg mark
    yaw = TUBE_START[2]
    if stage == "start":
        return tube_fk({}, a_origin, yaw)
    if stage == "bent_a":
        return tube_fk(qa, a_origin, yaw)
    if stage == "slid":
        return tube_fk(qa, s_origin, yaw)
    if stage == "rolled":
        return tube_fk({j: -v for j, v in qa.items()}, s_origin, yaw)
    if stage in ("bent_b", "bent_same"):  # bend B wraps the free side (mid, A) round the die; leg B stays where it lay
        sgn = -1 if stage == "bent_b" else 1  # rolled: opposite to bend A in the tube's own frame (an offset)
        f = tube_fk({**{j: sgn * v for j, v in qa.items()}, **{j: -v for j, v in qb.items()}}, (0.0, 0.0), 0.0)
        ref = tube_fk({j: sgn * v for j, v in qa.items()}, s_origin, yaw)["leg_b"]
        lb = f["leg_b"]
        rot = ref[2] - lb[2]
        c, s_ = math.cos(rot), math.sin(rot)
        out = {}
        for n, (x, y, a_) in f.items():
            dx, dy = x - lb[0], y - lb[1]
            out[n] = (ref[0] + c * dx - s_ * dy, ref[1] + s_ * dx + c * dy, a_ + rot)
        return out
    raise ValueError(stage)


def tube_point(frames, x):
    """The tube's centreline point at its own x, and the tangent's heading there."""
    n = next(n for n, x0, x1 in M.tube_links() if x0 <= x <= x1)
    fx, fy, fa = frames[n]
    return np.array((fx + x * math.cos(fa), fy + x * math.sin(fa))), fa


SLIDE_X = -0.120  # the right hand slides the tube by leg B, 12 cm from the middle
SLIDE_EXTRA = float(os.environ.get("TUBE_SLIDE_EXTRA", "0.0015"))  # the hand's lag against the groove's friction (measured: with 4 mm extra the tube slid 74.1 mm and bend B sat 4.8 mm past mark B, d77H; with -0.8 mm it slid 68.7 mm and the roll dies rode over bend B, rev2_e1)
CRANK_X = 0.130  # the left hand rolls the tube by leg A, 2 cm from its end
UNLATCH_ROLL = math.radians(float(os.environ.get("TUBE_UNLATCH_ROLL", "14.7")))
HOLD_A_X = 0.140  # the left hand steadies leg A 1 cm from its end while the right hand re-grips
CARRY_X = float(os.environ.get("TUBE_CARRY_X", "0.03"))  # carried by the middle part, in line with its centre of mass (no roll in the pads)
TAKE_X = float(os.environ.get("TUBE_TAKE_X", "0.06"))  # taken here first, 12 cm from the latch post, while the latch holds it level
REGRIP = os.environ.get("TUBE_REGRIP", "0") == "1"
REGRIP_REST = os.environ.get("TUBE_REGRIP_REST", "0") == "1"
LEFT_BACK = float(os.environ.get("TUBE_LEFT_BACK", "0.030"))  # the finished tube is lifted by its middle part, 2 cm from its centre of mass (5 cm from the die axis, clear of the die and support)


def _pinch_yaw(heading):
    """Fingers closing across the tube, in the right wrist's window (-70..+160 deg)."""
    y = _wrap(heading)
    if y < math.radians(-70.0):
        y += math.pi
    return y - math.pi if y > math.radians(110.0) else y


def _pinch(plan, p, yaw, label, b="right", z=None, shut=None):
    z = TUBE_Z if z is None else z
    plan.go(b, 1.8, (*p, z + 0.05), PINCH_OPEN, f"{label}: over", yaw)
    plan.go(b, 0.3, label="settle")
    plan.go(b, 1.0, (*p, z), label="descend")
    plan.go(b, 0.6, finger=PINCH_SHUT if shut is None else shut, label="pinch")
    plan.go(b, 0.3, label="settle")


def _release(plan, b, lift=0.04):
    p = plan.last[b][0]
    plan.go(b, 0.5, finger=PINCH_OPEN, label="let go")
    plan.go(b, 0.6, (p[0], p[1], p[2] + lift), label="up")


PARK_SUP = math.radians(float(os.environ.get("TUBE_PARK_SUP", "-1.0")))  # where the support goes after bend A (d55-56L: parked at 5.5 deg its roll-die holder caught the arc in the slide)
BACK_OVER = math.radians(float(os.environ.get("TUBE_BACK_OVER", "1.0")))  # the hand lags the support's friction: d47-50L stopped at +3.5 deg for 0
EASE = math.radians(float(os.environ.get("TUBE_EASE", "2.0")))  # after a bend the hand stops pushing (about the arm's lag); dies still down


def _roll_point(p3, psi):
    """A point turned psi about the straight part's axis (the groove line through the tangent point)."""
    a = np.array((*T_DIR, 0.0))
    o = np.array((*T0, M.Z_T))
    r = np.asarray(p3, float) - o
    return o + r * math.cos(psi) + np.cross(a, r) * math.sin(psi) + a * (a @ r) * (1 - math.cos(psi))


ROLL_SIGN = 1.0  # +: the bent leg (on the die side) rises first and goes over the top
ROLL_END = math.radians(float(os.environ.get("TUBE_ROLL_END", "184")))  # past 180: the leg lags the hand (d47L: 174 for 180; d48L: 178 for 184)


ROLL0 = 0.0  # the left hand holds leg A level through the slide, so the roll starts from flat


def _crank_path(n=24, stage="slid", psi0=None, psi1=None):
    """Leg A's crank point and the wrist yaw as the tube turns about its straight part from psi0 to psi1 (default: from
    flat, half a turn over the top)."""
    psi0 = ROLL0 if psi0 is None else psi0
    psi1 = ROLL_SIGN * ROLL_END if psi1 is None else psi1
    f = _tube_frames(stage)
    p, _ = tube_point(f, CRANK_X)
    p2, _ = tube_point(f, CRANK_X + 0.01)
    shift = np.zeros(3) if stage == "slid" else np.array((*(-M.MARK_GAP * T_DIR), 0.0))  # the roll axis point T0 is fixed
    pts, yaws, yprev = [], [], None
    for k in range(n + 1):
        psi = psi0 + (psi1 - psi0) * k / n
        g = _roll_point(np.array((*p, M.Z_T)) - shift, psi) + shift
        d = _roll_point(np.array((*p2, M.Z_T)) - shift, psi) + shift - g
        y = math.atan2(d[1], d[0])
        y = y if yprev is None else yprev + _wrap(y - yprev)  # continuous heading
        yprev = y
        pts.append(g)
        yaws.append(y)
    lo, hi = -math.radians(160.0), math.radians(70.0)  # the left wrist's window; one offset of k*180 deg for the whole path
    offs = [m * math.pi for m in range(-3, 4) if lo <= min(yaws) + m * math.pi and max(yaws) + m * math.pi <= hi]
    m = min(offs, key=abs) if offs else 0.0
    return pts, [y + m for y in yaws]


def _droop():
    """How far the tube turns about its straight part once the latch is eased, the bent leg's weight turning it until
    leg A rests (measured, d39c: the crank point drops 11.3 mm)."""
    f = _tube_frames("bent_a")
    p, _ = tube_point(f, CRANK_X)
    r = abs(float((np.asarray(p) - np.asarray(T0)) @ np.asarray(V_DIR)))
    return -ROLL_SIGN * math.asin(DROOP_DZ / r)


DROOP_DZ = float(os.environ.get("TUBE_DROOP_DZ", "0.0113"))


CRANK_SQUEEZE = float(os.environ.get("TUBE_CRANK_SQUEEZE", "0.0008"))  # a light pinch: the leg turns about its own axis in the pads as it goes round
CRANK_DZ = float(os.environ.get("TUBE_PINCH_DZ", "0.0055"))


def _crank_take(plan):
    pts, yaws = _crank_path()
    _pinch(plan, pts[0][:2], yaws[0], "crank: take leg A", b="left", z=TOP + pts[0][2] + CRANK_DZ, shut=_fingers(M.TUBE_D - 0.001))


ROLL_STEP = math.radians(float(os.environ.get("TUBE_ROLL_STEP", "30")))  # a top-down pinch follows the leg's tilt only so far


def _crank_turn(plan, dur=3.0):
    """The left hand turns the tube half a turn about its straight part by leg A, over the top, in steps: it turns the
    leg ROLL_STEP, lets go (the latch, resting on the tube, holds it where it is), and takes it again where it lies."""
    a = "left"
    nstep = max(1, round(math.pi / ROLL_STEP))
    edges = [ROLL0 + (ROLL_SIGN * ROLL_END - ROLL0) * k / nstep for k in range(nstep + 1)]
    for k in range(nstep):
        pts, yaws = _crank_path(6, "slid", edges[k], edges[k + 1])
        if k:
            _pinch(plan, pts[0][:2], yaws[0], "take leg A again", b=a, z=TOP + pts[0][2] + CRANK_DZ, shut=PINCH_SHUT)
        for i in range(1, len(pts)):
            plan.go(a, dur / (len(pts) - 1), (pts[i][0], pts[i][1], TOP + pts[i][2] + CRANK_DZ), label="turn it", yaw=yaws[i], n=1)
        plan.go(a, 0.3, label="settle")
        if k < nstep - 1:
            _release(plan, a, 0.03)
    plan.go(a, 0.3, label="hold")


def demo():
    plan = _Plan()
    for arm in SIDE:
        _park(plan, arm)
    if DEV:
        plan.go("left", 50.0, label="wait")
        return _finish(plan)
    a, b = "left", "right"
    UP = M.CLEVIS_UP
    # 1. bend A: pull the short handle round; the roll dies wrap the tube round the die
    _take_handle(plan, 0.0, 0.0, "bend A")
    _turn_support(plan, 0.0, SWEEP_A, 16.0, "bend A")
    plan.go(a, 0.8, label="hold")
    phi = SWEEP_A
    if os.environ.get("TUBE_DEMO") == "none":
        _leave_handle(plan, phi, 0.0)
        _park(plan, a)
        return _finish(plan)
    # 2. stop pushing, swing the short handle up and away, turn the roll support back to 0 with it up (latch still firm)
    _turn_support(plan, phi, phi - EASE, 1.0, "ease off")
    phi -= EASE
    _swing_clevis(plan, phi, 0.0, LIFT_CMD, 4.0, "handle up")
    _leave_handle(plan, phi, UP)
    _take_handle(plan, phi, UP, "support back", RESET_R)
    _turn_support(plan, phi, PARK_SUP, 8.0, "support back", UP, RESET_R)  # near 0, its roll-die holder clear of the bent arc as it slides
    _leave_handle(plan, PARK_SUP, UP, RESET_R)
    _park(plan, a)
    plan.until(b, plan.t[a])
    if os.environ.get("TUBE_DEMO") == "one":  # wrong way: one bend, the support back to 0, hands clear
        _turn_support(plan, phi, phi - EASE, 1.0, "ease off")
        phi -= EASE
        _swing_clevis(plan, phi, 0.0, LIFT_CMD, 4.0, "handle up")
        _leave_handle(plan, phi, UP)
        _take_handle(plan, phi, UP, "support back", RESET_R)
        _turn_support(plan, phi, -BACK_OVER, 8.0, "support back to 0", UP, RESET_R)
        _leave_handle(plan, -BACK_OVER, UP, RESET_R)
        _park(plan, a)
        return _finish(plan)
    # 3. ease the latch to "just enough to hold the tube", slide the tube along the groove by the mark spacing
    f = _tube_frames("bent_a")
    p, hd = tube_point(f, SLIDE_X)
    y = _pinch_yaw(hd)
    same = os.environ.get("TUBE_DEMO") == "same"  # wrong way: slid to mark B but not rolled: the second bend goes the same way
    # 3a. ease the latch (seated, its lip presses the tube into the groove and the slide would drag the bent arc open);
    # the bent leg's weight turns the round tube a little until leg A rests; the left hand takes leg A where it lies
    # and lifts it level again (the right hand can't stop a round tube rolling)
    _latch_push(plan, *POST, 0.0, LOOSE + math.radians(1.0), "ease the latch", off=EASE_OFF, a="right")
    _park(plan, b)
    plan.until(a, plan.t[b])
    za = TOP + M.Z_T + CRANK_DZ
    dpts, dyaws = _crank_path(8, "bent_a", _droop(), 0.0)
    pa = dpts[-1][:2]
    _pinch(plan, dpts[0][:2], dyaws[0], "take leg A", b=a, z=TOP + dpts[0][2] + CRANK_DZ, shut=PINCH_SHUT)
    for k in range(1, len(dpts)):
        plan.go(a, 2.0 / (len(dpts) - 1), (dpts[k][0], dpts[k][1], TOP + dpts[k][2] + CRANK_DZ), label="lift leg A level", yaw=dyaws[k], n=1)
    plan.until(b, plan.t[a])
    _pinch(plan, p, y, "take leg B")
    plan.until(a, plan.t[b])
    # 3b. both hands slide the tube along the groove by the mark spacing (mark B to the 45 deg mark)
    d = (M.MARK_GAP + SLIDE_EXTRA) * T_DIR
    plan.go(b, 5.0, (*(p + d), TUBE_Z), label="slide it to mark B", n=8)
    plan.go(a, 5.0, (*(pa + d), za), label="slide it to mark B", n=8)
    plan.go(b, 0.3, label="settle")
    plan.go(a, 0.3, label="settle")
    _release(plan, b)
    # 3c. the latch back down onto the tube, "just enough to hold the tube in place" (step 4): it rests on it by its
    # spring and holds the tube where it is put while the left hand turns it
    _latch_to(plan, EASED, math.radians(4.0), "latch down onto the tube")
    _park(plan, b)
    plan.until(a, plan.t[b])
    if not same:
        # 4. the left hand rolls the tube half a turn about its straight part by the bent leg, over the top
        _crank_turn(plan)
    else:
        plan.go(a, 0.3, label="hold")
    plan.until(b, plan.t[a])
    # 5. latch firm (the left hand still holds leg A, 15 cm away), then let go of leg A
    _latch_to(plan, 0.0, FIRM, "push the latch firmly")
    _park(plan, b)
    plan.until(a, plan.t[b])
    _release(plan, a, 0.03)
    # 6. the support the last few degrees to 0, lower the handle onto the tube, take the grip, bend B
    if PARK_SUP > 0:
        _take_handle(plan, PARK_SUP, UP, "support to 0", RESET_R)
        _turn_support(plan, PARK_SUP, -BACK_OVER, 2.0, "support to 0", UP, RESET_R)
        plan.go(a, 0.5, label="settle")
    else:
        _take_handle(plan, 0.0, UP, "handle down", RESET_R)
    _swing_clevis(plan, 0.0, UP, math.radians(3.0), 3.0, "handle down", RESET_R)
    _leave_handle(plan, 0.0, 0.0, RESET_R)
    _take_handle(plan, 0.0, 0.0, "bend B")
    _turn_support(plan, 0.0, SWEEP_B, 16.0, "bend B")
    plan.go(a, 0.8, label="hold")
    phi = SWEEP_B
    _turn_support(plan, phi, phi - EASE, 1.0, "ease off")
    phi -= EASE
    _swing_clevis(plan, phi, 0.0, LIFT_CMD, 4.0, "handle up")
    _leave_handle(plan, phi, UP)
    # the raised handle and roll support back round to the start (the 0 mark), then both hands clear
    _take_handle(plan, phi, UP, "support back", RESET_R)
    _turn_support(plan, phi, -BACK_OVER, 8.0, "support back to 0", UP, RESET_R)
    _leave_handle(plan, -BACK_OVER, UP, RESET_R)
    _park(plan, a)
    plan.until(b, plan.t[a])
    return _finish(plan)


def _take_out_and_lay(plan, stage):
    """Open the latch, lift the tube out by its middle and lay it on the template with its ends along the lanes, then
    let go."""
    a, b = "left", "right"
    f = _tube_frames(stage)
    x = TAKE_X if stage != "bent_a" else SLIDE_X
    p, hd = tube_point(f, x)
    y0 = _pinch_yaw(hd)
    # the right hand takes the tube while the latch still holds it level (12 cm from the latch post), the left hand
    # opens the latch; then the left steadies leg A while the right takes the tube again nearer its centre of mass
    _pinch(plan, p, y0, "take the tube")
    plan.until(a, plan.t[b])
    _latch_push(plan, *POST, 0.0, M.LATCH["open"], "open the latch", off=OPEN_OFF, a="left", back=LEFT_BACK)
    if REGRIP_REST and stage == "bent_b":  # take it again on the line of its centre of mass, where it rests once unlatched
        plan.until(b, plan.t[a])
        _release(plan, b, 0.03)
        x = CARRY_X
        p, hd = tube_point(f, x)
        g = min((_roll_point((*p, M.Z_T), sg * UNLATCH_ROLL) for sg in (1.0, -1.0)), key=lambda c: c[2])
        p = g[:2]
        y0 = _pinch_yaw(hd)
        _pinch(plan, p, y0, "take it again by its middle", z=TOP + g[2] + 0.0055)
        plan.go(b, 1.0, (*p, TUBE_Z), label="up to the groove's height")
    if REGRIP and stage == "bent_b":
        _park(plan, a)  # round by its park pose, clear of the right arm (straight across it hit the right arm: d71L)
        ha, hha = tube_point(f, HOLD_A_X)
        ya = _pinch_yaw(hha)
        ya = ya if ya <= math.radians(70.0) else ya - math.pi
        _pinch(plan, ha, ya, "steady leg A", b=a)
        plan.until(b, plan.t[a])
        _release(plan, b, 0.03)
        x = CARRY_X
        p, hd = tube_point(f, x)
        y0 = _pinch_yaw(hd)
        _pinch(plan, p, y0, "take it nearer its middle")
        plan.until(a, plan.t[b])
        _release(plan, a, 0.04)
    _park(plan, a)
    plan.until(b, plan.t[a])
    plan.go(b, 1.5, (*(p + OUT * V_DIR), TUBE_Z), label="out from under the groove's lip")
    p = p + OUT * V_DIR
    plan.go(b, 4.0, (*p, TUBE_Z + LIFT), label="lift it out")
    lay_shape = {k: v for k, v in ideal_q().items() if stage != "bent_a" or k in ZONE_A}
    opts = []
    for lay_yaw in (LAID_YAW, LAID_YAW + math.pi):  # either way round lies on the lanes; take the easier wrist turn
        q_, hq = tube_point(tube_fk(lay_shape, LANE_AT, lay_yaw), x)
        y_ = y0 + _wrap(hq - hd)
        if math.radians(-70.0) <= y_ <= math.radians(160.0):
            opts.append((abs(y_ - math.radians(20.0)), q_, y_))
    _, q, y1 = min(opts, key=lambda o: o[0])
    drop = TOP + M.TUBE_R + 0.010  # tips stay above the table; the tube drops a few mm
    plan.go(b, 4.0, (*q, TUBE_Z + LIFT), label="to the template", yaw=y0)  # carried clear of the bender first
    plan.go(b, 3.0, (*q, TUBE_Z + LIFT), label="turn it to the lanes", yaw=y1, n=4)
    plan.go(b, 1.5, (*q, drop), label="lower")
    plan.go(b, 0.5, label="settle")
    _release(plan, b, 0.06)
    _park(plan, b)
    return _finish(plan)


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


reset_latch()

if __name__ == "__main__":
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad = ik_errors(wps)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:12], len(bad))
    print("tube start", np.round(TUBE_START, 4), "lanes", np.round(LANES, 3).tolist())
