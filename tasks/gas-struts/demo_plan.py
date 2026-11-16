"""The scripted demo for task.py (the strut replacement), and its wrong-way variants (GS_DEMO=...).

With the lid shut, the left arm unclips the tagged strut's lower socket (fingers round the clip's two finger loops,
pushes them outboard: the clip opens and the socket comes off its ball), so the strut hangs from its upper stud; then
unclips the upper socket the same way: the tagged strut drops on the table beside the enclosure. The right arm lifts the lid by its handle to
the open stop and holds it there (one good strut cannot); the left arm picks the new strut off the battens and fits
it (rod-end socket onto the body stud, then the body-end socket onto the lid stud) and moves clear. The right arm lets
go: the lid holds itself open. Then the right arm takes the handle again, pushes the lid down past its balance point to
22 degrees and lets go: it falls shut.
Waypoints only: every motion is the arms' own, through IK, with real contact; positions of parts are read from the
simulator just before each grip (a scripted demo may do this) and the waypoints rewritten.

  noclose    as the demo, but the lid is never closed (expected 0.60)
  none       the lid is lifted, held and let go; nothing is replaced (expected 0)
  wrongside  the right arm removes the GOOD (right) strut and drops it; the tagged strut stays on (expected 0)
"""

from __future__ import annotations

import math

import numpy as np
import yam_arena.demo as _demo
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.urdf import axis_angle_matrix

import gsparts as G


class Grip(float):
    """A Waypoint yaw that carries a full TCP rotation (approach = column 2)."""

    def __new__(cls, R):
        obj = super().__new__(cls, 0.0)
        obj.R = np.asarray(R, float)
        return obj


_plain = _demo.grip_rotation


def _grip_rotation(yaw: float = 0.0):
    R = getattr(yaw, "R", None)
    return _plain(float(yaw)) if R is None else R


_demo.grip_rotation = _grip_rotation
Y, Z = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
DOWN_Y, DOWN_X = _plain(0.0), _plain(math.pi / 2)   # straight down, closing along world y / world x
assert np.allclose(DOWN_Y[:, 2], [0, 0, -1], atol=1e-6)

PAD_IN = 0.006          # the pads' centre beyond the TCP along the approach
GRIP_TUBE = 0.0         # fingers shut: squeezes the 10 mm body about 22 N (contact near -4.4 mm; r1 slipped at -3 mm)
GRIP_BAR = -0.002       # the 12 mm handle bar
LOOSE = -0.012          # open just enough to let go (the inner finger stays clear of the wall)
_b, _l = G.stud_points(1)
PITCH = math.degrees(math.atan2(_l[2] - _b[2], _b[0] - _l[0]))  # the fitted strut rises this much toward the robot (36 deg)
X_GRIP = 0.060          # grip point along the strut from the rod end: the middle of the 10 mm body
HOLD_DEG, PRESS_DEG = 45.0, 49.0  # the lid's stop, and the angle commanded to keep it pressed against the stop
TAIL_S = 15.0
CLOSE_DEG = 22.0       # pushed down to this (below the 28 deg balance point), then let go


ROLL = float(__import__("os").environ.get("GS_ROLL", "28"))  # deg: hand rolled about the strut, palm outboard


def roll(side):
    """The hand's roll about the strut's axis (its local y), palm outboard: the gripper body then clears the open
    lid above the strut (CPU: 0 mm unrolled, about 20 mm at 20 to 30 deg). The strut itself is not rolled: its socket
    mouths still face the studs."""
    R = axis_angle_matrix(Y, math.radians(ROLL))
    Rm = axis_angle_matrix(Y, -math.radians(ROLL))
    a = (DOWN_Y @ R)[:, 2]
    return R if a[1] * -side > 0 else Rm


def rot_y(deg):
    return axis_angle_matrix(Y, math.radians(deg))


def rot_z(rad):
    return axis_angle_matrix(Z, rad)


def tcp_for(point, R):
    return tuple(np.asarray(point, float) - R[:, 2] * PAD_IN)


def bar_pose(T, y, deg):
    """TCP and rotation on the handle bar at y, the lid open deg; the left arm (y > 0) reaches it with the fingers
    closing along -x, the right arm along +x (each arm's own IK branch)."""
    th = math.radians(deg)
    x, z = G.lid_point(0.18, -0.030, th)
    R = rot_y(-deg) @ (_plain(-math.pi / 2) if y > 0 else DOWN_X)
    return tcp_for((x, y, T.TOP + z), R), Grip(R)


def strut_pose(T, side, out, skew=0.0):
    """TCP and rotation gripping strut ``side`` at X_GRIP, the strut in its fitted line moved ``out`` m outboard and
    turned ``skew`` rad about the vertical through the grip (rod end toward the wall for skew > 0)."""
    b, l = G.stud_points(side, math.radians(HOLD_DEG))
    t = (l - b) / np.linalg.norm(l - b)
    p = b + X_GRIP * t + np.array([0.0, side * out, 0.0])
    R = rot_z(-side * skew) @ rot_y(PITCH) @ DOWN_Y @ roll(side)
    return tcp_for(p, R), Grip(R)


def lid_arc(T, y, label, deg_from=0.0, deg_to=PRESS_DEG, n=7, dur=0.55):
    out = []
    for k in range(1, n + 1):
        d = deg_from + (deg_to - deg_from) * k / n
        pos, R = bar_pose(T, y, d)
        out.append(Waypoint(dur, pos, GRIP_BAR, f"{label} {d:.0f} deg", R))
    return out


# -- aiming from the scene ----------------------------------------------------------------------------------------
# WHY measured aim: a round tube between flat pads turns freely about the closing axis (PhysX gives a line contact no
# torsional grip), so the strut's angle in the hand is not what the plan commands (dev5: 15 deg flatter, sockets 13 and
# 23 mm off their balls). The demo therefore reads where each socket really is just before pushing it (a scripted demo
# may read simulator state) and corrects the arm's target; after the rod-end socket is on, the strut's line is set by
# that ball and the grip point alone, so the tube end is swung onto its ball about the lower ball.
MEASURES: list = []  # filled by plan(); applied by task.py's hook at the given times


def _hang(v):
    return math.atan2(v[1], v[0])


PITCH_T = float(__import__("os").environ.get("GS_PITCH_T", str(PITCH)))  # the hand's pitch for the tube end (the strut keeps its angle in the hand)


def _tube_rot(v, side):
    """Closing axis square to the strut's horizontal line v (yawed from the nominal -x), pitched PITCH_T, rolled."""
    return rot_z(math.remainder(_hang(v) - math.pi, 2 * math.pi)) @ rot_y(PITCH_T) @ DOWN_Y @ roll(side)


def _aimed(dur, finger, label, kind, side, S, R):
    w = Waypoint(dur, tcp_for(S, R), finger, label, Grip(R))
    w.aim = dict(kind=kind, side=side, S=np.asarray(S, float), R=np.asarray(R, float))
    return w


def _pinch_rot(Yw, a_pref):
    """Closing along the stud axis Yw (across the loop's faces), approach the unit vector square to Yw nearest a_pref."""
    a = np.asarray(a_pref, float); a = a - Yw * (Yw @ a); a /= np.linalg.norm(a)
    c = Yw if Yw[1] <= 0 else -Yw      # the fingers are alike: pick the closing sign inside both wrists' yaw windows
    return np.column_stack([c, np.cross(a, c), a])


def apply_loop(m, socks):
    """Rewrite a release sequence from the measured socket: its centre c, rotation R, the clip loop's middle, the clip's
    opening direction dW and the stud axis Yw (mouth direction)."""
    c, R, jaw_mm, loop, dW, Yw = socks[m["strut"]][m["end"]]
    Rp = _pinch_rot(np.asarray(Yw, float), m["a_pref"])
    loop = np.asarray(loop, float)
    pts = {"above": loop - Rp[:, 2] * 0.06, "at": loop, "pinch": loop, "open": loop + dW * OPEN_M,
           "off": loop + dW * OPEN_M - np.asarray(Yw) * OFF_M}
    for w in m["wps"]:
        if w.loop in pts:
            w.pos = tcp_for(pts[w.loop], Rp)
        w.yaw = Grip(Rp)
    m["state"] = dict(c=np.round(c, 4).tolist(), jaw=round(jaw_mm, 2))
    m["done"] = True


def apply_measure(m, sockets):
    """Rewrite the aimed waypoints after measure ``m`` from the measured socket centres (world)."""
    if m["kind"] == "loop":
        return apply_loop(m, sockets)
    if "strut_new" in sockets:     # v2: the new strut is fitted on the left
        sockets = {("L", "body"): sockets["strut_new"]["body"][0], ("L", "lid"): sockets["strut_new"]["lid"][0]}
    side = m["side"]
    c_r = np.asarray(sockets[("R" if side < 0 else "L", "body")], float)
    c_t = np.asarray(sockets[("R" if side < 0 else "L", "lid")], float)
    if m.get("from") is not None:               # the pose the arm was last sent to, as rewritten by earlier measures
        R_m = m["from"].yaw.R
        P = np.asarray(m["from"].pos, float) + R_m[:, 2] * PAD_IN
    else:
        P, R_m = m["P"], m["R"]
    if m["kind"] == "body":
        o = c_r - P                                   # pad centre -> rod-end socket, as actually held
        xg = float(np.linalg.norm(o)); u = -o / xg    # the strut's real direction in the hand (toward the tube end)
        m["state"].update(xg=xg)
        for w in m["wps"]:
            if w.aim["kind"] == "body":
                w.pos = tcp_for(w.aim["S"] - o, w.aim["R"])
            else:  # the swing: grip point on the line from the (soon snapped) ball toward the tube-end target
                b = m["ball"]
                v = (w.aim["S"] - b) / np.linalg.norm(w.aim["S"] - b)
                Rw = _tube_rot(v, side)
                w.pos, w.yaw = tcp_for(b + xg * v, Rw), Grip(Rw)
    else:
        u = (c_t - c_r) / np.linalg.norm(c_t - c_r)
        xg = float((P - c_r) @ u)
        perp = P - (c_r + xg * u)                     # the pads' offset from the strut's axis (sag, grip)
        for w in m["wps"]:
            v = (w.aim["S"] - c_r) / np.linalg.norm(w.aim["S"] - c_r)
            Rw = _tube_rot(v, side)
            w.pos, w.yaw = tcp_for(c_r + xg * v + perp, Rw), Grip(Rw)
    m["done"] = True


def fit_strut(T, side, variant="", t0=0.0):
    """Waypoints for one strut by the arm on its side, starting at absolute time t0; registers its aim measures."""
    s = side
    x0, y0, top = T.START_X + G.STRUT_LEN / 2 - X_GRIP, s * T.START_Y, T.TOP   # over the middle of the body
    b, l = G.stud_points(s, math.radians(HOLD_DEG))
    t_nom = (l - b) / np.linalg.norm(l - b)
    yv = np.array([0.0, 1.0, 0.0])
    Rp = DOWN_Y @ roll(s)
    ap = Rp[:, 2]
    # pads centred on the body's axis (r2: 4 mm shallower, the body squirted out past the fingertips)
    pick = tuple(np.array([x0, y0, top + T.AXIS_Z]) - ap * PAD_IN)
    above = tuple(np.array(pick) - ap * 0.08)
    wp = [Waypoint(1.5, above, FINGER_OPEN, "over the strut, rolled", Grip(Rp)),
          Waypoint(1.3, pick, FINGER_OPEN, "down to its body", Grip(Rp)),
          Waypoint(0.8, pick, GRIP_TUBE, "grip the body", Grip(Rp)),
          Waypoint(1.0, (x0, y0, top + 0.10), GRIP_TUBE, "lift", Grip(Rp))]
    R0 = rot_y(PITCH) @ DOWN_Y @ roll(s)
    pre_pad = b + X_GRIP * t_nom + s * 0.040 * yv
    # WHY outboard first, in front of the enclosure: a straight line from the pick to beside the studs cuts through the
    # lid's side flange (dev3: the gripper and strut hit it and knocked the held lid out of the other hand)
    wp += [Waypoint(1.2, (x0, s * 0.29, top + 0.10), GRIP_TUBE, "outboard, in front of the enclosure", Grip(Rp)),
           Waypoint(1.5, tuple(np.asarray(tcp_for(pre_pad, R0)) + [-0.08, 0, 0.02]), GRIP_TUBE,
                    "forward beside the enclosure, pitching", Grip(R0)),
           Waypoint(1.0, tcp_for(pre_pad, R0), GRIP_TUBE, "beside the studs, pitched", Grip(R0)),
           Waypoint(0.6, tcp_for(pre_pad, R0), GRIP_TUBE, "steady: measure the rod-end socket", Grip(R0))]
    mA = dict(t=t0 + sum(w.duration for w in wp), side=s, kind="body", P=pre_pad, R=R0, ball=b, state={}, wps=[])
    body = [_aimed(1.2, GRIP_TUBE, "rod-end socket 10 mm out from its ball", "body", s, b + s * 0.010 * yv, R0),
            _aimed(1.5, GRIP_TUBE, "push the rod-end socket onto the body stud", "body", s, b - s * 0.008 * yv, R0),
            _aimed(0.6, GRIP_TUBE, "hold", "body", s, b - s * 0.008 * yv, R0)]
    out_t = l + s * 0.020 * yv
    swing = _aimed(1.8, GRIP_TUBE, "swing the tube end up beside its ball, 20 mm out", "tube", s, out_t, R0)
    settle = _aimed(0.6, GRIP_TUBE, "steady: measure the tube-end socket", "tube", s, out_t, R0)
    mA["wps"] = body + [swing, settle]
    wp += body + [swing, settle]
    mB = dict(t=t0 + sum(w.duration for w in wp), side=s, kind="tube", P=None, R=None, wps=[])
    aim = _aimed(0.8, GRIP_TUBE, "aim: tube-end socket 20 mm out from its ball", "tube", s, out_t, R0)
    aim2 = _aimed(0.5, GRIP_TUBE, "steady: measure again", "tube", s, out_t, R0)
    wp += [aim, aim2]
    mC = dict(t=t0 + sum(w.duration for w in wp), side=s, kind="tube", P=None, R=None, wps=[])
    if variant == "half":
        S = l + s * 0.008 * yv   # WHY 8 mm short: aimed 4 mm short it still snapped on (w_half, 8 mm ball)
        push = [_aimed(1.5, GRIP_TUBE, "tube-end socket only part way on (not snapped)", "tube", s, S, R0),
                _aimed(0.6, LOOSE, "let go", "tube", s, S, R0)]
    else:
        S = l - s * 0.015 * yv
        push = [_aimed(1.5, GRIP_TUBE, "push the tube-end socket onto the lid stud", "tube", s, S, R0),
                _aimed(0.6, GRIP_TUBE, "hold", "tube", s, S, R0),
                _aimed(0.6, LOOSE, "let go", "tube", s, S, R0)]
    wp += push
    mB["wps"] = [aim, aim2] + push
    mC["wps"] = push
    out_w = _aimed(1.0, LOOSE, "out, away from the strut", "tube", s, S + s * 0.045 * yv, R0)
    wp += [out_w]
    mB["wps"].append(out_w); mC["wps"].append(out_w)
    wp += [Waypoint(1.2, (b[0] - 0.06, s * 0.28, top + 0.12), LOOSE, "up, straight down", 0.0)]
    # B and C measure with the pose the arm was sent to (its pad centre and rotation, as rewritten by A or B)
    mB["from"], mC["from"] = settle, aim2
    MEASURES.extend([mA, mB, mC])
    return wp, R0


OPEN_M, OFF_M = 0.004, 0.018   # push the clip loops 4 mm outboard (the jaws open), then 18 mm more (the socket comes off)
GRIP_TAB = -0.003              # fingers 6 mm apart round the 2.5 mm loops: the inner finger pushes, nothing squeezes
TAB_OPEN = -0.009              # fingers 18 mm apart (fully open, the inner finger would be in the wall)


def _lw(dur, finger, label, role):
    w = Waypoint(dur, (0.2, 0.2, 0.9), finger, label)
    w.loop = role
    return w


def release_end(strut, end, a_pref, t0, side):
    """Waypoints that put the fingers round the two clip loops at ``end`` of ``strut`` and push them outboard: the
    jaws open, then the socket slides off its ball. Positions are filled in by a measure at t0."""
    wps = [_lw(1.5, TAB_OPEN, f"over the {end}-end clip loops", "above"),
           _lw(1.0, TAB_OPEN, "down round the loops", "at"),
           _lw(0.6, GRIP_TAB, "fingers in close", "pinch"),
           _lw(1.0, GRIP_TAB, "push the loops outboard: the clip opens", "open"),
           _lw(1.2, GRIP_TAB, "push on: the socket comes off its ball", "off")]
    MEASURES.append(dict(t=t0, kind="loop", strut=strut, end=end, a_pref=a_pref, side=side, wps=wps, state={}))
    return wps


def carry(dur, pos, finger, label, m_wps):
    """A waypoint that keeps the hand rotation set by the last measure (absolute TCP position)."""
    w = _lw(dur, finger, label, "carry")
    w.pos = tuple(pos)
    m_wps.append(w)
    return w


def remove_strut(T, strut, side, t0):
    """With the lid shut: unclip the lower (rod-end) socket (the strut then hangs from its upper stud, its lower end
    on the table), then unclip the upper socket: the strut drops beside the enclosure. Returns waypoints."""
    s, top = side, T.TOP
    wp = [Waypoint(1.5, (0.30, s * 0.30, top + 0.15), FINGER_OPEN, "up beside the enclosure", 0.0),
          Waypoint(0.5, (0.30, s * 0.30, top + 0.15), TAB_OPEN, "steady: measure the lower socket", 0.0)]
    wp += release_end(strut, "body", (0.0, 0.0, -1.0), t0 + sum(w.duration for w in wp), s)
    m = MEASURES[-1]["wps"]
    wp += [carry(1.0, (0.31, s * 0.28, top + 0.13), TAB_OPEN, "up", m),
           carry(1.5, (0.24, s * 0.28, top + 0.13), TAB_OPEN, "steady: measure the upper socket", m)]
    wp += release_end(strut, "lid", (0.0, 0.0, -1.0), t0 + sum(w.duration for w in wp), s)
    m = MEASURES[-1]["wps"]
    wp += [carry(1.0, (0.22, s * 0.30, top + 0.15), TAB_OPEN, "up: the strut has dropped", m),
           Waypoint(1.0, (0.14, s * 0.30, top + 0.17), FINGER_OPEN, "up, straight down", 0.0)]
    return wp


def plan(T, variant: str = ""):
    MEASURES.clear()
    top = T.TOP
    # WHY far out and back: at (0.16, +-0.27) the parked grippers hid the fitted struts from the top camera (r4, v1)
    CLEAR_L, CLEAR_R = (0.08, 0.38, top + 0.15), (0.08, -0.38, top + 0.15)
    globals().update(CLEAR_L=CLEAR_L, CLEAR_R=CLEAR_R)
    yR = -0.12
    # left arm: remove the tagged strut with the lid shut (or, wrongside: the right arm removes the good one)
    left = [Waypoint(0.5, CLEAR_L, FINGER_OPEN, "start")]
    right = [Waypoint(0.5, CLEAR_R, FINGER_OPEN, "start")]
    if variant == "wrongside":
        right += remove_strut(T, "strut_R", -1, 0.5)
        right += [Waypoint(1.5, CLEAR_R, FINGER_OPEN, "well clear")]
        left += [Waypoint(sum(w.duration for w in right) - 0.5, CLEAR_L, FINGER_OPEN, "stay clear")]
        t_rem = sum(w.duration for w in right)
    elif variant == "none":
        t_rem = 0.5
    else:
        left += remove_strut(T, "strut_old", 1, 0.5)
        t_rem = sum(w.duration for w in left)
    # right arm (left arm in no variant): lift the lid by its handle to the stop and hold it there
    bar0, Rb0 = bar_pose(T, yR, 0.0)
    lifter = right if variant != "wrongside" else right
    lifter += [Waypoint(max(0.5, t_rem - sum(w.duration for w in lifter)), (bar0[0], bar0[1], bar0[2] + 0.05),
                        FINGER_OPEN, "over the handle", Rb0),
               Waypoint(1.2, bar0, FINGER_OPEN, "down to the bar", Rb0),
               Waypoint(0.8, bar0, GRIP_BAR, "grip the handle", Rb0)]
    lifter += lid_arc(T, yR, "lift the lid")
    t_open = sum(w.duration for w in lifter)
    hold_pos, Rh = bar_pose(T, yR, PRESS_DEG)
    if variant in ("", "noclose"):
        left += [Waypoint(max(0.5, t_open - sum(w.duration for w in left)), (0.14, 0.30, top + 0.17), FINGER_OPEN,
                          "wait for the lid")]
        fl, _ = fit_strut(T, 1, "", sum(w.duration for w in left))
        left += fl
    left += [Waypoint(1.5, (CLEAR_L[0] + 0.06, CLEAR_L[1], CLEAR_L[2]), FINGER_OPEN, "clear"),
             Waypoint(1.5, CLEAR_L, FINGER_OPEN, "well clear")]
    t_done = max(sum(w.duration for w in left), t_open + 2.0)
    right += [Waypoint(t_done - sum(w.duration for w in right), hold_pos, GRIP_BAR, "hold the lid at its stop", Rh),
              Waypoint(0.6, hold_pos, LOOSE, "let go of the lid", Rh),
              Waypoint(1.0, tuple(np.asarray(hold_pos) - Rh.R[:, 2] * 0.06), LOOSE, "back off", Rh),
              Waypoint(1.5, (CLEAR_R[0] + 0.06, CLEAR_R[1], CLEAR_R[2]), FINGER_OPEN, "clear"),
              Waypoint(1.5, CLEAR_R, FINGER_OPEN, "well clear"),
              Waypoint(5.0, CLEAR_R, FINGER_OPEN, "wait: the lid holds itself open")]
    if variant == "":
        # close: take the handle at the stop, push the lid down past its balance point to 22 deg, let go
        rb45, Rr45 = bar_pose(T, yR, HOLD_DEG)
        right += [Waypoint(2.0, tuple(np.asarray(rb45) - Rr45.R[:, 2] * 0.05), FINGER_OPEN, "toward the handle", Rr45),
                  Waypoint(1.0, rb45, FINGER_OPEN, "onto the bar", Rr45),
                  Waypoint(0.8, rb45, GRIP_BAR, "take the handle", Rr45)]
        right += lid_arc(T, yR, "push the lid down", deg_from=HOLD_DEG, deg_to=CLOSE_DEG, n=6, dur=0.5)
        rbc, Rrc = bar_pose(T, yR, CLOSE_DEG)
        right += [Waypoint(0.5, rbc, FINGER_OPEN, "let go: it falls shut", Rrc),
                  Waypoint(1.2, tuple(np.asarray(rbc) - Rrc.R[:, 2] * 0.08), FINGER_OPEN, "back off", Rrc),
                  Waypoint(1.5, (CLEAR_R[0] + 0.06, CLEAR_R[1], CLEAR_R[2]), FINGER_OPEN, "clear"),
                  Waypoint(1.5, CLEAR_R, FINGER_OPEN, "well clear")]
    end = max(sum(w.duration for w in arm) for arm in (left, right)) + TAIL_S
    for arm, pos in ((left, CLEAR_L), (right, CLEAR_R)):
        arm.append(Waypoint(end - sum(w.duration for w in arm), pos, FINGER_OPEN, "hold still, clear"))
    return {"left": left, "right": right}
