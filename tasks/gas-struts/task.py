"""On the station table: a small steel machine enclosure (400 x 200 x 80 mm) facing the robot with its hinge side, its
blue aluminium lid shut on a piano hinge along the front top edge (open stop at 45 degrees), a black bar handle on the
lid, and a gas strut (GS-10-30 class, 92 mm extended, a steel ball socket with a clip and finger loop on each end) on
8 mm ball studs on each side, rod end down. The left strut has failed (no gas force) and carries a red defect tag. The
new strut lies across two wooden battens in front of the enclosure, on the left.
Replace the tagged gas strut with the new one, check that the lid holds itself open, then close the lid.
"""

import math
import os
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import gsparts as G  # noqa: E402
import gskit  # noqa: E402
import strutkit as K  # noqa: E402

SCENE = "station"
TOP = G.TOP
INSTRUCTION = "Replace the tagged gas strut with the new one, check that the lid holds itself open, then close the lid."

# the new strut's start: across the battens along x, rod end away from the robot, sockets' mouths facing -y, the way
# they face the left side wall once fitted (the left studs point +y out of the wall)
START_X, START_Y, AXIS_Z = G.STRUT_START_X, 0.140, G.BATTEN_H + 0.0065  # resting on the battens by its 13 mm sockets
SHUT = 0.0


def _fitted(name, side, **kw):
    b, l = G.stud_points(side, SHUT)
    return G.strut(name, b, l, (0, -side, 0), **kw)


def _new():
    rod = np.array([START_X + G.STRUT_LEN / 2, START_Y, TOP + AXIS_Z])
    tube = np.array([START_X - G.STRUT_LEN / 2, START_Y, TOP + AXIS_Z])
    return G.strut("strut_new", rod, tube, (0, -1, 0))


ENCLOSURE = G.enclosure()
STRUT_R = _fitted("strut_R", -1)                                  # the good strut, right side
STRUT_OLD = _fitted("strut_old", 1, failed=True, tagged=True)     # the failed, tagged strut, left side
STRUT_NEW = _new()
OBJECTS = [ENCLOSURE, STRUT_R, STRUT_OLD, STRUT_NEW]
for spec in OBJECTS:
    K._SPECS[spec.name] = spec

TRACKER = gskit.Tracker()
LAST = {"reading": None, "pts": None, "t": 0.0, "rate": 0.0, "sockets": {}}
DEBUG = int(os.environ.get("GS_DEBUG", "0"))
DBG = {}


# -- the hook (runs after every control step) -----------------------------------------------------------------------


def _np(x):
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _hand_points(base):
    from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET

    origin = _np(base.scene.env_origins[0])
    pts = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        names = list(robot.body_names)
        pose = _np(robot.data.body_link_pose_w)[0]
        g = pose[names.index("gripper")]
        x, y, z, w = g[3:7]
        rz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        tcp = g[:3] + rz * TCP_OFFSET[2]
        pts += [g[:3] - origin, tcp - origin, tcp + rz * PAD_TIP_PAST_TCP - origin]
        pts += [pose[names.index(n)][:3] - origin for n in ("tip_left", "tip_right")]
    return np.array(pts)


def _sockets(st, spec):
    """end -> (centre, rotation, larger jaw opening mm, middle of the two finger loops, release push direction,
    mouth direction) for a strut; end "body" = rod end, "lid" = tube end."""
    out = {}
    for end, link, xc in (("body", "rod", 0.0), ("lid", "tube", G.STRUT_LEN)):
        p, q = st.links[link]
        c, R = gskit.socket_centre(np.r_[p, q], xc)
        jaw_mm = max(abs(st.joints[f"jaw_{link}_{k}_slide"]) for k in "ab") * 1000
        e = -1.0 if link == "rod" else 1.0
        loop = c + R @ np.array([e * G.TAB_PAD, G.JAW_Y + 0.0009, 0.0])
        out[end] = (c, R, jaw_mm, loop, -R[:, 1], R[:, 1])
    return out


def read(base, t):
    origin = _np(base.scene.env_origins[0])
    enc = K.read_state(base.scene[ENCLOSURE.name], ENCLOSURE, origin)
    lid_deg = -math.degrees(enc.joints["lid_hinge"])
    th = math.radians(lid_deg)
    studs = {s: dict(zip(("body", "lid"), G.stud_points(s, th))) for s in (1, -1)}
    socks = {spec.name: _sockets(K.read_state(base.scene[spec.name], spec, origin), spec)
             for spec in (STRUT_R, STRUT_OLD, STRUT_NEW)}

    def on(c, R, jaw_mm, ball):
        return bool(np.linalg.norm(R.T @ (np.asarray(ball) - c)) * 1000 < gskit.SEAT_MM and jaw_mm < gskit.JAW_SHUT_MM)

    new_on = {e: on(*socks["strut_new"][e][:3], studs[1][e]) for e in ("body", "lid")}
    old_on = any(on(*socks["strut_old"][e][:3], studs[s][k]) for e in ("body", "lid") for s in (1, -1)
                 for k in ("body", "lid"))
    pts = _hand_points(base)
    dt = t - LAST["t"]
    speed = 0.0 if LAST["pts"] is None or dt <= 0 else float(np.max(np.linalg.norm(pts - LAST["pts"], axis=1)) / dt)
    LAST["pts"], LAST["t"] = pts, t
    LAST["sockets"] = socks
    if DEBUG:
        for nm, ss in socks.items():
            for e, (c, R, jaw_mm, *_rest) in ss.items():
                side = -1 if nm == "strut_R" else 1
                DBG[(nm[6:], e)] = (tuple(np.round(1000 * (R.T @ (studs[side][e] - c)), 1)), round(jaw_mm, 2))
    return gskit.Reading(t=t, lid_deg=lid_deg, new_on=new_on, old_on=old_on,
                         old_pts=[socks["strut_old"][e][0] for e in ("body", "lid")],
                         new_pos={e: socks["strut_new"][e][0] for e in ("body", "lid")},
                         hand_points=pts, hand_speed=speed)


def _apply_test_force(base, want, r):
    """The inspection's forces, applied by the task file (this task's own test values): a 30 N pull outward along the
    stud on one socket of the new strut; or a lift on the handle (at most 8 N) raising the lid to its stop."""
    import torch

    org = _np(base.scene.env_origins[0])
    for spec in (ENCLOSURE, STRUT_NEW):
        a = base.scene[spec.name]
        nb = len(a.body_names)
        com = getattr(a.data.body_com_pos_w, "torch", a.data.body_com_pos_w)
        f = torch.zeros((1, nb, 3), device=com.device)
        p = com[0:1].clone()
        if want is not None and want[0] == "pull" and spec is STRUT_NEW:
            i = list(a.body_names).index("rod" if want[1] == "body" else "tube")
            f[0, i, 1] = gskit.PULL_N          # outward along the left studs' axis (+y)
            p[0, i] = torch.tensor(r.new_pos[want[1]] + org, dtype=p.dtype, device=com.device)
        if want is not None and want[0] == "raise" and spec is ENCLOSURE:
            i = list(a.body_names).index("lid")
            th = math.radians(r.lid_deg)
            x, z = G.lid_point(0.18, -0.030, th)
            n = np.array([-math.sin(th), 0.0, math.cos(th)])
            # a hand lifting the handle, at most 8 N, easing off near the stop (fin2/cpu or spike: raise case)
            mag = max(0.0, min(gskit.RAISE_N, 1.0 + 0.2 * (45.0 - r.lid_deg) - 0.03 * LAST.get("rate", 0.0)))
            f[0, i] = torch.tensor(mag * n, dtype=f.dtype, device=com.device)
            p[0, i] = torch.tensor(np.array([x, 0.0, TOP + z]) + org, dtype=p.dtype, device=com.device)
        a.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=torch.zeros_like(f), positions=p,
                                                                 body_ids=list(range(nb)), is_global=True)


@K.on_step
def _step(env):
    base = env.unwrapped
    if not LAST.get("placed"):
        _place_start(base)
        LAST["placed"] = True
    t = LAST["t"] + float(base.step_dt)
    r = read(base, t)
    if abs(t - 0.5) < 0.5 * float(base.step_dt):
        print(f"GS start check t={t:.2f}: tagged strut on its studs={r.old_on}", flush=True)
    prev = LAST["reading"]
    LAST["rate"] = 0.0 if prev is None or t <= prev.t else (r.lid_deg - prev.lid_deg) / (t - prev.t)
    TRACKER.step(r)
    _apply_test_force(base, TRACKER.wants_force(), r)
    dp = sys.modules.get("demo_plan")  # the scripted demo's measured aim (only when the demo is being played)
    if dp is not None:
        for m in dp.MEASURES:
            if not m.get("done") and t >= m["t"] - 1e-6:
                dp.apply_measure(m, LAST["sockets"])
                if DEBUG:
                    print(f"GSM t={t:.2f} {m['kind']}: rewrote {len(m['wps'])} waypoints {m.get('state', {})}",
                          flush=True)
    LAST["reading"] = r
    if DEBUG and int(round(t / base.step_dt)) % DEBUG == 0:
        print(f"GS t={t:6.2f} lid={r.lid_deg:6.2f} new={[int(v) for v in r.new_on.values()]} old_on={int(r.old_on)} "
              f"rest={int(gskit.old_resting(r))} held={int(TRACKER.held)} closed={int(TRACKER.closed)} "
              f"insp={int(TRACKER.inspected)} phase={TRACKER.phase} res={TRACKER.results} clear={TRACKER.clear(r)} "
              f"v={r.hand_speed:.3f} score={gskit.ladder(r, TRACKER)}", flush=True)
        fing = []
        for arm in ("left_robot", "right_robot"):
            rob = base.scene[arm]
            q = _np(rob.data.joint_pos)[0]
            fing.append(1000 * q[list(rob.joint_names).index("joint7")])
        tl, tr_ = r.hand_points[1], r.hand_points[6]
        print("GSS " + " ".join(f"{k[0]}.{k[1]}:ball{v[0]}jaw{v[1]}" for k, v in DBG.items()), flush=True)
        op = np.round(r.old_pts, 3).tolist()
        print("GSX tcpL=(%.3f,%.3f,%.3f) tcpR=(%.3f,%.3f,%.3f) fingL=%.1f fingR=%.1f" % (*tl, *tr_, *fing)
              + f" old={op}", flush=True)


def _place_start(base):
    """Every assembly to its start state: the struts' root poses and joints (the fitted ones with the rod pushed in),
    the lid shut. Called by the reset hook and again just before the first control step."""
    import torch

    org = _np(base.scene.env_origins[0])
    for spec in OBJECTS:
        a = base.scene[spec.name]
        if not spec.fixed:
            dev = getattr(a.data.joint_pos, "torch", a.data.joint_pos).device
            pose = torch.tensor([[*(np.asarray(spec.pos) + org), *spec.quat]], dtype=torch.float32, device=dev)
            a.write_root_pose_to_sim_index(root_pose=pose)
            a.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev))
        K.rest(a, spec)


@K.on_reset
def _reset(env):
    # WHY root pose too, and again before the first step: the arena's reset can settle the scene with every joint at 0
    # around this hook, so a strut fitted with the lid shut (rod pushed in 17 mm) dropped off its studs before the
    # robot moved (fin2 r1, r2, r3)
    _place_start(env.unwrapped)
    TRACKER.reset()
    LAST.update(reading=None, pts=None, t=0.0, rate=0.0, sockets={}, placed=False)


# -- scoring ------------------------------------------------------------------------------------------------------


def score(objects) -> float:
    return gskit.ladder(LAST["reading"], TRACKER)


# WHY no success(): the episode runs on to the run limit, so the last frames show the end state and the inspection


def demo():
    from demo_plan import plan

    # WHY globals(): yam_arena loads this file under its own module name without adding it to sys.modules
    return plan(SimpleNamespace(**globals()), os.environ.get("GS_DEMO", ""))
