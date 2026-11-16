"""v2 no-robot mechanism check (GPU): the enclosure and lid, struts (good L/R, failed F), external forces for hands.
Cases: probe probehalf release twopull close failed rodup raise start (see each case_ function).

  none   lid at its 45 deg stop, no struts, let go
  two    both struts fitted, lid let go at the stop; 30 N pull on each socket; lid pushed to 35 deg and let go
  one    right strut only, let go at the stop
  snap   lid held at its stop; right strut 9 mm out from both studs, a hand (force at most CAP N, 0.4 N m) pushes it on
  half   as snap but pushed only 3 mm on; then the hand and the lid let go

    source ~/my_robotics_projects/_shared/IsaacLab-Arena/yam.env && python dev/spike.py [case ...]   # from the task folder
"""
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from pxr import UsdGeom  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import gsparts as G  # noqa: E402
import strutkit as K  # noqa: E402

CASES = sys.argv[1:] or ["probe", "probehalf", "two"]
CAP = float(os.environ.get("CAP", "4.0"))
DT = 1 / 240
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda:0"))
stage = sim_utils.get_current_stage()
UsdGeom.Xform.Define(stage, "/World")
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(), translation=(0, 0, G.TOP))

ENC = G.enclosure()
SIDES = {"R": -1, "L": 1, "F": 1}
STR = {}
for k, s in SIDES.items():
    b, l = G.stud_points(s)
    STR[k] = G.strut(f"strut_{k}", b, l, (0, -s, 0), failed=(k == "F"), tagged=(k == "F"))


def make(spec, path):
    K._SPECS[spec.name] = spec
    K.spawn(path, SimpleNamespace(assembly_name=spec.name), translation=spec.pos, orientation=spec.quat)
    init = ArticulationCfg.InitialStateCfg(pos=tuple(spec.pos), rot=tuple(spec.quat))
    return Articulation(ArticulationCfg(prim_path=path, spawn=None, actuators=K.actuators(spec), init_state=init))


enc = make(ENC, "/World/enclosure")
PROBE = G.probe("probe")
prb = make(PROBE, "/World/probe")
pstud = make(G.probe_stud("probe_stud"), "/World/probe_stud")
arts = {k: make(v, f"/World/strut_{k}") for k, v in STR.items()}
sim.reset()
dev = sim.device
ALL = [enc, prb, pstud, *arts.values()]


def np_(x):
    return getattr(x, "torch", x)[0].detach().cpu().double().numpy()


def upd():
    for a in ALL:
        a.update(DT)


def lid_angle():
    return -math.degrees(float(np_(enc.data.joint_pos)[0]))


def set_lid(deg):
    q = torch.tensor([[-math.radians(deg)]], device=dev)
    enc.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), joint_ids=[0])


REV = set()   # struts fitted rod end up


def park(k, away=True):
    """Move a strut far away (out of play) or put it back on its studs at the current lid angle (gas joint set to the
    compression that length needs)."""
    s = SIDES[k]
    a = arts[k]
    if away:
        pose = torch.tensor([[0.0, s * 2.0 + (0.3 if k == "F" else 0.0), 0.2, *STR[k].quat]], dtype=torch.float32,
                            device=dev)
        spec = STR[k]
    else:
        b, l = G.stud_points(s, math.radians(lid_angle()))
        if k in REV:
            b, l = l, b
        spec = G.strut("tmp", b, l, (0, -s, 0), failed=(k == "F"))
        pose = torch.tensor([[*spec.pos, *spec.quat]], dtype=torch.float32, device=dev)
    a.write_root_pose_to_sim_index(root_pose=pose)
    a.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev))
    K.rest(a, STR[k])
    gi = list(a.joint_names).index("gas")
    q0 = next(j.init for j in spec.joints if j.name == "gas") if not away else 0.0
    qq = torch.tensor([[q0]], device=dev)
    a.write_joint_state_to_sim_index(position=qq, velocity=torch.zeros_like(qq), joint_ids=[gi])


def place_offset(k, dy):
    s = SIDES[k]
    b, l = G.stud_points(s, math.radians(lid_angle()))
    off = np.array([0, s * dy, 0])  # dy > 0: outboard
    spec = G.strut("tmp", b + off, l + off, (0, -s, 0))
    pose = torch.tensor([[*spec.pos, *spec.quat]], dtype=torch.float32, device=dev)
    arts[k].write_root_pose_to_sim_index(root_pose=pose)
    arts[k].write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev))
    K.rest(arts[k], STR[k])
    return spec


BODY = {}


def body_ids(a):
    return {n: i for i, n in enumerate(a.body_names)}


forces = {}  # name -> list of (body, force(3), point(3))


def apply():
    for name, a in [("enclosure", enc), ("probe", prb), *[(f"strut_{k}", v) for k, v in arts.items()]]:
        nb = len(a.body_names)
        f = torch.zeros((1, nb, 3), device=dev)
        p = torch.zeros((1, nb, 3), device=dev)
        com = getattr(a.data.body_com_pos_w, "torch", a.data.body_com_pos_w)[0]
        p[0] = com
        for bname, fv, pt in forces.get(name, []):
            i = body_ids(a)[bname]
            f[0, i] = torch.tensor(fv, dtype=torch.float32, device=dev)
            if pt is not None:
                p[0, i] = torch.tensor(pt, dtype=torch.float32, device=dev)
        a.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=torch.zeros_like(f), positions=p,
                                                                 body_ids=list(range(nb)), is_global=True)
        a.write_data_to_sim()


def step(n=1):
    for _ in range(n):
        apply()
        sim.step(render=False)
        upd()


def bar_point():
    th = math.radians(lid_angle())
    x, z = G.lid_point(0.18, -0.030, th)
    return np.array([x, 0.0, G.TOP + z]), np.array([-math.sin(th), 0, math.cos(th)])


def hold_lid(push=6.0):
    pt, n = bar_point()
    forces["enclosure"] = [("lid", list(push * n), list(pt))]


def socket_report(k):
    """Ball centre relative to each socket centre, in the strut frame (mm), and jaw openings (mm)."""
    a, s = arts[k], SIDES[k]
    bp = np_(a.data.body_link_pose_w)
    bi = body_ids(a)
    th = math.radians(lid_angle())
    b, l = G.stud_points(s, th)
    out = []
    for link, xc, ball in (("rod", 0.0, b), ("tube", G.STRUT_LEN, l)):
        pose = bp[bi[link]]
        R = K.quat_matrix_xyzw(pose[3:7])
        c = pose[:3] + R @ np.array([xc, 0, 0])
        rel = R.T @ (ball - c)
        out.append(f"{link}: ball {rel[0]*1000:+5.1f} {rel[1]*1000:+5.1f} {rel[2]*1000:+5.1f}")
    q = np_(a.data.joint_pos)
    jn = list(a.joint_names)
    out.append("jaws " + " ".join(f"{abs(q[jn.index(j)])*1000:.2f}" for j in jn if "jaw" in j))
    out.append(f"gas {q[jn.index('gas')]*1000:.1f}")
    return " | ".join(out)


def seated(k, tol=0.0012):
    a, s = arts[k], SIDES[k]
    bp = np_(a.data.body_link_pose_w); bi = body_ids(a)
    b, l = G.stud_points(s, math.radians(lid_angle()))
    res = []
    for link, xc, ball in (("rod", 0.0, b), ("tube", G.STRUT_LEN, l)):
        pose = bp[bi[link]]; R = K.quat_matrix_xyzw(pose[3:7])
        rel = R.T @ (ball - (pose[:3] + R @ np.array([xc, 0, 0])))
        res.append(bool(np.linalg.norm(rel) < tol))
    return res


def start(lid_deg, fitted):
    forces.clear()
    set_lid(lid_deg)
    for k in arts:
        if k in fitted:
            park(k, away=False)
        else:
            park(k, away=True)
    step(1)


def log(tag, extra=""):
    print(f"[{tag:5s}] lid {lid_angle():6.2f} deg {extra}", flush=True)


def case_none():
    start(45, [])
    for i in range(5):
        step(48); log("none")


def case_one():
    start(45, ["R"])
    hold_lid(); step(120); log("one", "held " + socket_report("R"))
    forces.clear()
    for i in range(10):
        step(120); log("one", socket_report("R"))


def pull(k, end):
    """30 N outward along the stud axis on one socket for 1 s; returns its movement (mm)."""
    a, s = arts[k], SIDES[k]
    link, xc = ("rod", 0.0) if end == "body" else ("tube", G.STRUT_LEN)
    bi = body_ids(a)
    def centre():
        pose = np_(a.data.body_link_pose_w)[bi[link]]
        return pose[:3] + K.quat_matrix_xyzw(pose[3:7]) @ np.array([xc, 0, 0])
    c0 = centre()
    for i in range(360):
        fmag = min(30.0, 30.0 * i / 240)
        forces[f"strut_{k}"] = [(link, [0, s * fmag, 0], list(centre()))]
        step(1)
        if i % 24 == 0 or (os.environ.get("PULL_DIAG") and fmag > 16 and i % 2 == 0):
            pose = np_(a.data.body_link_pose_w)[bi[link]]
            Rl = K.quat_matrix_xyzw(pose[3:7])
            tilt = math.degrees(math.acos(max(-1, min(1, abs(float(Rl[:, 1] @ np.array([0, 1.0, 0])))))))
            log("pullN", f"{k} {end} {fmag:4.1f} N moved {np.linalg.norm(centre()-c0)*1000:6.2f} mm mouth tilt {tilt:5.1f} deg "
                + socket_report(k))
        if np.linalg.norm(centre() - c0) > 0.02:
            break
    d = np.linalg.norm(centre() - c0) * 1000
    forces.pop(f"strut_{k}")
    step(120)
    return d


def case_pulldiag():
    start(45, ["R", "L"])
    hold_lid(); step(120); forces.clear(); step(240)
    log("diag", "R " + socket_report("R"))
    pull("R", "body")
    log("diag", "after R " + socket_report("R"))


def case_two():
    start(45, ["R", "L"])
    hold_lid(); step(120); log("two", "held  R " + socket_report("R"))
    forces.clear()
    for i in range(4):
        step(120); log("two", "R " + socket_report("R") + " || L " + socket_report("L"))
    for k in ("R", "L"):
        for end in ("body", "lid"):
            d = pull(k, end)
            log("pull", f"{k} {end}: moved {d:.2f} mm while pulled; seated now {seated(k)}")
    # push the lid down to 35 deg with a hand on the bar, then let go
    for _ in range(400):
        pt, n = bar_point()
        e = lid_angle() - 35.0
        forces["enclosure"] = [("lid", list(-min(25.0, max(-25.0, 4.0 * e + 6)) * n), list(pt))]
        step(1)
    log("two", "pushed to ~35: " + socket_report("R"))
    forces.clear()
    for i in range(6):
        step(120); log("two", f"let go t={0.5*(i+1):.1f}s " + socket_report("R"))


def hand_push(k, dy_total, speed=0.010, settle=1.0):
    """A hand holding strut k pushes it toward its studs by dy_total (m) at speed; force at most CAP, torque 0.4."""
    s = SIDES[k]
    a = arts[k]
    bi = body_ids(a)
    spec0 = place_offset(k, 0.009)
    step(2)
    root0 = np_(a.data.body_link_pose_w)[bi["tube"]].copy()
    R0 = K.quat_matrix_xyzw(root0[3:7])
    mass = 0.042
    t = 0.0
    n = int((dy_total / speed + settle) / DT)
    peak = 0.0
    for i in range(n):
        t += DT
        d = min(dy_total, speed * t)
        target = root0[:3] + np.array([0, -s * d, 0])
        pose = np_(a.data.body_link_pose_w)[bi["tube"]]
        vel = np_(a.data.body_link_vel_w)[bi["tube"]] if hasattr(a.data, "body_link_vel_w") else np.zeros(6)
        e = target - pose[:3]
        f = 3000 * e - 30 * vel[:3]
        f[0] = max(-20, min(20, f[0])); f[2] = max(-20, min(20, f[2]))
        f[1] = max(-CAP, min(CAP, f[1]))
        peak = max(peak, abs(f[1]))
        f[2] += mass * 9.81
        R = K.quat_matrix_xyzw(pose[3:7])
        Er = R0 @ R.T
        ang = np.array([Er[2, 1] - Er[1, 2], Er[0, 2] - Er[2, 0], Er[1, 0] - Er[0, 1]]) / 2
        tq = 20.0 * ang - 0.2 * vel[3:6]
        tn = np.linalg.norm(tq)
        if tn > 2.0: tq *= 2.0 / tn
        # torque as a force couple is awkward: apply torque via two equal and opposite forces? use wrench torques instead
        HAND["f"], HAND["tq"], HAND["k"] = f, tq, k
        step(1)
        if i % 60 == 0:
            log("push", f"d {d*1000:4.1f} mm f_y {f[1]:+5.2f} N  " + socket_report(k))
    HAND["k"] = None
    return peak


HAND = {"k": None, "f": None, "tq": None}
_apply0 = apply


def apply():  # noqa: F811
    _apply0()
    if HAND["k"] is not None:
        a = arts[HAND["k"]]
        nb = len(a.body_names)
        bi = body_ids(a)
        f = torch.zeros((1, nb, 3), device=dev); tq = torch.zeros((1, nb, 3), device=dev)
        f[0, bi["tube"]] = torch.tensor(HAND["f"], dtype=torch.float32, device=dev)
        tq[0, bi["tube"]] = torch.tensor(HAND["tq"], dtype=torch.float32, device=dev)
        a.permanent_wrench_composer.add_forces_and_torques_index(forces=f, torques=tq, body_ids=list(range(nb)),
                                                                 is_global=True)
        a.write_data_to_sim()


def case_snap():
    start(45, [])
    hold_lid(); step(60)
    peak = hand_push("R", 0.0115)
    log("snap", f"peak push {peak:.2f} N; seated {seated('R')}; " + socket_report("R"))
    step(240); log("snap", "hand gone 1 s: " + socket_report("R"))
    for end in ("body", "lid"):
        d = pull("R", end)
        log("pull", f"R {end}: moved {d:.2f} mm; seated {seated('R')}")


def case_half():
    start(45, [])
    hold_lid(); step(60)
    hand_push("R", 0.003)
    log("half", "pushed 3 mm: " + socket_report("R"))
    forces.clear()
    for i in range(6):
        step(120); log("half", "all let go: " + socket_report("R"))


PJ = {n: i for i, n in enumerate(prb.joint_names)}
PB = {n: i for i, n in enumerate(prb.body_names)}


def probe_set(q=None, target=None):
    if q is not None:
        qq = torch.tensor([[q]], device=dev)
        prb.write_joint_state_to_sim_index(position=qq, velocity=torch.zeros_like(qq), joint_ids=[PJ["push"]])
    if target is not None:
        prb.set_joint_position_target_index(target=torch.tensor([[target]], device=dev), joint_ids=[PJ["push"]])
    prb.write_data_to_sim()


def probe_report():
    q = np_(prb.data.joint_pos)
    return f"push q {q[PJ['push']]*1000:+6.2f} mm jaws {q[PJ['jaw_a_slide']]*1000:5.2f} {q[PJ['jaw_b_slide']]*1000:5.2f} mm"


def probe_pull(peak=40.0, secs=2.0):
    q0 = float(np_(prb.data.joint_pos)[PJ["push"]])
    probe_set(target=q0)
    n = int(secs / DT)
    for i in range(n + 120):
        f = min(peak, peak * i / n)
        forces["probe"] = [("carriage", [0.0, -f, 0.0], None)]
        step(1)
        if i % 24 == 0:
            log("ppull", f"{f:5.1f} N " + probe_report())
        if float(np_(prb.data.joint_pos)[PJ["push"]]) < q0 - 0.006:
            log("ppull", f"OFF at {f:.1f} N " + probe_report())
            break
    forces.pop("probe", None)


def probe_push(to, secs=2.0):
    K.rest(prb, PROBE)
    probe_set(q=-0.009, target=-0.009)
    step(24)
    n = int(secs / DT)
    for i in range(n + 120):
        tgt = -0.009 + (to + 0.009) * min(1.0, i / n)
        probe_set(target=tgt)
        step(1)
        if i % 24 == 0:
            q = float(np_(prb.data.joint_pos)[PJ["push"]])
            log("ppush", f"target {tgt*1000:+6.2f} drive {min(4.0, 2000*(tgt-q)):4.2f} N " + probe_report())


def case_probe():
    start(0, [])
    probe_push(0.003)
    log("probe", "pushed: " + probe_report())
    probe_pull()


def case_probehalf():
    start(0, [])
    probe_push(-0.006)
    log("phalf", "3 mm on: " + probe_report())
    probe_pull(peak=10.0)


def jaw_d_world(a, bi, link):
    """World direction the strut's jaw on ``link`` opens along, and the jaw link name."""
    jn = next(n for n in a.body_names if n.startswith(f"jaw_{link}"))
    sj = 1 if jn.endswith("_a") else -1
    R = K.quat_matrix_xyzw(np_(a.data.body_link_pose_w)[bi[link]][3:7])
    return R @ np.array([0.0, -math.sin(G.JAW_BETA), sj * math.cos(G.JAW_BETA)]), jn


def case_release():
    """Probe socket snapped on; both finger loops pushed outboard (ramped to LOOP N in all), then the carriage pulled
    off by its drive (at most 4 N) with the push held: does it come off, and at what drive force?"""
    start(0, [])
    probe_push(0.003)
    log("rel", "snapped: " + probe_report())
    loop_n = float(os.environ.get("LOOP", "3.0"))   # total push on the two loops, outboard (-y), shared
    for i in range(240):
        fmag = loop_n * min(1.0, i / 120)
        forces["probe"] = [("jaw_a", [0.0, -fmag / 2, 0.0], None), ("jaw_b", [0.0, -fmag / 2, 0.0], None)]
        step(1)
        if i % 24 == 0:
            log("loop", f"{fmag:4.2f} N on the loop: " + probe_report())
    q0 = float(np_(prb.data.joint_pos)[PJ["push"]])
    peak = 0.0
    for i in range(480):
        tgt = q0 - 0.014 * min(1.0, i / 240)
        probe_set(target=tgt)
        step(1)
        q = float(np_(prb.data.joint_pos)[PJ["push"]])
        peak = max(peak, min(4.0, abs(2000 * (tgt - q))))
        if i % 48 == 0:
            log("off", f"target {tgt*1000:+6.2f} drive {min(4.0, abs(2000*(tgt-q))):4.2f} N " + probe_report())
    q = float(np_(prb.data.joint_pos)[PJ["push"]])
    log("rel", f"pulled with the loop held: moved {abs(q-q0)*1000:.1f} mm, peak drive {peak:.2f} N: "
        + ("OFF" if abs(q - q0) > 0.008 else "STILL ON"))
    forces.clear(); step(60)


def case_twopull():
    start(45, ["R", "L"])
    hold_lid(); step(120); forces.clear(); step(240)
    log("2pull", "let go: R " + socket_report("R") + " || L " + socket_report("L"))
    for k in ("L", "R"):
        for end in ("body", "lid"):
            d = pull(k, end)
            log("pull", f"{k} {end}: moved {d:.2f} mm while pulled; seated now {seated(k)}")


def let_go_from(deg, fitted, secs=8.0, tag="close"):
    start(deg, fitted)
    forces.clear()
    t_shut = None
    for i in range(int(secs / DT)):
        step(1)
        if t_shut is None and lid_angle() <= 3.0:
            t_shut = i * DT
        if i % 120 == 0:
            log(tag, f"t {i*DT:4.1f} s " + " ".join(f"{k}:{seated(k)}" for k in fitted))
    log(tag, f"released at {deg} deg with {fitted}: within 3 deg of shut after "
        + (f"{t_shut:.2f} s" if t_shut is not None else "never") + f"; final {lid_angle():.2f} deg")


def case_close():
    for deg in (25.0, 20.0):
        let_go_from(deg, ["R", "L"], tag="close")


def case_failed():
    let_go_from(45.0, ["R", "F"], secs=5.0, tag="failed")


def case_rodup():
    REV.add("L")
    let_go_from(25.0, ["R", "L"], tag="rodup")
    REV.discard("L")


def case_raise():
    """The inspection's lift (task.py law, applied at 20 Hz), from shut with two good struts; then let go 3 s."""
    start(0.0, ["R", "L"])
    forces.clear()
    prev, F = lid_angle(), 0.0
    reached = None
    for i in range(int(4.0 / DT)):
        if i % 12 == 0:
            deg = lid_angle(); rate = (deg - prev) / (12 * DT) if i else 0.0; prev = deg
            F = max(0.0, min(8.0, 1.0 + 0.2 * (45.0 - deg) - 0.03 * rate))
            if deg >= 44.0:
                reached = i * DT
                break
            pt, n = bar_point()
            forces["enclosure"] = [("lid", list(F * n), list(pt))]
        step(1)
    log("raise", f"reached 44 deg after {reached} s at {lid_angle():.2f} deg")
    forces.clear()
    for i in range(int(3.0 / DT)):
        step(1)
        if i % 120 == 0:
            log("raise", f"let go t {i*DT:.1f} s")


def case_start():
    """The start: lid shut, good strut right and the failed one left, both fitted (compressed): does it all sit?"""
    start(0.0, ["R", "F"])
    forces.clear()
    for i in range(6):
        step(120); log("start", "R " + socket_report("R") + " || F " + socket_report("F"))
    hold_lid(); 
    for i in range(int(2.0 / DT)):
        pt, n = bar_point(); forces["enclosure"] = [("lid", list(6.0 * n), list(pt))]; step(1)
    log("start", "lifted by 6 N for 2 s")
    forces.clear(); step(480); log("start", "let go 2 s")


t0 = time.perf_counter()
for c in CASES:
    print(f"==== {c}  (strut {G.STRUT_LEN*1000:.1f} mm)", flush=True)
    globals()[f"case_{c}"]()
print(f"wall {time.perf_counter()-t0:.0f} s", flush=True)
app.close()
