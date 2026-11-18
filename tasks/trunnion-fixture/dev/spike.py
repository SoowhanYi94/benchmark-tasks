"""No-robot spike: the fixture, the 1.2 kg gearbox, the flange and its screws on the GPU pipeline, worked by a dev "hand" (forces).

Cases (run several in one go; each restarts from the start state):
  turn     close the bar, turn the cradle over by a capped torque, push the bolt, let go: gearbox kept, bolt home?
  open     the same with the bar open: the gearbox must fall out
  flange   after turn: drop the flange (screws glued, tips 2 mm proud) at offsets/yaws: does it seat?
  screws   after turn + seated flange: glue off, threads on, run the targets down; head gaps; then the 10 N lift (and the lift before)
  driver   after turn + seated flange: the driver stood in screw 0's socket and let go: does it stay?

    source ~/my_robotics_projects/_shared/IsaacLab-Arena/yam.env && python dev/spike.py turn open flange screws    # from the task folder
"""

import argparse
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("cases", nargs="*", default=["turn", "open", "flange", "screws", "driver"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import task as T  # noqa: E402
from trunnion_fixture import mech as M  # noqa: E402

DT = 1 / 240
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda:0", physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2)))
stage = sim_utils.get_current_stage()
UsdGeom.Xform.Define(stage, "/World")
table_mat = M.look(stage, "/World/Looks/t", (0.5, 0.5, 0.5))
M.cube(stage, "/World/table", (-0.1, -0.5, T.TOP - 0.03), (0.7, 0.5, T.TOP), table_mat, M.surface(stage, "/World/Physics/t", 0.5))
x = UsdGeom.Xform.Define(stage, "/World/fixture")
x.AddTranslateOp().Set(Gf.Vec3d(T.AX, T.Y0, T.TOP))
M.build_fixture(stage, "/World/fixture", (T.AX, T.Y0, T.TOP))
objs = {}
for name in T.PARTS:
    p = f"/World/{name}"
    x = UsdGeom.Xform.Define(stage, p)
    x.AddTranslateOp().Set(Gf.Vec3d(*T.START[name]))
    x.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    M.build_part(stage, p, name)
for name in T.PARTS:
    objs[name] = RigidObject(RigidObjectCfg(prim_path=f"/World/{name}", spawn=None))
fx = Articulation(ArticulationCfg(prim_path="/World/fixture", spawn=None, actuators={}))
sim.reset()
J = {n: i for i, n in enumerate(fx.joint_names)}
B = {n: i for i, n in enumerate(fx.body_names)}
print("joints", fx.joint_names, "bodies", fx.body_names, flush=True)
dev = sim.device
SC_PATH = {i: f"/World/screw{i}" for i in range(4)}


def npy(t):
    return (t.torch if hasattr(t, "torch") else t).detach().cpu().numpy().astype(float)


def states():
    s = {}
    for n, o in objs.items():
        p = npy(o.data.root_pos_w)[0]
        s[n] = NS(pos=p, quat_xyzw=npy(o.data.root_quat_w)[0], lin_vel=npy(o.data.root_lin_vel_w)[0], ang_vel=npy(o.data.root_ang_vel_w)[0])
    q, qd = npy(fx.data.joint_pos)[0], npy(fx.data.joint_vel)[0]
    s["fixture"] = NS(pos=np.array((q[J["roll"]], q[J["bar_slide"]], -q[J["bolt_slide"]])), lin_vel=np.array((qd[J["roll"]], qd[J["bar_slide"]], -qd[J["bolt_slide"]])))
    far = np.array((0.0, 0.9, 1.5))
    for a in ("left_tcp", "right_tcp"):
        s[a] = NS(pos=far)
    return s


FIX_F = {}  # body -> (force xyz, torque xyz), world


def apply():
    names = list(FIX_F)
    if names:
        f = torch.tensor([[FIX_F[n][0] for n in names]], dtype=torch.float32, device=dev)
        tq = torch.tensor([[FIX_F[n][1] for n in names]], dtype=torch.float32, device=dev)
        fx.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=tq, body_ids=[B[n] for n in names], is_global=True)


def part_force(name, f):
    o = objs[name]
    o.permanent_wrench_composer.set_forces_and_torques_index(forces=torch.tensor([[list(map(float, f))]], dtype=torch.float32, device=dev),
                                                             torques=torch.zeros((1, 1, 3), dtype=torch.float32, device=dev), is_global=True)


HAND = {"roll": None}
wall = [0.0, 0]


def step(n, hook=None):
    t0 = time.perf_counter()
    for k in range(n):
        if HAND["roll"] is not None and k % 2 == 0:
            q, qd = npy(fx.data.joint_pos)[0][J["roll"]], npy(fx.data.joint_vel)[0][J["roll"]]
            tq = float(np.clip(1.5 * (HAND["roll"] - q) - 0.05 * qd, -0.4, 0.4))  # a capped hand: 0.4 N m = 6.7 N at the 60 mm crank
            FIX_F["cradle"] = ((0.0, 0.0, 0.0), (0.0, tq, 0.0))
        apply()
        if hook:
            hook()
        fx.write_data_to_sim()
        for o in objs.values():
            o.write_data_to_sim()
        sim.step(render=False)
        fx.update(DT)
        for o in objs.values():
            o.update(DT)
        wall[1] += 1
    wall[0] += time.perf_counter() - t0


def write_pose(name, pos, yaw=0.0, roll_y=0.0):
    qz = np.array((0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
    pose = torch.tensor([[*map(float, pos), *map(float, qz)]], dtype=torch.float32, device=dev)
    objs[name].write_root_pose_to_sim_index(root_pose=pose)
    objs[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=dev))


def put_flange(pos, yaw=0.0):
    """The flange (upright, yaw) with its glued screws, at pos."""
    write_pose("flange", pos, yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    for i in range(4):
        o = np.array(M.screw_offset(i))
        write_pose(f"screw{i}", np.asarray(pos) + np.array((c * o[0] - s * o[1], s * o[0] + c * o[1], o[2])), yaw)


def restart():
    T.reset_latch()
    FIX_F.clear()
    HAND["roll"] = None
    for i in range(4):
        for j in range(4):
            M.set_thread(SC_PATH[i], j, False)
        M.set_glue(SC_PATH[i], True)
    for o in objs.values():
        o.permanent_wrench_composer.reset() if hasattr(o.permanent_wrench_composer, "reset") else None
        part_force_zero(o)
    fx.write_joint_state_to_sim_index(position=torch.zeros((1, len(J)), device=dev), velocity=torch.zeros((1, len(J)), device=dev)) \
        if hasattr(fx, "write_joint_state_to_sim_index") else fx.write_joint_state_to_sim(torch.zeros((1, len(J)), device=dev), torch.zeros((1, len(J)), device=dev))
    for n in T.PARTS:
        write_pose(n, T.START[n])
    FIX_F["cradle"] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    FIX_F["bar"] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    FIX_F["bolt"] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    step(240)


def part_force_zero(o):
    o.permanent_wrench_composer.set_forces_and_torques_index(forces=torch.zeros((1, 1, 3), dtype=torch.float32, device=dev),
                                                             torques=torch.zeros((1, 1, 3), dtype=torch.float32, device=dev), is_global=True)


def report(tag):
    s = states()
    f = s["fixture"]
    g, off = T.flange_gaps(s)
    print(f"[{tag}] roll {math.degrees(f.pos[0]):7.2f} bar {1000 * f.pos[1]:5.1f} bolt {1000 * f.pos[2]:5.1f} gb_in {T.gb_in(s)} off {np.round(1000 * T.gb_offset(s), 2).tolist()} "
          f"tilt {math.degrees(T.gb_tilt(s)):.1f} | flange gaps {np.round(1000 * np.array(g), 2).tolist()} off {1000 * off:.2f} on {T.flange_on(s)} | heads "
          f"{[round(1000 * T.head_gap(s, i), 2) for i in range(4)]} | drv {np.round(1000 * s['driver'].pos, 1).tolist()} bad {T._trial['bad']} {T._trial['why']} "
          f"| sim {wall[1] * DT / max(wall[0], 1e-9):.2f}x", flush=True)
    return s


def turn(bar=True, bolt=True):
    if bar:
        FIX_F["bar"] = ((0.0, 4.0, 0.0), (0.0, 0.0, 0.0))
        step(480)
        FIX_F["bar"] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        step(60)
        report("bar pushed 4 N for 2 s")
    for k in range(1, 61):
        HAND["roll"] = math.radians(180) * k / 60
        step(24)
        T.note_step(states(), 0.1)
    step(240)
    report("hand holds the crank at 180 deg")
    if bolt:
        FIX_F["bolt"] = ((0.0, -4.0, 0.0), (0.0, 0.0, 0.0))
        step(360)
        FIX_F["bolt"] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    HAND["roll"] = None
    FIX_F["cradle"] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    step(480)
    for _ in range(5):
        T.note_step(states(), 0.05)
    return report("bolt pushed 4 N, hand off, 2 s")


def seat_flange(dx=0.0, dy=0.0, yaw=0.0, drop=0.004):
    s = states()
    c = s["gearbox"].pos + T._rot(s["gearbox"].quat_xyzw) @ np.array((0.0, 0.0, -M.GB["face"]))  # the real output axis on face J
    put_flange((c[0] + dx, c[1] + dy, c[2] + drop), yaw)
    step(480)
    return report(f"flange dropped from {1000 * drop:.0f} mm at dx {1000 * dx:.1f} dy {1000 * dy:.1f} yaw {math.degrees(yaw):.1f}")


def lift(tag, n=10.0):
    s0 = states()
    g0 = max(T.flange_gaps(s0)[0])
    mx = 0.0
    for k in range(1, 361):
        part_force("flange", (0.0, 0.0, n * min(1.0, k / 120)))
        step(1)
        if k % 6 == 0:
            mx = max(mx, max(T.flange_gaps(states())[0]) - g0)
            if mx > 0.001:
                break
    part_force_zero(objs["flange"])
    step(120)
    print(f"LIFT {tag}: {n} N, flange rose at most {1000 * mx:.3f} mm (pass <= {1000 * T.LIFT_RISE} mm): {'PASS' if mx <= T.LIFT_RISE else 'FAIL'}", flush=True)


for case in args.cases:
    print(f"==== {case}", flush=True)
    restart()
    report("start")
    if case == "open":
        turn(bar=False)
        continue
    s = turn()
    if case == "turn":
        continue
    if case == "flange":
        for dx, dy, yaw in ((0, 0, 0), (0.002, 0.0, 0.0), (0.0, -0.003, 0.0), (0.0, 0.0, math.radians(4)), (0.002, 0.002, math.radians(-6)), (0.0035, 0.0, 0.0), (-0.003, 0.002, math.radians(3))):
            seat_flange(dx, dy, yaw)
        continue
    s = seat_flange()
    if case == "screws":
        lift("glued screws, not run (expect FAIL)")
        s = seat_flange()
        for i in range(4):
            j, off, depth = T._hole_of(s, i)
            print(f"screw {i}: hole {j} off {1000 * off:.2f} mm depth {1000 * depth:.2f} mm", flush=True)
            M.set_glue(SC_PATH[i], False)
            M.set_thread(SC_PATH[i], j, True, -depth)
            T._trial["screws"][i].update(on=True, hole=j, target=-depth)
        step(120)
        report("threads on, targets at the catch")
        tg = {i: T._trial["screws"][i]["target"] for i in range(4)}
        for k in range(1, 121):  # run 0 and 2 down 4 mm, the rest down to the seat + overrun, over 6 s
            for i in range(4):
                end = (tg[i] - 0.004) if k <= 60 or i in (1, 3) else T.SEAT_TRAVEL - T.OVERRUN
                end = T.SEAT_TRAVEL - T.OVERRUN if i in (1, 3) else (tg[i] - 0.004 if k <= 60 else T.SEAT_TRAVEL - T.OVERRUN)
                M.set_thread_target(SC_PATH[i], T._trial["screws"][i]["hole"], tg[i] + (end - tg[i]) * min(1.0, k / 60 if k <= 60 else (k - 60) / 60))
            step(12)
            if k == 60:
                report("1, 3 seated; 0, 2 4 mm down")
                lift("two screws seated, two 4 mm short")
        s = report("all run to the seat")
        for i in range(4):
            j, off, depth = T._hole_of(s, i)
            fl = T._local(s, "flange", s[f"screw{i}"].pos)
            print(f"seated? screw {i}: hole {j} off {1000 * off:.2f} depth {1000 * depth:.2f} mm, target {1000 * T._trial['screws'][i]['target']:.2f}, in flange {np.round(1000 * fl, 2).tolist()}", flush=True)
        lift("four seated")
        lift("four seated, 20 N", 20.0)
        report("after the lifts")
    if case == "driver":
        x, y = T.screw_xy(0)
        head = T.output_axis_turned()[2] + M.SCREW_START_Z + M.SC["head_h"]
        write_pose("driver", (x, y, head - M.SC["socket_d"] + 0.0005))
        step(720)
        report("driver stood in screw 0's socket, 3 s")
print(f"done; sim speed {wall[1] * DT / max(wall[0], 1e-9):.2f}x real time", flush=True)
app.close()
