"""Feasibility check, no robots: does the mimic-coupled screw and locknut work on the GPU pipeline?

A torque on the screw's body (the "hand") turns it down onto a 2 mm blade lying on the valve stem; then a torque on the
locknut turns it down onto the rocker. Checks: screw travel per turn (PITCH), no jump past half a turn (unwrapped
angle), the blade stops the screw at a 2 mm gap, the loose nut rides along with the screw, turning the nut leaves the
screw where it is, the lock (raised screw clutch once the nut is seated) and sim speed.

    source ~/IsaacLab-Arena/yam.env && python dev/mech_check.py      # from tasks/valve-lash
"""

import argparse
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from valve_lash import fixture as F  # noqa: E402

DT = 1 / 240
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=DT, device="cuda:0", use_newton_actuators=False,
    physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2)))
stage = sim_utils.get_current_stage()

UsdGeom.Xform.Define(stage, "/World")
F.build_frame(stage, "/World/frame")
F.build_mech(stage, "/World/mech", (0.0, 0.0, 0.0))
# a kinematic 2 mm blade on the stem (as a hand would hold it)
blade = UsdGeom.Xform.Define(stage, "/World/blade").GetPrim()
UsdPhysics.RigidBodyAPI.Apply(blade).CreateKinematicEnabledAttr(True)
c = UsdGeom.Cube.Define(stage, "/World/blade/geom")
c.CreateSizeAttr(1.0)
c.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.03, F.STEM_TOP + 0.001))
c.AddScaleOp().Set(Gf.Vec3f(0.012, 0.11, 0.002))
UsdPhysics.CollisionAPI.Apply(c.GetPrim())

mech = Articulation(ArticulationCfg(prim_path="/World/mech", spawn=None, actuators={}))
t0 = time.perf_counter()
sim.reset()
print(f"reset {time.perf_counter() - t0:.1f} s; joints {mech.joint_names}; bodies {mech.body_names}", flush=True)
J = {n: i for i, n in enumerate(mech.joint_names)}
B = {n: i for i, n in enumerate(mech.body_names)}
wall = [0.0, 0]


def z(body):
    return float(mech.data.body_pos_w.torch[0, B[body], 2])


def yaw(body):
    x, y, zz, w = mech.data.body_quat_w.torch[0, B[body]].cpu().numpy()
    return math.atan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz))


def q(name):
    return float(mech.data.joint_pos.torch[0, J[name]])


def step(n, screw=0.0, nut=0.0):
    t = time.perf_counter()
    tq = torch.zeros(1, 2, 3, device=sim.device)
    tq[0, 0, 2], tq[0, 1, 2] = screw, nut
    for _ in range(n):
        mech.permanent_wrench_composer.set_forces_and_torques_index(
            torques=tq, body_ids=[B["screw"], B["nut"]], is_global=True)
        mech.write_data_to_sim()
        sim.step(render=False)
        mech.update(DT)
        wall[1] += 1
    wall[0] += time.perf_counter() - t


def report(tag):
    gap = F.TIP_Z0 + z("screw") - F.STEM_TOP
    nut_gap = F.NUT_Z0 + z("nut") - F.ROCKER_TOP
    print(f"[{tag}] turn {math.degrees(q('screw_turn')):+8.1f} deg slide {1000 * q('screw_slide'):+7.3f} mm "
          f"(expect {1000 * F.PITCH * q('screw_turn') / (2 * math.pi):+7.3f}) | gap {1000 * gap:6.3f} mm | "
          f"nut turn {math.degrees(q('nut_turn')):+8.1f} slide {1000 * q('nut_slide'):+7.3f} nut gap {1000 * nut_gap:6.3f} mm "
          f"| screw yaw {math.degrees(yaw('screw')):+7.1f}", flush=True)


step(60)
report("start")
for k in range(8):  # a clockwise torque (from above): screw goes down, nut rides along
    step(60, screw=-0.5)
    report(f"screw torque -0.5, {0.25 * (k + 1):.2f} s")
step(240, screw=-0.5)
report("screw torque held on the blade 1 s more")
step(120)
report("released 0.5 s")
g0 = z("screw")
for k in range(6):
    step(60, nut=-0.3)
    report(f"nut torque -0.3, {0.25 * (k + 1):.2f} s")
print(f"screw moved {1000 * (z('screw') - g0):+.3f} mm while the nut turned", flush=True)
step(120)
report("nut released")
seated = F.NUT_Z0 + z("nut") - F.ROCKER_TOP <= F.SEATED
print(f"seated: {seated}", flush=True)
lock = {}
F.set_lock(mech, True, lock)
a0, g0 = q("screw_turn"), z("screw")
for k in range(6):
    step(1200, screw=+2.0 if k % 2 == 0 else -2.0)  # 5 s each way, 30 s in all
    report(f"locked, {'+' if k % 2 == 0 else '-'}2 N*m on the screw, {5 * (k + 1)} s")
print(f"locked screw turned {math.degrees(q('screw_turn') - a0):+.3f} deg, moved {1000 * (z('screw') - g0):+.4f} mm in 30 s",
      flush=True)
F.set_lock(mech, False, lock)
step(240, screw=+0.5)
report("unlocked, +0.5 N*m on the screw 1 s (goes up)")
sim_s = wall[1] * DT
print(f"speed {sim_s:.1f} s sim in {wall[0]:.1f} s = {sim_s / wall[0]:.2f}x real time", flush=True)
app.close()
