"""Mechanism check, no robots: does the key turn the pinion by contact, do the jaws follow (mimic joints), clamp the
bar with a preload, hold it when the key is let go, and keep it against a 30 N pull out of the chuck?

    source ~/IsaacLab-Arena/yam.env && python dev/chuck_check.py      # from tasks/chuck-key
Env: TORQUE (N*m cap on the key, 1.0), TARGET_DEG (690), RENDER=1 (video).
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

RENDER = os.environ.get("RENDER") == "1"
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = RENDER
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chuck_key import lathe as L  # noqa: E402

OUT = Path(os.environ.get("OUT", Path.home() / "yam-eval/chuck-key/chuck_check"))
OUT.mkdir(parents=True, exist_ok=True)
TORQUE = float(os.environ.get("TORQUE", 1.0))
DT = 1 / 240
PULL = float(os.environ.get("PULL", 30.0))

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=DT, device="cuda:0", render_interval=8,
    physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2)))
stage = sim_utils.get_current_stage()
table = UsdGeom.Cube.Define(stage, "/World/table")
table.CreateSizeAttr(1.0)
table.AddTranslateOp().Set(Gf.Vec3d(0, 0.05, -0.02))
table.AddScaleOp().Set(Gf.Vec3f(0.6, 0.6, 0.04))
UsdPhysics.CollisionAPI.Apply(table.GetPrim())
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800.0)

UsdGeom.Xform.Define(stage, "/World/lathe")
L.build_lathe(stage, "/World/lathe", (0.0, 0.0, 0.0))
key = UsdGeom.Xform.Define(stage, "/World/key")
key.AddTranslateOp().Set(Gf.Vec3d(0.0, L.SOCKET_Y, L.KEY_SEATED_Z + 0.001))
L.build_key(stage, "/World/key")
bar = UsdGeom.Xform.Define(stage, "/World/bar")
BAR_Y0 = -0.05 + L.BAR_LEN / 2  # 50 mm out of the jaws
REST_Z = L.AXIS_Z - (L.R_OPEN - L.BAR_R) / math.sin(math.radians(30))  # on the two lower jaws
bar.AddTranslateOp().Set(Gf.Vec3d(0.0, BAR_Y0, REST_Z + 0.0005))
L.build_bar(stage, "/World/bar")

mech = Articulation(ArticulationCfg(prim_path="/World/lathe", spawn=None, actuators={}))
key_a = RigidObject(RigidObjectCfg(prim_path="/World/key", spawn=None))
bar_a = RigidObject(RigidObjectCfg(prim_path="/World/bar", spawn=None))
cam = None
if RENDER:
    from isaaclab.sensors import Camera, CameraCfg

    cam = Camera(CameraCfg(prim_path="/World/cam", width=640, height=400, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 10.0))))
sim.reset()
if cam is not None:
    cam.set_world_poses_from_view(torch.tensor([[0.17, -0.20, 0.20]], device=sim.device),
                                  torch.tensor([[0.0, 0.02, 0.05]], device=sim.device))
names = list(mech.joint_names)
print("joints", names, flush=True)
iP = names.index("pinion_turn")
iJ = [names.index(f"jaw_{k}_slide") for k in range(3)]
iG = [names.index(f"jaw_{k}_give") for k in range(3)]
frames, st = [], {"steps": 0, "wall": 0.0}


def step(n=1, torque=0.0, pull=0.0):
    t = time.perf_counter()
    for _ in range(n):
        key_a.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.zeros(1, 1, 3, device=sim.device),
            torques=torch.tensor([[[0.0, 0.0, torque]]], device=sim.device), is_global=True)
        bar_a.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.tensor([[[0.0, -pull, 0.0]]], device=sim.device), is_global=True)
        for a in (mech, key_a, bar_a):
            a.write_data_to_sim()
        sim.step(render=cam is not None and st["steps"] % 8 == 0)
        for a in (mech, key_a, bar_a):
            a.update(DT)
        st["steps"] += 1
        if cam is not None and st["steps"] % 8 == 0:
            cam.update(DT * 8, force_recompute=True)
            frames.append(cam.data.output["rgb"].torch[0, :, :, :3].cpu().numpy().astype(np.uint8))
    st["wall"] += time.perf_counter() - t


def show(tag):
    q = mech.data.joint_pos.torch[0].cpu().numpy()
    b = bar_a.data.root_pos_w.torch[0].cpu().numpy()
    qb = bar_a.data.root_quat_w.torch[0].cpu().numpy()  # xyzw
    x_, y_, z_, w_ = qb
    d = np.array((2 * (x_ * y_ - w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ + w_ * x_)))  # bar axis
    b = b + d * (0.0125 - b[1]) / d[1]  # its axis at the jaws' mid-depth
    k = key_a.data.root_pos_w.torch[0].cpu().numpy()
    gaps = []
    for j in range(3):
        u = L.jaw_dir(j)
        gaps.append(1000 * (L.jaw_face_r(q[iJ[j]] + q[iG[j]]) - (u[0] * b[0] + u[1] * (b[2] - L.AXIS_Z)) - L.BAR_R))
    print(f"[{tag}] t={st['steps'] * DT:5.2f} pinion {math.degrees(q[iP]):7.1f} deg (expect jaws {1000 * L.GEAR * q[iP]:.2f} mm) "
          f"jaws {np.round(1000 * q[iJ], 3).tolist()} give {np.round(1000 * q[iG], 3).tolist()} mm, gaps {np.round(gaps, 3).tolist()} mm | bar "
          f"{np.round(1000 * (b - (0, BAR_Y0, L.AXIS_Z)), 2).tolist()} mm tilt {math.degrees(math.acos(min(1, abs(d[1])))):.2f} deg | key z {1000 * (k[2] - L.KEY_SEATED_Z):+.2f} mm", flush=True)


step(int(0.5 / DT))
show("settled")
TARGET = math.radians(float(os.environ.get("TARGET_DEG", 690)))
OMEGA = math.radians(45)
k = 0
while float(mech.data.joint_pos.torch[0, iP]) < TARGET and k < int(20 / DT):
    w = float(key_a.data.root_ang_vel_w.torch[0, 2])
    step(1, torque=min(TORQUE, max(0.0, 0.25 + 0.5 * (OMEGA - w))))
    k += 1
    if k % 120 == 0:
        show("turning")
show("reached target")
step(int(1.0 / DT), torque=0.0)
show("key let go 1 s")
y0 = float(bar_a.data.root_pos_w.torch[0, 1])
n = int(0.5 / DT)
for i in range(n):
    step(1, pull=PULL * (i + 1) / n)
step(int(1.0 / DT), pull=PULL)
show(f"{PULL:.0f} N pull")
print(f"pull: bar moved {1000 * (y0 - float(bar_a.data.root_pos_w.torch[0, 1])):+.3f} mm out", flush=True)
step(int(0.5 / DT), torque=-TORQUE)
show("loosening")
step(int(3.0 / DT), torque=-TORQUE)
show("loosened")
print(f"speed {st['steps'] * DT / st['wall']:.2f}x real time", flush=True)
if frames:
    import imageio.v2 as imageio

    imageio.mimsave(OUT / "chuck_check.mp4", frames, fps=30)
    print("video", OUT / "chuck_check.mp4")
app.close()
