"""Feasibility check, no robots: the spindle, its nut and the stack on the GPU pipeline.

The five stack parts are put on the spindle (pushed home) and the nut is written to the thread's start and threaded on
(as the yam_wheel embodiment does). Checks: with the spindle unlocked a torque on the nut turns the spindle, not the
nut on its thread; locked, it runs the nut on at PITCH per turn, seats it on the outer flange and squeezes the stack
(crush); a 90 N push along the axis doesn't turn the nut; the judge's spin (spindle driven, wheel braked) and tug, on
the tightened stack and on a loose one; and sim speed.

    source ~/IsaacLab-Arena/yam.env && python dev/mech_check.py      # from tasks/wheel-mount
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

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wheel_mount import fixture as F  # noqa: E402
import task as T  # noqa: E402

DT = 1 / 240
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=DT, device="cuda:0", use_newton_actuators=False,
    physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2)))
stage = sim_utils.get_current_stage()
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/frame")
F.build_frame(stage, "/World/frame")
F.build_mech(stage, "/World/mech", (0.0, 0.0, 0.0))
parts = {}
for name in F.ORDER:
    x = UsdGeom.Xform.Define(stage, f"/World/{name}")
    x.AddTranslateOp().Set(Gf.Vec3d(0.0, F.HOME[name] - 0.0003, F.ZA - (0.0005 if name == "inner_flange" else 0.0019)))
    F.build_part(stage, f"/World/{name}", name)
    parts[name] = RigidObject(RigidObjectCfg(prim_path=f"/World/{name}", spawn=None))
mech = Articulation(ArticulationCfg(prim_path="/World/mech", spawn=None, actuators={}))
sim.reset()
print(f"joints {mech.joint_names}; bodies {mech.body_names}", flush=True)
J = {n: i for i, n in enumerate(mech.joint_names)}
B = {n: i for i, n in enumerate(mech.body_names)}
LS = {}
wall = [0.0, 0]


def q(name):
    return float(mech.data.joint_pos.torch[0, J[name]])


def wheel_angle():
    x, y, z, w = parts["wheel"].data.root_quat_w.torch[0].cpu().numpy()  # xyzw? isaac lab 3: xyzw
    return 2 * math.atan2(y, w)


HAND = {"target": None}


def nut_abs():
    return q("spindle_turn") + q("nut_turn")


def step(n, nut=0.0, spindle=0.0, nut_push=0.0, brake=0.0, tug=0.0, hand=0.0):
    """``hand``: turn the nut like a hand would, its angle (from the world) ramped at ``hand`` rad/s by a stiff PD
    capped at 0.5 N*m."""
    t = time.perf_counter()
    tq = torch.zeros(1, 2, 3, device=sim.device)
    f = torch.zeros(1, 2, 3, device=sim.device)
    tq[0, 0, 1], tq[0, 1, 1], f[0, 1, 1] = spindle, nut, nut_push
    w = parts["wheel"]
    if hand:
        HAND["target"] = nut_abs()
    for _ in range(n):
        F.set_clutch(mech, "nut_turn", F.NUT_DAMPING, F.nut_hold(-q("nut_crush")))
        if hand:
            HAND["target"] += hand * DT
            w_abs = float(mech.data.joint_vel.torch[0, J["spindle_turn"]] + mech.data.joint_vel.torch[0, J["nut_turn"]])
            tq[0, 1, 1] = max(-1.0, min(1.0, 20.0 * (HAND["target"] - nut_abs()) - 0.2 * w_abs))
        mech.permanent_wrench_composer.set_forces_and_torques_index(
            forces=f, torques=tq, body_ids=[B["spindle"], B["nut"]], is_global=True)
        w.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.tensor([[[0.0, -tug, 0.0]]], device=sim.device),
            torques=torch.tensor([[[0.0, brake, 0.0]]], device=sim.device), is_global=True)
        mech.write_data_to_sim()
        w.write_data_to_sim()
        sim.step(render=False)
        mech.update(DT)
        for p in parts.values():
            p.update(DT)
        wall[1] += 1
    wall[0] += time.perf_counter() - t


def face_y():
    return F.NUT_STAND_Y + q("nut_py") + q("nut_thread") + q("nut_crush")


def report(tag):
    def tilt(n):
        x, y, z, w = parts[n].data.root_quat_w.torch[0].cpu().numpy()
        r_yy = 1 - 2 * (x * x + z * z)
        return math.degrees(math.acos(max(-1.0, min(1.0, r_yy))))

    def dz(n):
        p = parts[n].data.root_pos_w.torch[0].cpu().numpy()
        return 1000 * (p[2] - F.ZA), 1000 * p[0]

    ys = " ".join(f"{n[:5]} {1000 * (float(parts[n].data.root_pos_w.torch[0, 1]) - F.HOME[n]):+5.2f}"
                  f"(t{tilt(n):.1f} z{dz(n)[0]:+.1f} x{dz(n)[1]:+.1f})" for n in F.ORDER)
    print(f"[{tag}] spindle {math.degrees(q('spindle_turn')):+7.1f} nut {math.degrees(q('nut_turn')):+7.1f} deg "
          f"thread {1000 * q('nut_thread'):6.3f} crush {-1000 * q('nut_crush'):6.3f} mm face {1000 * face_y():+7.2f} "
          f"| parts dy {ys} mm | wheel {math.degrees(wheel_angle()):+6.1f} deg", flush=True)


F.lock_here(mech, "spindle_turn", LS)
F.lock_here(mech, "nut_turn", LS)
step(120)
report("start: stack on the spindle, nut on its stand")
F.lock_here(mech, "nut_px", LS, at=0.0)
F.lock_here(mech, "nut_pz", LS, at=0.0)
F.lock_here(mech, "nut_py", LS, at=F.WALL_Q)
F.set_limits(mech, "nut_turn", None, LS)
step(60)
report("threaded on at the thread's start")
F.set_limits(mech, "spindle_turn", None, LS)
step(240, hand=math.radians(45))
report("spindle free, the hand turns the nut 45 deg (spindle should turn with it, nut not advance)")
F.lock_here(mech, "spindle_turn", LS)
RATE = math.radians(45)


def turn_to(deg_on):
    """Turn the nut on (relative to the spindle) to ``deg_on`` degrees from where it caught."""
    d = math.radians(deg_on) - q("nut_turn")
    if abs(d) > 1e-4:
        step(max(1, round(abs(d) / RATE / DT)), hand=math.copysign(RATE, d))
    step(60)


def judge(tag):
    F.set_limits(mech, "spindle_turn", None, LS)
    F.set_clutch(mech, "spindle_turn", T.SPIN_D, 1.0)
    c0, s0, w0 = -q("nut_crush"), q("spindle_turn"), wheel_angle()
    slip = 0.0
    for k in range(round(T.SPIN_S / DT / 10)):
        turned = w0 - wheel_angle()
        step(10, spindle=-T.SPIN_T, brake=T.BRAKE * min(1.0, max(0.0, turned / T.BRAKE_RAMP)))
        slip = max(slip, abs((wheel_angle() - w0) - (q("spindle_turn") - s0)))
    spun = q("spindle_turn") - s0
    F.set_clutch(mech, "spindle_turn", F.SPINDLE_DAMPING, F.SPINDLE_HOLD)
    step(60)
    y0, move = float(parts["wheel"].data.root_pos_w.torch[0, 1]), 0.0
    for k in range(round(T.TUG_S / DT / 6)):
        tug = T.TUG_N * min(1.0, (k + 1) * 6 * DT / T.TUG_RAMP)
        step(6, tug=tug)
        move = max(move, y0 - float(parts["wheel"].data.root_pos_w.torch[0, 1]))
    step(120)
    ok_s, ok_t = slip <= T.SLIP_MAX, move < T.TUG_MOVE
    print(f"JUDGE [{tag}] crush {1000 * c0:.3f} -> {-1000 * q('nut_crush'):.3f} mm | spin {math.degrees(spun):+.1f} deg"
          f" slip {math.degrees(slip):.2f} deg {'ok' if ok_s else 'SLIPPED'} | tug moved {1000 * move:+.3f} mm "
          f"{'ok' if ok_t else 'MOVED'} | hold {F.nut_hold(c0):.3f} N*m", flush=True)
    F.lock_here(mech, "spindle_turn", LS)


SNUG_DEG = F.RUN_DOWN / F.PITCH * 360
for past in (30, 50, 70, 120, 230):
    turn_to(SNUG_DEG + past)
    report(f"turned {past} past snug")
    judge(f"{past} deg past snug")
t0 = q("nut_turn")
step(480, nut_push=90.0)
print(f"90 N push along +y turned the nut {math.degrees(q('nut_turn') - t0):+.2f} deg", flush=True)
sim_s = wall[1] * DT
print(f"speed {sim_s:.1f} s sim in {wall[0]:.1f} s = {sim_s / wall[0]:.2f}x real time", flush=True)
app.close()
