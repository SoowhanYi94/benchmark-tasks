"""`yam_tig`: the stock yam_arena embodiment plus the welding bench, the TIG torch and the filler rod (tig_seam.mech). The
scorer also gets `left_tcp` and `right_tcp` (each gripper's fingertip midpoint), and `torch` and `rod` in their own frames
(grip centre and sleeve centre). After each step the task's note_step runs, and the bead pieces and torch effects it names
are shown; the rest are hidden. The harness is unchanged."""

import math
import os
import re

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("TIG_DEBUG") == "1"
BENCH = "bench"
TOOLS = ("torch", "rod")
FX = re.compile(r"^((bead_lo|bead_hi|cold|cut|pool)_|hole_s)\d+$")


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


class TigEmbodiment(YamArenaEmbodiment):
    _fx = None

    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        out = super().reset(scene, seed=seed)
        self._show(set())
        return out

    def _ensure_sim(self, goal):
        import atexit

        from yam_arena import inspect_embodiment as base

        if base._APP is None:  # Kit must be up before Arena is imported (its scene module needs PhysxSchema)
            from isaaclab.app import AppLauncher

            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            base._APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
            atexit.register(base._close_app_at_exit)
        from yam_arena import scene

        original = scene.task_object
        m = self.task.module

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd

            from . import mech

            stage = omni.usd.get_context().get_stage()
            _place(stage, prim_path, translation, orientation)
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            if name == BENCH:
                return mech.build_bench(stage, prim_path, m.SEGS)
            if name == "torch":
                return mech.build_torch(stage, prim_path)
            return mech.build_rod(stage, prim_path)

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name != BENCH and spec.name not in TOOLS:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = ObjectType.BASE if spec.name == BENCH else ObjectType.RIGID
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                                            rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original

    def _show(self, on):
        """Show the named effect prims, hide the others (only those that change)."""
        import omni.usd
        from pxr import UsdGeom

        if getattr(self, "_base", None) is None:
            return
        stage = omni.usd.get_context().get_stage()
        if self._fx is None:
            self._fx = {}
            for prim in stage.Traverse():
                n, p = prim.GetName(), str(prim.GetPath())
                if ("/bench/" in p and FX.match(n)) or "/torch/fx_" in p:
                    self._fx[n] = [UsdGeom.Imageable(prim), None]
        for n, rec in self._fx.items():
            want = n in on
            if rec[1] != want:
                rec[0].MakeVisible() if want else rec[0].MakeInvisible()
                rec[1] = want

    def _extra_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        base = self._base
        origin = base.scene.env_origins[0].cpu().numpy().astype(float)
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            p = np.asarray(ik.tcp_pose(arr(base.scene[robot].data.joint_pos.torch[0, arm_ids]), np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=p)
        world = base.arena_world
        for name in TOOLS:
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n != BENCH and n not in TOOLS]  # these come from _extra_states
            last["s"] = {**original(env, names), **self._extra_states()}
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if "s" in last:
            st = last["s"]
            m.note_step(st, 1.0 / self.info.control_hz)
            self._show(m.shown())
            if DEBUG:
                w = m.weld_state(st)
                r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
                tr = m._trial
                tq = st["torch"]
                tilt = math.degrees(math.acos(min(1.0, m._rot(tq.quat_xyzw)[2, 2])))
                print(f"TIG t={self._t:6.2f} h {1000 * w['h']:5.2f} lat {1000 * w['lat']:4.1f} arc {int(tr['arc'])} tip {r(w['tip'])} tilt {tilt:4.1f} "
                      f"rtip {r(w['rtip'])} rod ahead {1000 * w['ahead']:6.1f} rh {1000 * w['rh']:5.1f} rlat {1000 * w['rlat']:5.1f} gap {1000 * w['gap']:5.1f} hands {w['torch_hand']},{w['rod_hand']} "
                      f"L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} torch {r(tq.pos)} rod {r(st['rod'].pos)} dabs {tr['dabs']}/{tr['dabs_cold']} "
                      f"good {int(m.good_segments().sum())} dirty {int(tr['dirty'])} burnt {int(tr['burnt'].sum())} rested {int(m.tools_rested(st))} "
                      f"clear {int(m.hands_clear(st))} still {tr['still']:.2f} score {m.score(st):.3f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return TigEmbodiment(**kwargs)
