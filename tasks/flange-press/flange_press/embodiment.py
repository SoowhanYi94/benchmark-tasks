"""`yam_flange`: the stock yam_arena embodiment plus the straightening press (flange_press.mech), the harness unchanged. The
scorer also gets `left_tcp` and `right_tcp` (each gripper's fingertip midpoint), `press` (pos: the flange's turn from its start
in radians, the ram's travel down in m, the pointer's reading in m) and `handle` (the crank handle's world position)."""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("FLANGE_DEBUG") == "1"
FRAME, MECH = "press_frame", "press_mech"
OWN = (FRAME, MECH)


def _place(stage, prim_path, translation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return x.GetPrim()


def spawn_frame(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import omni.usd

    from . import mech

    stage = omni.usd.get_context().get_stage()
    _place(stage, prim_path, translation)
    return mech.build_frame(stage, prim_path)


def spawn_mech(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import omni.usd

    from . import mech

    stage = omni.usd.get_context().get_stage()
    _place(stage, prim_path, translation)
    return mech.build_mech(stage, prim_path, tuple(float(v) for v in translation))


class FlangeEmbodiment(YamArenaEmbodiment):
    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        return super().reset(scene, seed=seed)

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

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in OWN:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = {FRAME: spawn_frame, MECH: spawn_mech}[spec.name]
            kind = {FRAME: ObjectType.BASE, MECH: ObjectType.ARTICULATION}[spec.name]
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original

    def _extra_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        base = self._base
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            p = np.asarray(ik.tcp_pose(arr(base.scene[robot].data.joint_pos.torch[0, arm_ids]), np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=p)
        art = base.scene[MECH]
        q = arr(art.data.joint_pos.torch[0])
        qd = arr(art.data.joint_vel.torch[0])
        i = art.joint_names.index
        m = self.task.module
        turn = float(q[i("flange_turn")])
        s = np.array((turn, -q[i("ram_slide")], m.reading(turn)))
        out["press"] = ObjectState(pos=s, quat_xyzw=still, lin_vel=np.array((qd[i("flange_turn")], 0.0, 0.0)), ang_vel=zero, low=s, high=s)
        h = m.handle_pos(turn)
        out["handle"] = ObjectState(pos=h, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=h, high=h)
        return out

    def _set_pointer(self, reading):
        """The dial is a read-out of the flange's angle, written each step (a real press would have a cam and an indicator)."""
        import torch

        art = self._base.scene[MECH]
        art.write_joint_position_to_sim_index(position=torch.tensor([[float(reading)]], device=art.device, dtype=torch.float32),
                                              joint_ids=[art.joint_names.index("pointer_slide")])

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}

        def object_states(env, names):
            names = [n for n in names if n not in OWN]  # these come from _extra_states
            last["s"] = {**original(env, names), **self._extra_states()}
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if "s" in last:
            st = last["s"]
            self._set_pointer(st["press"].pos[2])
            self.task.module.note_step(st, 1.0 / self.info.control_hz)
            if DEBUG and round(self._t * self.info.control_hz) % round(self.info.control_hz / 4) == 0:
                m = self.task.module
                p = st["press"].pos
                print(f"FLANGE t={self._t:5.2f} turn {math.degrees(p[0]):+7.1f} ram {1000 * p[1]:5.1f} dial {1000 * p[2]:5.1f} handle {np.round(1000 * st['handle'].pos, 0).tolist()} "
                      f"good {m._trial['good']} bad {m._trial['bad']} clear {m.hands_clear(st)} still {m._trial['still']:.2f} L {np.round(1000 * st['left_tcp'].pos, 0).tolist()} "
                      f"R {np.round(1000 * st['right_tcp'].pos, 0).tolist()} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return FlangeEmbodiment(**kwargs)
