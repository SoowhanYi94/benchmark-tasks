"""`yam_escapement`: the stock yam_arena embodiment plus the chute, stops and parts (two_stop_escapement.mech), the
harness unchanged. The task file lists placeholder boxes that are spawned as those bodies, tilted as the chute is. The
scorer also gets `left_tcp` and `right_tcp` (the point midway between each gripper's fingertips) each step. A part's
state is its own frame (mech.PART), not a bounding box."""

import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("ESCAPEMENT_DEBUG") == "1"
FRAME, STOPS = "chute_frame", "chute_stops"
PARTS = tuple(f"part_{i + 1}" for i in range(6))
OWN = (FRAME, STOPS, *PARTS)
_RED = {}  # prim name -> shows red, filled by make_object


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


def spawn_frame(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import omni.usd

    from . import mech

    stage = omni.usd.get_context().get_stage()
    _place(stage, prim_path, translation, mech.tilt_xyzw())
    return mech.build_frame(stage, prim_path)


def spawn_stops(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import omni.usd
    from pxr import UsdGeom

    from . import mech

    stage = omni.usd.get_context().get_stage()
    prim = _place(stage, prim_path, translation, mech.tilt_xyzw())
    m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return mech.build_mech(stage, prim_path, tuple(m.ExtractTranslation()), m.ExtractRotationQuat())


def spawn_part(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import omni.usd

    from . import mech

    stage = omni.usd.get_context().get_stage()
    _place(stage, prim_path, translation, mech.tilt_xyzw())
    return mech.build_part(stage, prim_path, _RED[prim_path.rstrip("/").rsplit("/", 1)[-1]])


class EscapementEmbodiment(YamArenaEmbodiment):
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

        from . import mech

        original = scene.task_object
        _RED.update({n: n == self.task.module.RED_NAME for n in PARTS})

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in OWN:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = {FRAME: spawn_frame, STOPS: spawn_stops}.get(spec.name, spawn_part)
            kind = {FRAME: ObjectType.BASE, STOPS: ObjectType.ARTICULATION}.get(spec.name, ObjectType.RIGID)
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=mech.tilt_xyzw()))

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
        origin = base.scene.env_origins[0].cpu().numpy().astype(float)
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            p = np.asarray(ik.tcp_pose(arr(base.scene[robot].data.joint_pos.torch[0, arm_ids]), np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=p)
        world = base.arena_world
        for name in PARTS:
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        return out

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
            self.task.module.note_step(last["s"], 1.0 / self.info.control_hz)
        if DEBUG and "s" in last and round(self._t * self.info.control_hz) % round(self.info.control_hz / 4) == 0:
            m, st = self.task.module, last["s"]
            mech_art = self._base.scene[STOPS]
            q = mech_art.data.joint_pos.torch[0].detach().cpu().numpy()
            print(f"ESC t={self._t:5.2f} joints {dict(zip(mech_art.joint_names, np.round(q, 4).tolist()))} "
                  f"s {[round(1000 * float(m.chute_y(st[n])), 1) for n in PARTS]} "
                  f"released {m.released(st)} held {m.red_held(st)} clear {m.hands_clear(st)} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return EscapementEmbodiment(**kwargs)
