"""`yam_dog`: the stock yam_arena embodiment plus the rail, the carrier and the dogs (runback_dog.mech), the harness
unchanged. The scorer also gets `left_tcp` and `right_tcp` (each gripper's fingertip midpoint) and `carrier` (pos: how far the
carrier has moved from home, m). Each dog's state is its own frame."""

import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("DOG_DEBUG") == "1"
FRAME, MECH = "rail_frame", "carrier_mech"
DOGS = ("dog_a", "dog_b", "dog_c")
OWN = (FRAME, MECH, *DOGS)


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


def spawn_dog(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import omni.usd

    from . import mech

    stage = omni.usd.get_context().get_stage()
    _place(stage, prim_path, translation)
    return mech.build_dog(stage, prim_path)


class DogEmbodiment(YamArenaEmbodiment):
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
        turned = self.task.module.TURNED

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in OWN:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = {FRAME: spawn_frame, MECH: spawn_mech}.get(spec.name, spawn_dog)
            kind = {FRAME: ObjectType.BASE, MECH: ObjectType.ARTICULATION}.get(spec.name, ObjectType.RIGID)
            quat = (0.0, 0.0, 1.0, 0.0) if spec.name == turned else (0.0, 0.0, 0.0, 1.0)  # the backwards dog: 180 deg about z
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=quat))

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
        art = base.scene[MECH]
        q = arr(art.data.joint_pos.torch[0])
        c = np.array((q[art.joint_names.index("carrier_slide")], q[art.joint_names.index("pin_slide")], 0.0))
        out["carrier"] = ObjectState(pos=c, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=c, high=c)
        world = base.arena_world
        for name in DOGS:
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
            r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
            print(f"DOG t={self._t:5.2f} carrier {r(st['carrier'].pos)[:2]} forward {[m.forward(st, n) for n in DOGS]} "
                  f"a {r(st['dog_a'].pos)} b {r(st['dog_b'].pos)} c {r(st['dog_c'].pos)} clear {m.hands_clear(st)} still {m._trial['still']:.2f} "
                  f"L {np.round(1000 * st['left_tcp'].pos, 0).tolist()} R {np.round(1000 * st['right_tcp'].pos, 0).tolist()} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return DogEmbodiment(**kwargs)
