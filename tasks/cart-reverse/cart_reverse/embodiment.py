"""`yam_cart`: the stock yam_arena embodiment plus the tow cart (cart_reverse.mech) and the chock. The scorer also gets `left_tcp`, `right_tcp`
(each gripper's fingertip midpoint), `cart` (its frame: on the table under the axle's middle, +x toward the hitch) and `hitch` (pos[0]: the
drawbar's angle from the cart's axis in radians; pos[1]: how far the bar is lifted in radians). The harness is unchanged."""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("CART_DEBUG") == "1"
CART = "cart"


class CartEmbodiment(YamArenaEmbodiment):
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
        m = self.task.module

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd
            from pxr import Gf, UsdGeom

            from . import mech

            stage = omni.usd.get_context().get_stage()
            x = UsdGeom.Xform.Define(stage, prim_path)
            x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            if name == CART:
                qx, qy, qz, qw = m.CART_QUAT
                x.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
                return mech.build_cart(stage, prim_path)
            x.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            return mech.build_part(stage, prim_path, *m.PARTS[name])

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name != CART and spec.name not in m.PARTS:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = ObjectType.ARTICULATION if spec.name == CART else ObjectType.RIGID
            rot = m.CART_QUAT if spec.name == CART else (0.0, 0.0, 0.0, 1.0)
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg, initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=rot))

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
        for name in (CART, *self.task.module.PARTS):
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        art = base.scene[CART]
        q = art.data.joint_pos.torch[0]
        s = np.array((float(q[art.joint_names.index("hitch")]), -float(q[art.joint_names.index("lift")]), 0.0))
        out["hitch"] = ObjectState(pos=s, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=s, high=s)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n != CART and n not in m.PARTS]  # these come from _extra_states
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
            if DEBUG and round(self._t * self.info.control_hz) % round(self.info.control_hz / 4) == 0:
                r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
                print(f"CART t={self._t:6.2f} axle {r(st['cart'].pos)} heading {math.degrees(m.heading(st)):7.2f} tilt {math.degrees(m.tilt(st)):5.2f} hitch {math.degrees(st['hitch'].pos[0]):6.2f} lift {math.degrees(st['hitch'].pos[1]):5.2f} "
                      f"handle {r(m.handle_at(st))} chock {r(st['chock'].pos)} L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} slip {1000 * m._trial['slip']:.1f} bad {m._trial['bad']} why {m._trial['why']} still {m._trial['still']:.2f} near {1000 * m._trial['body_shift']:.1f} {1000 * m._trial['chock_shift']:.1f} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return CartEmbodiment(**kwargs)
