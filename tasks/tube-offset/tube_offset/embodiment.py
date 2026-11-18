"""`yam_tube_offset`: the stock yam_arena embodiment plus the task's own articulations (the hand tube bender and the tube,
built by tube_offset.mech). The task file supplies:

- ``ARTS``: the names of its articulated objects (each also a placeholder ``Box`` in ``OBJECTS``);
- ``build(stage, name, prim_path, origin)``: writes that object's USD and returns its root prim;
- ``on_step(emb, states, dt)``: called after every control step (it writes the tube's plastic-hinge anchors and the
  latch spring through ``emb.art(name)``);
- ``reset_latch()``: forgets the trial at every reset.

The scorer gets ``left_tcp`` and ``right_tcp`` (each gripper's fingertip midpoint), ``<art>/<body>`` for every body
of every articulation (pose and velocity, table frame), and ``states["q"]["<art>/<joint>"]`` (joint positions, rad or m).
"""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("TUBE_DEBUG") == "1"


class TubeOffsetEmbodiment(YamArenaEmbodiment):
    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        self.store = {}
        return super().reset(scene, seed=seed)

    def art(self, name):
        return self._base.scene[name]

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
        yaws = {o.name: float(getattr(o, "yaw", 0.0)) for o in m.OBJECTS}

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            x = UsdGeom.Xform.Define(stage, prim_path)
            x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            h = yaws.get(name, 0.0) / 2  # authored turned as placed, so nothing starts inside the bender before the reset
            x.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, 0.0, math.sin(h))))
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            return m.build(stage, name, prim_path, tuple(float(v) for v in (translation or (0.0, 0.0, 0.0))))

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in m.ARTS:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            y = spec.yaw / 2
            return Object(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=(0.0, 0.0, math.sin(y), math.cos(y))))

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
        self.origin = origin
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out, q = {}, {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, finger_ids = self._ids[robot]
            jp = base.scene[robot].data.joint_pos.torch[0]
            p = np.asarray(ik.tcp_pose(arr(jp[arm_ids]), np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=p)
            q[f"{arm}/finger"] = float(arr(jp[finger_ids]).mean())
            q[f"{arm}/wrist"] = arr(jp[arm_ids]).tolist()
        for name in self.task.module.ARTS:
            a = base.scene[name]
            pos, quat = a.data.body_link_pos_w.torch[0], a.data.body_link_quat_w.torch[0]
            lin, ang = a.data.body_link_lin_vel_w.torch[0], a.data.body_link_ang_vel_w.torch[0]
            P, Q, L, A = arr(pos) - origin, arr(quat), arr(lin), arr(ang)
            for i, b in enumerate(a.body_names):
                out[f"{name}/{b}"] = ObjectState(pos=P[i], quat_xyzw=Q[i], lin_vel=L[i], ang_vel=A[i], low=P[i], high=P[i])
            jq = arr(a.data.joint_pos.torch[0])
            for i, j in enumerate(a.joint_names):
                q[f"{name}/{j}"] = float(jq[i])
        out["q"] = q
        out["t"] = float(self._t)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n not in m.ARTS]  # these come from _extra_states
            last["s"] = {**original(env, names), **self._extra_states()}
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if "s" in last:
            m.on_step(self, last["s"], 1.0 / self.info.control_hz)
        return result


def make_embodiment(**kwargs):
    return TubeOffsetEmbodiment(**kwargs)
