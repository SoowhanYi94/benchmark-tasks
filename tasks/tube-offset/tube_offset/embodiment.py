"""`yam_tube_offset`: the stock yam_arena embodiment plus the hand bender and the tube (tube_offset.mech). Every step it writes the tube's two
hinge limits from the task's `ratchet`. The scorer also gets `left_tcp`, `right_tcp` (each gripper's fingertip midpoint), `tube_mid`, `tube_nose`,
`tube_tail` (the three segments' poses), `tube_q` (pos: the two bends in radians) and `handle` (pos[0]: its sweep in radians)."""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("TUBE_DEBUG") == "1"
BENDER, TUBE = "bender", "tube"


class TubeOffsetEmbodiment(YamArenaEmbodiment):
    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        self._written = None
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
            x.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            if name == BENDER:
                return mech.build_bender(stage, prim_path, tuple(float(v) for v in translation), m.LANES)
            return mech.build_tube(stage, prim_path)

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in (BENDER, TUBE):
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
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            p = np.asarray(ik.tcp_pose(arr(base.scene[robot].data.joint_pos.torch[0, arm_ids]), np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=p)
        tube = base.scene[TUBE]
        pos, quat = tube.data.body_link_pos_w.torch[0], tube.data.body_link_quat_w.torch[0]
        lin, ang = tube.data.body_link_lin_vel_w.torch[0], tube.data.body_link_ang_vel_w.torch[0]
        for n in ("mid", "nose", "tail"):
            i = tube.body_names.index(n)
            p = arr(pos[i]) - origin
            out[f"tube_{n}"] = ObjectState(pos=p, quat_xyzw=arr(quat[i]), lin_vel=arr(lin[i]), ang_vel=arr(ang[i]), low=p, high=p)
        q = tube.data.joint_pos.torch[0]
        s = np.array((float(q[tube.joint_names.index("bend_1")]), float(q[tube.joint_names.index("bend_2")]), 0.0))
        out["tube_q"] = ObjectState(pos=s, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=s, high=s)
        b = base.scene[BENDER]
        h = np.array((float(b.data.joint_pos.torch[0][b.joint_names.index("sweep")]), 0.0, 0.0))
        out["handle"] = ObjectState(pos=h, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=h, high=h)
        return out

    def _write_limits(self, limits):
        """Only when a limit's moved: a band not being bent stays locked where it's set."""
        import torch

        tube = self._base.scene[TUBE]
        flat = np.round(np.array(limits, float), 4)
        if self._written is not None and np.all(np.abs(flat - self._written) < 1e-3):
            return
        self._written = flat
        ids = [tube.joint_names.index("bend_1"), tube.joint_names.index("bend_2")]
        lim = np.array(limits, float)
        lim[:, 0] -= 0.002
        lim[:, 1] += 0.002
        t = torch.tensor(lim[None], device=tube.device, dtype=torch.float32)
        tube.write_joint_position_limit_to_sim_index(limits=t, joint_ids=ids, warn_limit_violation=False)

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n not in (BENDER, TUBE)]  # these come from _extra_states
            last["s"] = {**original(env, names), **self._extra_states()}
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if "s" in last:
            st = last["s"]
            dt = 1.0 / self.info.control_hz
            self._write_limits(m.ratchet(st, dt))
            m.note_step(st, dt)
            if DEBUG and round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0:
                r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
                d = lambda v: round(math.degrees(v), 2)  # noqa: E731
                print(f"COND t={self._t:6.2f} phi {d(st['handle'].pos[0]):7.2f} gap {[None if g is None else round(1000 * g, 1) for g in (m.shoe_gap(st, 0, -1), m.shoe_gap(st, 1, -1))]} q {d(st['tube_q'].pos[0]):6.2f} {d(st['tube_q'].pos[1]):6.2f} peak {d(m._trial['peak'][0]):5.1f} {d(m._trial['peak'][1]):5.1f} "
                      f"on {m._trial['on']} done {m._trial['done']} mid {r(st['tube_mid'].pos)} yaw {d(m._heading(st['tube_mid'])):7.2f} J1 {r(m.band_at(st, 0))} J2 {r(m.band_at(st, 1))} "
                      f"ang {round(float(np.linalg.norm(st['tube_mid'].ang_vel)), 3)} L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} bad {m._trial['why']} go {m.let_go(st)} still {m._trial['still']:.2f} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return TubeOffsetEmbodiment(**kwargs)
