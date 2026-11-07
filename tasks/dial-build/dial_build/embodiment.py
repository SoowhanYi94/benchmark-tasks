"""`yam_dial`: the stock yam_arena embodiment plus the assembly dial (dial_build.mech), two blocks and two pegs. The stop pad's travel
stops the dial for good, through the turn joint's limit. The scorer also gets `left_tcp`, `right_tcp` (each gripper's fingertip
midpoint) and `dial` (pos[0]: the turn in radians; pos[1]: the pad's travel in m; pos[2]: 1.0 once stopped)."""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("DIAL_DEBUG") == "1"
DIAL = "dial"


class DialEmbodiment(YamArenaEmbodiment):
    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        self._stopped = False
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
            if name == DIAL:
                return mech.build_dial(stage, prim_path, tuple(float(v) for v in translation), m.PAD_AT_REL, m.FIXTURES)
            return mech.build_part(stage, prim_path, *m.PARTS[name])

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name != DIAL and spec.name not in m.PARTS:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = ObjectType.ARTICULATION if spec.name == DIAL else ObjectType.RIGID
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
        origin = base.scene.env_origins[0].cpu().numpy().astype(float)
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            p = np.asarray(ik.tcp_pose(arr(base.scene[robot].data.joint_pos.torch[0, arm_ids]), np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=p)
        world = base.arena_world
        for name in self.task.module.PARTS:
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        art = base.scene[DIAL]
        q, qd = art.data.joint_pos.torch[0], art.data.joint_vel.torch[0]
        i, j = art.joint_names.index("turn"), art.joint_names.index("pad_slide")
        s = np.array((float(q[i]), -float(q[j]), 1.0 if getattr(self, "_stopped", False) else 0.0))
        out["dial"] = ObjectState(pos=s, quat_xyzw=still, lin_vel=np.array((float(qd[i]), float(qd[j]), 0.0)), ang_vel=zero, low=s, high=s)
        return out

    def _stop(self, turn):
        """The stop pad: the dial stops where it is, for good."""
        import torch

        art = self._base.scene[DIAL]
        i = art.joint_names.index("turn")
        limits = torch.tensor([[[-1.0 * math.pi / 180, float(turn)]]], device=art.device, dtype=torch.float32)
        art.write_joint_position_limit_to_sim_index(limits=limits, joint_ids=[i], warn_limit_violation=False)
        self._stopped = True

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n != DIAL and n not in m.PARTS]  # these come from _extra_states
            last["s"] = {**original(env, names), **self._extra_states()}
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if "s" in last:
            st = last["s"]
            if not self._stopped and st["dial"].pos[1] >= m.PAD_PRESSED:
                self._stop(st["dial"].pos[0])
            m.note_step(st, 1.0 / self.info.control_hz)
            if DEBUG and round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0:
                r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
                off = {n: r(m._offset(st, n, k)[:2]) for k, n in enumerate(("block_a", "block_b"))}
                print(f"DIAL t={self._t:6.2f} turn {math.degrees(st['dial'].pos[0]):7.2f} rate {math.degrees(st['dial'].lin_vel[0]):5.2f} pad {1000 * st['dial'].pos[1]:4.1f} "
                      f"A {r(st['block_a'].pos)} {off['block_a']} B {r(st['block_b'].pos)} {off['block_b']} pegA {r(st['peg_a'].pos)} pegB {r(st['peg_b'].pos)} L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} "
                      f"seat {[m.block_seated(st, k) for k in (0, 1)]} peg {[m.peg_seated(st, k) for k in (0, 1)]} bad {m._trial['bad']} clear {m.hands_clear(st)} still {m._trial['still']:.2f} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return DialEmbodiment(**kwargs)
