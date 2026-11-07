"""`yam_deadweight`: the stock yam_arena embodiment plus the dead-weight tester (deadweight_test.mech) and its slotted weights.

The piston is a prismatic joint on a soft spring drive, the oil's compliance: the weights' real mass sinks it, and each pump stroke moves
the spring's target up. The scorer also gets `left_tcp`, `right_tcp` (each gripper's fingertip midpoint), and `rig`: pos = (piston height m,
lever degrees, knob degrees), lin_vel = their rates, ang_vel[2] = the rotor's rate in rad/s. The needle and the flag are read-outs written
each step, with no colliders."""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("DW_DEBUG") == "1"
RIG = "rig"


class DeadweightEmbodiment(YamArenaEmbodiment):
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
            x.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            if name == RIG:
                return mech.build_tester(stage, prim_path, tuple(float(v) for v in translation), m.GEOM)
            return mech.build_weight(stage, prim_path, m.WEIGHT, m.GEOM["spin_damping"], m.BRASS)

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name != RIG and spec.name not in m.WEIGHTS:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = ObjectType.ARTICULATION if spec.name == RIG else ObjectType.RIGID
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
        for name in self.task.module.WEIGHTS:
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        art = base.scene[RIG]
        q, qd = art.data.joint_pos.torch[0], art.data.joint_vel.torch[0]
        i = art.joint_names.index
        s = np.array((float(q[i("rise")]), math.degrees(float(q[i("pump")])), math.degrees(float(q[i("trim")]))))
        rate = np.array((float(qd[i("rise")]), math.degrees(float(qd[i("pump")])), math.degrees(float(qd[i("trim")]))))
        out[RIG] = ObjectState(pos=s, quat_xyzw=still, lin_vel=rate, ang_vel=np.array((0.0, 0.0, float(qd[i("spin")]))), low=s, high=s)
        return out

    def _write_rig(self, st):
        """The oil's spring target, and the two read-outs."""
        import torch

        m = self.task.module
        art = self._base.scene[RIG]
        i = art.joint_names.index
        # the spring's target: an articulation with no actuator models takes it straight from the view
        view = art.root_view
        targets = view.get_dof_position_targets()
        t = (targets.torch if hasattr(targets, "torch") else torch.as_tensor(targets)).clone()
        t = t.reshape(1, -1)
        names = list(view.shared_metatype.dof_names)
        t[0, names.index("rise")] = float(m._trial["oil"])
        import warp as wp

        view.set_dof_position_targets(wp.from_torch(t, dtype=wp.float32), art._ALL_INDICES)
        ids = [i("needle_turn"), i("flag_slide")]
        pos = torch.tensor([[-math.radians(m.needle_error(st)), m.GEOM["flag_gain"] * float(st[RIG].pos[0])]], device=art.device, dtype=torch.float32)
        art.write_joint_position_to_sim_index(position=pos, joint_ids=ids)

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n != RIG and n not in m.WEIGHTS]  # these come from _extra_states
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
            self._write_rig(st)
            if DEBUG:
                r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
                n = getattr(self, "_n", 0) + 1
                self._n = n
                fast = m._trial["spinning"] or os.environ.get("DW_DEBUG_ALL") == "1"
                if n % (2 if fast else 10) == 0:
                    rig = st[RIG]
                    print(f"DW t={n / self.info.control_hz:6.2f} h {1000 * rig.pos[0]:5.1f} oil {1000 * m._trial['oil']:5.1f} lever {rig.pos[1]:6.1f} knob {rig.pos[2]:6.1f} rotor {rig.ang_vel[2]:6.2f} "
                          f"w {[round(float(st[n2].ang_vel[2]), 2) for n2 in m.WEIGHTS]} off {[r(m.offset_on_stem(st, n2)) for n2 in m.WEIGHTS]} needle {m.needle_error(st):6.1f} stick {m._trial['stick']:5.1f} "
                          f"freed {m._trial['freed']} set {m._trial['set_spinning']} L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} on {[m.weight_on(st, n2) for n2 in m.WEIGHTS]} "
                          f"bad {m._trial['bad']} still {m._trial['still']:.2f} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return DeadweightEmbodiment(**kwargs)
