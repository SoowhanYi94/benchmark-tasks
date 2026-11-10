"""`yam_hot_tap`: the stock yam_arena embodiment plus the hot-tap line, the tapping machine and the blank (hot_tap.mech). The task file's note_step says
when the cut and the spray happen; here those switch visuals. The scorer also gets `left_tcp`, `right_tcp` (each gripper's fingertip midpoint), `machine`
(its frame), `line` (pos: gate travel toward open, plug travel, both in m) and `feed` (pos[0]: bar travel in m)."""

import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("HT_DEBUG") == "1"
LINE, MACHINE = "line", "machine"


class HotTapEmbodiment(YamArenaEmbodiment):
    _paths = {}
    _shown = (False, False)

    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        self._shown = (False, False)
        self._show(False, False)
        return super().reset(scene, seed=seed)

    def _show(self, cut, spray):
        if LINE not in self._paths:
            return
        import omni.usd

        from . import mech

        stage = omni.usd.get_context().get_stage()
        L, M = self._paths[LINE], self._paths[MACHINE]
        mech.set_visible(stage, f"{L}/plug/tab", not cut)
        mech.set_visible(stage, f"{L}/anchor/hole", cut)
        mech.set_visible(stage, f"{M}/bar/tab", cut)
        mech.set_visible(stage, f"{L}/anchor/spray", spray)

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
        emb = self

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd
            from pxr import Gf, UsdGeom

            from . import mech

            stage = omni.usd.get_context().get_stage()
            x = UsdGeom.Xform.Define(stage, prim_path)
            x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
            x.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            emb._paths = {**emb._paths, name: prim_path}
            if name == LINE:
                return mech.build_line(stage, prim_path, tuple(float(v) for v in translation))
            if name == MACHINE:
                return mech.build_machine(stage, prim_path)
            return mech.build_part(stage, prim_path, *m.PARTS[name])

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in (LINE, MACHINE) and spec.name not in m.PARTS:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = ObjectType.ARTICULATION if spec.name in (LINE, MACHINE) else ObjectType.RIGID
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
        for name in (MACHINE, *self.task.module.PARTS):
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        art = base.scene[MACHINE]
        i = art.joint_names.index("feed")
        s = np.array((-float(art.data.joint_pos.torch[0][i]), 0.0, 0.0))
        out["feed"] = ObjectState(pos=s, quat_xyzw=still, lin_vel=np.array((-float(art.data.joint_vel.torch[0][i]), 0.0, 0.0)), ang_vel=zero, low=s, high=s)
        art = base.scene[LINE]
        q, qd = art.data.joint_pos.torch[0], art.data.joint_vel.torch[0]
        g, u = art.joint_names.index("gate_slide"), art.joint_names.index("plug_slide")
        s = np.array((-float(q[g]), -float(q[u]), 0.0))
        out["line"] = ObjectState(pos=s, quat_xyzw=still, lin_vel=np.array((-float(qd[g]), -float(qd[u]), 0.0)), ang_vel=zero, low=s, high=s)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n not in (LINE, MACHINE) and n not in m.PARTS]  # these come from _extra_states
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
            shown = (m._trial["cut"], m._trial["spray"])
            if shown != self._shown:
                self._show(*shown)
                self._shown = shown
            if DEBUG and round(self._t * self.info.control_hz) % max(1, round(self.info.control_hz / 2)) == 0:
                r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
                print(f"HT t={self._t:6.2f} mach {r(st['machine'].pos)} pitch {m.pitch_deg(st):6.2f} roll {m.roll_deg(st):6.2f} feed {1000 * st['feed'].pos[0]:5.1f} gate {1000 * st['line'].pos[0]:5.1f} plug {1000 * st['line'].pos[1]:4.1f} "
                      f"blank {r(st['blank'].pos)} L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} seat {m.seated(st)} off {m.on_stand(st)} shut {m.gate_shut(st)} blankon {m.blank_on(st)} cut {m._trial['cut']} spray {m._trial['spray']} clear {m.hands_clear(st)} still {m._trial['still']:.2f} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return HotTapEmbodiment(**kwargs)
