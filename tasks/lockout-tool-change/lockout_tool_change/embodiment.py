"""`yam_lockout`: the stock yam_arena embodiment plus the machine's control box (lockout_tool_change.button), the
harness itself unchanged.

The task file lists a placeholder box, `control_box`, whose position places the control box; here it is spawned as
the box, its sprung START button and its run lamp. The scorer gets three more states each step: `start_button` (its
pos is the cap's top centre: pushed down, z drops), and `left_tcp` / `right_tcp` (each gripper's TCP).

The run lamp: the machine starts (lamp lit) when START is pressed while the isolator is energised (task.energised),
and stops (lamp dark) when it is de-energised. With the isolator at OFF it stays dark: zero energy, verified.
Set LOCKOUT_DEBUG=1 to print every score term, the parts, the button, the lamp and the trial flags each second
and whenever a flag changes.
"""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("LOCKOUT_DEBUG") == "1"
PLACEHOLDER = "control_box"
_PATH = {}


def spawn_control_box(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import omni.usd
    from pxr import Gf, UsdGeom

    from . import button

    m = _task_module()
    stage = omni.usd.get_context().get_stage()
    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    w = UsdGeom.XformCache().GetLocalToWorldTransform(x.GetPrim())
    _PATH["root"] = prim_path
    return button.build(stage, prim_path, tuple(w.ExtractTranslation()), w.ExtractRotationQuat(), m.CONTROL["size"],
                        m.BUTTON_R, m.BUTTON_PROUD, m.BUTTON_TRAVEL, m.LAMP["offset"], m.LAMP["radius"])


def _task_module():
    return _TASK["module"]


_TASK = {}


class LockoutEmbodiment(YamArenaEmbodiment):
    _running = False

    def reset(self, scene, *, seed=None):
        self._running = False
        obs = super().reset(scene, seed=seed)
        self._lamp(False)
        return obs

    def _lamp(self, on):
        import omni.usd

        from . import button

        if "root" in _PATH:
            button.set_lamp(omni.usd.get_context().get_stage(), _PATH["root"], on)

    def _ensure_sim(self, goal):
        import atexit

        from yam_arena import inspect_embodiment as base

        _TASK["module"] = self.task.module
        if base._APP is None:  # Kit must be up before Arena is imported (its scene module needs PhysxSchema)
            from isaaclab.app import AppLauncher

            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            base._APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
            atexit.register(base._close_app_at_exit)
        from yam_arena import scene

        original_object = scene.task_object

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name == PLACEHOLDER:
                cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
                cfg.func = spawn_control_box
                return Object(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg,
                              initial_pose=Pose(position_xyz=tuple(spec.pos),
                                                rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))
            return original_object(spec)

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original_object

    def _debug(self, m, st, running):
        flags = {k: m._trial.get(k) for k in _FLAGS}
        step = round(self._t * self.info.control_hz)
        if step % self.info.control_hz == 0 or flags != getattr(self, "_flags", None):
            print(_debug_line(m, st, self._t, running), flush=True)
        self._flags = {k: m._trial.get(k) for k in _FLAGS}

    def _extra_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        base, m = self._base, self.task.module
        origin = base.scene.env_origins[0].cpu().numpy().astype(float)
        box = base.scene[PLACEHOLDER]
        i = box.body_names.index("cap")
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        top = arr(box.data.body_pos_w.torch[0, i]) - origin + (0.0, 0.0, m.CONTROL["size"][2] + m.BUTTON_PROUD)
        zero, ident = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {"start_button": ObjectState(pos=top, quat_xyzw=ident, lin_vel=arr(box.data.body_lin_vel_w.torch[0, i]),
                                           ang_vel=zero, low=top, high=top)}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            q = arr(base.scene[robot].data.joint_pos.torch[0, arm_ids])
            p = np.asarray(ik.tcp_pose(q, np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=ident, lin_vel=zero, ang_vel=zero, low=p, high=p)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}

        def object_states(env, names):
            names = [n for n in names if n != PLACEHOLDER]  # the button comes from _extra_states
            states = {**original(env, names), **self._extra_states()}
            last["s"] = states
            return states

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        st = last.get("s")
        if st is not None:
            m = self.task.module
            running = m.energised(st) and (self._running or m.pressed(st))
            if running != self._running:
                self._lamp(running)
            self._running = running
            if DEBUG:
                self._debug(m, st, running)
        return result


_FLAGS = ("trials", "violated", "locked_ever", "verified_ever", "verified", "lock_ok")


def _debug_line(m, st, t, running):
    r = lambda o: np.round(1000 * np.asarray(o.pos[:3]), 0).astype(int).tolist()  # noqa: E731
    tr = m._trial
    return (f"LOCKOUT t={t:5.1f} score {m.score(st):.2f} | off {m.handle_at(m.OFF_X, st)} on {m.handle_at(m.ON_X, st)} "
            f"lock_behind {m.lock_behind(st)} on_mark {m.lock_on_mark(st)} locked {m.locked_out(st)} safe {m.safe(st)} "
            f"scrap {m.worn_in_scrap(st)} seated {m.new_cutter_seated(st)} restored {m.restored(st)} | "
            f"button {1000 * (m.BUTTON_TOP - st['start_button'].pos[2]):+5.1f} mm lamp {running} | handle {r(st['isolator_handle'])} "
            f"tilt {np.degrees(m._tilt(st['isolator_handle'].quat_xyzw)):.0f} rest {st['isolator_handle'].resting()} "
            f"padlock {r(st['padlock'])} low {1000 * (st['padlock'].low[2] - m.TOP):.0f} rest {st['padlock'].resting()} "
            f"worn {r(st['worn_cutter'])} new {r(st['new_cutter'])} L {r(st['left_tcp'])} R {r(st['right_tcp'])} | "
            f"{ {k: tr.get(k) for k in _FLAGS} }")


def make_embodiment(**kwargs):
    return LockoutEmbodiment(**kwargs)
