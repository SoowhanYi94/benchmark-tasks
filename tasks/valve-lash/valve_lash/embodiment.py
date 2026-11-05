"""`yam_valve`: the stock yam_arena embodiment plus the valve-lash fixture and the feeler gauge (valve_lash.fixture),
the harness itself unchanged.

The task file lists placeholder boxes: `head_frame` and `head_mech` place the fixture (spawned here as its fixed
frame and its screw-and-locknut articulation) and `gauge` is spawned as the feeler gauge (handle and blade, one free
body). The scorer gets more states each step: `screw` and `nut` (each body's frame: its z is how far it has moved
from the start), `gauge_root` (the gauge's own frame: origin under the handle's centre), and `left_tcp` /
`right_tcp` (each gripper's fingertip centre).

The lock: while the locknut is seated (its underside within `fixture.SEATED` of the rocker's top), the screw's clutch
is replaced by a hard hold (its joint limits closed on where it is), so the screw no longer turns; loosen the nut and
it turns again. `blade_used` (pos[0] 1 or 0) latches for the trial once the blade has rested on the stem with the
screw's tip down on it (task.blade_touched); only a reset clears it.
"""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

from . import fixture as F

DEBUG = os.environ.get("VALVE_DEBUG") == "1"
PLACEHOLDERS = ("head_frame", "head_mech")


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


def _stage():
    import omni.usd

    return omni.usd.get_context().get_stage()


def spawn_frame(prim_path, cfg, translation=None, orientation=None, **kwargs):
    stage = _stage()
    _place(stage, prim_path, translation, orientation)
    return F.build_frame(stage, prim_path)


def spawn_mech(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import UsdGeom

    stage = _stage()
    prim = _place(stage, prim_path, translation, orientation)
    m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return F.build_mech(stage, prim_path, tuple(m.ExtractTranslation()), m.ExtractRotationQuat())


def spawn_gauge(prim_path, cfg, translation=None, orientation=None, **kwargs):
    stage = _stage()
    _place(stage, prim_path, translation, orientation)
    return F.build_gauge(stage, prim_path)


class ValveEmbodiment(YamArenaEmbodiment):
    _locked = None
    _blade_used = False

    def reset(self, scene, *, seed=None):
        self._last = None
        self._blade_used = False  # a new trial: the harness resets here, which a robot can't do
        if self._locked:
            self._set_lock(False)  # put the screw's limits back before the reset moves it home
        self._locked = None
        obs = super().reset(scene, seed=seed)
        self._set_lock(False)
        return obs

    def _mech(self):
        return self._base.scene["head_mech"]

    def _set_lock(self, locked):
        if locked == self._locked:
            return
        if not hasattr(self, "_lock_state"):
            self._lock_state = {}
        F.set_lock(self._mech(), locked, self._lock_state)
        self._locked = locked
        if DEBUG:
            print(f"LOCK t={self._t:5.2f} {'locked' if locked else 'free'}", flush=True)

    def step(self, action):
        if self._last is not None:
            self._set_lock(bool(self.task.module.nut_gap(self._last) <= F.SEATED))
        return super().step(action)

    def _ensure_sim(self, goal):
        import atexit

        from yam_arena import inspect_embodiment as base

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

            funcs = {"head_frame": (spawn_frame, ObjectType.BASE), "head_mech": (spawn_mech, ObjectType.ARTICULATION),
                     "gauge": (spawn_gauge, ObjectType.RIGID)}
            if spec.name in funcs:
                func, kind = funcs[spec.name]
                cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
                cfg.func = func
                return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                              initial_pose=Pose(position_xyz=tuple(spec.pos),
                                                rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))
            return original_object(spec)

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original_object

    def _extra_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        base = self._base
        origin = base.scene.env_origins[0].cpu().numpy().astype(float)
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero = np.zeros(3)
        out = {}
        mech = self._mech()
        for name in ("screw", "nut"):
            i = mech.body_names.index(name)
            p = arr(mech.data.body_pos_w.torch[0, i]) - origin
            out[name] = ObjectState(pos=p, quat_xyzw=arr(mech.data.body_quat_w.torch[0, i]),
                                    lin_vel=arr(mech.data.body_lin_vel_w.torch[0, i]),
                                    ang_vel=arr(mech.data.body_ang_vel_w.torch[0, i]), low=p, high=p)
        g = base.scene["gauge"]
        p = arr(g.data.root_pos_w.torch[0]) - origin
        out["gauge_root"] = ObjectState(pos=p, quat_xyzw=arr(g.data.root_quat_w.torch[0]),
                                        lin_vel=arr(g.data.root_lin_vel_w.torch[0]),
                                        ang_vel=arr(g.data.root_ang_vel_w.torch[0]), low=p, high=p)
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            q = arr(base.scene[robot].data.joint_pos.torch[0, arm_ids])
            p, _ = ik.tcp_pose(q, np.array(BASES[arm]))
            p = np.asarray(p, float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=np.array((0.0, 0.0, 0.0, 1.0)), lin_vel=zero,
                                            ang_vel=zero, low=p, high=p)
        return out

    def _task_step(self):
        from yam_arena import scene
        from yam_arena.task_file import ObjectState

        original = scene.object_states

        def object_states(env, names):
            names = [n for n in names if n not in PLACEHOLDERS]
            states = {**original(env, names), **self._extra_states()}
            if self.task.module.blade_touched(states):
                self._blade_used = True
            u = np.array((float(self._blade_used), 0.0, 0.0))
            states["blade_used"] = ObjectState(pos=u, quat_xyzw=np.array((0.0, 0.0, 0.0, 1.0)), lin_vel=np.zeros(3),
                                               ang_vel=np.zeros(3), low=u, high=u)
            self._last = states
            return states

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if DEBUG and self._last is not None and round(self._t * self.info.control_hz) % round(self.info.control_hz / 4) == 0:
            m, st = self.task.module, self._last
            g = st["gauge_root"]
            print(f"VALVE t={self._t:5.2f} gap {1000 * m.gap(st):6.3f} mm nut gap {1000 * m.nut_gap(st):6.3f} mm "
                  f"screw yaw {math.degrees(m.yaw(st['screw'].quat_xyzw)):+7.1f} nut yaw {math.degrees(m.yaw(st['nut'].quat_xyzw)):+7.1f} "
                  f"blade under axis {1000 * (m._blade_bottom_at_axis(st) - m.TOP):.2f} | blade in gap {m.blade_in_gap(st)} clear {m.blade_clear(st)} hands {m.hands_clear(st)} | gauge "
                  f"{np.round(1000 * g.pos, 1).tolist()} tilt {math.degrees(m._tilt(g.quat_xyzw)):.1f} | L {np.round(1000 * st['left_tcp'].pos, 0).tolist()} "
                  f"R {np.round(1000 * st['right_tcp'].pos, 0).tolist()} locked {self._locked} used {self._blade_used} | score {m.score(st):.2f}",
                  flush=True)
        return result


def make_embodiment(**kwargs):
    return ValveEmbodiment(**kwargs)
