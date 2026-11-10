"""`yam_pulley`: the stock yam_arena embodiment plus the gearmotor, its pulley and two pry bars (pulley_pry.mech).

The task file lists placeholder boxes: `machine` (static: frame, gearmotor, bearing housing, shaft, slide rails, bar rests, V-block),
`pulley` (a free body) and `bar_l`, `bar_r` (free bodies). The pulley's seized fit is one maximal joint (`seat`) whose only effect is a
capped axial drag; the task's note_step says when the pulley has cleared the seized band, and the joint is then switched off. The
scorer also gets `left_tcp`, `right_tcp` (each gripper's fingertip midpoint) and `seat` (pos[0]: 1.0 while the seat joint is on).

Debug only: PULLEY_PUSH="mode:F:ramp_s" applies a force with no arm: mode `both`/`left`/`right` pushes each named bar's handle
toward -x (F rising from 0 over ramp_s, 210 mm from the tip, plus 0.85 N up to stand in for the hand's support), `pull` pulls the
pulley's hub along +x on its axis."""

import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("PULLEY_DEBUG") == "1"
PUSH = os.environ.get("PULLEY_PUSH", "")
BARS = ("bar_l", "bar_r")
HANDLE = 0.21
LIFT = 0.0


def _place(stage, prim_path, translation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


class PulleyEmbodiment(YamArenaEmbodiment):
    _seat = None
    _last = None

    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        self._set_seat(True)
        self._last = None
        return super().reset(scene, seed=seed)

    def _set_seat(self, on):
        if os.environ.get("PULLEY_NOSEAT") == "1":  # debugging only: no seized fit at all
            on = False
        if self._seat is not None:
            self._seat.GetAttribute("physics:jointEnabled").Set(bool(on))
        self._seat_on = bool(on)

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

            from . import mech

            stage = omni.usd.get_context().get_stage()
            _place(stage, prim_path, translation)
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            if name == "machine":
                return mech.build_machine(stage, prim_path, m.RACK, m.VBLOCK, debug_stand=m.SEATED)
            if name == "pulley":
                prim, emb._seat = mech.build_pulley(stage, prim_path, tuple(float(v) for v in translation), m.DRAG, m.DRAG_DAMPING)
                return prim
            return mech.build_bar(stage, prim_path, 1 if name == "bar_l" else -1, "rubber_red" if name == "bar_l" else "rubber_yel")

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in m.SPAWNED:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = ObjectType.BASE if spec.name == "machine" else ObjectType.RIGID
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
        for name in ("pulley", *BARS):
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        s = np.array((1.0 if self._seat_on else 0.0, getattr(self, "_push_f", 0.0), 0.0))
        out["seat"] = ObjectState(pos=s, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=s, high=s)
        return out

    def _apply_push(self):
        """Debug only: the PULLEY_PUSH force, set before each control step."""
        import torch

        mode, fmax, ramp = PUSH.split(":")
        f = min(float(fmax), float(fmax) * self._t / float(ramp))
        self._push_f = f
        st, base = self._last, self._base
        origin = base.scene.env_origins[0].cpu().numpy().astype(float)
        if st is None:  # the first step: the bars as spawned
            from yam_arena.task_file import ObjectState

            st = {n: ObjectState(pos=np.array(o.pos, float), quat_xyzw=np.array((0.0, 0.0, 0.0, 1.0)), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=None, high=None)
                  for o in self.task.module.OBJECTS for n in (o.name,)}
        if mode == "pull":
            p = st["pulley"]
            at = np.asarray(p.pos) + _rot(p.quat_xyzw) @ np.array((0.045, 0.0, 0.0)) + origin
            obj = base.scene["pulley"]
            obj.permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor([[[f, 0.0, 0.0]]], device=base.device), positions=torch.tensor([[at.tolist()]], device=base.device, dtype=torch.float32),
                body_ids=[0], is_global=True)
            return
        for name, sg in (("bar_l", 1), ("bar_r", -1)):
            b = st[name]
            at = np.asarray(b.pos) + _rot(b.quat_xyzw) @ np.array((0.0, sg * HANDLE, 0.0)) + origin
            k = {"both": 1.0, "left": float(name == "bar_l"), "right": float(name == "bar_r"), "skew": 1.0 if name == "bar_l" else 0.5}[mode]
            base.scene[name].permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor([[[-f * k, 0.0, LIFT]]], device=base.device), positions=torch.tensor([[at.tolist()]], device=base.device, dtype=torch.float32),
                body_ids=[0], is_global=True)

    def step(self, action):
        if PUSH:
            self._apply_push()
        return super().step(action)

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n not in m.SPAWNED]  # these come from _extra_states
            last["s"] = {**original(env, names), **self._extra_states()}
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if "s" in last:
            st = last["s"]
            self._last = st
            m.note_step(st, 1.0 / self.info.control_hz)
            if self._seat_on and m._trial["loose"]:
                self._set_seat(False)
            if DEBUG and round(self._t * self.info.control_hz) % max(1, round(self.info.control_hz / 4)) == 0:
                r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
                yaw, pitch = m.cock(st)
                print(f"PULLEY t={self._t:6.2f} F {getattr(self, '_push_f', 0.0):5.1f} travel {1000 * m.travel(st):6.2f} yaw {yaw:5.2f} pitch {pitch:5.2f} "
                      f"pos {np.round(1e4 * (np.asarray(st['pulley'].pos) - (m.XP, 0.0, m.TOP + m.ZS0))).tolist()} q {np.round(st['pulley'].quat_xyzw, 4).tolist()} barL {r(st['bar_l'].pos)} {m.bar_turn(st, 'bar_l'):5.1f}/{m.bar_tilt(st, 'bar_l'):4.1f} barR {r(st['bar_r'].pos)} {m.bar_turn(st, 'bar_r'):5.1f}/{m.bar_tilt(st, 'bar_r'):4.1f} "
                      f"L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} seat {self._seat_on} trial {m._trial} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return PulleyEmbodiment(**kwargs)
