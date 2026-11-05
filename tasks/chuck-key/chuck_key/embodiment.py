"""`yam_chuck`: the stock yam_arena embodiment plus the lathe (chuck_key.lathe), the harness itself unchanged.

The task file lists three placeholder boxes: `lathe` (spawned as the lathe's articulation: headstock, chuck, pinion,
jaws, key holder, V-block), `bar` and `key` (spawned as the round bar and the chuck key, free rigid bodies). The scorer
gets each step: `bar` and `key` (their body frames: centre / shoulder, and orientation), `jaws` (pos = each jaw face's
radius from the spindle axis, m), `pinion` (pos = its angle and its angle when all three jaws first touched the bar, rad; nan until then), `left_tcp` / `right_tcp` (each gripper's fingertip
centre) and `pull_test`.

The judge's pull: while the task's `pull_ready(objects)` holds (jaws on the bar, hands clear, key out), a force pulling the bar out of the chuck (-y) ramps to
`PULL_N` over `PULL_RAMP_S` and stays there. If the bar slips `PULL_SLIP` the pull stops and counts as failed until
the key turns the pinion another 2 degrees or a hand comes back within HANDS_OFF of the bar. `pull_test`'s pos is
(force N, slip m, seconds held at the full force); a failed pull reads force 0.
"""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("CHUCK_DEBUG") == "1"
CUSTOM = ("lathe", "bar", "key")
RETEST_TURN = math.radians(2)


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


def _spawner(build):
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = _place(stage, prim_path, translation, orientation)
        origin = tuple(UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation())
        return build(stage, prim_path, origin)

    return spawn


def _build(name):
    from . import lathe

    return {"lathe": lambda st, p, o: lathe.build_lathe(st, p, o),
            "bar": lambda st, p, o: lathe.build_bar(st, p),
            "key": lambda st, p, o: lathe.build_key(st, p)}[name]


class ChuckEmbodiment(YamArenaEmbodiment):
    _pull = None
    _last = None

    def reset(self, scene, *, seed=None):
        self._pull = {"force": 0.0, "y0": None, "failed": False, "held": 0.0, "turn": 0.0}
        self._last = None
        self._contact = None
        obs = super().reset(scene, seed=seed)
        self._set_pull(0.0)
        return obs

    def _obj(self, name):
        return self._base.scene[name]

    def _set_pull(self, force):
        import torch

        f = torch.tensor([[[0.0, -float(force), 0.0]]], device=self._base.device)
        self._obj("bar").permanent_wrench_composer.set_forces_and_torques_index(forces=f, is_global=True)

    def step(self, action):
        m = self.task.module
        p = self._pull
        last = self._last
        if last is not None and m.pull_ready(last):
            y = float(last["bar"].pos[1])
            if p["y0"] is None:
                p["y0"] = y
            if not p["failed"] and p["y0"] - y >= m.PULL_SLIP:
                p.update(failed=True, force=0.0, held=0.0, turn=float(last["pinion"].pos[0]))
            elif not p["failed"]:
                full = p["force"] >= m.PULL_N
                p["force"] = min(m.PULL_N, p["force"] + m.PULL_N / (m.PULL_RAMP_S * self.info.control_hz))
                p["held"] = p["held"] + 1.0 / self.info.control_hz if full else 0.0
        else:
            # a failed pull stays failed until the robot tightens further or goes back to the bar
            retest = last is None or not m.hands_off(last) or abs(float(last["pinion"].pos[0]) - p["turn"]) >= RETEST_TURN
            p.update(force=0.0, y0=None, held=0.0, failed=p["failed"] and not retest)
        if p["failed"] and last is not None and abs(float(last["pinion"].pos[0]) - p["turn"]) >= RETEST_TURN:
            p.update(failed=False, y0=None, force=0.0, held=0.0)
        if DEBUG and (p["y0"] is not None or p["force"]):
            print(f"PULL t={self._t:5.2f} {p}", flush=True)
        self._set_pull(p["force"])
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

            if spec.name not in CUSTOM:
                return original_object(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = _spawner(_build(spec.name))
            kind = ObjectType.ARTICULATION if spec.name == "lathe" else ObjectType.RIGID
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(spec.pos),
                                            rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original_object

    def _states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        from . import lathe

        base = self._base
        origin = base.scene.env_origins[0].cpu().numpy().astype(float)
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, unit = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))

        def point(p):
            p = np.asarray(p, float)
            return ObjectState(pos=p, quat_xyzw=unit, lin_vel=zero, ang_vel=zero, low=p, high=p)

        out = {}
        for name in ("bar", "key"):
            d = self._obj(name).data
            pos = arr(d.root_pos_w.torch[0]) - origin
            out[name] = ObjectState(pos=pos, quat_xyzw=arr(d.root_quat_w.torch[0]),  # Isaac Lab 3: already xyzw
                                   
                                    lin_vel=arr(d.root_lin_vel_w.torch[0]), ang_vel=arr(d.root_ang_vel_w.torch[0]),
                                    low=pos, high=pos)
        mech = self._obj("lathe")
        names = list(mech.joint_names)
        q = arr(mech.data.joint_pos.torch[0])
        out["jaws"] = point([lathe.jaw_face_r(q[names.index(f"jaw_{k}_slide")] + q[names.index(f"jaw_{k}_give")])
                             for k in range(3)])
        out["pinion"] = point((q[names.index("pinion_turn")], 0.0, 0.0))
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            p, _ = ik.tcp_pose(arr(base.scene[robot].data.joint_pos.torch[0, arm_ids]), np.array(BASES[arm]))
            out[f"{arm}_tcp"] = point(p)
        pull = self._pull or {"force": 0.0, "y0": None, "held": 0.0}
        slip = 0.0 if pull["y0"] is None else pull["y0"] - float(out["bar"].pos[1])
        out["pull_test"] = point((pull["force"], slip, pull["held"]))
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states

        def object_states(env, names):
            names = [n for n in names if n not in CUSTOM]
            states = {**original(env, names), **self._states()}
            m = self.task.module
            angle = float(states["pinion"].pos[0])
            states["pinion"] = states["pinion"].__class__(**{**states["pinion"].__dict__, "pos": np.array((angle, math.nan, 0.0))})
            if not m.inserted(states):
                self._contact = None  # the bar came out: first contact is measured afresh
            elif self._contact is None and m.clamped(states):
                self._contact = angle
            if self._contact is not None:
                states["pinion"].pos[1] = self._contact
            self._last = states
            return states

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if DEBUG and self._last is not None and round(self._t * self.info.control_hz) % round(self.info.control_hz / 4) == 0:
            m, st = self.task.module, self._last
            print(f"CHUCK t={self._t:5.2f} pinion {math.degrees(st['pinion'].pos[0]):6.1f} deg | bar stick-out "
                  f"{1000 * m.stick_out(st):5.1f} mm offset {np.round(1000 * m.bar_offset(st), 2).tolist()} tilt "
                  f"{math.degrees(m.bar_tilt(st)):.2f} gaps {np.round(1000 * m.jaw_gaps(st), 2).tolist()} | ins {m.inserted(st)} "
                  f"clamp {m.clamped(st)} past {math.degrees(m.turned_past_contact(st)):.0f} hands_off {m.hands_off(st)} pull {np.round(st['pull_test'].pos, 4).tolist()} | key "
                  f"{np.round(1000 * st['key'].pos, 1).tolist()} tilt {math.degrees(m.key_tilt(st)):.1f} on_chuck "
                  f"{m.key_on_chuck(st)} home {m.key_home(st)} | L {np.round(1000 * st['left_tcp'].pos).tolist()} "
                  f"R {np.round(1000 * st['right_tcp'].pos).tolist()} | score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return ChuckEmbodiment(**kwargs)
