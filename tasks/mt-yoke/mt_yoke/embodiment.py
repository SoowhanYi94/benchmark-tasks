"""`yam_mt`: the stock yam_arena embodiment plus the test bench, the AC yoke, the powder bulb (a free articulation) and three
marker pegs (mt_yoke.mech). The scorer also gets `left_tcp` and `right_tcp` (each gripper's fingertip midpoint; low/high
its two pads), `yoke`, `bulb` (its body's frame) and `peg0`-`peg2` in their own frames, and `squeeze` (pos[0]: the bulb
plunger's travel in m; lin_vel[0]: its rate). After each step the task's note_step runs, and the powder, indications, lamp
and puff it names are shown; the rest are hidden. The harness is unchanged."""

import math
import os
import re

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("MT_DEBUG") == "1"
BENCH, YOKE, BULB = "bench", "yoke", "bulb"
PEGS = ("peg0", "peg1", "peg2")
OWN = (BENCH, YOKE, BULB) + PEGS
FX = re.compile(r"^(dust_\d+|ind_\w+|fx_\w+)$")


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


class MtEmbodiment(YamArenaEmbodiment):
    _fx = None
    _yaw = {}

    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        out = super().reset(scene, seed=seed)
        self._show(set())
        return out

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

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd

            from . import mech

            stage = omni.usd.get_context().get_stage()
            _place(stage, prim_path, translation, orientation)
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            build = {BENCH: mech.build_bench, YOKE: mech.build_yoke, BULB: mech.build_bulb}.get(name, mech.build_peg)
            return build(stage, prim_path)

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in OWN:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = {BENCH: ObjectType.BASE, BULB: ObjectType.ARTICULATION}.get(spec.name, ObjectType.RIGID)
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                                            rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original

    def _show(self, on):
        """Show the named effect prims, hide the others (only those that change)."""
        import omni.usd
        from pxr import UsdGeom

        if getattr(self, "_base", None) is None:
            return
        stage = omni.usd.get_context().get_stage()
        if self._fx is None:
            self._fx = {}
            for prim in stage.Traverse():
                n, p = prim.GetName(), str(prim.GetPath())
                if FX.match(n) and any(f"/{o}/" in p for o in (BENCH, YOKE, BULB)):
                    self._fx[n] = [UsdGeom.Imageable(prim), None]
        for n, rec in self._fx.items():
            want = n in on
            if rec[1] != want:
                rec[0].MakeVisible() if want else rec[0].MakeInvisible()
                rec[1] = want

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
            arm_ids, finger_ids = self._ids[robot]
            p, R = ik.tcp_pose(arr(base.scene[robot].data.joint_pos.torch[0, arm_ids]), np.array(BASES[arm]))
            p, R = np.asarray(p, float), np.asarray(R, float)
            half = -float(np.mean(arr(base.scene[robot].data.joint_pos.torch[0, finger_ids])))  # each pad's distance from the midpoint
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p - half * R[:, 0], high=p + half * R[:, 0])
            self._yaw[arm] = math.degrees(math.atan2(R[0, 0], -R[1, 0]))  # debug only: the grip's yaw (0 closes along world y)
        world = base.arena_world
        for name in (YOKE,) + PEGS:
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        art = base.scene[BULB]
        d = art.data
        pose = arr(d.root_link_pose_w.torch[0])
        p = pose[:3] - origin
        out[BULB] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(d.root_link_lin_vel_w.torch[0]), ang_vel=arr(d.root_link_ang_vel_w.torch[0]), low=p, high=p)
        j = art.joint_names.index("squeeze")
        q, qd = float(arr(d.joint_pos.torch[0])[j]), float(arr(d.joint_vel.torch[0])[j])
        out["squeeze"] = ObjectState(pos=np.array((q, 0.0, 0.0)), quat_xyzw=still, lin_vel=np.array((qd, 0.0, 0.0)), ang_vel=zero, low=zero, high=zero)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n not in OWN]  # these come from _extra_states
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
            self._show(m.shown())
            if DEBUG and round(self._t * self.info.control_hz) % 5 == 0:
                r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
                f = m.feet(st)
                tilt = math.degrees(math.acos(min(1.0, m._rot(st[YOKE].quat_xyzw)[2, 2])))
                print(f"MT t={self._t:6.2f} field {int(m._trial['field'])} dir {m.direction(st)} feet {r(f[:, :2].ravel())} fz {r(f[:, 2] - m.PLATE_Z)} tilt {tilt:4.1f} "
                      f"yh {m.yoke_hand(st)} bh {m.bulb_hand(st)} gapL {1000 * m.pad_gap(st, 'left_tcp'):4.1f} gapR {1000 * m.pad_gap(st, 'right_tcp'):4.1f} "
                      f"yawL {self._yaw.get('left', 0):6.1f} yawR {self._yaw.get('right', 0):6.1f} hb {math.degrees(m._turn(m._rot(st[BULB].quat_xyzw))):6.1f} hy {math.degrees(m._turn(m._rot(st[YOKE].quat_xyzw))):6.1f} sq {1000 * m.squeeze(st):4.1f} noz {r(m.nozzle(st))} nzh {1000 * (m.nozzle(st)[2] - m.PLATE_Z):5.1f} L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} "
                      f"yoke {r(st[YOKE].pos)} bulb {r(st[BULB].pos)} pegs {[r(st[n].pos) for n in PEGS]} | {m.report(st)} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return MtEmbodiment(**kwargs)
