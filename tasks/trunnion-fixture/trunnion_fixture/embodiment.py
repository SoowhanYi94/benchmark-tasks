"""`yam_trunnion`: the stock yam_arena embodiment plus the rollover fixture, the gearbox, its flange and screws and the screwdriver
(trunnion_fixture.mech). Each control step it hands the task file the state, then carries out what the task file's screw rule decides
(glue off, thread on, thread target) and the scorer's lift force on the flange. The scorer also gets `left_tcp`, `right_tcp` (each gripper's
fingertip midpoint) and `fixture` (pos: the roll in radians, the bar's travel in m, the bolt's travel in m; lin_vel: their rates)."""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("TRUNNION_DEBUG") == "1"
DEBUG_FROM, DEBUG_TO = (float(v) for v in os.environ.get("TRUNNION_DEBUG_SPAN", "-1,-1").split(","))
FIXTURE = "fixture"
PATHS = {}


class TrunnionEmbodiment(YamArenaEmbodiment):
    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        self._placed = False
        out = super().reset(scene, seed=seed)
        try:
            from . import mech

            for i in range(4):
                p = PATHS.get(f"screw{i}")
                if p:
                    for j in range(4):
                        mech.set_thread(p, j, False)
                    mech.set_glue(p, False)
                    mech.set_glue(p, True)
        except Exception as e:  # noqa: BLE001  (the stage is not up before the first reset)
            print("TRN reset switches skipped:", e, flush=True)
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
        m = self.task.module

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd
            from pxr import Gf, UsdGeom

            from . import mech

            stage = omni.usd.get_context().get_stage()
            x = UsdGeom.Xform.Define(stage, prim_path)
            x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            qx, qy, qz, qw = m.START_Q.get(name, (0.0, 0.0, 0.0, 1.0))
            x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
            PATHS[name] = prim_path
            if name == FIXTURE:
                return mech.build_fixture(stage, prim_path, tuple(float(v) for v in translation))
            return mech.build_part(stage, prim_path, name)

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name != FIXTURE and spec.name not in m.PARTS:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = ObjectType.ARTICULATION if spec.name == FIXTURE else ObjectType.RIGID
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(float(v) for v in m.START_Q.get(spec.name, (0.0, 0.0, 0.0, 1.0)))))

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original

    def _origin(self):
        return self._base.scene.env_origins[0].cpu().numpy().astype(float)

    def _extra_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        base = self._base
        origin = self._origin()
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            p, rm = ik.tcp_pose(arr(base.scene[robot].data.joint_pos.torch[0, arm_ids]), np.array(BASES[arm]))
            p = np.asarray(p, float)
            self._yaw = getattr(self, "_yaw", {})
            self._yaw[arm] = math.degrees(math.atan2(float(np.asarray(rm)[1, 0]), float(np.asarray(rm)[0, 0])))
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=p)
        world = base.arena_world
        for name in self.task.module.PARTS:
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        art = base.scene[FIXTURE]
        q, qd = art.data.joint_pos.torch[0], art.data.joint_vel.torch[0]
        i, j, k = (art.joint_names.index(n) for n in ("roll", "bar_slide", "bolt_slide"))
        s = np.array((float(q[i]), float(q[j]), -float(q[k])))
        out[FIXTURE] = ObjectState(pos=s, quat_xyzw=still, lin_vel=np.array((float(qd[i]), float(qd[j]), -float(qd[k]))), ang_vel=zero, low=s, high=s)
        return out

    def _place_start(self):
        """Put the free parts back exactly where they start, at rest, just before the first step (the arena's reset can disturb them)."""
        import torch

        m = self.task.module
        origin = self._origin()
        for name in m.PARTS:
            asset = self._base.scene[name]
            dev = asset.data.root_pos_w.torch.device if hasattr(asset.data.root_pos_w, "torch") else "cuda:0"
            pose = torch.tensor([[*(np.asarray(m.START[name]) + origin), *m.START_Q.get(name, (0.0, 0.0, 0.0, 1.0))]], dtype=torch.float32, device=dev)
            asset.write_root_pose_to_sim_index(root_pose=pose)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=dev))

    def _apply(self):
        from . import mech

        m = self.task.module
        for a in m.take_actions():
            p = PATHS[f"screw{a[1]}"]
            if a[0] == "glue":
                mech.set_glue(p, a[2])
            elif a[0] == "thread":
                mech.set_thread(p, a[2], a[3], a[4])
            else:
                mech.set_thread_target(p, a[2], a[3])
            if a[0] != "target":
                print(f"TRN t={self._t:6.2f} switch {a}", flush=True)
        import torch

        fl = self._base.scene["flange"]
        f = float(m.lift_force())
        if f != getattr(self, "_lift", 0.0):
            dev = fl.data.root_pos_w.torch.device if hasattr(fl.data.root_pos_w, "torch") else "cuda:0"
            fl.permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor([[[0.0, 0.0, f]]], dtype=torch.float32, device=dev),
                torques=torch.zeros((1, 1, 3), dtype=torch.float32, device=dev), body_ids=[0], env_ids=torch.tensor([0], device=dev), is_global=True)
            self._lift = f
        tq = np.asarray(m.driver_torque(self._last_state), float) if getattr(self, "_last_state", None) is not None else np.zeros(3)
        if np.any(tq != 0.0) or np.any(getattr(self, "_tq", np.zeros(3)) != 0.0):  # the thread's resistance on the driver (the task file's rule)
            dr = self._base.scene["driver"]
            dev = dr.data.root_pos_w.torch.device if hasattr(dr.data.root_pos_w, "torch") else "cuda:0"
            dr.permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.zeros((1, 1, 3), dtype=torch.float32, device=dev),
                torques=torch.tensor([[[float(v) for v in tq]]], dtype=torch.float32, device=dev), body_ids=[0], env_ids=torch.tensor([0], device=dev), is_global=True)
            self._tq = tq

    def _task_step(self):
        from yam_arena import scene

        if not getattr(self, "_placed", True):
            self._place_start()
            self._placed = True
        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n != FIXTURE and n not in m.PARTS]  # these come from _extra_states
            last["s"] = {**original(env, names), **self._extra_states()}
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if "s" in last:
            st = last["s"]
            if not getattr(self, "_start_logged", False):
                d = max(float(np.linalg.norm(np.asarray(st[n].pos) - np.asarray(m.START[n]))) for n in m.PARTS)
                print(f"TRN start check: largest part offset from its start {1000 * d:.2f} mm, gearbox in {m.gb_in(st)}, roll {math.degrees(st[FIXTURE].pos[0]):.2f}", flush=True)
                self._start_logged = True
            m.note_step(st, 1.0 / self.info.control_hz)
            self._last_state = st
            self._apply()
            if DEBUG and (round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0 or (DEBUG_FROM <= self._t <= DEBUG_TO)):
                r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
                f = st[FIXTURE]
                gaps, off = m.flange_gaps(st)
                dr = m._rot(st["driver"].quat_xyzw)
                sc = " ".join(f"{i}:{'T' if s['on'] else '-'}{s['hole'] if s['on'] else ''} g{1000 * m.head_gap(st, i):.1f} b{int(m.bit_in(st, i))} @{r(st[f'screw{i}'].pos[:2])} t{s['turn']:.1f} s{math.degrees(s['snug']):.0f} h{np.round(1000 * np.array(m._hole_of(st, i)[1:]), 1).tolist()} {np.round(np.array(m.bit_parts(st, i)) * (1000, 1000, 57.3), 1).tolist()}" for i, s in enumerate(m._trial["screws"]))
                print(f"TRN t={self._t:6.2f} roll {math.degrees(f.pos[0]):7.2f} bar {1000 * f.pos[1]:5.1f} bolt {1000 * f.pos[2]:4.1f} gb {r(m.gb_offset(st))} tilt {math.degrees(m.gb_tilt(st)):4.1f} "
                      f"fl gaps {r(gaps)} off {1000 * off:.1f} {r(m._local(st, 'gearbox', st['flange'].pos)[:2])} | {sc} | drv {r(st['driver'].pos)} L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} "
                      f"lift {m._trial['lift']} ok {m._trial['lift_ok']} max {1000 * m._trial['lift_max']:.2f} bad {m._trial['bad']} {m._trial['why']} clear {m.hands_clear(st)} "
                      f"still {m._trial['still']:.2f} score {m.score(st):.2f} | Ryaw {self._yaw.get('right', 0):.1f} drvyaw {math.degrees(math.atan2(dr[1, 0], dr[0, 0])):.1f} drvtilt {math.degrees(math.acos(min(1.0, dr[2, 2]))):.1f} tq {np.round(getattr(self, '_tq', np.zeros(3)), 2).tolist()}", flush=True)
        return result


def make_embodiment(**kwargs):
    return TrunnionEmbodiment(**kwargs)
