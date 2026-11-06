"""`yam_wheel`: the stock yam_arena embodiment plus the bench grinder, its spindle and nut, the loose wheel-mounting
parts and the parts tray (wheel_mount.fixture), the harness itself unchanged.

The task file lists placeholder boxes: `grinder_frame` places the fixed frame (grinder body and tray),
`grinder_mech` the spindle-nut-button articulation, and one box per loose part (`inner_flange`, `blotter_a`,
`wheel`, `blotter_b`, `outer_flange`, `decoy`), spawned here as that part (a free rigid ring). The scorer gets more
states each step: `<part>_root` (each part's own frame: its centre, its axis along y), `spindle` (pos = spindle angle
rad, the lock button's travel m (down positive), 1 if the spindle is locked), `nut` (pos = the nut's slides x, y, z
from its stand, m), `thread` (pos = the nut's turn on the spindle rad, its advance along the thread m, the crush m:
how far the stack has pushed its face back), `latch` (pos = 1 once the nut is threaded on, 1 once the wheel has been
cracked by overtightening, 0), `judge` (pos = phase: 0 idle, 1 spin, 2 tug, 3 passed, 4 failed the spin (the wheel
slipped), 5 failed the tug (the wheel moved); the wheel's slip on the spindle in the spin, rad; how far the tug moved
the wheel, m), and `left_tcp` / `right_tcp` (each gripper's
fingertip centre).

Mechanism rules (wheel_mount.fixture has the geometry):
- The thread's start: its free run before the seat is fixture.RUN_DOWN (1.81 mm).
- Threading on: until it is on the thread the nut's slides are free and it can't turn; once its face is within
  CAPTURE_Y of the thread's start and its centre within CAPTURE_R of the axis, its slides lock (it is snapped onto
  the axis) and it turns on its thread. It can't come back off. Only a reset undoes this.
- The spindle is held still until the nut is on (nothing else can turn it). After that it turns freely in its
  bearings unless the lock button is pressed down PRESSED or more.
- Tightening: the torque it takes to turn the nut on is its free-run drag plus a term in the clamp force (the crush
  spring's force; fixture.nut_hold), updated each step.
- Overtightening: the nut's turn is recorded when it first seats (the crush reaches the task's SEATED); once it has
  turned on the task's CRACK_TURN (240 degrees) past that, the wheel is cracked for the trial (latched; only a reset
  clears it). (The rigid stack yields by contact at high load, so the crush alone stops rising near 1 mm.)
- The judge: while the task's `judge_ready(objects)` holds (the stack complete, the nut seated, hands clear), the
  spindle is driven (SPIN_T against a SPIN_D damper) with the wheel held back by a brake (BRAKE, ramped in over the wheel's first BRAKE_RAMP of turn) for SPIN_S; only the stack's
  friction, set by the clamp force, drives the wheel, so a loose nut lets it slip: if it slips on the spindle more
  than SLIP_MAX the test fails. Then a tug pulls the wheel along -y (TUG_N, ramped over TUG_RAMP, TUG_S in all): the
  crush spring gives unless its preload exceeds the pull, so if the wheel moves TUG_MOVE the test fails. A test that passed or failed stays so until `judge_ready` stops holding (a hand comes back, say).
"""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

from . import fixture as F

DEBUG = os.environ.get("WHEEL_DEBUG") == "1"
PLACEHOLDERS = ("grinder_frame", "grinder_mech")
PART_NAMES = tuple(F.PARTS)


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


def _part_spawner(name):
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        stage = _stage()
        _place(stage, prim_path, translation, orientation)
        return F.build_part(stage, prim_path, name)

    return spawn


class WheelEmbodiment(YamArenaEmbodiment):
    _last = None

    def reset(self, scene, *, seed=None):
        self._last = None
        if getattr(self, "_ls", None) and self._base is not None:
            mech = self._mech()
            for j in list(self._ls.get("limits", {})):
                F.set_limits(mech, j, None, self._ls)  # put the built limits back before the reset moves things home
            F.set_clutch(mech, "spindle_turn", F.SPINDLE_DAMPING, F.SPINDLE_HOLD)
            F.set_clutch(mech, "nut_turn", F.NUT_DAMPING, F.NUT_HOLD)
        self._ls = {}
        self._hold = F.NUT_HOLD
        self._wall_q = F.SNUG_Y - F.RUN_DOWN - F.NUT_STAND_Y
        self._engaged = self._cracked = self._decoy = False
        self._seat_turn = None  # a new trial: the harness resets here, which a robot can't do
        self._spindle = None
        self._judge = {"phase": 0}
        self._wrench = None
        obs = super().reset(scene, seed=seed)
        mech = self._mech()
        F.lock_here(mech, "nut_turn", self._ls)
        F.set_limits(mech, "nut_py", (-0.02, self._wall_q), self._ls)
        if DEBUG:
            print(f"RESET seed={seed} run-down {1000 * (F.SNUG_Y - F.NUT_STAND_Y - self._wall_q):.2f} mm", flush=True)
        self._set_spindle("locked")
        self._apply(0.0, 0.0, 0.0)
        return obs

    def _mech(self):
        return self._base.scene["grinder_mech"]

    def _q(self, name):
        mech = self._mech()
        return float(mech.data.joint_pos.torch[0, mech.joint_names.index(name)])

    def _set_spindle(self, mode):
        """locked (hard, where it is), free (its bearings) or test (the judge's damper)."""
        if mode == self._spindle:
            return
        mech = self._mech()
        if mode == "locked":
            F.lock_here(mech, "spindle_turn", self._ls)
        else:
            F.set_limits(mech, "spindle_turn", None, self._ls)
        if mode == "test":
            F.set_clutch(mech, "spindle_turn", self.task.module.SPIN_D, 1.0)
        elif self._spindle == "test":
            F.set_clutch(mech, "spindle_turn", F.SPINDLE_DAMPING, F.SPINDLE_HOLD)
        self._spindle = mode
        if DEBUG:
            print(f"SPINDLE t={self._t:5.2f} {mode}", flush=True)

    def _apply(self, spindle_torque, brake, tug):
        """The judge's wrenches: a torque on the spindle about y, and on the wheel a torque about y and a force
        along -y (world frame)."""
        import torch

        if self._wrench == (spindle_torque, brake, tug):
            return
        dev = self._base.device
        mech = self._mech()
        mech.permanent_wrench_composer.set_forces_and_torques_index(
            torques=torch.tensor([[[0.0, float(spindle_torque), 0.0]]], device=dev),
            body_ids=[mech.body_names.index("spindle")], is_global=True)
        self._base.scene["wheel"].permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.tensor([[[0.0, -float(tug), 0.0]]], device=dev),
            torques=torch.tensor([[[0.0, float(brake), 0.0]]], device=dev), is_global=True)
        self._wrench = (spindle_torque, brake, tug)

    def _capture(self):
        mech = self._mech()
        F.lock_here(mech, "nut_px", self._ls, at=0.0)
        F.lock_here(mech, "nut_pz", self._ls, at=0.0)
        F.lock_here(mech, "nut_py", self._ls, at=self._wall_q)
        F.set_limits(mech, "nut_turn", None, self._ls)
        self._engaged = True
        if DEBUG:
            print(f"NUT t={self._t:5.2f} threaded on", flush=True)

    def _wheel_angle(self):
        x, y, z, w = self._base.scene["wheel"].data.root_quat_w.torch[0].cpu().numpy().astype(float)
        a = 2 * math.atan2(y, w)
        j = self._judge
        if "a_raw" in j:
            j["a"] += (a - j["a_raw"] + math.pi) % (2 * math.pi) - math.pi
        else:
            j["a"] = a
        j["a_raw"] = a
        return j["a"]

    def step(self, action):
        st, m = self._last, self.task.module
        if st is not None:
            self._decoy = self._decoy or m.decoy_on(st)  # latched: taking it off again doesn't clear it
            if not self._engaged:
                px, py, pz = st["nut"].pos
                if abs(px) <= F.CAPTURE_R and abs(pz) <= F.CAPTURE_R and py >= self._wall_q - F.CAPTURE_Y:
                    self._capture()
            if self._engaged:
                hold = F.nut_hold(-self._q("nut_crush"))
                if abs(hold - self._hold) > 0.002:
                    F.set_clutch(self._mech(), "nut_turn", F.NUT_DAMPING, hold)
                    self._hold = hold
            if self._engaged and self._seat_turn is None and -self._q("nut_crush") >= m.SEATED:
                self._seat_turn = self._q("nut_turn")
                if DEBUG:
                    print(f"SEATED t={self._t:5.2f} at turn {math.degrees(self._seat_turn):.1f} deg", flush=True)
            if self._seat_turn is not None and self._q("nut_turn") - self._seat_turn >= m.CRACK_TURN:
                if DEBUG and not self._cracked:
                    print(f"CRACKED t={self._t:5.2f}", flush=True)
                self._cracked = True
            self._run_judge(st, m)
        return super().step(action)

    def _run_judge(self, st, m):
        j, dt = self._judge, 1.0 / self.info.control_hz
        pressed = -self._q("lock_press") >= F.PRESSED
        if not m.judge_ready(st):
            if j["phase"]:
                self._judge = j = {"phase": 0}
            self._apply(0.0, 0.0, 0.0)
            self._set_spindle("free" if self._engaged and not pressed else "locked")
            return
        w_angle = self._wheel_angle()
        if j["phase"] == 0:
            j.update(phase=1, t=0.0, s0=self._q("spindle_turn"), w0=w_angle, slip=0.0, move=0.0)
            self._set_spindle("test")
            if DEBUG:
                print(f"JUDGE t={self._t:5.2f} start, crush {-1000 * self._q('nut_crush'):.3f} mm", flush=True)
        if j["phase"] == 1:
            j["t"] += dt
            j["slip"] = (w_angle - j["w0"]) - (self._q("spindle_turn") - j["s0"])
            if abs(j["slip"]) > m.SLIP_MAX:
                j["phase"] = 4
            elif j["t"] >= m.SPIN_S - 1e-9:
                if abs(self._q("spindle_turn") - j["s0"]) < m.SPIN_MIN:
                    j["phase"] = 4
                else:
                    j.update(phase=2, t=0.0, y0=float(st["wheel_root"].pos[1]))
                    self._set_spindle("free")
            else:
                turned = j["w0"] - w_angle  # the wheel's turn with the spindle (the spin runs it negative)
                self._apply(-m.SPIN_T, m.BRAKE * min(1.0, max(0.0, turned / m.BRAKE_RAMP)), 0.0)
        if j["phase"] == 2:
            j["t"] += dt
            j["move"] = j["y0"] - float(st["wheel_root"].pos[1])
            if j["move"] >= m.TUG_MOVE:
                j["phase"] = 5
            elif j["t"] >= m.TUG_S - 1e-9:
                j["phase"] = 3
            else:
                self._apply(0.0, 0.0, m.TUG_N * min(1.0, j["t"] / m.TUG_RAMP))
        if j["phase"] in (3, 4, 5):
            if DEBUG and not j.get("said"):
                j["said"] = True
                print(f"JUDGE t={self._t:5.2f} phase {j['phase']} slip {math.degrees(j['slip']):+.2f} deg move "
                      f"{1000 * j['move']:+.3f} mm crush {-1000 * self._q('nut_crush'):.3f} mm", flush=True)
            self._apply(0.0, 0.0, 0.0)
            self._set_spindle("free")

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

            funcs = {"grinder_frame": (spawn_frame, ObjectType.BASE),
                     "grinder_mech": (spawn_mech, ObjectType.ARTICULATION),
                     **{n: (_part_spawner(n), ObjectType.RIGID) for n in PART_NAMES}}
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
        zero, ident = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))

        def point(p):
            p = np.asarray(p, float)
            return ObjectState(pos=p, quat_xyzw=ident, lin_vel=zero, ang_vel=zero, low=p, high=p)

        out = {}
        for name in PART_NAMES:
            g = base.scene[name]
            p = arr(g.data.root_pos_w.torch[0]) - origin
            out[f"{name}_root"] = ObjectState(pos=p, quat_xyzw=arr(g.data.root_quat_w.torch[0]),
                                              lin_vel=arr(g.data.root_lin_vel_w.torch[0]),
                                              ang_vel=arr(g.data.root_ang_vel_w.torch[0]), low=p, high=p)
        q = self._q
        out["spindle"] = point((q("spindle_turn"), -q("lock_press"), float(self._spindle == "locked")))
        out["nut"] = point((q("nut_px"), q("nut_py"), q("nut_pz")))
        out["thread"] = point((q("nut_turn"), q("nut_thread"), -q("nut_crush")))
        out["latch"] = point((float(self._engaged), float(self._cracked), float(self._decoy)))
        j = self._judge or {"phase": 0}
        out["judge"] = point((float(j["phase"]), j.get("slip", 0.0), j.get("move", 0.0)))
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            qa = arr(base.scene[robot].data.joint_pos.torch[0, arm_ids])
            p, _ = ik.tcp_pose(qa, np.array(BASES[arm]))
            out[f"{arm}_tcp"] = point(p)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states

        def object_states(env, names):
            names = [n for n in names if n not in PLACEHOLDERS]
            states = {**original(env, names), **self._extra_states()}
            self._last = states
            return states

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if DEBUG and self._last is not None and round(self._t * self.info.control_hz) % round(self.info.control_hz / 4) == 0:
            m, st = self.task.module, self._last
            ys = " ".join(f"{n[:4]} {m.describe(st, n)}" for n in F.ORDER + ("decoy",))
            th = st["thread"].pos
            print(f"WHEEL t={self._t:5.2f} {ys} | nut {np.round(1000 * st['nut'].pos, 1).tolist()} turn "
                  f"{math.degrees(th[0]):+7.1f} deg crush {1000 * th[2]:5.2f} mm | spindle "
                  f"{math.degrees(st['spindle'].pos[0]):+6.1f} btn {1000 * st['spindle'].pos[1]:4.1f} | latch "
                  f"{st['latch'].pos[:2].tolist()} judge {np.round(st['judge'].pos, 3).tolist()} | L "
                  f"{np.round(1000 * st['left_tcp'].pos, 0).tolist()} R {np.round(1000 * st['right_tcp'].pos, 0).tolist()}"
                  f" | prefix {m.prefix(st)} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return WheelEmbodiment(**kwargs)
