"""`yam_tip`: the stock yam_arena embodiment plus the rubber-matted lane and the marked block, the harness unchanged.
The task file lists two placeholder boxes that are spawned as them. The block's state is its own frame: its centre, its
+z the red face's outward normal. The scorer also gets `left_tcp` and `right_tcp` (each gripper's fingertip midpoint)."""

import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("TIP_DEBUG") == "1"
LANE, BLOCK = "tip_lane", "die_block"
OWN = (LANE, BLOCK)

COLORS = {"block": (0.42, 0.44, 0.47), "red": (0.90, 0.08, 0.06), "mat": (0.08, 0.08, 0.09), "curb": (0.85, 0.70, 0.10)}
MU_MAT, MU_CURB = 1.0, 0.05


def _physx(prim, schema, **attrs):
    """Apply a PhysX API schema by name and set its attributes (no PhysxSchema bindings here)."""
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


class _Builder:
    def __init__(self, stage, root):
        from pxr import Gf, Sdf, UsdPhysics, UsdShade

        self.stage, self.root, self.looks, self.phys = stage, root, {}, {}
        for name, color in COLORS.items():
            mat = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
            sh = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self.looks[name] = mat
        for name, mu in (("mat", MU_MAT), ("curb", MU_CURB)):
            mat = UsdShade.Material.Define(stage, f"{root}/Physics/{name}")
            pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            pm.CreateStaticFrictionAttr(mu)
            pm.CreateDynamicFrictionAttr(mu)
            self.phys[name] = mat

    def cube(self, path, lo, hi, look, phys=None):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.looks[look])
        if phys:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            _physx(c.GetPrim(), "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002, "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.phys[phys], UsdShade.Tokens.weakerThanDescendants, "physics")


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


def _rigid(prim, kinematic=False, mass=None):
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
    _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": 16,
                                         "physxRigidBody:solverVelocityIterationCount": 4,
                                         "physxRigidBody:maxDepenetrationVelocity": 1.0})
    if mass is not None:
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)




class TipEmbodiment(YamArenaEmbodiment):
    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        return super().reset(scene, seed=seed)

    def _ensure_sim(self, goal):
        import atexit
        import math

        from yam_arena import inspect_embodiment as base

        if base._APP is None:  # Kit must be up before Arena is imported (its scene module needs PhysxSchema)
            from isaaclab.app import AppLauncher

            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            base._APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
            atexit.register(base._close_app_at_exit)
        from yam_arena import scene

        original = scene.task_object
        m = self.task.module

        def spawn_lane(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            prim = _place(stage, prim_path, translation, orientation)
            _rigid(prim, kinematic=True)
            b = _Builder(stage, prim_path)
            hx, hy, t = m.LANE_X_HALF, m.LANE_Y_HALF, m.MAT_T
            b.cube(f"{prim_path}/mat", (-hx, -hy, 0.0), (hx, hy, t), "mat", "mat")
            for side, s in (("far", 1), ("near", -1)):
                x = sorted((s * hx, s * (hx + m.CURB_W)))
                b.cube(f"{prim_path}/curb_{side}", (x[0], -hy, 0.0), (x[1], hy, m.CURB_H), "curb", "curb")
            return prim

        def spawn_block(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            prim = _place(stage, prim_path, translation, orientation)
            _rigid(prim, mass=m.BLOCK_MASS)
            b = _Builder(stage, prim_path)
            h, e, band = m.SIDE / 2, 0.0005, m.BAND
            b.cube(f"{prim_path}/body", (-h, -h, -h), (h, h, h), "block", "mat")
            # shown only, a hair proud of the block: the red face (+z), and a red band on each face along its shared edge
            b.cube(f"{prim_path}/red", (-h, -h, h), (h, h, h + e), "red")
            b.cube(f"{prim_path}/band_px", (h, -h, h - band), (h + e, h, h), "red")
            b.cube(f"{prim_path}/band_nx", (-h - e, -h, h - band), (-h, h, h), "red")
            b.cube(f"{prim_path}/band_py", (-h, h, h - band), (h, h + e, h), "red")
            b.cube(f"{prim_path}/band_ny", (-h, -h - e, h - band), (h, -h, h), "red")
            return prim

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in OWN:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn_lane if spec.name == LANE else spawn_block
            kind = ObjectType.BASE if spec.name == LANE else ObjectType.RIGID
            rot = (0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2)) if spec.name == LANE else m.START_XYZW
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg, initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=rot))

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
        pose = arr(world.get_pose_w(BLOCK)[0])
        p = pose[:3] - origin
        out[BLOCK] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(BLOCK)[0]),
                                 ang_vel=arr(world.get_root_angular_velocity_w(BLOCK)[0]), low=p, high=p)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}

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
            self.task.module.note_step(last["s"], 1.0 / self.info.control_hz)
        if DEBUG and "s" in last and round(self._t * self.info.control_hz) % round(self.info.control_hz / 4) == 0:
            m, st = self.task.module, last["s"]
            b = st[BLOCK]
            print(f"TIP t={self._t:5.2f} pos {np.round(1000 * b.pos, 1).tolist()} red-z {m.red_up_cos(b):+.3f} in-lane {m.in_lane(b)} "
                  f"clear {m.hands_clear(st)} rest {b.resting()} L {np.round(1000 * st['left_tcp'].pos, 0).tolist()} "
                  f"R {np.round(1000 * st['right_tcp'].pos, 0).tolist()} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return TipEmbodiment(**kwargs)
