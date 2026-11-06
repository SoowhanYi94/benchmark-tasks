"""`yam_parallels`: the stock yam_arena embodiment plus the holed plate and the two pins, each one rigid body of several
boxes. Their states are their own frames (origin at the centre of the underside: the plate's, or a pin's collar). The scorer
also gets `left_tcp` and `right_tcp` (each gripper's fingertip midpoint). The harness is unchanged."""

import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("PARALLELS_DEBUG") == "1"
OWN = ("plate", "pin_a", "pin_b")


def _physx(prim, schema, **attrs):
    """Apply a PhysX API schema by name and set its attributes (no PhysxSchema bindings here)."""
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def _build(stage, path, boxes, mass, mu, color):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": 16,
                                         "physxRigidBody:solverVelocityIterationCount": 4,
                                         "physxRigidBody:maxDepenetrationVelocity": 1.0})
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    look = UsdShade.Material.Define(stage, f"{path}/Looks/part")
    sh = UsdShade.Shader.Define(stage, f"{path}/Looks/part/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    look.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    phys = UsdShade.Material.Define(stage, f"{path}/Physics/part")
    pm = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
    pm.CreateStaticFrictionAttr(mu)
    pm.CreateDynamicFrictionAttr(mu)
    for name, (lo, hi) in boxes.items():
        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(stage, f"{path}/{name}")
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(look)
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        _physx(c.GetPrim(), "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002, "physxCollision:restOffset": 0.0})
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")
    return prim


class ParallelsEmbodiment(YamArenaEmbodiment):
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

        def spawn_own(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            x = UsdGeom.Xform.Define(stage, prim_path)
            x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
            x.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            return _build(stage, prim_path, *m.PARTS[name])

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in OWN:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn_own
            return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

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
        for name in OWN:
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
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
            r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
            print(f"PLATE t={self._t:5.2f} plate {r(st['plate'].pos)} flat {m.flat(st['plate'])} supports {m.supports(st)} "
                  f"par {r(st['parallel_a'].pos)} {r(st['parallel_b'].pos)} pins {r(st['pin_a'].pos)} {r(st['pin_b'].pos)} in {m.pins_in(st)} "
                  f"L {r(st['left_tcp'].pos)} R {r(st['right_tcp'].pos)} clear {m.hands_clear(st)} still {m._trial['still']:.2f} score {m.score(st):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return ParallelsEmbodiment(**kwargs)
