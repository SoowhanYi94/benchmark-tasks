"""strutkit: articulated assemblies (fixed or loose) built from boxes, cylinders, capsules and spheres, for yam_arena
task files. Written for this task, after the partkit and mechanism modules of earlier tasks in this repository.

An :class:`Assembly` is one Isaac Lab articulation: a tree of rigid links joined by revolute or prismatic joints, each
with its stops, a spring (a drive toward ``target``, which may lie beyond a stop to preload it) and viscous damping.
``fixed=True`` bolts its root link to the world where it stands (a machine); otherwise it is a loose part that can be
picked up (a gas strut). Every link's frame is the assembly's frame at rest, so shapes and joint anchors are all given
in that one frame. Joint axes are the assembly frame's X, Y or Z.

Importing this module hooks ``yam_arena.scene`` (and the embodiment's step and reset) without modifying yam_arena:
``success()``/``score()`` receive an :class:`AssemblyState` (root pose, every joint's position, every link's pose);
``on_step(fn)`` and ``on_reset(fn)`` register callbacks run with the Isaac Lab environment after every control step and
after every reset. Pure Python at import time; Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

Color = tuple[float, float, float]


@dataclass(frozen=True)
class Shape:
    """One collider, also drawn. ``kind`` box (``size`` = x, y, z edges), cylinder or capsule (``size`` = radius,
    length along ``axis``), sphere (``size`` = (radius,)). ``rot`` is a (w, x, y, z) rotation in the assembly frame."""

    kind: str
    center: tuple
    size: tuple
    color: Color = (0.6, 0.6, 0.6)
    axis: str = "Z"
    rot: tuple = (1.0, 0.0, 0.0, 0.0)
    material: str = "default"
    contact_offset: float = 0.001
    visible: bool = True
    torsion: float = 0.0   # m: torsional contact patch radius (a pad's soft contact area on a round part), 0 = none


@dataclass(frozen=True)
class Link:
    name: str
    mass: float
    shapes: tuple
    com: tuple | None = None   # centre of mass in the assembly frame (m); None = from the shapes


@dataclass(frozen=True)
class Joint:
    """``child`` moves relative to ``parent`` about or along ``axis`` through ``anchor`` (assembly frame, m).
    Positions in rad or m, 0 at rest; ``lower``/``upper`` are hard stops."""

    name: str
    kind: str
    parent: str
    child: str
    anchor: tuple
    axis: str
    lower: float
    upper: float
    stiffness: float = 0.0
    damping: float = 0.0
    target: float = 0.0
    rot: tuple = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z): the joint frame's rotation, turning ``axis`` off the frame axes
    effort: float | None = None        # the drive's force or torque limit (None: unlimited)
    init: float = 0.0                  # position at reset


@dataclass(frozen=True)
class Assembly(Box):
    """It subclasses Box only so yam_arena's task-file checks accept it. ``pos`` is the assembly frame's origin in the
    world, ``quat`` its rotation (x, y, z, w); ``size`` is a rough bounding box. Loose assemblies need ``mass`` set
    (the total, for yam_arena's checks); the links carry the real masses."""

    links: tuple = ()
    joints: tuple = ()
    materials: dict = field(default_factory=dict, compare=False, hash=False)
    fixed: bool = False
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    self_collisions: bool = False

    def _check(self) -> None:
        names = [link.name for link in self.links]
        if len(set(names)) != len(names) or not names:
            raise ValueError(f"{self.name}: link names must be unique and present")
        for j in self.joints:
            if j.parent not in names or j.child not in names or j.kind not in ("revolute", "prismatic"):
                raise ValueError(f"{self.name}: bad joint {j.name}")
        for link in self.links:
            for s in link.shapes:
                if s.material not in self.materials:
                    raise ValueError(f"{self.name}.{link.name}: no material {s.material!r}")

    @property
    def root(self) -> str:
        return self.links[0].name


@dataclass(frozen=True)
class AssemblyState(ObjectState):
    joints: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)  # name -> (pos (3,), quat_xyzw (4,))


def quat_matrix_xyzw(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_quat_xyzw(m) -> tuple:
    m = np.asarray(m, float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return (x, y, z, w)


# -- spawning --------------------------------------------------------------------------------------------------------

_SPECS: dict = {}


def spawn(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the assembly under ``prim_path`` (an Xform carrying the articulation root)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Assembly = _SPECS[cfg.assembly_name]
    stage = get_current_stage()
    dbl = UsdGeom.XformOp.PrecisionDouble
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=dbl).Set(Gf.Vec3d(*(translation if translation is not None else spec.pos)))
    x, y, z, w = orientation if orientation is not None else spec.quat
    root.AddOrientOp(precision=dbl).Set(Gf.Quatd(w, x, y, z))

    phys = {}
    for name, m in spec.materials.items():
        static, dynamic = m[0], m[1]
        combine = m[2] if len(m) > 2 else "average"
        path = f"{prim_path}/Looks/phys_{name}"
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                             friction_combine_mode=combine, restitution=0.0)
        mat.func(path, mat)
        phys[name] = path
    looks = {}

    def look(color):
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.45)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def author(path, s: Shape):
        if s.kind == "box":
            g = UsdGeom.Cube.Define(stage, path)
            g.CreateSizeAttr(1.0)
        elif s.kind == "sphere":
            g = UsdGeom.Sphere.Define(stage, path)
            g.CreateRadiusAttr(s.size[0])
        else:
            g = (UsdGeom.Cylinder if s.kind == "cylinder" else UsdGeom.Capsule).Define(stage, path)
            g.CreateAxisAttr(s.axis)
            g.CreateRadiusAttr(s.size[0])
            g.CreateHeightAttr(s.size[1])
        g.AddTranslateOp(precision=dbl).Set(Gf.Vec3d(*s.center))
        if tuple(s.rot) != (1.0, 0.0, 0.0, 0.0):
            g.AddOrientOp().Set(Gf.Quatf(*s.rot))
        if s.kind == "box":
            g.AddScaleOp().Set(Gf.Vec3f(*s.size))
        prim = g.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(s.contact_offset)
        cp.CreateRestOffsetAttr(0.0)
        if s.torsion:
            cp.CreateTorsionalPatchRadiusAttr(s.torsion)
            cp.CreateMinTorsionalPatchRadiusAttr(0.8 * s.torsion)
        bind_physics_material(str(prim.GetPath()), phys[s.material], stage=stage)
        bind_visual_material(str(prim.GetPath()), look(s.color), stage=stage)
        if not s.visible:
            UsdGeom.Imageable(prim).MakeInvisible()

    bodies = {}
    for link in spec.links:
        path = f"{prim_path}/{link.name}"
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp(precision=dbl).Set(Gf.Vec3d(0, 0, 0))
        xf.AddOrientOp(precision=dbl).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mapi = UsdPhysics.MassAPI.Apply(prim)
        mapi.CreateMassAttr(link.mass)
        if link.com is not None:
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(*link.com))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.5)
        rb.CreateAngularDampingAttr(0.5)
        for i, s in enumerate(link.shapes):
            author(f"{path}/{s.kind}_{i}", s)
        bodies[link.name] = Sdf.Path(path)

    if spec.fixed:
        base = stage.GetPrimAtPath(bodies[spec.root])
        tw = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        fj = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/{spec.root}/fixed_to_world")
        fj.CreateBody1Rel().SetTargets([bodies[spec.root]])
        fj.CreateLocalPos0Attr(Gf.Vec3f(tw.ExtractTranslation()))
        fj.CreateLocalRot0Attr(Gf.Quatf(tw.ExtractRotationQuat()))
        fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    for j in spec.joints:
        rev = j.kind == "revolute"
        uj = (UsdPhysics.RevoluteJoint if rev else UsdPhysics.PrismaticJoint).Define(stage, f"{prim_path}/{j.child}/{j.name}")
        uj.CreateAxisAttr(j.axis)
        uj.CreateBody0Rel().SetTargets([bodies[j.parent]])
        uj.CreateBody1Rel().SetTargets([bodies[j.child]])
        uj.CreateLocalPos0Attr(Gf.Vec3f(*j.anchor))
        uj.CreateLocalRot0Attr(Gf.Quatf(*j.rot))
        uj.CreateLocalPos1Attr(Gf.Vec3f(*j.anchor))
        uj.CreateLocalRot1Attr(Gf.Quatf(*j.rot))
        unit = math.degrees(1.0) if rev else 1.0
        uj.CreateLowerLimitAttr(j.lower * unit)
        uj.CreateUpperLimitAttr(j.upper * unit)
        d = UsdPhysics.DriveAPI.Apply(uj.GetPrim(), "angular" if rev else "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(j.stiffness / unit)
        d.CreateDampingAttr(j.damping / unit)
        d.CreateTargetPositionAttr(j.target * unit)

    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(spec.self_collisions)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()


def actuators(spec: Assembly):
    from isaaclab.actuators import ImplicitActuatorCfg

    out = {}
    for j in spec.joints:
        kw = {} if j.effort is None else {"effort_limit_sim": j.effort}
        out[j.name] = ImplicitActuatorCfg(joint_names_expr=[j.name], stiffness=j.stiffness, damping=j.damping, **kw)
    return out


def _arena_object(spec: Assembly):
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    _SPECS[spec.name] = spec

    @configclass
    class AssemblySpawnerCfg(SpawnerCfg):
        func: object = clone(spawn)
        assembly_name: str = ""

    class AssemblyObject(Object):
        def _generate_articulation_cfg(self):
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators(spec))
            cfg = self._add_initial_pose_to_cfg(cfg)
            # WHY: the arena's own reset writes init_state.joint_pos (zeros by default) after our hook; a strut
            # fitted with the lid shut must start with its rod pushed in (fin2 r1: both fitted struts fell off)
            cfg.init_state.joint_pos = {j.name: float(j.init) for j in spec.joints}
            return cfg

    return AssemblyObject(name=spec.name, object_type=ObjectType.ARTICULATION,
                          spawner_cfg=AssemblySpawnerCfg(assembly_name=spec.name),
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                                            rotation_xyzw=tuple(float(v) for v in spec.quat)))


# -- state, reset ----------------------------------------------------------------------------------------------------


def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    return _t(x).detach().cpu().numpy().astype(np.float64)


def read_state(asset, spec: Assembly, origin=np.zeros(3)) -> AssemblyState:
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    names = list(asset.joint_names)
    q = _np(asset.data.joint_pos)[0]
    joints = {j.name: float(q[names.index(j.name)]) for j in spec.joints}
    bp = _np(asset.data.body_link_pose_w)[0]
    links = {n: (bp[i, :3] - origin, bp[i, 3:7]) for i, n in enumerate(asset.body_names)}
    pts = np.array([p for p, _ in links.values()])
    return AssemblyState(pos=pose[:3] - origin, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6],
                         low=pts.min(0) - 0.01, high=pts.max(0) + 0.01, joints=joints, links=links)


def rest(asset, spec: Assembly) -> None:
    """Every joint to its ``init`` (0 unless set) and still; each spring's target set (a target beyond a stop preloads
    it)."""
    import torch

    names = list(asset.joint_names)
    ids = [names.index(j.name) for j in spec.joints]
    dev = _t(asset.data.joint_pos).device
    zeros = torch.zeros((1, len(ids)), dtype=torch.float32, device=dev)
    inits = torch.tensor([[j.init for j in spec.joints]], dtype=torch.float32, device=dev)
    targets = torch.tensor([[j.target for j in spec.joints]], dtype=torch.float32, device=dev)
    asset.write_joint_state_to_sim_index(position=inits, velocity=zeros, joint_ids=ids)
    asset.set_joint_position_target_index(target=targets, joint_ids=ids)
    asset.set_joint_velocity_target_index(target=zeros, joint_ids=ids)
    asset.write_data_to_sim()


# -- hooking yam_arena -----------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    _STEP.append(fn)
    return fn


def on_reset(fn):
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def state(environment, name: str) -> AssemblyState:
    base = environment.unwrapped
    return read_state(base.scene[name], _SPECS[name], _np(base.scene.env_origins[0]))


def _patch_scene(scene) -> None:
    if getattr(scene, "_strutkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Assembly):
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = state(environment, n)
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._strutkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_strutkit_patched", False):
        return
    base_task_step, base_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return base_task_step(self, *a, **k)

    def reset(self, *a, **k):
        obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._strutkit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
