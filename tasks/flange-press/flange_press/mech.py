"""Builds the straightening press into a USD stage. A plain flange on a turntable, cranked by a free-turning handle; a dial
pointer over a scale, reading how near the flange's high spot is to the ram; and a sprung ram over the rim. Dimensions in m,
relative to the turntable's centre on the table."""

import numpy as np

R_RIM, Z_FLANGE0, Z_FLANGE1 = 0.12, 0.02, 0.03
CRANK_R = 0.05
HANDLE = dict(half=0.012, z1=0.08)  # above the table, on the flange at CRANK_R
RAM = dict(y=0.105, half_x=0.03, half_y=0.015, z0=0.07, z1=0.11, travel=0.03)  # over the rim on the +y side; it stops 1 cm above the flange
SPRING_RAM = 50.0  # N/m, about 1 N at the top, 2.5 N at the bottom
DAMP_FLANGE = 0.1  # N m s/rad: the flange stays where it's left
DIAL = dict(x=-0.20, y=0.0, travel=0.10)  # a scale along y beside the turntable, its pointer sliding along it
MASS = {"anchor": 1.0, "flange": 0.3, "handle": 0.02, "ram": 0.1, "pointer": 0.01}
ITERS, VEL_ITERS, MAX_DEPEN = 16, 4, 1.0
MU = 0.5
COLORS = {"frame": (0.16, 0.17, 0.19), "steel": (0.30, 0.32, 0.35), "flange": (0.70, 0.72, 0.75), "handle": (0.95, 0.55, 0.05),
          "ram": (0.10, 0.45, 0.85), "scale": (0.92, 0.92, 0.88), "pointer": (0.90, 0.08, 0.06)}


def _api():
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    return Gf, Sdf, UsdGeom, UsdPhysics, UsdShade


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
        Gf, Sdf, _, UsdPhysics, UsdShade = _api()
        self.stage, self.root, self.looks = stage, root, {}
        for name, color in COLORS.items():
            mat = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
            sh = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self.looks[name] = mat
        self.phys = UsdShade.Material.Define(stage, f"{root}/Physics/mat")
        pm = UsdPhysics.MaterialAPI.Apply(self.phys.GetPrim())
        pm.CreateStaticFrictionAttr(MU)
        pm.CreateDynamicFrictionAttr(MU)

    def rigid(self, prim, kinematic=False, mass=None):
        _, _, _, UsdPhysics, _ = _api()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
        _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS,
                                             "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                             "physxRigidBody:maxDepenetrationVelocity": float(MAX_DEPEN)})
        if mass is not None:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)

    def cube(self, path, lo, hi, look, collide=True):
        Gf, _, UsdGeom, UsdPhysics, UsdShade = _api()
        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.looks[look])
        if collide:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            _physx(c.GetPrim(), "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002, "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.phys, UsdShade.Tokens.weakerThanDescendants, "physics")

    def disc(self, path, r, z0, z1, look):
        """A round face to look at. It has no collider; the square inside it takes contact."""
        Gf, _, UsdGeom, _, UsdShade = _api()
        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(float(z1 - z0))
        c.CreateAxisAttr("Z")
        c.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(z0 + z1) / 2))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.looks[look])


def build_frame(stage, path):
    """The fixed parts: the pedestal, the ram's guide posts and the scale."""
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, kinematic=True)
    b.cube(f"{path}/pedestal", (-0.04, -0.04, 0.0), (0.04, 0.04, Z_FLANGE0 - 0.001), "frame")
    gy = RAM["y"] + RAM["half_y"] + 0.03  # behind the rim's reach
    for s, x in (("r", 0.045), ("l", -0.045)):
        b.cube(f"{path}/post_{s}", (x - 0.008, gy - 0.008, 0.0), (x + 0.008, gy + 0.008, RAM["z1"] + 0.02), "steel")
    b.cube(f"{path}/bridge", (-0.053, gy - 0.008, RAM["z1"] + 0.02), (0.053, gy + 0.008, RAM["z1"] + 0.03), "steel")
    d = DIAL
    b.cube(f"{path}/scale", (d["x"] - 0.02, d["y"] - d["travel"] / 2 - 0.01, 0.0), (d["x"] + 0.02, d["y"] + d["travel"] / 2 + 0.01, 0.012), "scale")
    return prim


def build_mech(stage, path, world_origin):
    """The articulation at ``path``: the flange and its handle, the ram, and the pointer the embodiment sets each step."""
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    _physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False,
                                            "physxArticulation:solverPositionIterationCount": ITERS,
                                            "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    bodies = {}
    for name in MASS:
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        b.rigid(bodies[name], mass=MASS[name])
    b.disc(f"{path}/flange/face", R_RIM, Z_FLANGE0, Z_FLANGE1, "flange")
    s = R_RIM * 0.7
    b.cube(f"{path}/flange/square", (-s, -s, Z_FLANGE0), (s, s, Z_FLANGE1), "flange")
    h, hy = HANDLE["half"], -CRANK_R
    b.cube(f"{path}/handle/grip", (-h, hy - h, Z_FLANGE1), (h, hy + h, HANDLE["z1"]), "handle")
    b.cube(f"{path}/ram/pad", (-RAM["half_x"], RAM["y"] - RAM["half_y"], RAM["z0"]), (RAM["half_x"], RAM["y"] + RAM["half_y"], RAM["z1"]), "ram")
    d = DIAL
    b.cube(f"{path}/pointer/flag", (d["x"] - 0.025, d["y"] - d["travel"] / 2 - 0.006, 0.012), (d["x"] + 0.025, d["y"] - d["travel"] / 2 + 0.006, 0.03), "pointer", collide=False)
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    flange = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/flange_turn")
    flange.CreateAxisAttr("Z")
    flange.CreateBody0Rel().SetTargets([bodies["anchor"].GetPath()])
    flange.CreateBody1Rel().SetTargets([bodies["flange"].GetPath()])
    d = UsdPhysics.DriveAPI.Apply(flange.GetPrim(), "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(float(np.radians(DAMP_FLANGE)))  # USD angular drives are per degree
    handle = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/handle_turn")
    handle.CreateAxisAttr("Z")
    handle.CreateBody0Rel().SetTargets([bodies["flange"].GetPath()])
    handle.CreateBody1Rel().SetTargets([bodies["handle"].GetPath()])
    handle.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(-CRANK_R), 0.0))
    handle.CreateLocalPos1Attr(Gf.Vec3f(0.0, float(-CRANK_R), 0.0))
    ram = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/ram_slide")
    ram.CreateAxisAttr("Z")
    ram.CreateLowerLimitAttr(-RAM["travel"])
    ram.CreateUpperLimitAttr(0.0)
    ram.CreateBody0Rel().SetTargets([bodies["anchor"].GetPath()])
    ram.CreateBody1Rel().SetTargets([bodies["ram"].GetPath()])
    d = UsdPhysics.DriveAPI.Apply(ram.GetPrim(), "linear")
    d.CreateTypeAttr("force")
    d.CreateTargetPositionAttr(0.02)
    d.CreateStiffnessAttr(SPRING_RAM)
    d.CreateDampingAttr(SPRING_RAM / 10)
    ptr = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/pointer_slide")
    ptr.CreateAxisAttr("Y")
    ptr.CreateLowerLimitAttr(0.0)
    ptr.CreateUpperLimitAttr(DIAL["travel"])
    ptr.CreateBody0Rel().SetTargets([bodies["anchor"].GetPath()])
    ptr.CreateBody1Rel().SetTargets([bodies["pointer"].GetPath()])
    return prim
