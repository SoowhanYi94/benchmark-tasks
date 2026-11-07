"""Builds the small press into a USD stage, as one articulation: a fixed bed with rails and a rear stop, columns, and a sprung
ram whose punch sits at one of three places across the bed. Dimensions in m, relative to the bed's centre on the table."""

import numpy as np

BED = dict(x0=-0.10, x1=0.06, half_y=0.095, z=0.02)
RAIL = dict(inner=0.079, z1=0.032, x0=-0.015, x1=0.05)  # they guide the last of the slide, 4 mm of play each side; in front of them a die lies on the open bed
STOP = dict(x0=0.025, z1=0.045)  # the die's rear edge meets this with its pocket under the ram
COLUMN = dict(x0=0.018, x1=0.04, inner=0.086, z1=0.10, ledge_x0=-0.019)  # the guard rests on the columns and their ledges
PUNCH = dict(half=0.015, tip=0.053, top=0.095)  # 3 cm square, its tip this high at rest, its top showing from above once the guard is off
TRAVEL = 0.03
HEAD = dict(half_x=0.0145, half_y=0.083, z0=0.083, z1=0.093)  # the ram's crosshead, which the punch passes through
PAD = dict(half_x=0.03, y0=0.098, y1=0.158, z0=0.085, z1=0.095)  # to the +y side, beyond the guard
SPRING = 50.0  # N/m, about 1 N at the top, 2.5 N at the bottom
MASS = {"anchor": 1.0, "ram": 0.1}
ITERS, VEL_ITERS, MU = 16, 4, 0.5
COLORS = {"bed": (0.10, 0.11, 0.12), "steel": (0.30, 0.32, 0.35), "ram": (0.10, 0.45, 0.85), "punch": (0.98, 0.45, 0.02)}


def _api():
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    return Gf, Sdf, UsdGeom, UsdPhysics, UsdShade


def physx(prim, schema, **attrs):
    """Apply a PhysX API schema by name and set its attributes (no PhysxSchema bindings here)."""
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def look(stage, path, color):
    Gf, Sdf, _, _, UsdShade = _api()
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def surface(stage, path, mu):
    _, _, _, UsdPhysics, UsdShade = _api()
    phys = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
    pm.CreateStaticFrictionAttr(mu)
    pm.CreateDynamicFrictionAttr(mu)
    return phys


def cube(stage, path, lo, hi, mat, phys=None):
    Gf, _, UsdGeom, UsdPhysics, UsdShade = _api()
    lo, hi = np.array(lo, float), np.array(hi, float)
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
    c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if phys is not None:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        physx(c.GetPrim(), "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002, "physxCollision:restOffset": 0.0})
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")
    return c


def rigid(prim, mass, iters=ITERS):
    _, _, _, UsdPhysics, _ = _api()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": iters, "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                        "physxRigidBody:maxDepenetrationVelocity": 1.0})
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


def build_part(stage, path, boxes, mass, mu, color):
    """A free part: one rigid body of several boxes, its origin at the centre of its underside."""
    prim = stage.GetPrimAtPath(path)
    rigid(prim, mass)
    mat, phys = look(stage, f"{path}/Looks/part", color), surface(stage, f"{path}/Physics/part", mu)
    for name, (lo, hi) in boxes.items():
        cube(stage, f"{path}/{name}", lo, hi, mat, phys)
    return prim


def build_press(stage, path, world_origin, punch_y):
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False, "physxArticulation:solverPositionIterationCount": ITERS,
                                           "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    m = {k: look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}
    phys = surface(stage, f"{path}/Physics/mat", MU)
    bodies = {}
    for name, mass in MASS.items():
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        rigid(bodies[name], mass)
    a, r = f"{path}/anchor", f"{path}/ram"
    b, hy = BED, BED["half_y"]
    cube(stage, f"{a}/bed", (b["x0"], -hy, 0.0), (b["x1"], hy, b["z"]), m["bed"], phys)
    for s, sign in (("l", 1), ("r", -1)):
        y0, y1 = sorted((sign * RAIL["inner"], sign * hy))
        cube(stage, f"{a}/rail_{s}", (RAIL["x0"], y0, b["z"]), (RAIL["x1"], y1, RAIL["z1"]), m["steel"], phys)
        y0, y1 = sorted((sign * COLUMN["inner"], sign * hy))
        cube(stage, f"{a}/column_{s}", (COLUMN["x0"], y0, b["z"]), (COLUMN["x1"], y1, COLUMN["z1"]), m["steel"], phys)
        cube(stage, f"{a}/ledge_{s}", (COLUMN["ledge_x0"], y0, COLUMN["z1"] - 0.01), (COLUMN["x0"], y1, COLUMN["z1"]), m["steel"], phys)
    cube(stage, f"{a}/stop", (STOP["x0"], -hy, b["z"]), (b["x1"], hy, STOP["z1"]), m["steel"], phys)
    p, h, pd = PUNCH, HEAD, PAD
    cube(stage, f"{r}/punch", (-p["half"], punch_y - p["half"], p["tip"]), (p["half"], punch_y + p["half"], p["top"]), m["punch"], phys)
    cube(stage, f"{r}/head", (-h["half_x"], -h["half_y"], h["z0"]), (h["half_x"], h["half_y"], h["z1"]), m["ram"])
    cube(stage, f"{r}/arm", (-h["half_x"], h["half_y"], pd["z0"]), (h["half_x"], pd["y0"], pd["z1"]), m["ram"])
    cube(stage, f"{r}/pad", (-pd["half_x"], pd["y0"], pd["z0"]), (pd["half_x"], pd["y1"], pd["z1"]), m["ram"], phys)
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    ram = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/ram_slide")
    ram.CreateAxisAttr("Z")
    ram.CreateLowerLimitAttr(-TRAVEL)
    ram.CreateUpperLimitAttr(0.0)
    ram.CreateBody0Rel().SetTargets([bodies["anchor"].GetPath()])
    ram.CreateBody1Rel().SetTargets([bodies["ram"].GetPath()])
    d = UsdPhysics.DriveAPI.Apply(ram.GetPrim(), "linear")
    d.CreateTypeAttr("force")
    d.CreateTargetPositionAttr(0.02)
    d.CreateStiffnessAttr(SPRING)
    d.CreateDampingAttr(SPRING / 10)
    return prim
