"""Builds the assembly dial into a USD stage, as one articulation: a steel base, a disc turned at a steady rate by a velocity drive,
with two square pockets, and a sprung stop pad on the fixed frame. Dimensions in m, relative to the dial's foot on the table."""

import math

import numpy as np

DISC = dict(r=0.15, half=0.106, z0=0.012, z1=0.02)  # the collider is a square inside the round face
RATE = 6.0  # degrees per second
POCKET = dict(radius=0.10, inner=0.038, wall=0.008, h=0.01)  # half inner width: 76 mm across
PAD = dict(half=0.03, z0=0.05, z1=0.06, travel=0.03)  # relative to its own place on the table
SPRING = 80.0
MASS = {"anchor": 1.0, "disc": 0.6, "pad": 0.05}
ITERS, VEL_ITERS, MU = 16, 4, 0.6
COLORS = {"steel": (0.30, 0.32, 0.35), "dark": (0.12, 0.13, 0.14), "disc": (0.72, 0.74, 0.77), "pocket": (0.10, 0.45, 0.85), "pad": (0.85, 0.12, 0.08), "patch": (0.95, 0.75, 0.10)}


def _api():
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    return Gf, Sdf, UsdGeom, UsdPhysics, UsdShade


def physx(prim, schema, **attrs):
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


def rigid(prim, mass):
    _, _, _, UsdPhysics, _ = _api()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS, "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                        "physxRigidBody:maxDepenetrationVelocity": 1.0})
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


def build_part(stage, path, boxes, mass, mu, color):
    prim = stage.GetPrimAtPath(path)
    rigid(prim, mass)
    mat, phys = look(stage, f"{path}/Looks/part", color), surface(stage, f"{path}/Physics/part", mu)
    for name, (lo, hi) in boxes.items():
        cube(stage, f"{path}/{name}", lo, hi, mat, phys)
    return prim


def _face(stage, path, r, z0, z1, mat):
    Gf, _, UsdGeom, _, UsdShade = _api()
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(z1 - z0))
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(z0 + z1) / 2))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


def build_dial(stage, path, world_origin, pad_at, fixtures):
    """fixtures: each pocket's start angle in radians. Pockets' sides lie along world x and y when the fixture's at 90 degrees."""
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
    a, d, p = f"{path}/anchor", f"{path}/disc", f"{path}/pad"
    cube(stage, f"{a}/base", (-0.09, -0.09, 0.0), (0.09, 0.09, DISC["z0"] - 0.0005), m["dark"], phys)
    _face(stage, f"{d}/face", DISC["r"], DISC["z0"], DISC["z1"], m["disc"])
    cube(stage, f"{d}/square", (-DISC["r"] * 0.89, -DISC["r"] * 0.45, DISC["z0"]), (DISC["r"] * 0.89, DISC["r"] * 0.45, DISC["z1"]), m["disc"], phys)
    cube(stage, f"{d}/square2", (-DISC["r"] * 0.45, -DISC["r"] * 0.89, DISC["z0"]), (DISC["r"] * 0.45, DISC["r"] * 0.89, DISC["z1"]), m["disc"], phys)
    cube(stage, f"{d}/square3", (-DISC["r"] * 0.7, -DISC["r"] * 0.7, DISC["z0"]), (DISC["r"] * 0.7, DISC["r"] * 0.7, DISC["z1"]), m["disc"], phys)
    pk = POCKET
    for k, th in enumerate(fixtures):
        # the disc's own frame is its pose at the start; a pocket's walls are square to its pose when the fixture reaches 90 degrees
        cx, cy = pk["radius"] * math.cos(th), pk["radius"] * math.sin(th)
        roll = th - math.pi / 2
        q = Gf.Quatf(math.cos(roll / 2), Gf.Vec3f(0.0, 0.0, math.sin(roll / 2)))
        fx = UsdGeom.Xform.Define(stage, f"{d}/fixture_{k}")
        fx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.0))
        fx.AddOrientOp().Set(q)
        i, w, z0, z1 = pk["inner"], pk["wall"], DISC["z1"], DISC["z1"] + pk["h"]
        o = i + w
        cube(stage, f"{d}/fixture_{k}/w_n", (-o, i, z0), (o, o, z1), m["pocket"], phys)
        cube(stage, f"{d}/fixture_{k}/w_s", (-o, -o, z0), (o, -i, z1), m["pocket"], phys)
        cube(stage, f"{d}/fixture_{k}/w_e", (i, -i, z0), (o, i, z1), m["pocket"], phys)
        cube(stage, f"{d}/fixture_{k}/w_w", (-o, -i, z0), (-i, i, z1), m["pocket"], phys)
    cube(stage, f"{a}/pad_post", (pad_at[0] - 0.012, pad_at[1] - 0.012, 0.0), (pad_at[0] + 0.012, pad_at[1] + 0.012, PAD["z0"] - PAD["travel"] - 0.001), m["steel"], phys)
    h = PAD["half"]
    cube(stage, f"{p}/top", (pad_at[0] - h, pad_at[1] - h, PAD["z0"]), (pad_at[0] + h, pad_at[1] + h, PAD["z1"]), m["pad"], phys)
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    turn = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/turn")
    turn.CreateAxisAttr("Z")
    turn.CreateLowerLimitAttr(-1.0)
    turn.CreateUpperLimitAttr(3600.0)
    turn.CreateBody0Rel().SetTargets([bodies["anchor"].GetPath()])
    turn.CreateBody1Rel().SetTargets([bodies["disc"].GetPath()])
    dr = UsdPhysics.DriveAPI.Apply(turn.GetPrim(), "angular")
    dr.CreateTypeAttr("force")
    dr.CreateStiffnessAttr(0.0)
    dr.CreateDampingAttr(50.0)
    dr.CreateTargetVelocityAttr(float(RATE))
    dr.CreateMaxForceAttr(1000.0)
    slide = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/pad_slide")
    slide.CreateAxisAttr("Z")
    slide.CreateLowerLimitAttr(-PAD["travel"])
    slide.CreateUpperLimitAttr(0.0)
    slide.CreateBody0Rel().SetTargets([bodies["anchor"].GetPath()])
    slide.CreateBody1Rel().SetTargets([bodies["pad"].GetPath()])
    dp = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
    dp.CreateTypeAttr("force")
    dp.CreateTargetPositionAttr(0.02)
    dp.CreateStiffnessAttr(SPRING)
    dp.CreateDampingAttr(SPRING / 10)
    return prim
