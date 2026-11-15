"""Builds the hand bender and the tube into a USD stage. The bender is one fixed articulation: a low-friction bed, a former, a guide
on the former's far side, and a handle swept about the former on a free sleeve knob. The tube is a second, floating articulation: three
straight segments joined by two vertical hinges at its bands, whose limits the task writes at run time. Dimensions in m. The bender's
are in table xy, its origin at the table's height; the tube's are in its own frame, on the middle segment's underside."""

import math

import numpy as np

BED_H = 0.004
BED = dict(x0=0.04, x1=0.46, y0=-0.22, y1=0.52)
PIVOT = (0.24, -0.08)  # the former's and the handle's axis: this far toward the right arm, which can't reach the sweep's end from 0
BAND_AT = (0.29, -0.08)  # the mark on the former: a band sits here to be bent
FORMER = dict(r=0.024, core=0.009, h=0.05)  # drawn radius; the collider's a slim core, so a bend has no hard stop short of ruin
RAIL = dict(gap=0.015, w=0.01, h=0.018, y0=-0.11, y1=-0.03)  # the guide: half gap across the band line, and its y range from the mark
HANDLE = dict(r=0.15, z0=0.075, z1=0.085, half=0.01, stop=145.0)  # stop in degrees: far past a good bend, and only there
SHOE = dict(rho=0.08, half=0.008, z0=0.003)
KNOB = dict(half=0.018, z1=0.115)
TUBE = dict(w=0.02, h=0.02, n=0.105, m=0.14, t=0.105)  # steel, 20 mm across
GRIP = dict(along=0.04, h=0.04)  # an aluminium grip block on the middle segment, as wide as the tube, so a shoe passes beside it
MASS = {"mid": 0.0779 + 0.0864, "nose": 0.0584, "tail": 0.0584}
BAND_LEN = 0.03
MU_SLICK, MU_GRIP = 0.05, 0.9
ITERS, VEL_ITERS = 32, 8
COLORS = {"steel": (0.30, 0.32, 0.35), "dark": (0.12, 0.13, 0.14), "bed": (0.86, 0.86, 0.82), "former": (0.72, 0.74, 0.77), "handle": (0.15, 0.3, 0.7),
          "knob": (0.85, 0.12, 0.08), "tube": (0.62, 0.65, 0.68), "band": (0.95, 0.75, 0.10), "block": (0.90, 0.45, 0.08), "mark": (0.85, 0.12, 0.08), "lane": (0.10, 0.60, 0.30)}


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
                                        "physxRigidBody:maxDepenetrationVelocity": 1.0, "physxRigidBody:angularDamping": 2.0, "physxRigidBody:linearDamping": 0.5})
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


def _face(stage, path, xy, r, z0, z1, mat):
    Gf, _, UsdGeom, _, UsdShade = _api()
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(z1 - z0))
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(xy[0], xy[1], float(z0 + z1) / 2))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


def _revolute(stage, path, body0, body1, at, lower, upper, damping):
    Gf, _, _, UsdPhysics, _ = _api()
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateAxisAttr("Z")
    j.CreateLowerLimitAttr(float(lower))
    j.CreateUpperLimitAttr(float(upper))
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, at)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, at)))
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(float(damping))
    return j


def build_bender(stage, path, world_origin, lanes):
    """lanes: the template's painted rectangles, (x0, y0, x1, y1) each."""
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False, "physxArticulation:solverPositionIterationCount": ITERS,
                                           "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    m = {k: look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}
    slick = surface(stage, f"{path}/Physics/slick", MU_SLICK)
    grip = surface(stage, f"{path}/Physics/grip", MU_GRIP)
    bodies = {}
    for name, mass in (("anchor", 3.0), ("handle", 0.2), ("knob", 0.03)):
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        rigid(bodies[name], mass)
    a, h, k = f"{path}/anchor", f"{path}/handle", f"{path}/knob"
    px, py = PIVOT
    cube(stage, f"{a}/bed", (BED["x0"], BED["y0"], 0.0), (BED["x1"], BED["y1"], BED_H), m["bed"], slick)
    for i, (x0, y0, x1, y1) in enumerate(lanes):
        cube(stage, f"{a}/lane_{i}", (x0, y0, BED_H), (x1, y1, BED_H + 0.0005), m["lane"])
    z0, z1 = BED_H, BED_H + FORMER["h"]
    _face(stage, f"{a}/former", PIVOT, FORMER["r"], z0, z1, m["former"])
    cube(stage, f"{a}/former_core", (px - FORMER["core"], py - FORMER["core"], z0), (px + FORMER["core"], py + FORMER["core"], z1), m["former"], slick)
    cube(stage, f"{a}/mark", (px, py - 0.004, z1), (px + FORMER["r"], py + 0.004, z1 + 0.0005), m["mark"])  # points at the band's place
    bx, by = BAND_AT
    for s, sign in (("in", -1), ("out", 1)):
        x = bx + sign * RAIL["gap"]
        cube(stage, f"{a}/rail_{s}", (min(x, x + sign * RAIL["w"]), by + RAIL["y0"], z0), (max(x, x + sign * RAIL["w"]), by + RAIL["y1"], z0 + RAIL["h"]), m["steel"], slick)
    # the handle: a beam from the pivot, with a shoe hanging beside the tube's line and the sleeve knob on its end
    zb0, zb1 = BED_H + HANDLE["z0"], BED_H + HANDLE["z1"]
    hw = HANDLE["half"]
    _face(stage, f"{h}/hub", PIVOT, 0.016, zb0, zb1, m["handle"])
    cube(stage, f"{h}/beam", (px, py - hw, zb0), (px + HANDLE["r"], py + hw, zb1), m["handle"], slick)
    s = SHOE["half"]
    cube(stage, f"{h}/shoe", (px + SHOE["rho"] - s, py - s, BED_H + SHOE["z0"]), (px + SHOE["rho"] + s, py + s, zb0), m["dark"], slick)
    kx = px + HANDLE["r"]
    q = KNOB["half"]
    cube(stage, f"{k}/sleeve", (kx - q, py - q, zb1), (kx + q, py + q, BED_H + KNOB["z1"]), m["knob"], grip)
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    _revolute(stage, f"{path}/sweep", bodies["anchor"].GetPath(), bodies["handle"].GetPath(), (px, py, 0.0), -2.0, HANDLE["stop"], 0.01)
    _revolute(stage, f"{path}/sleeve", bodies["handle"].GetPath(), bodies["knob"].GetPath(), (kx, py, 0.0), -400.0, 400.0, 0.001)
    return prim


def build_tube(stage, path):
    """Mid is the root link. Local +y runs from the first band (nose side) to the second (tail side)."""
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    m = {k: look(stage, f"{path}/Looks/{k}", COLORS[k]) for k in ("tube", "band", "block")}
    slick = surface(stage, f"{path}/Physics/slick", MU_SLICK)
    grip = surface(stage, f"{path}/Physics/grip", MU_GRIP)
    bodies = {}
    for name, mass in MASS.items():
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        rigid(bodies[name], mass)
    root = bodies["mid"]
    UsdPhysics.ArticulationRootAPI.Apply(root)
    physx(root, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False, "physxArticulation:solverPositionIterationCount": ITERS,
                                           "physxArticulation:solverVelocityIterationCount": VEL_ITERS, "physxArticulation:sleepThreshold": 0.0,
                                           "physxArticulation:stabilizationThreshold": 0.0})
    w, h, hm = TUBE["w"] / 2, TUBE["h"], TUBE["m"] / 2
    n, t = TUBE["n"], TUBE["t"]
    cube(stage, f"{path}/mid/body", (-w, -hm, 0.0), (w, hm, h), m["tube"], slick)
    cube(stage, f"{path}/mid/block", (-w, -GRIP["along"] / 2, h), (w, GRIP["along"] / 2, h + GRIP["h"]), m["block"], grip)
    cube(stage, f"{path}/nose/body", (-w, -hm - n, 0.0), (w, -hm, h), m["tube"], slick)
    cube(stage, f"{path}/tail/body", (-w, hm, 0.0), (w, hm + t, h), m["tube"], slick)
    b = BAND_LEN / 2
    cube(stage, f"{path}/mid/band_1", (-w, -hm, h), (w, -hm + b, h + 0.0005), m["band"])
    cube(stage, f"{path}/nose/band_1", (-w, -hm - b, h), (w, -hm, h + 0.0005), m["band"])
    cube(stage, f"{path}/mid/band_2", (-w, hm - b, h), (w, hm, h + 0.0005), m["band"])
    cube(stage, f"{path}/tail/band_2", (-w, hm, h), (w, hm + b, h + 0.0005), m["band"])
    _revolute(stage, f"{path}/bend_1", root.GetPath(), bodies["nose"].GetPath(), (0.0, -hm, 0.0), -0.1, 0.1, 0.002)
    _revolute(stage, f"{path}/bend_2", root.GetPath(), bodies["tail"].GetPath(), (0.0, hm, 0.0), -0.1, 0.1, 0.002)
    return root
