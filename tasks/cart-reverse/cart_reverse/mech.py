"""Builds the tow cart into a USD stage as one free articulation on the table: a chassis on two wheels on a fixed axle and a low-friction
skid under the nose, with a drawbar hinged at the hitch. In the cart's own frame: origin on the table under the axle's middle, +x from the
axle toward the hitch. Dimensions in m. Wheels are revolute joints with sphere colliders: the rolling direction is free, sideways they grip."""

import numpy as np

WB = 0.12  # axle to hitch
BAR = 0.10  # hitch to the handle's centre
WHEEL = dict(r=0.02, y=0.08, w=0.016, mass=0.03)
DECK = dict(x0=-0.03, x1=0.10, half=0.045, z0=0.03, z1=0.045)
TOTE = dict(x0=-0.02, x1=0.09, half=0.04, z1=0.075)
HITCH_Z = 0.05
HANDLE = dict(half_x=0.015, half_y=0.015, z0=0.04, z1=0.075)  # gripped across y, centred BAR from the hitch
HITCH_LIMIT = 75.0  # degrees either way, the hinge's stop: the jackknife
LIFT_LIMIT = 20.0  # degrees the bar can swing up; it rests level
MASS = dict(chassis=0.35, yoke=0.01, bar=0.06)
MU = dict(wheel=1.0, skid=0.02, body=0.5)
ITERS, VEL_ITERS = 16, 4
COLORS = {"deck": (0.72, 0.74, 0.77), "tote": (0.15, 0.40, 0.75), "wheel": (0.08, 0.08, 0.09), "bar": (0.30, 0.32, 0.35), "handle": (0.75, 0.16, 0.01), "nose": (0.30, 0.32, 0.35)}


def _api():
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    return Gf, Sdf, UsdGeom, UsdPhysics, UsdShade


def physx(prim, schema, **attrs):
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float, str: Sdf.ValueTypeNames.Token}
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


def surface(stage, path, mu, combine=None):
    _, _, _, UsdPhysics, UsdShade = _api()
    phys = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
    pm.CreateStaticFrictionAttr(mu)
    pm.CreateDynamicFrictionAttr(mu)
    if combine:
        physx(phys.GetPrim(), "PhysxMaterialAPI", **{"physxMaterial:frictionCombineMode": combine})
    return phys


def _collide(prim, phys):
    _, _, _, UsdPhysics, UsdShade = _api()
    UsdPhysics.CollisionAPI.Apply(prim)
    physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002, "physxCollision:restOffset": 0.0})
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def cube(stage, path, lo, hi, mat, phys=None):
    Gf, _, UsdGeom, _, UsdShade = _api()
    lo, hi = np.array(lo, float), np.array(hi, float)
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
    c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if phys is not None:
        _collide(c.GetPrim(), phys)
    return c


def sphere(stage, path, at, r, mat, phys, visible=True):
    Gf, _, UsdGeom, _, UsdShade = _api()
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(float(r))
    s.AddTranslateOp().Set(Gf.Vec3d(*at))
    UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(mat)
    if not visible:
        s.CreatePurposeAttr("guide")
    _collide(s.GetPrim(), phys)


def rigid(prim, mass, com=None):
    Gf, _, _, UsdPhysics, _ = _api()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS, "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                        "physxRigidBody:maxDepenetrationVelocity": 1.0})
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(mass)
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))


def build_part(stage, path, boxes, mass, mu, color):
    prim = stage.GetPrimAtPath(path)
    rigid(prim, mass)
    mat, phys = look(stage, f"{path}/Looks/part", color), surface(stage, f"{path}/Physics/part", mu)
    for name, (lo, hi) in boxes.items():
        cube(stage, f"{path}/{name}", lo, hi, mat, phys)
    return prim


def _damp(joint, damping):
    """A little bearing drag, as a drive with no spring, so a free joint doesn't creep."""
    _, _, _, UsdPhysics, _ = _api()
    d = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(float(damping))


def build_cart(stage, path):
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False, "physxArticulation:solverPositionIterationCount": ITERS,
                                           "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    m = {k: look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}
    grip = surface(stage, f"{path}/Physics/wheel", MU["wheel"], "max")
    skid = surface(stage, f"{path}/Physics/skid", MU["skid"], "min")
    body = surface(stage, f"{path}/Physics/body", MU["body"])
    bodies = {}
    masses = {"chassis": MASS["chassis"], "wheel_l": WHEEL["mass"], "wheel_r": WHEEL["mass"], "yoke": MASS["yoke"], "bar": MASS["bar"]}
    for name, mass in masses.items():
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        # the load sits over the axle, so the wheels carry it and grip sideways
        rigid(bodies[name], mass, (0.02, 0.0, 0.045) if name == "chassis" else None)
    c = f"{path}/chassis"
    d = DECK
    cube(stage, f"{c}/deck", (d["x0"], -d["half"], d["z0"]), (d["x1"], d["half"], d["z1"]), m["deck"], body)
    cube(stage, f"{c}/tote", (TOTE["x0"], -TOTE["half"], d["z1"]), (TOTE["x1"], TOTE["half"], TOTE["z1"]), m["tote"], body)
    cube(stage, f"{c}/axle", (-0.006, -WHEEL["y"], 0.014), (0.006, WHEEL["y"], d["z0"]), m["bar"])
    cube(stage, f"{c}/nose", (d["x1"], -0.012, d["z0"]), (WB + 0.012, 0.012, d["z1"]), m["nose"], body)
    sphere(stage, f"{c}/skid", (0.10, 0.0, 0.01), 0.01, m["bar"], skid)
    cube(stage, f"{c}/skid_post", (0.094, -0.006, 0.01), (0.106, 0.006, d["z0"]), m["bar"])
    for name, sign in (("wheel_l", 1), ("wheel_r", -1)):
        w = f"{path}/{name}"
        y = sign * WHEEL["y"]
        sphere(stage, f"{w}/collide", (0.0, y, WHEEL["r"]), WHEEL["r"], m["wheel"], grip)
        j = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/spin_{name}")
        j.CreateAxisAttr("Y")
        j.CreateBody0Rel().SetTargets([bodies["chassis"].GetPath()])
        j.CreateBody1Rel().SetTargets([bodies[name].GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(y), float(WHEEL["r"])))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, float(y), float(WHEEL["r"])))
        _damp(j, 0.0002)
    b = f"{path}/bar"
    x_h = WB + BAR
    cube(stage, f"{path}/yoke/pin", (WB - 0.008, -0.008, d["z1"]), (WB + 0.008, 0.008, HITCH_Z + 0.008), m["bar"])
    cube(stage, f"{b}/rod", (WB, -0.006, HITCH_Z - 0.006), (x_h, 0.006, HITCH_Z + 0.006), m["bar"], body)
    cube(stage, f"{b}/handle", (x_h - HANDLE["half_x"], -HANDLE["half_y"], HANDLE["z0"]), (x_h + HANDLE["half_x"], HANDLE["half_y"], HANDLE["z1"]), m["handle"], body)
    hitch = Gf.Vec3f(float(WB), 0.0, float(HITCH_Z))
    h = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/hitch")
    h.CreateAxisAttr("Z")
    h.CreateLowerLimitAttr(-HITCH_LIMIT)
    h.CreateUpperLimitAttr(HITCH_LIMIT)
    h.CreateBody0Rel().SetTargets([bodies["chassis"].GetPath()])
    h.CreateBody1Rel().SetTargets([bodies["yoke"].GetPath()])
    h.CreateLocalPos0Attr(hitch)
    h.CreateLocalPos1Attr(hitch)
    _damp(h, 0.02)
    p = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/lift")
    p.CreateAxisAttr("Y")  # negative raises the bar
    p.CreateLowerLimitAttr(-LIFT_LIMIT)
    p.CreateUpperLimitAttr(0.0)
    p.CreateBody0Rel().SetTargets([bodies["yoke"].GetPath()])
    p.CreateBody1Rel().SetTargets([bodies["bar"].GetPath()])
    p.CreateLocalPos0Attr(hitch)
    p.CreateLocalPos1Attr(hitch)
    _damp(p, 0.02)
    return prim
