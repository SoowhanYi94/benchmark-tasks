"""Builds the rollover fixture into a USD stage, as one articulation: a steel base with two pillars, a cradle on a horizontal axle along y,
a retaining bar sliding along y on the cradle, a free-turning knob sleeve on the crank's pin, and a lock bolt sliding along y in the far
pillar. The bolt enters a notch in the cradle's end plate by real contact, only when the cradle is turned over. All link frames coincide
at the start (a foot on the table, under the axle). (u, v, w) in comments: the cradle's own axes from the axle, x, y, z when level."""

import numpy as np

AXLE = 0.075  # above the table
CRANK = dict(r=0.04, y=-0.146, half_x=0.015, half_y=0.012, half_z=0.02)  # the knob sleeve: its centre at u=-r, v=y
RAIL = dict(inner=0.046, wall=0.008, v0=-0.098, v1=0.058, w0=-0.031, w1=0.020)
LEDGE = dict(u0=0.030, w0=-0.031, w1=-0.026, v=0.041)  # the part's underside rests on w1; open between the ledges
STUB = dict(u0=0.022, v0=0.041, v1=0.047, w1=-0.018)  # corner stubs stop the part along y; fingers pass between them
BAR = dict(half_u=0.0455, half_v=0.015, w0=0.009, w1=0.017, park=-0.075, travel=0.075, lug_u=0.02, lug_v=0.008, lug_w=0.05)
PLATE = dict(half_u=0.04, half_w=0.045, v0=0.058, v1=0.068, notch_u=0.008, notch_w=-0.015)  # the far end plate, the bolt's notch toward -w
PILLAR = dict(half_x=0.03, near=(-0.120, -0.108), far=(0.072, 0.084), near_top=0.087, far_top=0.115)
BOLT = dict(half=0.006, z=0.100, tip=0.074, end=0.130, travel=0.015, lug_half_x=0.02, lug_y=0.008, lug_top=0.145)
ROLL_LIMIT = 195.0  # degrees: the cradle's far stop lies past turned over, so only the bolt holds it there
MASS = {"anchor": 1.0, "cradle": 0.6, "bar": 0.095, "bolt": 0.066, "knob": 0.03}
CRADLE_COM = (-0.004, -0.02, AXLE - 0.008)  # level: slightly toward the crank side, so it rests on its near stop
ROLL_DAMP, SLIDE_DAMP, KNOB_DAMP = 0.002, 5.0, 0.00002  # N m per deg/s, N per m/s, N m per deg/s
ITERS, VEL_ITERS, MU = 16, 4, 0.3
COLORS = {"steel": (0.30, 0.32, 0.35), "dark": (0.12, 0.13, 0.14), "cradle": (0.62, 0.64, 0.68), "bar": (0.10, 0.40, 0.85), "bolt": (0.85, 0.12, 0.08), "knob": (0.95, 0.75, 0.10)}


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


def rigid(prim, mass, com=None):
    Gf, _, _, UsdPhysics, _ = _api()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS, "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                        "physxRigidBody:maxDepenetrationVelocity": 1.0})
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(mass)
    if com is not None:
        api.CreateCenterOfMassAttr(Gf.Vec3f(*com))


def build_part(stage, path, boxes, mass, mu):
    """boxes: name -> (lo, hi, color)."""
    prim = stage.GetPrimAtPath(path)
    rigid(prim, mass)
    phys = surface(stage, f"{path}/Physics/part", mu)
    for name, (lo, hi, color) in boxes.items():
        cube(stage, f"{path}/{name}", lo, hi, look(stage, f"{path}/Looks/{name}", color), phys)
    return prim


def _uvw(lo, hi):
    """A box given in the cradle's axes from the axle, in the links' common frame."""
    return (lo[0], lo[1], lo[2] + AXLE), (hi[0], hi[1], hi[2] + AXLE)


def _joint(stage, kind, path, b0, b1, at, axis, lo, hi):
    Gf, _, _, UsdPhysics, _ = _api()
    j = kind.Define(stage, path)
    j.CreateAxisAttr(axis)
    j.CreateLowerLimitAttr(float(lo))
    j.CreateUpperLimitAttr(float(hi))
    j.CreateBody0Rel().SetTargets([b0])
    j.CreateBody1Rel().SetTargets([b1])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, at)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, at)))
    return j


def _damp(joint, kind, damping):
    dr = _api()[3].DriveAPI.Apply(joint.GetPrim(), kind)
    dr.CreateTypeAttr("force")
    dr.CreateStiffnessAttr(0.0)
    dr.CreateDampingAttr(float(damping))
    dr.CreateMaxForceAttr(1000.0)


def build_fixture(stage, path, world_origin):
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": True, "physxArticulation:solverPositionIterationCount": ITERS,
                                           "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    m = {k: look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}
    phys = surface(stage, f"{path}/Physics/mat", MU)
    bodies = {}
    for name, mass in MASS.items():
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        rigid(bodies[name], mass, CRADLE_COM if name == "cradle" else None)
    a, c, b, k, l = (f"{path}/{n}" for n in ("anchor", "cradle", "bar", "knob", "bolt"))
    # base and pillars
    pl = PILLAR
    cube(stage, f"{a}/base", (-0.07, -0.135, 0.0), (0.07, 0.10, 0.006), m["dark"], phys)
    cube(stage, f"{a}/near", (-pl["half_x"], pl["near"][0], 0.006), (pl["half_x"], pl["near"][1], pl["near_top"]), m["steel"], phys)
    cube(stage, f"{a}/far", (-pl["half_x"], pl["far"][0], 0.006), (pl["half_x"], pl["far"][1], BOLT["z"] - BOLT["half"]), m["steel"], phys)
    cube(stage, f"{a}/far_top", (-pl["half_x"], pl["far"][0], BOLT["z"] + BOLT["half"]), (pl["half_x"], pl["far"][1], pl["far_top"]), m["steel"], phys)
    # cradle: two rails, ledges, end beams, corner stubs, end plates, crank
    r, le, st, pa = RAIL, LEDGE, STUB, PLATE
    for s, sign in (("p", 1), ("m", -1)):
        lo, hi = sorted((sign * r["inner"], sign * (r["inner"] + r["wall"])))
        cube(stage, f"{c}/rail_{s}", *_uvw((lo, r["v0"], r["w0"]), (hi, r["v1"], r["w1"])), m["cradle"], phys)
        lo, hi = sorted((sign * le["u0"], sign * r["inner"]))
        cube(stage, f"{c}/ledge_{s}", *_uvw((lo, -le["v"], le["w0"]), (hi, le["v"], le["w1"])), m["cradle"], phys)
    for vs, name in ((1, "far"), (-1, "near")):
        v0, v1 = sorted((vs * st["v0"], vs * st["v1"]))
        cube(stage, f"{c}/beam_{name}", *_uvw((-r["inner"], v0, le["w0"]), (r["inner"], v1, le["w1"])), m["cradle"], phys)
        for us, un in ((1, "p"), (-1, "m")):
            u0, u1 = sorted((us * st["u0"], us * r["inner"]))
            cube(stage, f"{c}/stub_{name}_{un}", *_uvw((u0, v0, le["w1"]), (u1, v1, st["w1"])), m["dark"], phys)
    cube(stage, f"{c}/plate_top", *_uvw((-pa["half_u"], pa["v0"], pa["notch_w"]), (pa["half_u"], pa["v1"], pa["half_w"])), m["cradle"], phys)
    for us, un in ((1, "p"), (-1, "m")):
        u0, u1 = sorted((us * pa["notch_u"], us * pa["half_u"]))
        cube(stage, f"{c}/plate_{un}", *_uvw((u0, pa["v0"], -pa["half_w"]), (u1, pa["v1"], pa["notch_w"])), m["cradle"], phys)
    cube(stage, f"{c}/axle_far", *_uvw((-0.008, pa["v1"], -0.008), (0.008, pl["far"][0], 0.008)), m["dark"])
    cube(stage, f"{c}/plate_near", *_uvw((-0.054, -0.106, -0.031), (0.054, r["v0"], 0.020)), m["cradle"], phys)
    cube(stage, f"{c}/axle_near", *_uvw((-0.008, pl["near"][1], -0.008), (0.008, -0.106, 0.008)), m["dark"])
    cube(stage, f"{c}/crank", *_uvw((-CRANK["r"] - 0.008, -0.128, -0.008), (0.008, -0.122, 0.008)), m["dark"], phys)
    cube(stage, f"{c}/pin", *_uvw((-CRANK["r"] - 0.004, CRANK["y"] + CRANK["half_y"], -0.004), (-CRANK["r"] + 0.004, -0.128, 0.004)), m["dark"])
    # the knob sleeve, square to the world because it turns freely on the pin
    cr = CRANK
    cube(stage, f"{k}/sleeve", *_uvw((-cr["r"] - cr["half_x"], cr["y"] - cr["half_y"], -cr["half_z"]), (-cr["r"] + cr["half_x"], cr["y"] + cr["half_y"], cr["half_z"])), m["knob"], phys)
    # the retaining bar, parked toward the crank; its lug on the crank side
    br = BAR
    cube(stage, f"{b}/plate", *_uvw((-br["half_u"], br["park"] - br["half_v"], br["w0"]), (br["half_u"], br["park"] + br["half_v"], br["w1"])), m["bar"], phys)
    cube(stage, f"{b}/lug", *_uvw((-br["lug_u"], br["park"] - br["half_v"] - br["lug_v"], br["w0"]), (br["lug_u"], br["park"] - br["half_v"], br["lug_w"])), m["bar"], phys)
    # the lock bolt, withdrawn; its lug at the far end
    bo = BOLT
    cube(stage, f"{l}/rod", (-bo["half"], bo["tip"], bo["z"] - bo["half"]), (bo["half"], bo["end"], bo["z"] + bo["half"]), m["bolt"], phys)
    cube(stage, f"{l}/lug", (-bo["lug_half_x"], bo["end"] - bo["lug_y"], bo["z"] - bo["half"]), (bo["lug_half_x"], bo["end"], bo["lug_top"]), m["bolt"], phys)
    # joints
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    P = {n: bodies[n].GetPath() for n in bodies}
    _damp(_joint(stage, UsdPhysics.RevoluteJoint, f"{path}/roll", P["anchor"], P["cradle"], (0.0, 0.0, AXLE), "Y", 0.0, ROLL_LIMIT), "angular", ROLL_DAMP)
    _damp(_joint(stage, UsdPhysics.PrismaticJoint, f"{path}/bar_slide", P["cradle"], P["bar"], (0.0, 0.0, AXLE), "Y", 0.0, br["travel"]), "linear", SLIDE_DAMP)
    _damp(_joint(stage, UsdPhysics.RevoluteJoint, f"{path}/knob_spin", P["cradle"], P["knob"], (-cr["r"], cr["y"], AXLE), "Y", -720.0, 720.0), "angular", KNOB_DAMP)
    _damp(_joint(stage, UsdPhysics.PrismaticJoint, f"{path}/bolt_slide", P["anchor"], P["bolt"], (0.0, bo["tip"], bo["z"]), "Y", -bo["travel"], 0.0), "linear", SLIDE_DAMP)
    return prim
