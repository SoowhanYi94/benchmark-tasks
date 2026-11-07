"""Builds the turbine casing ring into a USD stage as one fixed body: a base plate, a covered ring groove with a slot in its cover,
a gate where the cover is cut away, a stop at the gate's far edge and a low step on the groove's floor. Each part is a free rigid body
of boxes. Dimensions in m, relative to the ring's foot on the table."""

import math

import numpy as np

R = 0.10  # the groove's mean radius
FLOOR = 0.008  # the groove's floor above the table
GROOVE = dict(r0=0.082, r1=0.118, h=0.032)  # 36 wide, 32 tall above the floor
OUTER = dict(r0=0.072, r1=0.128)  # the walls' faces toward the hub and outside
COVER_T = 0.006
SLOT = dict(r0=0.089, r1=0.111)  # 22 wide
POCKET = dict(r0=0.078, r1=0.122)  # the groove's walls stand back this far inside the gate
STEP_H = 0.006
WALL_DEG = 5.0  # facets: 72 round each wall
MU = 0.15  # lubricated: the ring and a segment's body
COLORS = {"plate": (0.20, 0.21, 0.23), "floor": (0.07, 0.07, 0.08), "ring": (0.62, 0.64, 0.67), "hub": (0.42, 0.44, 0.47), "stop": (0.75, 0.10, 0.08), "step": (0.90, 0.72, 0.08)}


def top():
    return FLOOR + GROOVE["h"] + COVER_T


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


def box(stage, path, c, size, yaw, mat, phys=None):
    """A box with its centre at c, edges `size`, turned yaw radians about z."""
    Gf, _, UsdGeom, UsdPhysics, UsdShade = _api()
    b = UsdGeom.Cube.Define(stage, path)
    b.CreateSizeAttr(1.0)
    b.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
    b.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2))))
    b.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    UsdShade.MaterialBindingAPI.Apply(b.GetPrim()).Bind(mat)
    if phys is not None:
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
        physx(b.GetPrim(), "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002, "physxCollision:restOffset": 0.0})
        UsdShade.MaterialBindingAPI.Apply(b.GetPrim()).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")
    return b


def arc_box(r0, r1, a0, a1, z0, z1):
    """(centre, size, yaw) of a facet between radii r0 and r1 and angles a0 and a1 in radians, about the origin."""
    mid, half = (a0 + a1) / 2, (a1 - a0) / 2
    rc = (r0 + r1) / 2
    length = 2 * r1 * math.tan(half)  # long enough to close the outer edge; adjacent facets overlap slightly on the inner edge
    return (rc * math.cos(mid), rc * math.sin(mid), (z0 + z1) / 2), (r1 - r0, length, z1 - z0), mid


def rigid(prim, mass, kinematic=False):
    _, _, _, UsdPhysics, _ = _api()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
    physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": 16, "physxRigidBody:solverVelocityIterationCount": 4,
                                        "physxRigidBody:maxDepenetrationVelocity": 1.0})
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


def build_part(stage, path, boxes, mass, mus, colors):
    """boxes: name -> (centre, size, yaw, tag); mus and colors by tag."""
    prim = stage.GetPrimAtPath(path)
    rigid(prim, mass)
    mats = {t: look(stage, f"{path}/Looks/{t}", c) for t, c in colors.items()}
    phys = {t: surface(stage, f"{path}/Physics/{t}", m) for t, m in mus.items()}
    for name, (c, size, yaw, tag) in boxes.items():
        box(stage, f"{path}/{name}", c, size, yaw, mats[tag], phys[tag])
    return prim


def _disc(stage, path, r, z0, z1, mat):
    Gf, _, UsdGeom, _, UsdShade = _api()
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(z1 - z0))
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(z0 + z1) / 2))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


def build_ring(stage, path, gate_deg, gate_half_deg, step_deg, step_half_deg):
    """Angles in degrees from +x, about the ring's centre. Segments travel clockwise from the gate, so a run ends at the stop, on the gate's anticlockwise edge."""
    prim = stage.GetPrimAtPath(path)
    rigid(prim, 50.0, kinematic=True)
    m = {k: look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}
    phys = surface(stage, f"{path}/Physics/mat", MU)
    z0, z_cover, z_top = FLOOR, FLOOR + GROOVE["h"], top()
    # the floor is one flat plate, with no seams for a sliding part to catch on
    box(stage, f"{path}/plate", (0.0, 0.0, FLOOR / 2), (2 * OUTER["r1"] + 0.01, 2 * OUTER["r1"] + 0.01, FLOOR), 0.0, m["plate"], phys)
    _disc(stage, f"{path}/groove_floor", GROOVE["r1"] + 0.004, FLOOR, FLOOR + 0.0002, m["floor"])
    _disc(stage, f"{path}/hub", OUTER["r0"] + 0.0005, FLOOR, z_top, m["hub"])
    n = round(360.0 / WALL_DEG)
    for i in range(n):
        a0, a1 = i * WALL_DEG, (i + 1) * WALL_DEG
        mid = (a0 + a1) / 2
        d = (mid - gate_deg + 180.0) % 360.0 - 180.0
        in_gate = abs(d) < gate_half_deg
        r0, r1 = (POCKET["r0"], POCKET["r1"]) if in_gate else (GROOVE["r0"], GROOVE["r1"])
        th0, th1 = math.radians(a0), math.radians(a1)
        for tag, ra, rb in (("in", OUTER["r0"], r0), ("out", r1, OUTER["r1"])):
            box(stage, f"{path}/wall_{tag}_{i}", *arc_box(ra, rb, th0, th1, z0, z_top), m["ring"], phys)
        if not in_gate:
            for tag, ra, rb in (("in", r0, SLOT["r0"]), ("out", SLOT["r1"], r1)):
                box(stage, f"{path}/cover_{tag}_{i}", *arc_box(ra, rb, th0, th1, z_cover, z_top), m["ring"], phys)
    # the stop: a block filling the groove at the gate's far edge, its face at that edge, with its top showing red in the slot's line
    g0, g1 = math.radians(gate_deg + gate_half_deg - 5.5), math.radians(gate_deg + gate_half_deg)
    box(stage, f"{path}/stop", *arc_box(POCKET["r0"], POCKET["r1"], g0, g1, z0, z_cover), m["stop"], phys)
    # the step: a low bar across the floor, shown by a yellow patch on the cover
    s0, s1 = math.radians(step_deg - step_half_deg), math.radians(step_deg + step_half_deg)
    box(stage, f"{path}/step", *arc_box(GROOVE["r0"], GROOVE["r1"], s0, s1, z0, z0 + STEP_H), m["step"], phys)
    for tag, ra, rb in (("in", OUTER["r0"], SLOT["r0"]), ("out", SLOT["r1"], OUTER["r1"])):
        box(stage, f"{path}/step_mark_{tag}", *arc_box(ra, rb, s0 - 0.14, s1 + 0.14, z_top, z_top + 0.0005), m["step"])
    return prim
