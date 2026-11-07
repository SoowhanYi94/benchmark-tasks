"""Builds the dead-weight tester into a USD stage, as one articulation. Dimensions in m. The articulation's origin is the table's
origin, so every body's frame is given in table coordinates relative to the table top."""

import math

import numpy as np

ITERS, VEL_ITERS, MU = 16, 4, 0.6
COLORS = {"steel": (0.30, 0.32, 0.35), "dark": (0.10, 0.11, 0.12), "platen": (0.70, 0.72, 0.75), "bar": (0.65, 0.12, 0.70), "lever": (0.85, 0.12, 0.08),
          "knob": (0.10, 0.45, 0.85), "face": (0.93, 0.93, 0.90), "needle": (0.03, 0.03, 0.03), "mark": (0.85, 0.05, 0.05), "band": (0.10, 0.65, 0.20), "flag": (0.95, 0.45, 0.05)}


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


def cube(stage, path, lo, hi, mat, phys=None, visible=True):
    Gf, _, UsdGeom, UsdPhysics, UsdShade = _api()
    lo, hi = np.array(lo, float), np.array(hi, float)
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
    c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if not visible:
        c.CreatePurposeAttr("guide")
    if phys is not None:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        physx(c.GetPrim(), "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002, "physxCollision:restOffset": 0.0})
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")
    return c


def disc_face(stage, path, r, z0, z1, mat):
    """A round face with no collider."""
    Gf, _, UsdGeom, _, UsdShade = _api()
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(z1 - z0))
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(z0 + z1) / 2))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


def rigid(prim, mass, linear_damping=None, angular_damping=None):
    _, _, _, UsdPhysics, _ = _api()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    attrs = {"physxRigidBody:solverPositionIterationCount": ITERS, "physxRigidBody:solverVelocityIterationCount": VEL_ITERS, "physxRigidBody:maxDepenetrationVelocity": 1.0}
    if angular_damping is not None:
        attrs["physxRigidBody:angularDamping"] = float(angular_damping)
    physx(prim, "PhysxRigidBodyAPI", **attrs)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


def _slotted(stage, path, r, half, back, h, mat):
    """The slotted weight's seen shape: a round plate with a slot from `back` out through +x, as a triangle strip about the origin."""
    Gf, _, UsdGeom, _, UsdShade = _api()
    open_at = math.asin(half / r)
    rim, bore = [], []
    n = 48
    for k in range(n + 1):
        th = open_at + (2 * math.pi - 2 * open_at) * k / n
        c, s = math.cos(th), math.sin(th)
        rim.append((r * c, r * s))
        # where this ray from the origin leaves the slot's rectangle: its sides at y=+/-half, its end at x=-back
        t = min(half / abs(s) if abs(s) > 1e-9 else 1e9, back / -c if c < -1e-9 else 1e9, r)
        bore.append((t * c, t * s))
    m = len(rim)
    pts = [(*p, z) for z in (0.0, h) for p in rim] + [(*p, z) for z in (0.0, h) for p in bore]
    rim0, rim1, bore0, bore1 = 0, m, 2 * m, 3 * m
    quads = []
    for k in range(m - 1):
        quads += [(bore1 + k, rim1 + k, rim1 + k + 1, bore1 + k + 1), (bore0 + k, bore0 + k + 1, rim0 + k + 1, rim0 + k),
                  (rim0 + k, rim0 + k + 1, rim1 + k + 1, rim1 + k), (bore0 + k, bore1 + k, bore1 + k + 1, bore0 + k + 1)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4] * len(quads))
    mesh.CreateFaceVertexIndicesAttr([i for q in quads for i in q])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def build_weight(stage, path, spec, angular_damping, color):
    """One slotted weight. Boxes collide: two arms each side of the slot, with flat sides for the fingers, and a back."""
    prim = stage.GetPrimAtPath(path)
    rigid(prim, spec["mass"], angular_damping=angular_damping)
    mat, phys = look(stage, f"{path}/Looks/part", color), surface(stage, f"{path}/Physics/part", MU)
    r, s, back, h = spec["r"], spec["slot"], spec["back"], spec["h"]
    flat = r - 0.001
    x0, x1 = -0.020, 0.025
    for name, lo, hi in (("arm_l", (x0, s, 0.0), (x1, flat, h)), ("arm_r", (x0, -flat, 0.0), (x1, -s, h)), ("back", (-r + 0.002, -0.026, 0.0), (-back, 0.026, h))):
        cube(stage, f"{path}/{name}", lo, hi, mat, phys, visible=False)
    _slotted(stage, f"{path}/seen", r, s, back, h, mat)
    return prim


def _body(stage, path, xyz, mass, inertia=None, angular_damping=None):
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
    x.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    rigid(x.GetPrim(), mass, angular_damping=angular_damping)
    if inertia is not None:
        UsdPhysics.MassAPI(x.GetPrim()).CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
    return x.GetPrim()


def _joint(kind, stage, path, b0, b1, at, axis, lo, hi):
    Gf, _, _, UsdPhysics, _ = _api()
    j = kind.Define(stage, path)
    j.CreateAxisAttr(axis)
    if lo is not None:
        j.CreateLowerLimitAttr(lo)
        j.CreateUpperLimitAttr(hi)
    j.CreateBody0Rel().SetTargets([b0])
    j.CreateBody1Rel().SetTargets([b1])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, at)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    return j


def _drive(j, kind, stiffness, damping, target=0.0):
    _, _, _, UsdPhysics, _ = _api()
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), kind)
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(float(stiffness))
    d.CreateDampingAttr(float(damping))
    d.CreateTargetPositionAttr(float(target))
    return d


def build_tester(stage, path, world_origin, g):
    """g: the task's geometry (see task.py). Bodies: anchor (fixed), piston (rises), rotor (turns in the piston), lever, knob, needle, flag."""
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False, "physxArticulation:solverPositionIterationCount": ITERS,
                                           "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    m = {k: look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}
    phys = surface(stage, f"{path}/Physics/mat", MU)

    a = f"{path}/anchor"
    _body(stage, a, (0.0, 0.0, 0.0), 2.0)
    px, py = g["piston_at"]
    plat = g["platen_z"]
    ch = g["column_half"]
    cube(stage, f"{a}/column", (px - ch, py - ch, 0.0), (px + ch, py + ch, plat - g["platen_t"] - 0.002), m["steel"], phys)
    # pipes from the column to the gauge and to the pump, seen only
    gx, gy = g["gauge_at"]
    lx, ly = g["lever_pivot"]
    cube(stage, f"{a}/pipe_a", (px - 0.005, py, 0.001), (px + 0.005, gy, 0.006), m["dark"])
    cube(stage, f"{a}/pipe_b", (px, gy - 0.005, 0.001), (gx, gy + 0.005, 0.006), m["dark"])
    cube(stage, f"{a}/pipe_c", (lx, gy - 0.005, 0.001), (px, gy + 0.005, 0.006), m["dark"])
    cube(stage, f"{a}/pipe_d", (lx - 0.005, gy, 0.001), (lx + 0.005, ly, 0.006), m["dark"])
    cube(stage, f"{a}/lever_foot", (lx - 0.02, ly - 0.02, 0.0), (lx + 0.02, ly + 0.02, g["lever_z"] - 0.008), m["steel"], phys)
    # the gauge: a dial face up, with a red mark at +x
    gr = g["gauge_r"]
    Gx = UsdGeom.Xform.Define(stage, f"{a}/gauge")
    Gx.AddTranslateOp().Set(Gf.Vec3d(gx, gy, 0.0))
    disc_face(stage, f"{a}/gauge/rim", gr + 0.006, 0.0, 0.012, m["dark"])
    disc_face(stage, f"{a}/gauge/face", gr, 0.012, 0.014, m["face"])
    cube(stage, f"{a}/gauge/mark", (gr * 0.55, -0.006, 0.0142), (gr, 0.006, 0.0146), m["mark"])
    # the float indicator: a plate up-facing beyond the column, the green band on it
    fx0, fy = g["flag_x0"], g["flag_y"]
    cube(stage, f"{a}/ind_plate", (fx0 - 0.012, fy - 0.03, 0.0), (fx0 + g["flag_gain"] * g["top_stop"] + 0.012, fy + 0.03, 0.01), m["face"])
    cube(stage, f"{a}/ind_band", (fx0 + g["flag_gain"] * g["band"][0], fy - 0.03, 0.01), (fx0 + g["flag_gain"] * g["band"][1], fy + 0.03, 0.0104), m["band"])
    cube(stage, f"{a}/ind_link", (px + ch, fy - 0.004, 0.002), (fx0 - 0.012, fy + 0.004, 0.008), m["dark"])
    # the knob's boss
    kx, ky = g["knob_at"]
    cube(stage, f"{a}/knob_boss", (kx - 0.012, ky - 0.012, 0.0), (kx + 0.012, ky + 0.012, g["knob_z"] - 0.002), m["steel"], phys)

    pi = f"{path}/piston"
    _body(stage, pi, (px, py, 0.0), g["piston_mass"], inertia=(1e-3, 1e-3, 1e-3))
    ro = f"{path}/rotor"
    _body(stage, ro, (px, py, 0.0), g["rotor_mass"], angular_damping=g["spin_damping"])
    p, pz = g["platen_half"], plat
    cube(stage, f"{ro}/platen", (-p, -p, pz - g["platen_t"]), (p, p, pz), m["platen"], phys)
    s, top = g["stem_half"], pz + g["stem_h"]
    cube(stage, f"{ro}/stem", (-s, -s, pz), (s, s, top), m["steel"], phys)
    b, bw = g["bar_half"], g["bar_width"]
    cube(stage, f"{ro}/bar", (-b, -bw, top), (b, bw, top + g["bar_t"]), m["bar"], phys)
    cube(stage, f"{ro}/bar2", (-bw, -b, top), (bw, b, top + g["bar_t"]), m["bar"], phys)  # a cross, so a sweep past it catches an arm at any angle

    le = f"{path}/lever"
    _body(stage, le, (lx, ly, g["lever_z"]), g["lever_mass"])
    # the lever's own frame: along +x from the pivot, turned up by the joint's start
    L = g["lever_len"]
    cube(stage, f"{le}/arm", (0.0, -0.006, -0.006), (L, 0.006, 0.006), m["steel"], phys)
    pw = g["lever_pad"]
    cube(stage, f"{le}/pad", (L - 2 * pw, -pw, 0.006), (L, pw, 0.014), m["lever"], phys)

    kn = f"{path}/knob"
    _body(stage, kn, (kx, ky, g["knob_z"]), g["knob_mass"])
    kb, kw, kt = g["knob_half"], g["knob_width"], g["knob_t"]
    cube(stage, f"{kn}/hub", (-0.018, -0.018, 0.0), (0.018, 0.018, 0.008), m["steel"], phys)
    cube(stage, f"{kn}/wing", (-kb, -kw, 0.008), (kb, kw, 0.008 + kt), m["knob"], phys)

    ne = f"{path}/needle"
    _body(stage, ne, (gx, gy, 0.0), 0.02, inertia=(1e-4, 1e-4, 1e-4))
    cube(stage, f"{ne}/hand", (-0.012, -0.004, 0.0148), (gr * 0.92, 0.004, 0.0162), m["needle"])

    fl = f"{path}/flag"
    _body(stage, fl, (fx0, fy, 0.0), 0.02, inertia=(1e-4, 1e-4, 1e-4))
    cube(stage, f"{fl}/flag", (-0.006, -g["flag_half"], 0.012), (0.006, g["flag_half"], 0.0135), m["flag"])

    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([a])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))

    rise = _joint(UsdPhysics.PrismaticJoint, stage, f"{path}/rise", a, pi, (px, py, 0.0), "Z", 0.0, g["top_stop"])
    _drive(rise, "linear", g["oil_k"], g["oil_c"])
    _joint(UsdPhysics.RevoluteJoint, stage, f"{path}/spin", pi, ro, (0.0, 0.0, 0.0), "Z", None, None)
    pump = _joint(UsdPhysics.RevoluteJoint, stage, f"{path}/pump", a, le, (lx, ly, g["lever_z"]), "Y", g["lever_up"], g["lever_down"])
    _drive(pump, "angular", g["lever_k"], g["lever_c"], g["lever_up"] - g["lever_preload"])
    trim = _joint(UsdPhysics.RevoluteJoint, stage, f"{path}/trim", a, kn, (kx, ky, g["knob_z"]), "Z", -g["knob_limit"], g["knob_limit"])
    _drive(trim, "angular", 0.0, g["knob_c"])
    _joint(UsdPhysics.RevoluteJoint, stage, f"{path}/needle_turn", a, ne, (gx, gy, 0.0), "Z", None, None)
    _joint(UsdPhysics.PrismaticJoint, stage, f"{path}/flag_slide", a, fl, (fx0, fy, 0.0), "X", -0.01, 0.2)
    return prim
