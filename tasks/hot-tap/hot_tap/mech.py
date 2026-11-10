"""Builds the hot-tap station into a USD stage. `build_line`: one fixed articulation with the pipe, an open-top valve body, its sliding gate
and a sprung coupon plug, and the machine's saddle. `build_machine`: one free articulation, a hollow housing with a boring bar on a prismatic feed.
`line` frame: on the table under the bar's line, x as world x relative to the axis, y as world y. `machine` frame: under the housing's middle,
on its underside; the bar feeds toward -y."""


import numpy as np

# Line, all in m in the `line` frame (x relative to the bar's axis). Bar axis height above the table: 0.05.
AXIS_Z = 0.05
PIPE_Y, PIPE_R = -0.155, 0.03
FLOOR_Z, WALL_TOP = 0.02, 0.075
CHANNEL_X, WALL_T = 0.02, 0.01  # channel half-width, wall thickness
STUB_Y0, VALVE_Y0, VALVE_Y1 = -0.125, -0.095, -0.045
GATE = dict(y=-0.08, t=0.007, half=0.026, z0=0.0205, z1=0.08, travel=0.05, post_x=0.012, post_y0=-0.096, post_y1=-0.0835, post_z1=0.115)
GATE_GAP = (-0.085, -0.075)  # walls stop here for the gate
RIB = dict(y0=-0.072, y1=-0.049, t=0.004, z1=0.035)
PLUG = dict(face=-0.112, depth=0.006, half_x=0.018, z0=0.035, z1=0.075, travel=0.006, spring=150.0)
SADDLE = dict(x=0.033, y0=VALVE_Y1, y1=0.113, wall=0.01, z1=0.035)
MASS_LINE = {"anchor": 5.0, "gate": 0.09, "plug": 0.03}

# Machine, in the `machine` frame. Hollow aluminium housing; solid colliders only.
BODY = dict(x=0.025, y=0.075, z=0.045)
BLOCK = dict(x=0.015, y=0.015, z0=0.045, z1=0.085, y_at=0.003)  # the grip block, over the balance point
BAR = dict(half=0.01, z0=0.02, z1=0.04, nose=-0.096, tail=0.117, post_y=0.105, post_x=0.012, post_y_half=0.012, post_z1=0.085, stroke=0.056)
MASS_MACHINE = {"body": 0.32, "bar": 0.11}

ITERS, VEL_ITERS, MU = 16, 4, 0.6
COLORS = {"pipe": (0.12, 0.32, 0.62), "steel": (0.30, 0.32, 0.35), "brass": (0.70, 0.55, 0.25), "gate": (0.85, 0.12, 0.08), "post": (0.95, 0.75, 0.10), "floor": (0.22, 0.23, 0.25),
          "saddle": (0.15, 0.16, 0.18), "housing": (0.72, 0.74, 0.77), "bar": (0.45, 0.47, 0.5), "coupon": (0.95, 0.5, 0.05), "hole": (0.05, 0.25, 0.75), "spray": (0.2, 0.55, 1.0)}


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


def disc(stage, path, x, y, z, r, mat, axis="Z", h=0.003, hidden=False):
    Gf, _, UsdGeom, _, UsdShade = _api()
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(h))
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if hidden:
        UsdGeom.Imageable(c.GetPrim()).MakeInvisible()
    return c


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


def _articulation(stage, path):
    _, _, _, UsdPhysics, _ = _api()
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False, "physxArticulation:solverPositionIterationCount": ITERS,
                                           "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    return prim, {k: look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}, surface(stage, f"{path}/Physics/mat", MU)


def _slide(stage, path, axis, parent, child, lower, upper, k, d, target=0.0):
    _, _, _, UsdPhysics, _ = _api()
    s = UsdPhysics.PrismaticJoint.Define(stage, path)
    s.CreateAxisAttr(axis)
    s.CreateLowerLimitAttr(float(lower))
    s.CreateUpperLimitAttr(float(upper))
    s.CreateBody0Rel().SetTargets([parent])
    s.CreateBody1Rel().SetTargets([child])
    dr = UsdPhysics.DriveAPI.Apply(s.GetPrim(), "linear")
    dr.CreateTypeAttr("force")
    dr.CreateTargetPositionAttr(float(target))
    dr.CreateStiffnessAttr(float(k))
    dr.CreateDampingAttr(float(d))
    return s


def build_line(stage, path, world_origin):
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    prim, m, phys = _articulation(stage, path)
    bodies = {}
    for name, mass in MASS_LINE.items():
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        rigid(bodies[name], mass)
    a, g, p = f"{path}/anchor", f"{path}/gate", f"{path}/plug"
    # pipe, along x, with a visual round section over a box collider
    cube(stage, f"{a}/pipe_box", (-0.12, PIPE_Y - PIPE_R, AXIS_Z - PIPE_R), (0.12, PIPE_Y + PIPE_R, AXIS_Z + PIPE_R), m["pipe"], phys)
    for k, sx in enumerate((-1, 1)):
        cube(stage, f"{a}/pipe_foot_{k}", (sx * 0.09 - 0.015, PIPE_Y - 0.035, 0.0), (sx * 0.09 + 0.015, PIPE_Y + 0.035, AXIS_Z - PIPE_R), m["steel"], phys)
    # stub and valve: one open-top channel, walls stopped short for the gate
    cube(stage, f"{a}/floor", (-CHANNEL_X - WALL_T, STUB_Y0, 0.0), (CHANNEL_X + WALL_T, VALVE_Y1, FLOOR_Z), m["floor"], phys)
    for sx, tag in ((-1, "w"), (1, "e")):
        x0, x1 = sorted((sx * CHANNEL_X, sx * (CHANNEL_X + WALL_T)))
        cube(stage, f"{a}/stub_{tag}", (x0, STUB_Y0, FLOOR_Z), (x1, VALVE_Y0, WALL_TOP), m["steel"], phys)
        cube(stage, f"{a}/valve_{tag}_a", (x0, VALVE_Y0, FLOOR_Z), (x1, GATE_GAP[0], WALL_TOP), m["brass"], phys)
        cube(stage, f"{a}/valve_{tag}_b", (x0, GATE_GAP[1], FLOOR_Z), (x1, VALVE_Y1, WALL_TOP), m["brass"], phys)
    # the blank's locating ribs, low on the floor under the bar
    cube(stage, f"{a}/rib_a", (-CHANNEL_X, RIB["y0"], FLOOR_Z), (CHANNEL_X, RIB["y0"] + RIB["t"], RIB["z1"]), m["steel"], phys)
    cube(stage, f"{a}/rib_b", (-CHANNEL_X, RIB["y1"], FLOOR_Z), (CHANNEL_X, RIB["y1"] + RIB["t"], RIB["z1"]), m["steel"], phys)
    # the saddle: a low pocket, the valve's own end is its front stop
    s = SADDLE
    cube(stage, f"{a}/saddle", (-s["x"] - s["wall"], s["y0"], 0.0), (s["x"] + s["wall"], s["y1"] + s["wall"], FLOOR_Z), m["saddle"], phys)
    for sx, tag in ((-1, "w"), (1, "e")):
        x0, x1 = sorted((sx * s["x"], sx * (s["x"] + s["wall"])))
        cube(stage, f"{a}/saddle_{tag}", (x0, s["y0"], FLOOR_Z), (x1, s["y1"], s["z1"]), m["saddle"], phys)
    cube(stage, f"{a}/saddle_rear", (-s["x"] - s["wall"], s["y1"], FLOOR_Z), (s["x"] + s["wall"], s["y1"] + s["wall"], s["z1"]), m["saddle"], phys)
    # what the cut uncovers: a blue disc in the stub, hidden until then
    disc(stage, f"{a}/hole", 0.0, PLUG["face"] - PLUG["depth"] / 2, WALL_TOP + 0.003, 0.016, m["hole"], hidden=True)
    disc(stage, f"{a}/spray", 0.0, VALVE_Y1 + 0.03, WALL_TOP + 0.05, 0.07, m["spray"], hidden=True)
    # gate, sliding across the bore in x; shut at 0, open toward -x
    G = GATE
    cube(stage, f"{g}/slab", (-G["half"], G["y"] - G["t"] / 2, G["z0"]), (G["half"], G["y"] + G["t"] / 2, G["z1"]), m["gate"], phys)
    cube(stage, f"{g}/post", (-G["post_x"], G["post_y0"], G["z1"]), (G["post_x"], G["post_y1"], G["post_z1"]), m["post"], phys)
    # the coupon: a sprung plug in the stub, standing for the pipe wall
    P = PLUG
    cube(stage, f"{p}/plug", (-P["half_x"], P["face"] - P["depth"], P["z0"]), (P["half_x"], P["face"], P["z1"]), m["steel"], phys)
    disc(stage, f"{p}/tab", 0.0, P["face"] - P["depth"] / 2, P["z1"] + 0.003, 0.016, m["coupon"])
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    _slide(stage, f"{path}/gate_slide", "X", bodies["anchor"].GetPath(), bodies["gate"].GetPath(), -G["travel"], 0.0, 0.0, 5.0)
    _slide(stage, f"{path}/plug_slide", "Y", bodies["anchor"].GetPath(), bodies["plug"].GetPath(), -P["travel"], 0.0, P["spring"], P["spring"] / 10)
    return prim


def build_machine(stage, path):
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    prim, m, phys = _articulation(stage, path)
    bodies = {}
    for name, mass in MASS_MACHINE.items():
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
    B, K, R = BODY, BLOCK, BAR
    rigid(bodies["body"], MASS_MACHINE["body"], (0.0, K["y_at"], 0.03))
    rigid(bodies["bar"], MASS_MACHINE["bar"], (0.0, 0.0065, 0.03))
    b, r = f"{path}/body", f"{path}/bar"
    cube(stage, f"{b}/housing", (-B["x"], -B["y"], 0.0), (B["x"], B["y"], B["z"]), m["housing"], phys)
    cube(stage, f"{b}/grip", (-K["x"], K["y_at"] - K["y"], K["z0"]), (K["x"], K["y_at"] + K["y"], K["z1"]), m["brass"], phys)
    cube(stage, f"{r}/shaft", (-R["half"], R["nose"], R["z0"]), (R["half"], R["tail"], R["z1"]), m["bar"], phys)
    cube(stage, f"{r}/post", (-R["post_x"], R["post_y"] - R["post_y_half"], R["z1"]), (R["post_x"], R["post_y"] + R["post_y_half"], R["post_z1"]), m["post"], phys)
    disc(stage, f"{r}/tab", 0.0, R["nose"] + 0.008, R["z1"] + 0.003, 0.016, m["coupon"], hidden=True)
    _slide(stage, f"{path}/feed", "Y", bodies["body"].GetPath(), bodies["bar"].GetPath(), -R["stroke"], 0.0, 0.0, 30.0)  # the feed's own drag holds the bar wherever it's left
    return prim


def set_visible(stage, path, on):
    _, _, UsdGeom, _, _ = _api()
    img = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    img.MakeVisible() if on else img.MakeInvisible()
