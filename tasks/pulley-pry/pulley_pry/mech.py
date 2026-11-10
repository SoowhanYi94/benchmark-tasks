"""USD builders for the pulley-pry scene. Dimensions in m. Collision is simple boxes; the look is separate visual-only geometry
(lathe meshes, cylinders) with physically based materials.

Frames: the machine is built in table coordinates (x forward, y left, z up from the table top). The pulley's own frame sits on the
shaft axis at its back face, x along the shaft. Each bar's frame sits on the bar's centre line at its tip end."""

import math

import numpy as np

ITERS, VEL_ITERS = 32, 8

# geometry (table frame); see task.py for how they're used
ZS = 0.075          # shaft axis above the table
XH = 0.25           # the bearing housing's front face
GAP = 0.019         # housing face to the pulley's back face
W = 0.065           # housing half-width
BEVEL = 0.006       # 45-degree bevel on the housing's front corners; its inner edge, W - BEVEL out, is each bar's fulcrum
BOSS = 0.028        # bearing-cap boss half-width (square collider), 12 mm proud of the face
BOSS_L = 0.012
SHAFT_R = 0.0095    # 19 mm shaft
SHAFT_END = XH + GAP + 0.050  # flush with the hub front at the start
SEIZED = 0.008      # the band of travel over which the fit is seized
PLATE_Z = 0.010
RAIL_Y = (0.060, 0.085)   # motor slide rails, each side
RAIL_X = (0.17, 0.28)
RAIL_TOP = ZS - 0.006

PULLEY = dict(r_o=0.045, r_rim=0.037, rim_w=0.026, web=0.010, r_hub=0.020, hub_l=0.050, bore=SHAFT_R + 0.0003, relief=SHAFT_R + 0.0011,
              land=0.025, mass=0.80)
BAR = dict(len=0.242, t=0.010, h=0.012, tip=0.004, tip_w=0.006, mass=0.24, grip=(0.170, 0.242))

# physically based looks: base colour, metallic, roughness
LOOKS = {
    "cast_iron": ((0.21, 0.22, 0.23), 0.75, 0.62),
    "machined": ((0.62, 0.63, 0.64), 1.0, 0.28),
    "steel": ((0.55, 0.56, 0.58), 1.0, 0.35),
    "forged": ((0.16, 0.17, 0.18), 0.9, 0.45),
    "paint_blue": ((0.10, 0.22, 0.42), 0.15, 0.45),
    "paint_grey": ((0.34, 0.37, 0.38), 0.15, 0.5),
    "frame": ((0.20, 0.27, 0.22), 0.2, 0.55),
    "rubber_red": ((0.62, 0.08, 0.05), 0.0, 0.8),
    "rubber_yel": ((0.85, 0.62, 0.05), 0.0, 0.8),
    "wood": ((0.52, 0.38, 0.22), 0.0, 0.75),
    "dark": ((0.06, 0.06, 0.065), 0.2, 0.6),
    "brass": ((0.78, 0.60, 0.30), 1.0, 0.3),
    "stripe": ((0.92, 0.90, 0.85), 0.0, 0.5),
}


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


def look(stage, path, name):
    Gf, Sdf, _, _, UsdShade = _api()
    color, metal, rough = LOOKS[name]
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def looks(stage, root, names=None):
    return {k: look(stage, f"{root}/Looks/{k}", k) for k in (names or LOOKS)}


def surface(stage, path, mu, combine=None):
    _, _, _, UsdPhysics, UsdShade = _api()
    phys = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
    pm.CreateStaticFrictionAttr(float(mu))
    pm.CreateDynamicFrictionAttr(float(mu))
    pm.CreateRestitutionAttr(0.0)
    if combine:
        physx(phys.GetPrim(), "PhysxMaterialAPI", **{"physxMaterial:frictionCombineMode": combine})
    return phys


def _collide(prim, phys, offset=0.002):
    _, _, _, UsdPhysics, UsdShade = _api()
    UsdPhysics.CollisionAPI.Apply(prim)
    physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": float(offset), "physxCollision:restOffset": 0.0})
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def collider(stage, path, centre, size, phys, rot_x=0.0, offset=0.002, rot_z=0.0):
    """An invisible collision box: centre, full size, turned rot_x about the x axis (or rot_z about z)."""
    Gf, _, UsdGeom, _, _ = _api()
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
    if rot_x:
        c.AddOrientOp().Set(Gf.Quatf(math.cos(rot_x / 2), Gf.Vec3f(math.sin(rot_x / 2), 0.0, 0.0)))
    elif rot_z:
        c.AddOrientOp().Set(Gf.Quatf(math.cos(rot_z / 2), Gf.Vec3f(0.0, 0.0, math.sin(rot_z / 2))))
    c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    _collide(c.GetPrim(), phys, offset)
    UsdGeom.Imageable(c.GetPrim()).MakeInvisible()
    return c


def vbox(stage, path, lo, hi, mat, rot=None):
    """A visual box from lo to hi; rot: (axis, angle) about the box's centre."""
    Gf, _, UsdGeom, _, UsdShade = _api()
    lo, hi = np.array(lo, float), np.array(hi, float)
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
    if rot is not None:
        ax, a = rot
        s = math.sin(a / 2)
        c.AddOrientOp().Set(Gf.Quatf(math.cos(a / 2), Gf.Vec3f(ax[0] * s, ax[1] * s, ax[2] * s)))
    c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    return c


def vcyl(stage, path, centre, r, h, axis, mat):
    Gf, _, UsdGeom, _, UsdShade = _api()
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(h))
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    return c


def mesh(stage, path, points, faces, mat):
    Gf, _, UsdGeom, _, UsdShade = _api()
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def lathe(stage, path, profile, mat, centre=(0.0, 0.0, 0.0), n=40):
    """A visual surface of revolution about the x axis: profile is [(r, u), ...], u along +x from centre."""
    pts, faces = [], []
    c = np.asarray(centre, float)
    for r, u in profile:
        for k in range(n):
            a = 2 * math.pi * k / n
            pts.append(c + (u, r * math.cos(a), r * math.sin(a)))
    m = len(profile)
    for i in range(m - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            faces.append((a, b, b + n, a + n))
    return mesh(stage, path, pts, faces, mat)


def rigid(prim, mass):
    _, _, _, UsdPhysics, _ = _api()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS, "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                        "physxRigidBody:maxDepenetrationVelocity": 0.3})
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))


def ring(stage, path, n, r_in, r_out, u0, u1, phys, offset=0.002):
    """A ring of n collision plates about the x axis: inner faces form an n-gon with inradius r_in."""
    half_t = r_out * math.tan(math.pi / n) + 0.001 if n > 4 else r_out
    for k in range(n):
        a = 2 * math.pi * k / n
        rc = (r_in + r_out) / 2
        # a plate whose normal points along (0, cos a, sin a): build it along local +z, then turn about x
        collider(stage, f"{path}_{k}", ((u0 + u1) / 2, -rc * math.sin(a), rc * math.cos(a)), (u1 - u0, 2 * half_t, r_out - r_in), phys, rot_x=a, offset=offset)


# -- the machine: frame, gearmotor, bearing housing, shaft, slide rails, bar rack, V-block -----------------------------------------
def build_machine(stage, path, rack, vblock, debug_stand=False):
    """Static. rack: {side: (x, y_in, y_out)} for each bar's two wooden rests; vblock: (x, y) of the V-block's centre."""
    lk = looks(stage, path)
    steel, seat = surface(stage, f"{path}/Physics/steel", 0.5), surface(stage, f"{path}/Physics/seat", 0.5)
    wood = surface(stage, f"{path}/Physics/wood", 0.6)
    if debug_stand:  # debugging only: frictionless rests under both handles, level with the rails, standing in for the hands
        ice = surface(stage, f"{path}/Physics/ice", 0.0, combine="min")
        for s, sg in (("l", 1), ("r", -1)):
            collider(stage, f"{path}/stand_{s}_c", (0.20, sg * 0.235, RAIL_TOP / 2), (0.16, 0.05, RAIL_TOP), ice)
    # base plate and slide rails
    collider(stage, f"{path}/plate_c", (0.205, 0.0, PLATE_Z / 2), (0.25, 0.20, PLATE_Z), steel)
    vbox(stage, f"{path}/plate", (0.08, -0.10, 0.0), (0.33, 0.10, PLATE_Z), lk["frame"])
    for s, sg in (("l", 1), ("r", -1)):
        y0, y1 = RAIL_Y
        collider(stage, f"{path}/rail_{s}_c", ((RAIL_X[0] + RAIL_X[1]) / 2, sg * (y0 + y1) / 2, (PLATE_Z + RAIL_TOP) / 2),
                 (RAIL_X[1] - RAIL_X[0], y1 - y0, RAIL_TOP - PLATE_Z), steel)
        vbox(stage, f"{path}/rail_{s}", (RAIL_X[0], sg * y0 if sg > 0 else -y1, PLATE_Z), (RAIL_X[1], y1 if sg > 0 else -y0, RAIL_TOP), lk["paint_grey"])
        vbox(stage, f"{path}/rail_{s}_slot", (RAIL_X[0] + 0.015, sg * 0.0705 - 0.003, RAIL_TOP - 0.0005), (RAIL_X[1] - 0.015, sg * 0.0705 + 0.003, RAIL_TOP + 0.0002), lk["dark"])
        for i, bx in enumerate((RAIL_X[0] + 0.012, RAIL_X[1] - 0.012)):
            vcyl(stage, f"{path}/rail_{s}_bolt{i}", (bx, sg * 0.0725, RAIL_TOP + 0.003), 0.0045, 0.006, "Z", lk["steel"])
    # gearmotor: a ribbed cast motor body behind the bearing housing
    collider(stage, f"{path}/motor_c", (0.14, 0.0, ZS), (0.10, 0.10, 0.10), steel)
    lathe(stage, f"{path}/motor", [(0.0, -0.05), (0.046, -0.05), (0.05, -0.044)] + [(r, -0.04 + 0.006 * i) for i in range(14) for r in ((0.05,) if i % 2 else (0.046,))]
          + [(0.05, 0.044), (0.048, 0.05), (0.0, 0.05)], lk["paint_blue"], centre=(0.14, 0.0, ZS))
    vbox(stage, f"{path}/motor_tbox", (0.115, -0.03, ZS + 0.045), (0.165, 0.03, ZS + 0.07), lk["paint_blue"])
    vbox(stage, f"{path}/motor_tlid", (0.113, -0.032, ZS + 0.068), (0.167, 0.032, ZS + 0.073), lk["paint_grey"])
    for s, sg in (("l", 1), ("r", -1)):
        vbox(stage, f"{path}/motor_foot_{s}", (0.10, sg * 0.04 if sg > 0 else -0.085, RAIL_TOP), (0.18, 0.085 if sg > 0 else -0.04, ZS - 0.03), lk["paint_blue"])
    # bearing housing (the gearbox's output end): a square cast block with a rounded top, and a bearing cap proud of its face
    # its front corners carry a 45-degree bevel (BEVEL a side) that leads a bar's end onto the face; the bevel's inner edge is the fulcrum
    zc, hz = (PLATE_Z + ZS + W) / 2, ZS + W - PLATE_Z
    collider(stage, f"{path}/housing_c", ((0.19 + XH) / 2, 0.0, zc), (XH - 0.19, 2 * (W - BEVEL), hz), steel)
    collider(stage, f"{path}/housing_back_c", ((0.19 + XH - BEVEL) / 2, 0.0, zc), (XH - BEVEL - 0.19, 2 * W, hz), steel)
    for s, sg in (("l", 1), ("r", -1)):
        c = BEVEL * math.sqrt(2)  # a square on its corner: one edge is the bevel
        collider(stage, f"{path}/bevel_{s}_c", (XH - BEVEL, sg * (W - BEVEL), zc), (c, c, hz), steel, rot_z=math.pi / 4)
    vbox(stage, f"{path}/housing", (0.19, -W, PLATE_Z), (XH, W, ZS), lk["paint_blue"])
    lathe(stage, f"{path}/housing_top", [(0.0, 0.0), (W, 0.0), (W, XH - 0.19), (0.0, XH - 0.19)], lk["paint_blue"], centre=(0.19, 0.0, ZS))
    vbox(stage, f"{path}/housing_face", (XH - 0.002, -W + 0.004, PLATE_Z + 0.004), (XH + 0.0005, W - 0.004, ZS), lk["machined"])
    for i, (by, bz) in enumerate(((0.052, ZS + 0.030), (-0.052, ZS + 0.030), (0.052, ZS - 0.045), (-0.052, ZS - 0.045))):
        vcyl(stage, f"{path}/housing_bolt{i}", (XH + 0.003, by, bz), 0.005, 0.006, "X", lk["steel"])
    collider(stage, f"{path}/boss_c", (XH + BOSS_L / 2, 0.0, ZS), (BOSS_L, 2 * BOSS, 2 * BOSS), steel)
    lathe(stage, f"{path}/boss", [(0.0, 0.0), (BOSS + 0.004, 0.0), (BOSS + 0.004, BOSS_L - 0.002), (BOSS + 0.002, BOSS_L), (0.0, BOSS_L)], lk["machined"], centre=(XH, 0.0, ZS))
    # the shaft: one square collider 19 mm across flats (its square fit in the bore stands in for the key), round to look at
    L = SHAFT_END - XH
    # the seat (fretted, friction 0.5: under the bore's land and the 8 mm band it is seized over) and the clean, oiled shaft end beyond it (0.1)
    x1 = XH + GAP + PULLEY["land"] + SEIZED
    collider(stage, f"{path}/shaft_c", ((XH + x1) / 2, 0.0, ZS), (x1 - XH, 2 * SHAFT_R, 2 * SHAFT_R), seat, offset=0.001)
    oiled = surface(stage, f"{path}/Physics/oiled", 0.1, combine="min")
    collider(stage, f"{path}/shaft_end_c", ((x1 + SHAFT_END) / 2, 0.0, ZS), (SHAFT_END - x1, 2 * SHAFT_R, 2 * SHAFT_R), oiled, offset=0.001)
    lathe(stage, f"{path}/shaft", [(0.0, 0.0), (SHAFT_R, 0.0), (SHAFT_R, L - 0.0015), (SHAFT_R - 0.0015, L), (0.0, L)], lk["machined"], centre=(XH, 0.0, ZS))
    vbox(stage, f"{path}/key", (XH + GAP + 0.004, -0.0025, ZS + SHAFT_R - 0.001), (SHAFT_END - 0.006, 0.0025, ZS + SHAFT_R + 0.0002), lk["dark"])
    vcyl(stage, f"{path}/centre", (SHAFT_END + 0.0002, 0.0, ZS), 0.002, 0.0004, "X", lk["dark"])
    # wooden bar rests, one pair per bar
    for s, (x, y_in, y_out) in rack.items():
        for i, y in enumerate((y_in, y_out)):
            collider(stage, f"{path}/rest_{s}{i}_c", (x, y, 0.02), (0.05, 0.03, 0.04), wood)
            vbox(stage, f"{path}/rest_{s}{i}", (x - 0.025, y - 0.015, 0.0), (x + 0.025, y + 0.015, 0.04), lk["wood"])
            vbox(stage, f"{path}/rest_{s}{i}_v", (x - 0.008, y - 0.0151, 0.034), (x + 0.008, y + 0.0151, 0.0405), lk["dark"])
    # V-block for the pulley: two hardwood cheeks at 45 degrees on a base, the V along x
    vx, vy = vblock
    collider(stage, f"{path}/vb_base_c", (vx, vy, 0.01), (0.07, 0.11, 0.02), wood)
    vbox(stage, f"{path}/vb_base", (vx - 0.035, vy - 0.055, 0.0), (vx + 0.035, vy + 0.055, 0.02), lk["wood"])
    for s, sg in (("a", 1), ("b", -1)):
        c = (vx, vy + sg * 0.030, 0.042)
        collider(stage, f"{path}/vb_{s}_c", c, (0.07, 0.05, 0.02), wood, rot_x=sg * math.pi / 4)
        vbox(stage, f"{path}/vb_{s}", np.subtract(c, (0.035, 0.025, 0.01)), np.add(c, (0.035, 0.025, 0.01)), lk["wood"], rot=((1.0, 0.0, 0.0), sg * math.pi / 4))
        vbox(stage, f"{path}/vb_{s}_post", (vx - 0.035, vy + sg * 0.047 - 0.008, 0.02), (vx + 0.035, vy + sg * 0.047 + 0.008, 0.045), lk["wood"])
    return stage.GetPrimAtPath(path)


# V-block: the cheeks' inner faces are 45-degree planes; where a 45 mm rim rests (centre height above the table)
def vblock_rest_z():
    # cheek 'a' centre (y=+0.030, z=0.042), 20 mm thick, inner face normal (0, -1, 1)/sqrt2 pointing into the V
    face_d = (-0.030 + 0.042) / math.sqrt(2) + 0.010  # the face plane: (-y + z)/sqrt2 = face_d
    return math.sqrt(2) * (face_d + PULLEY["r_o"])  # at y = 0: z/sqrt2 = face_d + r


# -- the pulley -------------------------------------------------------------------------------------------------------------------------
def build_pulley(stage, path, seat_world, drag, damping):
    """A cast-iron two-groove V-belt pulley, its own frame on the axis at the back face. The bore's back 25 mm is the fitted land
    (0.3 mm a side); the front 25 mm is relieved. `seat`: a maximal joint to the world whose only effect is a capped axial drag
    (the seized fit): every other direction is free and held by the real contact of the bore on the shaft."""
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    P = PULLEY
    prim = stage.GetPrimAtPath(path)
    rigid(prim, P["mass"])
    lk = looks(stage, path, ("cast_iron", "machined", "stripe"))
    bore, body = surface(stage, f"{path}/Physics/bore", 0.5), surface(stage, f"{path}/Physics/body", 0.3)
    ring(stage, f"{path}/land", 4, P["bore"], P["r_hub"], 0.0, P["land"], bore, offset=0.001)
    ring(stage, f"{path}/relief", 4, P["relief"], P["r_hub"], P["land"], P["hub_l"], bore, offset=0.001)
    ring(stage, f"{path}/web", 12, P["r_hub"] - 0.002, P["r_o"], 0.0, P["web"], body)
    ring(stage, f"{path}/rim", 12, P["r_rim"], P["r_o"], P["web"], P["rim_w"], body)
    # look: machined grooves and faces, cast web and hub
    g = [(P["r_o"], 0.0), (P["r_o"], 0.0025)]
    for u0 in (0.0025, 0.0135):
        g += [(P["r_o"] - 0.008, u0 + 0.0042), (P["r_o"] - 0.008, u0 + 0.0058), (P["r_o"], u0 + 0.010)]
    g += [(P["r_o"], P["rim_w"])]
    lathe(stage, f"{path}/rim_look", [(P["r_rim"] + 0.002, 0.0)] + g + [(P["r_rim"], P["rim_w"]), (P["r_rim"], P["web"] + 0.003)], lk["machined"])
    lathe(stage, f"{path}/body_look", [(P["bore"], -0.0001), (P["r_rim"] + 0.0021, -0.0001), (P["r_rim"], P["web"] + 0.003), (P["r_rim"] - 0.003, P["web"]),
                                       (P["r_hub"] + 0.003, P["web"]), (P["r_hub"], P["web"] + 0.003), (P["r_hub"], P["hub_l"] - 0.0015),
                                       (P["r_hub"] - 0.0015, P["hub_l"]), (P["bore"] + 0.001, P["hub_l"]), (P["bore"], P["hub_l"] - 0.001), (P["bore"], 0.0)], lk["cast_iron"])
    vbox(stage, f"{path}/keyway", (0.0, -0.0025, P["bore"] - 0.0005), (P["hub_l"] + 0.0002, 0.0025, P["bore"] + 0.0025), lk["cast_iron"])
    # a white paint stripe across the rim top and down the hub front, for the camera
    vbox(stage, f"{path}/stripe", (0.0005, -0.006, P["r_o"] - 0.0004), (P["rim_w"] - 0.0005, 0.006, P["r_o"] + 0.0006), lk["stripe"])
    vbox(stage, f"{path}/stripe_hub", (P["web"] + 0.003, -0.004, P["r_hub"] - 0.0004), (P["hub_l"] - 0.002, 0.004, P["r_hub"] + 0.0006), lk["stripe"])
    vcyl(stage, f"{path}/grub", (P["hub_l"] - 0.012, 0.0, P["r_hub"] + 0.0004), 0.003, 0.001, "Z", lk["machined"])
    j = UsdPhysics.Joint.Define(stage, f"{path}/seat")
    j.CreateBody1Rel().SetTargets([prim.GetPath()])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, seat_world)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateExcludeFromArticulationAttr(True)
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "transX")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(float(damping))
    d.CreateTargetVelocityAttr(0.0)
    d.CreateMaxForceAttr(float(drag))
    return prim, j.GetPrim()


# -- a pry bar ---------------------------------------------------------------------------------------------------------------------------
def build_bar(stage, path, side, grip_look):
    """A 10 x 12 mm forged steel pry bar, 242 mm long, its tip end bent 4 mm toward +x, a rubber sleeve on the handle.
    side +1 runs along +y from the tip, -1 along -y."""
    B, sg = BAR, side
    prim = stage.GetPrimAtPath(path)
    rigid(prim, B["mass"])
    lk = looks(stage, path, ("forged", "machined", grip_look))
    phys, grip = surface(stage, f"{path}/Physics/steel", 0.4), surface(stage, f"{path}/Physics/grip", 0.9)
    L, t, h = B["len"], B["t"], B["h"]
    collider(stage, f"{path}/shank_c", (0.0, sg * L / 2, 0.0), (t, L, h), phys, offset=0.001)
    collider(stage, f"{path}/tip_c", (t / 2 + B["tip"] / 2, sg * B["tip_w"] / 2, 0.0), (B["tip"], B["tip_w"], h), phys, offset=0.001)
    g0, g1 = B["grip"]
    collider(stage, f"{path}/grip_c", (0.0, sg * (g0 + g1) / 2, 0.0), (t + 0.006, g1 - g0, h + 0.004), grip, offset=0.001)
    y = lambda a, b: (min(sg * a, sg * b), max(sg * a, sg * b))  # noqa: E731
    ya, yb = y(0.012, g0)
    vbox(stage, f"{path}/shank", (-t / 2, ya, -h / 2), (t / 2, yb, h / 2), lk["forged"])
    ya, yb = y(0.0, 0.014)
    vbox(stage, f"{path}/neck", (-t / 2, ya, -h / 2), (t / 2, yb, h / 2), lk["forged"], rot=((0.0, 0.0, 1.0), -sg * 0.18))
    ya, yb = y(0.0, B["tip_w"])
    vbox(stage, f"{path}/tip", (t / 2 - 0.001, ya, -h / 2), (t / 2 + B["tip"], yb, h / 2), lk["machined"])
    ya, yb = y(g0, g1)
    vbox(stage, f"{path}/grip", (-t / 2 - 0.003, ya, -h / 2 - 0.002), (t / 2 + 0.003, yb, h / 2 + 0.002), lk[grip_look])
    for i, gy in enumerate(np.linspace(g0 + 0.01, g1 - 0.01, 5)):
        vbox(stage, f"{path}/grip_rib{i}", (-t / 2 - 0.0035, sg * gy - 0.0015, -h / 2 - 0.0025), (t / 2 + 0.0035, sg * gy + 0.0015, h / 2 + 0.0025), lk[grip_look])
    return prim
