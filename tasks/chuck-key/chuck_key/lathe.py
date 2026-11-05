"""The lathe's headstock and 3-jaw chuck, the chuck key, its holder and the bar's V-block, built into a USD stage.

Lathe frame (metres): origin on the table top, under the spindle axis, in the plane of the jaws' front faces;
x across the spindle (world x), y along it into the chuck (world y), z up. The constants below need no Isaac Sim, so
the task file imports them for its layout and scoring.

One articulation holds the mechanism, fixed to the world by its root link (`anchor`: headstock, chuck body, key
holder and V-block). The `pinion` turns about the vertical in a socket on top of the chuck; the three `jaw_k` slide
radially on prismatic joints. Each jaw slide follows the pinion rigidly through a mimic joint (NewtonMimicAPI, which
PhysX reads; JAW_PER_TURN of travel per turn), and carries its jaw on a sprung "give" joint (up to GIVE_MAX at
GIVE_K), so turning the key past first contact clamps the bar with a spring preload. The
pinion has a damping drive capped at BRAKE_TORQUE, which acts as friction: the clamp's reaction can't turn it back.
The key and the bar are free rigid bodies; the key turns the pinion only through contact (its square drive in the
socket's square hole).
"""

import math

import numpy as np

# -- chuck and headstock ----------------------------------------------------------------------------------------------
AXIS_Z = 0.042  # spindle axis above the table
JAW_DEPTH = 0.025  # the jaws stand 25 mm proud of the chuck's face (y 0 .. 0.025)
FACE_Y = 0.0255  # the chuck body's front face
BODY_Y = (FACE_Y, FACE_Y + 0.045)  # chuck body, 45 mm long
CHUCK_R = 0.035  # chuck body radius (70 mm chuck)
BORE_R = 0.021  # through bore (chuck and spindle), 42 mm
HEAD_Y = (BODY_Y[1], BODY_Y[1] + 0.09)  # headstock block behind the chuck
HEAD_X = 0.05  # half-width
HEAD_TOP = 0.085
HEAD_BORE = 0.021  # half-size of the headstock's square bore

JAW_ANGLES = (90.0, 210.0, 330.0)  # degrees round the spindle (x-z plane), from +x toward +z: one on top
R_OPEN = 0.015  # each jaw's gripping face from the axis when fully open (30 mm opening)
JAW_H = 0.018  # radial height of a jaw
JAW_W = 0.020  # width of a jaw (tangential)
S_MAX = 0.0068  # jaw travel from fully open to fully closed (the scroll's end stop; the faces 8.2 mm from the axis)
GIVE_MAX, GIVE_K, GIVE_D = 0.002, 5.0e4, 60.0  # each jaw's give: up to 2 mm, a 50 N/mm spring, damped
JAW_PER_TURN = 0.003  # jaw travel per turn of the key
GEAR = JAW_PER_TURN / (2 * math.pi)  # m per rad
MIMIC_COEF = GEAR * math.pi / 180  # m of jaw travel per degree of pinion (NewtonMimicAPI: follower = coef1 * leader)
TURN_MAX = S_MAX / GEAR  # rad, pinion travel open -> closed

# the key socket: on top of the chuck, 23 mm behind its face; a square hole in the pinion
SOCKET_Y = FACE_Y + 0.023
SOCKET_Z = AXIS_Z + CHUCK_R  # top of the pinion (flush with the top of the chuck)
SOCKET_HALF = 0.0055  # 11 mm square hole (0.5 mm of play round the drive)
SOCKET_DEPTH = 0.016
SOCKET_CHAMFER = 0.003  # 45 degree lead-in round the hole's mouth
SOCKET_WALL = 0.004
PINION_R = 0.0085

# -- chuck key (key frame: origin at the shoulder, on the key's axis; z up; T-bar along x) -----------------------------
DRIVE_HALF = 0.005  # 10 mm square drive
DRIVE_LEN = 0.012
SHAFT_R, SHAFT_LEN = 0.006, 0.040
TBAR_R, TBAR_HALF = 0.005, 0.045  # 10 mm square bar, 90 mm long
TBAR_Z = SHAFT_LEN + TBAR_R  # T-bar axis above the shoulder
KEY_MASS = 0.10

# -- key holder (a steel block with a square hole; position in the lathe frame) ---------------------------------------
HOLDER_XY = (-0.10, 0.14)
HOLDER_HALF = 0.016  # 32 mm square block
HOLDER_HOLE = 0.0075  # 15 mm square hole
HOLDER_TOP = 0.045
HOLDER_CHAMFER = 0.006  # a deep 45-degree lead-in: the key slides a few mm along the T-bar in the hand while it is turned
HOLDER_FLOOR = 0.010
KEY_HOME_Z = HOLDER_FLOOR + DRIVE_LEN  # key frame height when it stands in the holder
KEY_SEATED_Z = SOCKET_Z - 0.0025  # key frame height when it is seated (its shoulder on the lead-in)

# -- the bar and its V-block --------------------------------------------------------------------------------------------
BAR_R, BAR_LEN = 0.010, 0.120  # 20 mm round aluminium bar
BAR_MASS = 2700 * math.pi * BAR_R**2 * BAR_LEN  # aluminium, 0.10 kg
VBLOCK_XY = (-0.09, -0.08)  # centre of the V-block (two rails along y)
VBLOCK_GAP = 0.007  # half-gap between the rails
VBLOCK_RAIL = (0.012, 0.030, 0.035)  # each rail: width (x), length (y), height
BAR_START_Z = VBLOCK_RAIL[2] + math.sqrt(BAR_R**2 - VBLOCK_GAP**2)  # bar axis height on the V-block

# -- physics -----------------------------------------------------------------------------------------------------------
ITERS, VEL_ITERS = 32, 4
BRAKE_DAMPING = 20.0  # N*m*s/rad on the pinion ...
BRAKE_TORQUE = 0.25  # ... capped at this torque: it can't be turned by less, and holds against the clamp
MU_STEEL = 0.5
MASS = {"anchor": 2.0, "pinion": 0.03, "jaw": 0.04}
COLORS = {"cast": (0.30, 0.34, 0.36), "steel": (0.72, 0.72, 0.74), "alu": (0.82, 0.84, 0.87), "jaw": (0.55, 0.56, 0.58),
          "key": (0.12, 0.12, 0.13), "holder": (0.85, 0.55, 0.10), "dark": (0.03, 0.03, 0.03)}


def jaw_dir(k):
    """Unit vector (x, z) from the axis out through jaw k."""
    a = math.radians(JAW_ANGLES[k])
    return np.array((math.cos(a), math.sin(a)))


def jaw_face_r(s):
    """Radius of a jaw's gripping face at travel s (m)."""
    return R_OPEN - s


# -- USD ---------------------------------------------------------------------------------------------------------------


def _physx(prim, schema, **attrs):
    """Apply a PhysX API schema by name and set its attributes (no PhysxSchema bindings needed)."""
    from pxr import Sdf

    try:
        ok = prim.ApplyAPI(*schema.split(":"))
    except Exception:  # noqa: BLE001 (schema not registered by name in this process)
        ok = False
    if not ok:
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float,
             str: Sdf.ValueTypeNames.Token}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


class _Builder:
    def __init__(self, stage, root):
        self.stage, self.root = stage, root
        self.looks = {k: self._look(k, c) for k, c in COLORS.items()}
        self.phys = self._phys("steel", MU_STEEL)

    def _look(self, name, color):
        from pxr import Gf, Sdf, UsdShade

        mat = UsdShade.Material.Define(self.stage, f"{self.root}/Looks/{name}")
        sh = UsdShade.Shader.Define(self.stage, f"{self.root}/Looks/{name}/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.5)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _phys(self, name, mu):
        from pxr import UsdPhysics, UsdShade

        mat = UsdShade.Material.Define(self.stage, f"{self.root}/Physics/{name}")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(mu)
        pm.CreateDynamicFrictionAttr(mu)
        return mat

    def collider(self, prim, approx=None):
        from pxr import UsdPhysics, UsdShade

        UsdPhysics.CollisionAPI.Apply(prim)
        _physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.001, "physxCollision:restOffset": 0.0})
        if approx:
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.phys, UsdShade.Tokens.weakerThanDescendants, "physics")

    def rigid(self, prim, mass):
        from pxr import UsdPhysics

        UsdPhysics.RigidBodyAPI.Apply(prim)
        _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS,
                                             "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                             "physxRigidBody:maxDepenetrationVelocity": 0.2})
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))

    def paint(self, prim, look):
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.looks[look])

    def box(self, path, lo, hi, look, collide=True):
        from pxr import Gf, UsdGeom

        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        self.paint(c.GetPrim(), look)
        if collide:
            self.collider(c.GetPrim())
        return c

    def cyl(self, path, center, radius, height, axis, look, collide=True):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(height)
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        self.paint(c.GetPrim(), look)
        if collide:
            self.collider(c.GetPrim())
        return c

    def prism(self, path, pts_xz, y0, y1, look, along="y"):
        """A convex prism along y (or x) over the polygon pts_xz (counter-clockwise in x-z, or y-z)."""
        from pxr import UsdGeom, Vt

        n = len(pts_xz)
        v = [(x, y0, z) for x, z in pts_xz] + [(x, y1, z) for x, z in pts_xz]
        if along == "x":
            v = [(y, x, z) for x, y, z in v]
        faces = [list(range(n))[::-1], list(range(n, 2 * n))]
        faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray([tuple(map(float, p)) for p in v]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
        m.CreateSubdivisionSchemeAttr("none")
        self.paint(m.GetPrim(), look)
        self.collider(m.GetPrim(), "convexHull")
        return m


def _xform(stage, path, pos=(0.0, 0.0, 0.0), rot_y=None):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if rot_y is not None:
        x.AddOrientOp().Set(Gf.Quatf(math.cos(rot_y / 2), Gf.Vec3f(0.0, math.sin(rot_y / 2), 0.0)))
    return x.GetPrim()


def _sectors(b, path, r_in, r_out, y0, y1, n=24):
    """The chuck body: an annulus round the bore as n convex sectors; the ones under the key socket leave room for
    the pinion."""
    k = 0
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        pts = [(r * math.cos(a), AXIS_Z + r * math.sin(a)) for r, a in ((r_in, a0), (r_out, a0), (r_out, a1), (r_in, a1))]
        xs = [r_out * math.cos(a) for a in (a0, a1)]
        under_socket = math.sin((a0 + a1) / 2) > 0 and min(xs) < PINION_R + 0.001 and max(xs) > -PINION_R - 0.001
        spans = [(y0, y1)]
        if under_socket:
            spans = [(y0, SOCKET_Y - PINION_R - 0.0005), (SOCKET_Y + PINION_R + 0.0005, y1)]
        for s0, s1 in spans:
            b.prism(f"{path}/sector_{k}", pts, s0, s1, "cast")
            k += 1


def _quat_y(angle):
    from pxr import Gf

    return Gf.Quatf(math.cos(angle / 2), Gf.Vec3f(0.0, math.sin(angle / 2), 0.0))


def build_lathe(stage, path, world_origin, brake=(BRAKE_DAMPING, BRAKE_TORQUE)):
    """The lathe as one articulation at ``path`` (its prim's transform places the lathe frame at
    ``world_origin``, unrotated)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    _physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False,
                                            "physxArticulation:solverPositionIterationCount": ITERS,
                                            "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    anchor = _xform(stage, f"{path}/anchor")
    b.rigid(anchor, MASS["anchor"])
    a = f"{path}/anchor"
    # headstock: a block round a square bore
    hy0, hy1 = HEAD_Y
    zb0, zb1 = AXIS_Z - HEAD_BORE, AXIS_Z + HEAD_BORE
    b.box(f"{a}/head_low", (-HEAD_X, hy0, 0.0), (HEAD_X, hy1, zb0), "cast")
    b.box(f"{a}/head_high", (-HEAD_X, hy0, zb1), (HEAD_X, hy1, HEAD_TOP), "cast")
    b.box(f"{a}/head_l", (-HEAD_X, hy0, zb0), (-HEAD_BORE, hy1, zb1), "cast")
    b.box(f"{a}/head_r", (HEAD_BORE, hy0, zb0), (HEAD_X, hy1, zb1), "cast")
    _sectors(b, f"{a}/chuck", BORE_R, CHUCK_R, *BODY_Y)
    # key holder
    hx, hy = HOLDER_XY
    H, h = HOLDER_HALF, HOLDER_HOLE
    b.box(f"{a}/holder_floor", (hx - H, hy - H, 0.0), (hx + H, hy + H, HOLDER_FLOOR), "holder")
    T, c = HOLDER_TOP, HOLDER_CHAMFER
    b.box(f"{a}/holder_px", (hx + h, hy - H, HOLDER_FLOOR), (hx + H, hy + H, T - c), "holder")
    b.box(f"{a}/holder_nx", (hx - H, hy - H, HOLDER_FLOOR), (hx - h, hy + H, T - c), "holder")
    b.box(f"{a}/holder_py", (hx - h, hy + h, HOLDER_FLOOR), (hx + h, hy + H, T - c), "holder")
    b.box(f"{a}/holder_ny", (hx - h, hy - H, HOLDER_FLOOR), (hx + h, hy - h, T - c), "holder")
    for name, ctr, lo, hi, along in (("x", hx, hy - H, hy + H, "y"), ("y", hy, hx - H, hx + H, "x")):
        lead = [(h, T - c), (H, T - c), (H, T), (h + c, T)]
        b.prism(f"{a}/holder_lead_p{name}", [(ctr + u, z) for u, z in lead], lo, hi, "holder", along=along)
        b.prism(f"{a}/holder_lead_n{name}", [(ctr - u, z) for u, z in reversed(lead)], lo, hi, "holder", along=along)
    # V-block for the bar: two rails along y
    vx, vy = VBLOCK_XY
    w, ln, ht = VBLOCK_RAIL
    for s in (1, -1):
        x0 = vx + s * VBLOCK_GAP
        b.box(f"{a}/vblock_{'l' if s > 0 else 'r'}", (min(x0, x0 + s * w), vy - ln / 2, 0.0),
              (max(x0, x0 + s * w), vy + ln / 2, ht), "steel")

    # pinion with its square socket (pinion frame: origin at the socket's top centre)
    pin = _xform(stage, f"{path}/pinion", (0.0, SOCKET_Y, SOCKET_Z))
    b.rigid(pin, MASS["pinion"])
    p = f"{path}/pinion"
    S, W, D = SOCKET_HALF, SOCKET_WALL, SOCKET_DEPTH
    c = SOCKET_CHAMFER
    b.box(f"{p}/wall_px", (S, -S - W, -D), (S + W, S + W, -c), "steel")
    b.box(f"{p}/wall_nx", (-S - W, -S - W, -D), (-S, S + W, -c), "steel")
    b.box(f"{p}/wall_py", (-S, S, -D), (S, S + W, -c), "steel")
    b.box(f"{p}/wall_ny", (-S, -S - W, -D), (S, -S, -c), "steel")
    lead = [(S, -c), (S + W, -c), (S + W, 0.0), (S + c, 0.0)]  # the socket's 45 degree lead-in
    mirror = [(-u, z) for u, z in reversed(lead)]
    b.prism(f"{p}/lead_px", lead, -S - W, S + W, "steel")
    b.prism(f"{p}/lead_nx", mirror, -S - W, S + W, "steel")
    b.prism(f"{p}/lead_py", lead, -S - W, S + W, "steel", along="x")
    b.prism(f"{p}/lead_ny", mirror, -S - W, S + W, "steel", along="x")
    b.box(f"{p}/floor", (-S - W, -S - W, -D - 0.004), (S + W, S + W, -D), "steel")
    b.cyl(f"{p}/boss", (0.0, 0.0, -0.011), PINION_R, 0.0218, "Z", "steel", collide=False)
    b.box(f"{p}/hole", (-S, -S, -0.0005), (S, S, 0.0003), "dark", collide=False)  # the hole, seen from above
    rev = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/pinion_turn")
    rev.CreateAxisAttr("Z")
    rev.CreateBody0Rel().SetTargets([anchor.GetPath()])
    rev.CreateBody1Rel().SetTargets([pin.GetPath()])
    rev.CreateLocalPos0Attr(Gf.Vec3f(0.0, SOCKET_Y, SOCKET_Z))
    rev.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    rev.CreateLowerLimitAttr(0.0)
    rev.CreateUpperLimitAttr(math.degrees(TURN_MAX))
    if brake is not None:
        drv = UsdPhysics.DriveAPI.Apply(rev.GetPrim(), "angular")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(0.0)
        drv.CreateDampingAttr(float(brake[0]) * math.pi / 180)  # USD angular drives are per degree
        drv.CreateMaxForceAttr(float(brake[1]))
        drv.CreateTargetVelocityAttr(0.0)
    _physx(rev.GetPrim(), "PhysxJointAPI", **{"physxJoint:armature": 1e-4})

    # jaws: each body's frame sits on the axis in the jaws' front plane, its +x pointing in along the jaw's slide
    for k, ang in enumerate(JAW_ANGLES):
        rot = math.pi - math.radians(ang)
        jaw = _xform(stage, f"{path}/jaw_{k}", (0.0, 0.0, AXIS_Z), rot_y=rot)
        b.rigid(jaw, MASS["jaw"])
        pad = _xform(stage, f"{path}/pad_{k}", (0.0, 0.0, AXIS_Z), rot_y=rot)
        b.rigid(pad, MASS["jaw"])
        b.box(f"{path}/pad_{k}/block", (-(R_OPEN + JAW_H), 0.0005, -JAW_W / 2), (-R_OPEN, JAW_DEPTH, JAW_W / 2), "jaw")
        give = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/jaw_{k}_give")
        give.CreateAxisAttr("X")
        give.CreateBody0Rel().SetTargets([jaw.GetPath()])
        give.CreateBody1Rel().SetTargets([pad.GetPath()])
        give.CreateLowerLimitAttr(-GIVE_MAX)
        give.CreateUpperLimitAttr(0.0002)
        drv = UsdPhysics.DriveAPI.Apply(give.GetPrim(), "linear")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(float(GIVE_K))
        drv.CreateDampingAttr(float(GIVE_D))
        drv.CreateTargetPositionAttr(0.0)
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/jaw_{k}_slide")
        j.CreateAxisAttr("X")
        j.CreateBody0Rel().SetTargets([anchor.GetPath()])
        j.CreateBody1Rel().SetTargets([jaw.GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, AXIS_Z))
        j.CreateLocalRot0Attr(_quat_y(rot))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLowerLimitAttr(-0.0005)
        j.CreateUpperLimitAttr(S_MAX + 0.0005)
        jp = j.GetPrim()
        _physx(jp, "NewtonMimicAPI", **{"newton:mimicCoef0": 0.0, "newton:mimicCoef1": float(MIMIC_COEF)})
        jp.CreateRelationship("newton:mimicJoint").SetTargets([rev.GetPath()])

    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([anchor.GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    return prim


def build_key(stage, path):
    """The chuck key as one free rigid body at ``path`` (key frame: origin at the shoulder)."""
    from pxr import UsdGeom

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    b.rigid(prim, KEY_MASS)
    d = DRIVE_HALF
    b.box(f"{path}/drive", (-d, -d, -DRIVE_LEN), (d, d, 0.0), "key")
    b.cyl(f"{path}/shaft", (0.0, 0.0, SHAFT_LEN / 2), SHAFT_R, SHAFT_LEN, "Z", "key")
    b.box(f"{path}/tbar", (-TBAR_HALF, -TBAR_R, TBAR_Z - TBAR_R), (TBAR_HALF, TBAR_R, TBAR_Z + TBAR_R), "key")
    return prim


def build_bar(stage, path):
    """The round bar as one free rigid body at ``path`` (frame at its centre, axis along y)."""
    from pxr import UsdGeom

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    b.rigid(prim, BAR_MASS)
    n = 32  # a 32-sided convex prism: PhysX's own cylinder approximation is coarser
    pts = [(BAR_R * math.cos(2 * math.pi * (i + 0.5) / n) / math.cos(math.pi / n),
            BAR_R * math.sin(2 * math.pi * (i + 0.5) / n) / math.cos(math.pi / n)) for i in range(n)]
    b.prism(f"{path}/rod", pts, -BAR_LEN / 2, BAR_LEN / 2, "alu")
    return prim
