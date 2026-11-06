"""Builds the feed chute and its two stops into a USD stage. Every dimension is in the chute's frame: x across, y uphill
along the floor, z normal to the floor, origin on the floor's centre line at the low end. The chute body is fixed; the
stops are one articulation (anchor fixed to the world): the upper stop a pin on a spring-up slide, the lower stop a
see-saw lever on a spring-closed pivot. The parts are single rigid bodies with low friction, standing in for rollers."""

import math

import numpy as np

TILT = math.radians(8.0)
LENGTH = 0.43
INNER_HALF = 0.027  # part 50 wide in a 54 mm chute
WALL = 0.005
COVER_Z, COVER_T = 0.028, 0.006  # the cover's lips, over a 25 mm tall part
SLOT_HALF = 0.015  # 30 mm open slot along the top: the part shows through it, and it can't come out (50 mm wide)
COVER_FROM = 0.07  # where the lips start; below this the chute is open at the top, so the lever can move

MU_SLIDE, MU_KNOB = 0.05, 0.6

# A part: a 50 x 50 x 25 mm block with an 8 mm nose low on its downhill face. Noses touch the block ahead, so there is
# an 8 mm gap between neighbouring blocks for the upper pin to drop into. Origin: centre of the block's underside.
BODY = ((-0.025, -0.025, 0.0), (0.025, 0.025, 0.025))
NOSE = ((-0.010, -0.033, 0.0), (0.010, -0.025, 0.010))
PART = {"body": BODY, "nose": NOSE}
PITCH = 0.058
PART_MASS = 0.15  # about 2.4 g/cm3

STOP_FACE = 0.085  # the lower stop's uphill face: the first block rests against it
TIP_Z = 0.012  # both stops reach down to here when closed: the blocks hit them, the noses (10 mm) pass under
LOWER_T, UPPER_T, STOP_W = 0.006, 0.005, 0.016  # thickness along y, width across

# The lever's pivot is outside the chute, level with the blocks, so the stop's tip lifts almost straight up and doesn't
# push the first block back into the held queue as it opens.
PIVOT = (0.035, 0.020)  # y, z of the lever's pivot
ARM = STOP_FACE - LOWER_T / 2 - PIVOT[0]  # pivot to the stop's centre, 47 mm
KNOB_X = 0.065  # both knobs sit on the far side, clear of the parts' path and not hiding the slot from the camera
KNOB_L = -0.015  # y of the lower stop's knob
KNOB_H = 0.05  # both knobs are 50 x 50 mm pads
KNOB_T = 0.010
KNOB_L_TOP = 0.040
LEVER_MAX = 22.0  # deg: lifts the tip 18 mm, to 30 mm, clear of the 25 mm blocks

UPPER_Y = STOP_FACE + 0.054  # the pin's centre, in the gap between the first and the second block
UPPER_REST_TIP = 0.030  # clear of the blocks' tops at rest
UPPER_TRAVEL = UPPER_REST_TIP - TIP_Z  # 18 mm
KNOB_U_TOP = 0.070

SPRING_N_PER_M, SPRING_PRELOAD = 200.0, 0.01  # the pin: about 2 N to start, 5.6 N at the bottom
LEVER_NM_PER_DEG, LEVER_PRELOAD = 0.008, 5.0  # the lever: about 0.8 N at the knob to start, 4.8 N fully open
MASS = {"anchor": 1.0, "lever": 0.08, "pin": 0.04}
ITERS, VEL_ITERS, MAX_DEPEN = 16, 4, 1.0

COLORS = {"chute": (0.14, 0.15, 0.17), "part": (0.75, 0.77, 0.80), "red_part": (0.90, 0.08, 0.06),
          "lever": (0.95, 0.55, 0.05), "pin": (0.10, 0.45, 0.85), "steel": (0.30, 0.32, 0.35)}


def part_y(i):
    """The chute-frame y of the part at queue position ``i`` (0 is the first, against the lower stop)."""
    return STOP_FACE + 0.025 + i * PITCH


def to_world(local, origin):
    """A chute-frame point in the world, the chute's low end at ``origin``."""
    x, y, z = local
    c, s = math.cos(TILT), math.sin(TILT)
    return np.array((origin[0] + x, origin[1] + y * c - z * s, origin[2] + y * s + z * c))


def to_local(world, origin):
    d = np.asarray(world, float) - np.asarray(origin, float)
    c, s = math.cos(TILT), math.sin(TILT)
    return np.array((d[0], d[1] * c + d[2] * s, -d[1] * s + d[2] * c))


def tilt_xyzw():
    return (math.sin(TILT / 2), 0.0, 0.0, math.cos(TILT / 2))


def _api():
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    return Gf, Sdf, UsdGeom, UsdPhysics, UsdShade


def _physx(prim, schema, **attrs):
    """Apply a PhysX API schema by name and set its attributes (no PhysxSchema bindings here)."""
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


class _Builder:
    def __init__(self, stage, root):
        self.stage, self.root = stage, root
        self.looks = {k: self._look(k, c) for k, c in COLORS.items()}
        self.phys = {"slide": self._phys("slide", MU_SLIDE), "knob": self._phys("knob", MU_KNOB)}

    def _look(self, name, color):
        Gf, Sdf, _, _, UsdShade = _api()
        mat = UsdShade.Material.Define(self.stage, f"{self.root}/Looks/{name}")
        sh = UsdShade.Shader.Define(self.stage, f"{self.root}/Looks/{name}/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _phys(self, name, mu):
        _, _, _, UsdPhysics, UsdShade = _api()
        mat = UsdShade.Material.Define(self.stage, f"{self.root}/Physics/{name}")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(mu)
        pm.CreateDynamicFrictionAttr(mu)
        return mat

    def rigid(self, prim, kinematic=False, mass=None):
        _, _, _, UsdPhysics, _ = _api()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
        _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS,
                                             "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                             "physxRigidBody:maxDepenetrationVelocity": float(MAX_DEPEN)})
        if mass is not None:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)

    def cube(self, path, box, look, collide=True, mat="slide"):
        Gf, _, UsdGeom, UsdPhysics, UsdShade = _api()
        lo, hi = np.array(box[0], float), np.array(box[1], float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.looks[look])
        if collide:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            _physx(c.GetPrim(), "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002,
                                                        "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.phys[mat], UsdShade.Tokens.weakerThanDescendants,
                                                                "physics")
        return c


def build_frame(stage, path):
    """The chute as one kinematic body at ``path``; its prim's own transform places and tilts it."""
    _, _, UsdGeom, _, _ = _api()
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    b.rigid(prim, kinematic=True)
    outer, top = INNER_HALF + WALL, COVER_Z + COVER_T
    b.cube(f"{path}/floor", ((-outer, 0.0, -WALL), (outer, LENGTH, 0.0)), "chute")
    b.cube(f"{path}/back", ((-outer, LENGTH, -WALL), (outer, LENGTH + WALL, top)), "chute")
    for side, s in (("l", 1), ("r", -1)):
        x = sorted((s * INNER_HALF, s * outer))
        b.cube(f"{path}/wall_{side}", ((x[0], 0.0, 0.0), (x[1], LENGTH, top)), "chute")
        lip = sorted((s * SLOT_HALF, s * outer))
        b.cube(f"{path}/lip_{side}", ((lip[0], COVER_FROM, COVER_Z), (lip[1], LENGTH, top)), "chute")
        # shown only: the pin's guide
        b.cube(f"{path}/guide_{side}", ((s * 0.012 - 0.002, UPPER_Y - 0.008, top), (s * 0.012 + 0.002, UPPER_Y + 0.008, top + 0.012)),
               "steel", collide=False)
    return prim


def build_mech(stage, path, world_origin, world_rot):
    """The two stops as an articulation at ``path``; its frame is the chute's. Spring drives are set in USD, where the
    runtime writes didn't reach PhysX in our earlier vise task."""
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    _physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False,
                                            "physxArticulation:solverPositionIterationCount": ITERS,
                                            "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    bodies = {}
    for name in MASS:
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        b.rigid(bodies[name], mass=MASS[name])
    # the lower stop: a see-saw along y. Its knob end is downhill; pressing it lifts the stop out of the blocks' way.
    w, y0, y1, h = STOP_W / 2, STOP_FACE - LOWER_T, STOP_FACE, KNOB_H / 2
    out = INNER_HALF + WALL + 0.001  # the side arm runs just outside the far wall
    b.cube(f"{path}/lever/stop", ((-w, y0, TIP_Z), (w, y1, COVER_Z + COVER_T + 0.012)), "lever")
    b.cube(f"{path}/lever/knob", ((KNOB_X - h, KNOB_L - h, KNOB_L_TOP - KNOB_T), (KNOB_X + h, KNOB_L + h, KNOB_L_TOP)), "lever", mat="knob")
    # shown only: what joins the stop to the knob
    b.cube(f"{path}/lever/cross", ((-w, y0, COVER_Z + COVER_T + 0.012), (out + 0.006, y1, COVER_Z + COVER_T + 0.018)), "lever", collide=False)
    b.cube(f"{path}/lever/riser", ((out, y0, PIVOT[1]), (out + 0.006, y1, COVER_Z + COVER_T + 0.012)), "lever", collide=False)
    b.cube(f"{path}/lever/arm", ((out, KNOB_L, PIVOT[1] - 0.004), (out + 0.006, y1, PIVOT[1] + 0.004)), "lever", collide=False)
    b.cube(f"{path}/lever/link", ((out, KNOB_L - 0.004, PIVOT[1] - 0.004), (KNOB_X, KNOB_L + 0.004, KNOB_L_TOP - KNOB_T)), "lever", collide=False)
    # the upper stop: a pin sliding along z, its spring holding it clear of the blocks; pressing it drops it into the gap
    yl, yh = UPPER_Y - UPPER_T / 2, UPPER_Y + UPPER_T / 2
    b.cube(f"{path}/pin/pin", ((-w, yl, UPPER_REST_TIP), (w, yh, KNOB_U_TOP - KNOB_T)), "pin")
    b.cube(f"{path}/pin/knob", ((KNOB_X - h, UPPER_Y - h, KNOB_U_TOP - KNOB_T), (KNOB_X + h, UPPER_Y + h, KNOB_U_TOP)), "pin", mat="knob")
    b.cube(f"{path}/pin/link", ((w, yl, KNOB_U_TOP - KNOB_T), (KNOB_X, yh, KNOB_U_TOP - KNOB_T + 0.004)), "pin", collide=False)  # shown only
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    fix.CreateLocalRot0Attr(Gf.Quatf(world_rot))
    lever = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/lever_turn")
    lever.CreateAxisAttr("X")
    lever.CreateLowerLimitAttr(0.0)
    lever.CreateUpperLimitAttr(LEVER_MAX)
    pin = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/pin_slide")
    pin.CreateAxisAttr("Z")
    pin.CreateLowerLimitAttr(-UPPER_TRAVEL)
    pin.CreateUpperLimitAttr(0.0)
    for j, child, pt in ((lever, "lever", (0.0, PIVOT[0], PIVOT[1])), (pin, "pin", (0.0, UPPER_Y, UPPER_REST_TIP))):
        j.CreateBody0Rel().SetTargets([bodies["anchor"].GetPath()])
        j.CreateBody1Rel().SetTargets([bodies[child].GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(*pt))
        j.CreateLocalPos1Attr(Gf.Vec3f(*pt))
    d = UsdPhysics.DriveAPI.Apply(lever.GetPrim(), "angular")
    d.CreateTypeAttr("force")
    d.CreateTargetPositionAttr(-LEVER_PRELOAD)
    d.CreateStiffnessAttr(LEVER_NM_PER_DEG)
    d.CreateDampingAttr(LEVER_NM_PER_DEG / 20)
    d = UsdPhysics.DriveAPI.Apply(pin.GetPrim(), "linear")
    d.CreateTypeAttr("force")
    d.CreateTargetPositionAttr(SPRING_PRELOAD)
    d.CreateStiffnessAttr(SPRING_N_PER_M)
    d.CreateDampingAttr(SPRING_N_PER_M / 20)
    return prim


def build_part(stage, path, red):
    """A loose part at ``path``; its prim's own transform places it."""
    _, _, UsdGeom, _, _ = _api()
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    b.rigid(prim, mass=PART_MASS)
    for name, box in PART.items():
        b.cube(f"{path}/{name}", box, "red_part" if red else "part")
    return prim
