"""Builds the carrier rail and its dogs into a USD stage. A fixed rail runs along y with three dog stations, each with twin
studs. A carrier slides on it toward +y, pushed by a rod from a handle at its -y end, and has a spring pin on top. A dog is a
capped block on the studs with a ramp and a tooth hidden under the cap: facing forward, the ramp presses the pin down as the
carrier passes and the tooth then stops it running back; turned round, the tooth stops the carrier dead. All dimensions in m."""

import math

import numpy as np

STATIONS = (-0.07, 0.0, 0.07)  # y of the three dogs
START, STOP = -0.12, 0.11  # the carrier's y at home, and at its end stop
TRAVEL = STOP - START
RAIL_LOW, RAIL_HIGH, RAIL_HALF_X, RAIL_Z = -0.37, STOP + 0.04, 0.05, 0.02
STUD, STUD_X, STUD_TOP = 0.008, 0.03, 0.05  # twin studs, either side of the rail's centre
CARRIER = dict(half_x=0.012, half_y=0.02, z0=RAIL_Z + 0.001, z1=0.04)
ROD = dict(half_x=0.008, z1=0.035)
HANDLE = dict(y=START - TRAVEL, half_x=0.03, half_y=0.008, z1=0.09)
PIN_HALF, PIN_TOP, PIN_DROP = 0.003, 0.056, 0.012
PIN_BOTTOM = CARRIER["z1"] - 0.002  # inside the carrier (no self-collision): pressed fully down it must still clear the rail
SPRING_PIN = 25.0  # N/m, preloaded about 0.6 N at home, under a dog's weight
MASS = {"anchor": 1.0, "carrier": 0.3, "pin": 0.02}

# a dog, in its own frame (origin at the centre of its underside, on the rail's top): 9.2 x 5 cm, a cap with a grip on top
DOG_HALF_X, DOG_HALF_Y = 0.046, 0.025
CAP_Z0, CAP_Z1 = 0.04 + 0.02 - RAIL_Z, 0.07 - RAIL_Z + 0.02  # the cap's underside clears the pin
SLOT_HALF_Y, LEG_INNER = 0.014, 0.02  # slots over the studs: stud 8 mm, 2.8 cm long, open toward the rail
BACK_INNER = 0.039  # 5 mm of sideways play on the studs: under the carrier's 8 mm clearance to the legs, even at the 5 degrees the scorer allows
GRIP = dict(half=0.012, z1=CAP_Z1 + 0.034)
TIP_Z = 0.048 - RAIL_Z  # the ramp's lowest edge, under the pin's top
RAMP_Y0, RAMP_Y1, TOOTH_Y = -0.02, 0.012, 0.018
DOG_MASS = 0.15


def dog_boxes():
    """Everything except the ramp, as (lo, hi) boxes; symmetric but for the ramp and tooth, which the cap hides from above."""
    b = {"cap": ((-DOG_HALF_X, -DOG_HALF_Y, CAP_Z0), (DOG_HALF_X, DOG_HALF_Y, CAP_Z1)),
         "grip": ((-GRIP["half"], -GRIP["half"], CAP_Z1), (GRIP["half"], GRIP["half"], GRIP["z1"])),
         "tooth": ((-0.012, RAMP_Y1, TIP_Z), (0.012, TOOTH_Y, CAP_Z0))}
    for s, sign in (("r", 1), ("l", -1)):
        for e, ys in (("low", (-DOG_HALF_Y, -SLOT_HALF_Y)), ("high", (SLOT_HALF_Y, DOG_HALF_Y))):
            x0, x1 = sorted((sign * LEG_INNER, sign * DOG_HALF_X))
            b[f"leg_{s}_{e}"] = ((x0, ys[0], 0.0), (x1, ys[1], CAP_Z0))
        x0, x1 = sorted((sign * BACK_INNER, sign * DOG_HALF_X))
        b[f"back_{s}"] = ((x0, -SLOT_HALF_Y, 0.0), (x1, SLOT_HALF_Y, CAP_Z0))
    return b


ITERS, VEL_ITERS, MAX_DEPEN = 16, 4, 1.0
MU = 0.3
COLORS = {"frame": (0.16, 0.17, 0.19), "stud": (0.30, 0.32, 0.35), "stop": (0.90, 0.08, 0.06), "carrier": (0.10, 0.45, 0.85),
          "handle": (0.95, 0.55, 0.05), "pin": (0.85, 0.75, 0.15), "dog": (0.62, 0.64, 0.67)}


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
        Gf, Sdf, _, UsdPhysics, UsdShade = _api()
        self.stage, self.root, self.looks = stage, root, {}
        for name, color in COLORS.items():
            mat = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
            sh = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self.looks[name] = mat
        self.phys = UsdShade.Material.Define(stage, f"{root}/Physics/line")
        pm = UsdPhysics.MaterialAPI.Apply(self.phys.GetPrim())
        pm.CreateStaticFrictionAttr(MU)
        pm.CreateDynamicFrictionAttr(MU)

    def rigid(self, prim, kinematic=False, mass=None):
        _, _, _, UsdPhysics, _ = _api()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
        _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS,
                                             "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                             "physxRigidBody:maxDepenetrationVelocity": float(MAX_DEPEN)})
        if mass is not None:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)

    def _finish(self, c):
        _, _, _, UsdPhysics, UsdShade = _api()
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        _physx(c.GetPrim(), "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002, "physxCollision:restOffset": 0.0})
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.phys, UsdShade.Tokens.weakerThanDescendants, "physics")

    def cube(self, path, lo, hi, look):
        Gf, _, UsdGeom, _, UsdShade = _api()
        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.looks[look])
        self._finish(c)

    def ramp(self, path, look):
        """A plate tilted about x: its underside falls from the cap at RAMP_Y0 to TIP_Z at RAMP_Y1."""
        Gf, _, UsdGeom, _, UsdShade = _api()
        dy, dz = RAMP_Y1 - RAMP_Y0, CAP_Z0 - TIP_Z
        length, angle, t = math.hypot(dy, dz), -math.atan2(dz, dy), 0.006
        # the underside's midpoint, then in by half the thickness along the plate's normal
        cy, cz = (RAMP_Y0 + RAMP_Y1) / 2 - t / 2 * math.sin(angle), (CAP_Z0 + TIP_Z) / 2 + t / 2 * math.cos(angle)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(0.0, cy, cz))
        c.AddOrientOp().Set(Gf.Quatf(math.cos(angle / 2), Gf.Vec3f(math.sin(angle / 2), 0.0, 0.0)))
        c.AddScaleOp().Set(Gf.Vec3f(0.024, float(length), t))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.looks[look])
        self._finish(c)


def build_frame(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, kinematic=True)
    b.cube(f"{path}/base", (-RAIL_HALF_X, RAIL_LOW, 0.0), (RAIL_HALF_X, RAIL_HIGH, RAIL_Z), "frame")
    b.cube(f"{path}/stop", (-0.03, STOP + CARRIER["half_y"], RAIL_Z), (0.03, STOP + CARRIER["half_y"] + 0.012, 0.06), "stop")
    for i, y in enumerate(STATIONS):
        for s, x in (("r", STUD_X), ("l", -STUD_X)):
            b.cube(f"{path}/stud_{i}_{s}", (x - STUD / 2, y - STUD / 2, RAIL_Z), (x + STUD / 2, y + STUD / 2, STUD_TOP), "stud")
    return prim


def build_mech(stage, path, world_origin):
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    _physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False,
                                            "physxArticulation:solverPositionIterationCount": ITERS,
                                            "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    bodies = {}
    for name in MASS:
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        b.rigid(bodies[name], mass=MASS[name])
    c = CARRIER
    b.cube(f"{path}/carrier/body", (-c["half_x"], START - c["half_y"], c["z0"]), (c["half_x"], START + c["half_y"], c["z1"]), "carrier")
    b.cube(f"{path}/carrier/rod", (-ROD["half_x"], HANDLE["y"], c["z0"]), (ROD["half_x"], START - c["half_y"], ROD["z1"]), "handle")
    b.cube(f"{path}/carrier/handle", (-HANDLE["half_x"], HANDLE["y"] - HANDLE["half_y"], c["z0"]),
           (HANDLE["half_x"], HANDLE["y"] + HANDLE["half_y"], HANDLE["z1"]), "handle")
    b.cube(f"{path}/pin/tip", (-PIN_HALF, START - PIN_HALF, PIN_BOTTOM), (PIN_HALF, START + PIN_HALF, PIN_TOP), "pin")
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    for name, parent, child, axis, lo, hi, target, k, damp in (
            ("carrier_slide", "anchor", "carrier", "Y", 0.0, TRAVEL, None, 0.0, 1.5),
            ("pin_slide", "carrier", "pin", "Z", -PIN_DROP, 0.0, 0.025, SPRING_PIN, 1.0)):
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{name}")
        j.CreateAxisAttr(axis)
        j.CreateLowerLimitAttr(lo)
        j.CreateUpperLimitAttr(hi)
        j.CreateBody0Rel().SetTargets([bodies[parent].GetPath()])
        j.CreateBody1Rel().SetTargets([bodies[child].GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateTargetPositionAttr(target or 0.0)
        d.CreateStiffnessAttr(k)
        d.CreateDampingAttr(damp)
    return prim


def build_dog(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, mass=DOG_MASS)
    for name, (lo, hi) in dog_boxes().items():
        b.cube(f"{path}/{name}", lo, hi, "dog")
    b.ramp(f"{path}/ramp", "dog")
    return prim
