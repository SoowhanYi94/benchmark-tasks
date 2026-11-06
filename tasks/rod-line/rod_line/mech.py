"""Builds the rod line into a USD stage. A guide frame (fixed) holds two bars end to end along y, each on a spring-return
slide: the drive bar, with a handle pad at its -y end, and the driven bar, with the gate at its +y end. They meet at the
parted joint, one of four joints that all look alike at rest, because each joint is drawn as a collar split between what is
on either side. Sound joints are inside one rigid bar; only the parted one can open, showing a red strip fixed underneath.
Every joint has a stud either side, so the loose splice plate can be seated on any of them, but it only drives across the
parted one. All dimensions in m, from the line's centre on the table."""

import numpy as np

JOINTS = (-0.12, -0.04, 0.04, 0.12)  # y of the four joints
BAR_W, BAR_Z0, BAR_Z1 = 0.024, 0.03, 0.04  # width, underside and top
LINE_LOW, LINE_HIGH = -0.205, 0.205  # the bars' far ends
STUD, STUD_H, STUD_OFF = 0.008, 0.012, 0.03  # 8 mm square, 12 mm tall, 3 cm either side of each joint
COLLAR_L, COLLAR_W, COLLAR_H = 0.014, 0.034, 0.046  # the split collar drawn at each joint, half on either side
PAD = dict(y=-0.22, half_x=0.03, half_y=0.01, z1=0.09)  # the handle pad on the drive bar
GATE = dict(y=0.225, half_x=0.04, half_y=0.015, z1=0.09)  # the gate on the driven bar; it opens as it moves -y
STROKE = 0.04  # each bar slides this far toward -y
SPRING_N_PER_M, SPRING_PRELOAD = 25.0, 0.02  # about 0.5 N at home, 1.5 N at the end of the stroke, each bar
MASS = {"anchor": 1.0, "drive": 0.1, "driven": 0.1}
MU_LINE = 0.3

# the splice plate, in its own frame (origin at the centre of its underside, y along it): two slots over the studs
PLATE_L, PLATE_W, PLATE_T = 0.09, 0.048, 0.006
SLOT_HALF_X, SLOT_Y0, SLOT_Y1 = 0.012, 0.02, 0.04  # 2.4 cm wide, 2 cm long, either side of its middle
PLATE_GRIP = dict(half=0.012, z1=0.04)
PLATE_BOXES = {
    "rail_far": ((SLOT_HALF_X, -PLATE_L / 2, 0.0), (PLATE_W / 2, PLATE_L / 2, PLATE_T)),
    "rail_near": ((-PLATE_W / 2, -PLATE_L / 2, 0.0), (-SLOT_HALF_X, PLATE_L / 2, PLATE_T)),
    "web": ((-SLOT_HALF_X, -SLOT_Y0, 0.0), (SLOT_HALF_X, SLOT_Y0, PLATE_T)),
    "end_low": ((-SLOT_HALF_X, -PLATE_L / 2, 0.0), (SLOT_HALF_X, -SLOT_Y1, PLATE_T)),
    "end_high": ((-SLOT_HALF_X, SLOT_Y1, 0.0), (SLOT_HALF_X, PLATE_L / 2, PLATE_T)),
    "grip": ((-PLATE_GRIP["half"], -PLATE_GRIP["half"], PLATE_T), (PLATE_GRIP["half"], PLATE_GRIP["half"], PLATE_GRIP["z1"])),
}
PLATE_MASS = 0.06
ITERS, VEL_ITERS, MAX_DEPEN = 16, 4, 1.0
COLORS = {"frame": (0.16, 0.17, 0.19), "bar": (0.72, 0.74, 0.77), "red": (0.90, 0.08, 0.06), "pad": (0.95, 0.55, 0.05),
          "gate": (0.10, 0.45, 0.85), "stud": (0.30, 0.32, 0.35), "plate": (0.85, 0.75, 0.15)}


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
        pm.CreateStaticFrictionAttr(MU_LINE)
        pm.CreateDynamicFrictionAttr(MU_LINE)

    def rigid(self, prim, kinematic=False, mass=None):
        _, _, _, UsdPhysics, _ = _api()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
        _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS,
                                             "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                             "physxRigidBody:maxDepenetrationVelocity": float(MAX_DEPEN)})
        if mass is not None:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)

    def cube(self, path, lo, hi, look, collide=True):
        Gf, _, UsdGeom, UsdPhysics, UsdShade = _api()
        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.looks[look])
        if collide:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            _physx(c.GetPrim(), "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.002, "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(self.phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def build_frame(stage, path):
    """The guide frame as one kinematic body at ``path``: a base, a red strip under the bars, two low side rails."""
    _, _, UsdGeom, _, _ = _api()
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, kinematic=True)
    lo, hi = LINE_LOW - STROKE - 0.03, LINE_HIGH + 0.05
    b.cube(f"{path}/base", (-0.03, lo, 0.0), (0.03, hi, BAR_Z0 - 0.006), "frame")
    b.cube(f"{path}/strip", (-BAR_W / 2 + 0.002, lo, BAR_Z0 - 0.006), (BAR_W / 2 - 0.002, hi, BAR_Z0 - 0.001), "red")  # 1 mm under the bars, so no drag
    return prim


def build_mech(stage, path, world_origin, parted):
    """The two bars as an articulation at ``path``: they meet at joint index ``parted``. Spring drives are set in USD."""
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
    seam = JOINTS[parted]
    h = BAR_W / 2
    b.cube(f"{path}/drive/bar", (-h, LINE_LOW, BAR_Z0), (h, seam, BAR_Z1), "bar")
    b.cube(f"{path}/driven/bar", (-h, seam, BAR_Z0), (h, LINE_HIGH, BAR_Z1), "bar")
    b.cube(f"{path}/drive/pad", (-PAD["half_x"], PAD["y"] - PAD["half_y"], BAR_Z0), (PAD["half_x"], PAD["y"] + PAD["half_y"], PAD["z1"]), "pad")
    b.cube(f"{path}/drive/pad_link", (-h, PAD["y"] - PAD["half_y"], BAR_Z0), (h, LINE_LOW, BAR_Z1), "pad", collide=False)
    b.cube(f"{path}/driven/gate", (-GATE["half_x"], GATE["y"] - GATE["half_y"], BAR_Z0), (GATE["half_x"], GATE["y"] + GATE["half_y"], GATE["z1"]), "gate")
    b.cube(f"{path}/driven/gate_link", (-h, LINE_HIGH, BAR_Z0), (h, GATE["y"] - GATE["half_y"], BAR_Z1), "gate", collide=False)
    for i, y in enumerate(JOINTS):
        for side, sign in (("low", -1), ("high", 1)):
            body = "drive" if y + sign * 0.001 < seam else "driven"
            s = y + sign * STUD_OFF
            b.cube(f"{path}/{body}/stud_{i}_{side}", (-STUD / 2, s - STUD / 2, BAR_Z1), (STUD / 2, s + STUD / 2, BAR_Z1 + STUD_H), "stud")
            c0, c1 = sorted((y, y + sign * COLLAR_L / 2))
            b.cube(f"{path}/{body}/collar_{i}_{side}", (-COLLAR_W / 2, c0, BAR_Z0 - 0.004), (COLLAR_W / 2, c1, BAR_Z1 + 0.002), "stud", collide=False)
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    for name, child in (("drive_slide", "drive"), ("driven_slide", "driven")):
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{name}")
        j.CreateAxisAttr("Y")
        j.CreateLowerLimitAttr(-STROKE)
        j.CreateUpperLimitAttr(0.0)
        j.CreateBody0Rel().SetTargets([bodies["anchor"].GetPath()])
        j.CreateBody1Rel().SetTargets([bodies[child].GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateTargetPositionAttr(SPRING_PRELOAD)
        d.CreateStiffnessAttr(SPRING_N_PER_M)
        d.CreateDampingAttr(SPRING_N_PER_M / 10)
    return prim


def build_plate(stage, path):
    """The loose splice plate at ``path``; its prim's own transform places it."""
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, mass=PLATE_MASS)
    for name, (lo, hi) in PLATE_BOXES.items():
        b.cube(f"{path}/{name}", lo, hi, "plate")
    return prim
