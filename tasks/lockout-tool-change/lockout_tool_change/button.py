"""Builds the machine's control box into a USD stage as one articulation: `anchor` (fixed to the world: the box and
the run lamp) and `cap` (the green START button) on a prismatic joint along z, limited to 8 mm of travel, held up by
a linear drive acting as its spring (target 0). The box and the cap overlap without colliding (self-collisions off).

Frame: the placeholder's position (box centre on the table), z up. Sizes come from the task file.
"""

STIFFNESS = 600.0  # N/m: about 2.4 N to press it 4 mm, 4.8 N at the stop
DAMPING = 2.0  # N s/m
CAP_H = 0.016
MASS = {"anchor": 1.0, "cap": 0.02}
COLORS = {"box": (0.25, 0.26, 0.28), "cap": (0.15, 0.60, 0.25), "lamp": (0.12, 0.14, 0.12)}
LAMP_ON = (0.25, 1.0, 0.35)


def _physx(prim, schema, **attrs):
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def _look(stage, path, color):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _shape(prim, look, collide=True):
    from pxr import UsdPhysics, UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(look)
    if collide:
        UsdPhysics.CollisionAPI.Apply(prim)


def build(stage, path, world_origin, world_rot, box_size, button_r, button_proud, travel, lamp_offset, lamp_r):
    """The control box at ``path`` (already placed); ``world_origin`` / ``world_rot`` is that pose in the world."""
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    _physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False})
    looks = {k: _look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}
    bodies = {}
    for name in ("anchor", "cap"):
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(bodies[name])
        UsdPhysics.MassAPI.Apply(bodies[name]).CreateMassAttr(MASS[name])
    bx, by, bz = box_size
    box = UsdGeom.Cube.Define(stage, f"{path}/anchor/box")
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, bz / 2))
    box.AddScaleOp().Set(Gf.Vec3f(bx, by, bz))
    _shape(box.GetPrim(), looks["box"])
    lamp = UsdGeom.Sphere.Define(stage, f"{path}/anchor/lamp")
    lamp.CreateRadiusAttr(lamp_r)
    lamp.AddTranslateOp().Set(Gf.Vec3d(lamp_offset[0], lamp_offset[1], bz))
    _shape(lamp.GetPrim(), looks["lamp"])
    cap = UsdGeom.Cylinder.Define(stage, f"{path}/cap/cap")
    cap.CreateAxisAttr("Z")
    cap.CreateRadiusAttr(button_r)
    cap.CreateHeightAttr(CAP_H)
    cap.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, bz + button_proud - CAP_H / 2))
    _shape(cap.GetPrim(), looks["cap"])

    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    if world_rot is not None:
        fix.CreateLocalRot0Attr(Gf.Quatf(world_rot))
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/press")
    j.CreateAxisAttr("Z")
    j.CreateBody0Rel().SetTargets([bodies["anchor"].GetPath()])
    j.CreateBody1Rel().SetTargets([bodies["cap"].GetPath()])
    j.CreateLowerLimitAttr(-float(travel))
    j.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(STIFFNESS)
    drive.CreateDampingAttr(DAMPING)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateMaxForceAttr(50.0)
    return prim


def set_lamp(stage, path, on):
    """Light the run lamp (or not)."""
    from pxr import Gf

    sh = stage.GetPrimAtPath(f"{path}/Looks/lamp/shader")
    if sh:
        sh.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*(LAMP_ON if on else COLORS["lamp"])))
        sh.GetAttribute("inputs:emissiveColor").Set(Gf.Vec3f(*(LAMP_ON if on else (0.0, 0.0, 0.0))))
