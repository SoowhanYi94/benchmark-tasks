"""The valve-lash fixture and the feeler gauge, built into a USD stage. Pure geometry constants at the top (NumPy only,
so the task file and the tests can import them without Isaac Sim); the builders import pxr when called.

Fixture frame: origin on the table top under the adjusting screw's axis; x away from the arms, y left, z up (m).

- The frame (one kinematic body): cylinder-head block, valve stem with its spring, rocker pedestal and rocker arm.
  The rocker is fixed: its pushrod end sits on the cam's base circle, which is when lash is set.
- The mechanism (one articulation): anchor (fixed to the world) -> screw_slide (prismatic, z) -> screw_carrier ->
  screw_turn (revolute, z) -> screw (the adjusting screw with its T-bar) -> nut_slide (prismatic, z) -> nut_carrier
  -> nut_turn (revolute, z) -> nut (the locknut with its four spokes). Each prismatic joint follows its revolute
  joint through a PhysX mimic joint: PITCH per turn, right-handed (clockwise from above moves it down).
  The screw's tip and the nut are rigid colliders; the tip meets the valve stem (or the blade), the nut the rocker.
- Each revolute joint has a damping drive capped at a torque (a slipping clutch): it holds where it is left and
  turns once a hand pushes harder than the cap. The nut's cap is lower than the screw's, so turning the screw takes
  the loose nut round with it, and turning the nut leaves the screw where it is.
- The gauge (one free rigid body): an aluminium handle with a 2 mm steel blade along its -y end, flush with its
  underside.
"""

import math

import numpy as np

PITCH = 0.002  # m per turn (an M8 adjusting screw is about 1.25 mm; doubled so a turn is easier to see and do)
COEF = PITCH / 360.0  # mimic: q_slide = COEF * q_turn; PhysX reads the revolute leader in degrees (measured)

# -- frame ---------------------------------------------------------------------------------------------------------
HEAD = ((-0.085, -0.040, 0.0), (0.040, 0.040, 0.035))  # cast-iron cylinder-head block
STEM_R, STEM_TOP = 0.004, 0.060  # valve stem, 8 mm, its top 6 cm above the table
SPRING_R, SPRING_TOP = 0.011, 0.054  # valve spring and retainer round the stem
PEDESTAL = ((-0.075, -0.012, 0.035), (-0.050, 0.012, 0.072))  # the rocker's stand
PIVOT_X, PIVOT_Z = -0.0625, 0.078
ROCKER = ((-0.080, -0.009, 0.072), (0.012, -0.009 + 0.018, 0.084))  # steel rocker arm, 18 mm wide, 12 mm thick
ROCKER_BOTTOM, ROCKER_TOP = ROCKER[0][2], ROCKER[1][2]

# -- screw (body frame = fixture frame at the start) ---------------------------------------------------------------
GAP0 = 0.008  # starting lash: the screw's tip 8 mm above the stem
SCREW_R = 0.004
TIP_Z0 = STEM_TOP + GAP0
TIP_COLLIDER = 0.002  # only the tip's lowest 2 mm collides (the shank runs through the rocker's thread)
SCREW_TOP0 = 0.118
SLIDE_LIMITS = (-(GAP0 + 0.0005), 0.003)  # m; contact with the stem stops it first
TBAR_Z0 = SCREW_TOP0 - 0.003  # T-bar's rod centre
TBAR_R = 0.035  # the T-bar's pins stand at +-TBAR_R along the screw body's x
TBAR_ROD = 0.006
PIN_R, PIN_H = 0.005, 0.022  # each pin rises PIN_H above the rod's centre
SCREW_PIN_TOP0 = TBAR_Z0 + PIN_H

# -- locknut -------------------------------------------------------------------------------------------------------
NUT_R, NUT_H = 0.0075, 0.007
NUT_LIFT0 = 0.007  # the loosened nut starts 7 mm above the rocker
NUT_Z0 = ROCKER_TOP + NUT_LIFT0  # its underside
SPOKES = 4
SPOKE_R = 0.085  # the spokes' pins stand SPOKE_R from the axis
SPOKE_W = 0.005
SPOKE_Z0 = NUT_Z0 + 0.0045  # spoke centre
SPOKE_PIN_H = 0.025
NUT_PIN_TOP0 = SPOKE_Z0 + SPOKE_PIN_H
NUT_PHASE0 = math.radians(-45.0)  # one spoke points this way at the start (the others every 90 degrees)
NUT_SLIDE_LIMITS = (-0.015, 0.012)  # nut on the screw, relative

# -- clutch drives (revolute joints) -------------------------------------------------------------------------------
SCREW_HOLD = 0.06  # N*m the screw takes to turn in the rocker's thread
NUT_HOLD = 0.02  # N*m the loose locknut takes to turn on the screw
DAMPING = {"screw_turn": 50.0, "nut_turn": 10.0}  # N*m*s/rad below the cap (a joint creeps at load / DAMPING rad/s)
SEATED = 0.0002  # m: the screw locks once the nut's underside is within this of the rocker's top
LOCKED_JOINTS = ("screw_turn", "screw_slide")

MASS = {"anchor": 0.5, "screw_carrier": 0.005, "screw": 0.040, "nut_carrier": 0.005, "nut": 0.050}
COLORS = {"cast": (0.30, 0.31, 0.33), "steel": (0.72, 0.72, 0.75), "dark": (0.15, 0.15, 0.16),
          "spring": (0.20, 0.35, 0.65), "brass": (0.80, 0.62, 0.25), "alu": (0.80, 0.82, 0.85),
          "blade": (0.60, 0.62, 0.66)}

# -- feeler gauge (gauge frame: origin under the handle's centre, on its underside) -----------------------------------
HANDLE = (0.018, 0.060, 0.035)  # x, y, z
BLADE = (0.012, 0.135, 0.002)  # x, y (length), z (thickness)
BLADE_Y = (-HANDLE[1] / 2 - BLADE[1], -HANDLE[1] / 2)  # the blade runs from the handle's -y end, flush underneath
HANDLE_MASS, BLADE_MASS = 0.102, 0.025  # aluminium, steel
GAUGE_MASS = HANDLE_MASS + BLADE_MASS


def blade_box():
    """The blade's box in the gauge frame (low, high)."""
    return (np.array((-BLADE[0] / 2, BLADE_Y[0], 0.0)), np.array((BLADE[0] / 2, BLADE_Y[1], BLADE[2])))


def set_lock(mech, locked, state):
    """The lock (Isaac Lab articulation ``mech``): while the nut is seated, the screw's joints are held where they are
    by closing their limits on their present positions (hard limits: no creep); unlocking restores them. ``state``
    (a dict) keeps the original limits. Closing the limits clamps the articulation's default joint positions, which
    a reset uses, so they are put back to the start (all zero) each time."""
    import torch

    ids = [mech.joint_names.index(n) for n in LOCKED_JOINTS]
    lim = mech.data.joint_pos_limits.torch
    if "limits" not in state:
        state["limits"] = lim[:, ids].clone()
    if locked:
        q = mech.data.joint_pos.torch[:, ids]
        new = torch.stack((q - 1e-6, q + 1e-6), dim=-1)
    else:
        new = state["limits"]
    mech.write_joint_position_limit_to_sim_index(limits=new.contiguous(), joint_ids=ids, warn_limit_violation=False)
    mech.data._default_joint_pos.zero_()


# -- USD -----------------------------------------------------------------------------------------------------------
def _physx(prim, schema, **attrs):
    """Apply a PhysX API schema by name and set its attributes (no PhysxSchema bindings needed)."""
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
        self.phys = {"steel": self._phys("steel", 0.3), "grip": self._phys("grip", 1.0)}

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

    def collider(self, prim, mat="steel", offset=0.001):
        from pxr import UsdPhysics, UsdShade

        UsdPhysics.CollisionAPI.Apply(prim)
        _physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": float(offset),
                                             "physxCollision:restOffset": 0.0})
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.phys[mat], UsdShade.Tokens.weakerThanDescendants, "physics")

    def rigid(self, prim, kinematic=False, mass=None, inertia=None, iters=32):
        from pxr import Gf, UsdPhysics

        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
        _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": iters,
                                             "physxRigidBody:solverVelocityIterationCount": 1,
                                             "physxRigidBody:maxDepenetrationVelocity": 0.1})
        if mass is not None:
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr(float(mass))
            if inertia is not None:
                m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))

    def _paint(self, prim, look):
        from pxr import UsdShade

        if look:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.looks[look])

    def cube(self, path, box, look=None, collide=True, mat="steel", offset=0.001):
        from pxr import Gf, UsdGeom

        lo, hi = np.array(box[0], float), np.array(box[1], float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        self._paint(c.GetPrim(), look)
        if collide:
            self.collider(c.GetPrim(), mat, offset)
        return c

    def cyl(self, path, center, radius, height, axis="Z", look=None, collide=False, mat="steel"):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(float(radius))
        c.CreateHeightAttr(float(height))
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        self._paint(c.GetPrim(), look)
        if collide:
            self.collider(c.GetPrim(), mat)
        return c


def _zcyl(b, path, r, z0, z1, look, collide=False, mat="steel", xy=(0.0, 0.0)):
    return b.cyl(path, (xy[0], xy[1], (z0 + z1) / 2), r, z1 - z0, "Z", look, collide, mat)


def build_frame(stage, path):
    """The fixed frame as one kinematic body at ``path`` (its prim's own transform places the fixture)."""
    from pxr import UsdGeom

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    b.rigid(prim, kinematic=True)
    b.cube(f"{path}/head", HEAD, "cast")
    _zcyl(b, f"{path}/spring", SPRING_R, HEAD[1][2], SPRING_TOP - 0.003, "spring", collide=True)
    _zcyl(b, f"{path}/retainer", SPRING_R - 0.002, SPRING_TOP - 0.003, SPRING_TOP, "steel", collide=True)
    _zcyl(b, f"{path}/stem", STEM_R, SPRING_TOP, STEM_TOP, "steel", collide=True)
    b.cube(f"{path}/pedestal", PEDESTAL, "cast")
    b.cube(f"{path}/rocker", ROCKER, "steel")
    b.cyl(f"{path}/pivot", (PIVOT_X, 0.0, PIVOT_Z), 0.006, 0.030, "Y", "dark")
    return prim


def build_mech(stage, path, world_origin, world_rot=None):
    """The screw and the locknut as one articulation at ``path``; ``world_origin`` is where ``path``'s frame sits in
    the world (the anchor's fixed joint holds it there)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    _physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False,
                                            "physxArticulation:solverPositionIterationCount": 32,
                                            "physxArticulation:solverVelocityIterationCount": 1})
    bodies = {}
    for name in ("anchor", "screw_carrier", "screw", "nut_carrier", "nut"):
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        m = MASS[name]
        b.rigid(bodies[name], mass=m, inertia=(m * 4e-4, m * 4e-4, m * 4e-4))
    s, n = f"{path}/screw", f"{path}/nut"
    # the screw: shank (look only; it runs through the rocker), tip collider, hub and T-bar with two pins
    _zcyl(b, f"{s}/shank", SCREW_R, TIP_Z0, SCREW_TOP0 - 0.006, "brass")
    t = b.cube(f"{s}/tip", ((-0.0035, -0.0035, TIP_Z0), (0.0035, 0.0035, TIP_Z0 + TIP_COLLIDER)))  # flat contact
    t.CreateVisibilityAttr("invisible")
    _zcyl(b, f"{s}/hub", 0.006, SCREW_TOP0 - 0.008, SCREW_TOP0, "dark")
    h = TBAR_ROD / 2
    b.cube(f"{s}/tbar", ((-TBAR_R, -h, TBAR_Z0 - h), (TBAR_R, h, TBAR_Z0 + h)), "dark")
    for side, sx in (("a", 1), ("b", -1)):
        _zcyl(b, f"{s}/pin_{side}", PIN_R, TBAR_Z0 - h, SCREW_PIN_TOP0, "dark", collide=True, mat="grip",
              xy=(sx * TBAR_R, 0.0))
    # the locknut: body and four spokes with pins
    _zcyl(b, f"{n}/body", NUT_R, NUT_Z0, NUT_Z0 + NUT_H, "steel", collide=True)
    w = SPOKE_W / 2
    for k in range(SPOKES):
        a = NUT_PHASE0 + k * 2 * math.pi / SPOKES
        spoke = b.cube(f"{n}/spoke_{k}", ((NUT_R - 0.001, -w, SPOKE_Z0 - w), (SPOKE_R, w, SPOKE_Z0 + w)), "steel")
        _rotate_about_z(spoke, a)
        _zcyl(b, f"{n}/pin_{k}", PIN_R, SPOKE_Z0 - w, NUT_PIN_TOP0, "steel", collide=True, mat="grip",
              xy=(SPOKE_R * math.cos(a), SPOKE_R * math.sin(a)))
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    if world_rot is not None:
        fix.CreateLocalRot0Attr(Gf.Quatf(world_rot))
    axis_pt = Gf.Vec3f(0.0, 0.0, float(ROCKER_TOP))
    joints = {}
    for name, typ, b0, b1 in (("screw_slide", UsdPhysics.PrismaticJoint, "anchor", "screw_carrier"),
                              ("screw_turn", UsdPhysics.RevoluteJoint, "screw_carrier", "screw"),
                              ("nut_slide", UsdPhysics.PrismaticJoint, "screw", "nut_carrier"),
                              ("nut_turn", UsdPhysics.RevoluteJoint, "nut_carrier", "nut")):
        j = typ.Define(stage, f"{path}/{name}")
        j.CreateAxisAttr("Z")
        j.CreateBody0Rel().SetTargets([bodies[b0].GetPath()])
        j.CreateBody1Rel().SetTargets([bodies[b1].GetPath()])
        j.CreateLocalPos0Attr(axis_pt)
        j.CreateLocalPos1Attr(axis_pt)
        joints[name] = j
    lo, hi = SLIDE_LIMITS
    joints["screw_slide"].CreateLowerLimitAttr(lo)
    joints["screw_slide"].CreateUpperLimitAttr(hi)
    lo, hi = NUT_SLIDE_LIMITS
    joints["nut_slide"].CreateLowerLimitAttr(lo)
    joints["nut_slide"].CreateUpperLimitAttr(hi)
    for name in ("screw_turn", "nut_turn"):  # limited far out, so PhysX keeps the angle unwrapped
        joints[name].CreateLowerLimitAttr(-7200.0)
        joints[name].CreateUpperLimitAttr(7200.0)
    for name, cap in (("screw_turn", SCREW_HOLD), ("nut_turn", NUT_HOLD)):
        drv = UsdPhysics.DriveAPI.Apply(joints[name].GetPrim(), "angular")
        drv.CreateTypeAttr("force")
        drv.CreateTargetVelocityAttr(0.0)
        drv.CreateStiffnessAttr(0.0)
        drv.CreateDampingAttr(float(DAMPING[name] * math.pi / 180))  # USD angular drives are per degree
        drv.CreateMaxForceAttr(float(cap))
    for follower, leader in (("screw_slide", "screw_turn"), ("nut_slide", "nut_turn")):
        p = joints[follower].GetPrim()
        p.AddAppliedSchema("NewtonMimicAPI")
        from pxr import Sdf

        p.CreateRelationship("newton:mimicJoint").SetTargets([joints[leader].GetPath()])
        p.CreateAttribute("newton:mimicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        p.CreateAttribute("newton:mimicCoef0", Sdf.ValueTypeNames.Float).Set(0.0)
        p.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(float(COEF))
    return prim


def _rotate_about_z(gprim, angle):
    """Turn a translated, scaled cube about the fixture's z axis (prepends a rotation before its own ops)."""
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xformable(gprim)
    ops = x.GetOrderedXformOps()
    t = ops[0].Get()
    s = ops[1].Get()
    x.ClearXformOpOrder()
    x.AddRotateZOp().Set(math.degrees(angle))
    x.AddTranslateOp().Set(t)
    x.AddScaleOp().Set(s)


def build_gauge(stage, path, mass=GAUGE_MASS):
    """The feeler gauge as one free rigid body at ``path`` (its prim's transform places it; origin under the
    handle's centre on its underside)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    b.rigid(prim, mass=None, iters=32)
    hx, hy, hz = HANDLE
    b.cube(f"{path}/handle", ((-hx / 2, -hy / 2, 0.0), (hx / 2, hy / 2, hz)), "alu", mat="grip", offset=0.003)
    lo, hi = blade_box()
    b.cube(f"{path}/blade", (lo, hi), "blade", offset=0.0005)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    com_y = (BLADE_MASS * (BLADE_Y[0] + BLADE_Y[1]) / 2) / GAUGE_MASS
    com_z = (HANDLE_MASS * hz / 2 + BLADE_MASS * BLADE[2] / 2) / GAUGE_MASS
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, float(com_y), float(com_z)))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(2.0e-4, 3.0e-5, 2.0e-4))
    return prim
