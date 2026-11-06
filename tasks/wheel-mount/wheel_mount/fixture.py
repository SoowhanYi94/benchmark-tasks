"""The bench grinder, its loose wheel-mounting parts and the parts tray, built into a USD stage. Pure geometry
constants at the top (NumPy only, so the task file and the tests can import them without Isaac Sim); the builders
import pxr when called.

Fixture frame: origin on the table top under the spindle's axis, level with the face of the spindle's collar (the
shoulder the stack is clamped against); x away from the arms, y left, z up (m). The spindle runs along y, its nose
pointing to -y (the grinder's right-hand side, seen from the arms).

- The frame (one kinematic body): base, motor housing, the guard's back plate, the spindle-lock button's pedestal,
  the parts tray (rails and dividers) and the nut's stand.
- The mechanism (one articulation, self-collision off): anchor (fixed to the world) ->
    spindle_turn (revolute, y) -> spindle: the collar and the bare spindle (a 12 mm shaft with a flat on top, solid);
      -> nut_px, nut_pz, nut_py (prismatic x, z, y; free) -> nut_turn (revolute, y) -> nut_thread (prismatic, y, a
         mimic of nut_turn: PITCH per turn) -> nut (hex, hub and four spokes) -> nut_crush (prismatic, y, a stiff
         spring) -> nut_face (the nut's washer face, which presses on the stack).
    lock_press (prismatic, z) -> button: the spindle lock's push button, sprung up.
  The nut starts on its stand, coaxial with the spindle 23.5 cm out from the collar; until it is on the thread its
  slides are free (it moves like a loose part, but stays square to the spindle's axis) and it can't turn. Its y slide
  stops at the thread's start (WALL). The yam_wheel embodiment threads it on (`capture`) once it is pushed to the
  thread's start on the spindle's axis: its x, z and y slides lock and it can turn; turning it on (positively about +y,
  its top moving away from the arms) runs it toward the collar at PITCH per turn. The nut's own clutch (NUT_HOLD) is
  stronger than the spindle's bearings (SPINDLE_HOLD), so unless the spindle is locked, turning the nut turns the
  spindle with it and the nut doesn't advance. Once the nut seats, the stack's blotters squeeze (nut_crush, a spring of
  CRUSH_K: the clamp force) and the torque to turn the nut on rises with the clamp force (nut_hold). The thread's
  free run before the seat is RUN_DOWN.
- The loose parts (free rigid bodies): inner flange (D bore, keyed to the spindle's flat), two blotters, the wheel,
  the outer flange and a damaged wheel (a chip out of its rim and a crack). Each is a ring of convex segments, so its bore really
  fits round the spindle; it hangs on the spindle where it is left and slides along it.
"""

import math

import numpy as np

# -- spindle (fixture frame) -----------------------------------------------------------------------------------------
ZA = 0.085  # the spindle's axis height above the table
SPINDLE_R = 0.006  # 12 mm spindle
FLAT_Z = 0.0040  # its flat, on top, 2 mm deep
NOSE_Y = -0.060  # the nose's end
COLLAR_R, COLLAR_Y = 0.011, (0.0, 0.010)  # the shoulder the stack is clamped against (its face at y = 0)
SHAFT_IN = 0.025  # the spindle runs on into the motor housing (look only past the collar)

# -- grinder body (frame) -------------------------------------------------------------------------------------------
MOTOR_R, MOTOR_Y = 0.055, (0.025, 0.145)
BASE = ((-0.065, 0.015, 0.0), (0.065, 0.160, 0.032))
GUARD_R, GUARD_Y = 0.068, (0.012, 0.015)  # the guard's back plate; its front cover is off
BUTTON_XY = (-0.100, 0.120)  # the spindle lock's push button
BUTTON_R, BUTTON_H = 0.010, 0.012
BUTTON_Z0 = 0.040  # its pedestal's top; the cap stands BUTTON_Z0..BUTTON_Z0 + BUTTON_H when up
BUTTON_TRAVEL = 0.004
PRESSED = 0.0025  # the spindle is locked while the button is pressed at least this far
PEDESTAL = ((BUTTON_XY[0] - 0.014, BUTTON_XY[1] - 0.014, 0.0),  # its top 5 mm under the cap, so the cap can go down
            (BUTTON_XY[0] + 0.014, BUTTON_XY[1] + 0.014, BUTTON_Z0 - BUTTON_TRAVEL - 0.001))
BRACKET = ((BUTTON_XY[0] - 0.006, BUTTON_XY[1] - 0.010, 0.0), (BASE[0][0], BUTTON_XY[1] + 0.010, 0.008))

# -- the loose parts (each part's frame: its centre, its axis along y) ---------------------------------------------
BORE_R = 0.0075  # 15 mm bores on a 12 mm spindle
D_CHORD = 0.0044  # the inner flange's D bore: its flat 4.4 mm above the bore's centre, 11.9 mm across, under the spindle's 12 mm (keyed: it slides on only
#                    with its flat lined up with the spindle's, and then turns with the spindle)
PARTS = {  # name: outer radius, thickness, bore ("D" or "round"), mass (kg), look, physics material
    "inner_flange": (0.022, 0.006, "D", 0.060, "oxide", "steel"),
    "blotter_a": (0.026, 0.002, "round", 0.020, "paper", "paper"),
    "wheel": (0.050, 0.013, "round", 0.230, "wheel", "abrasive"),
    "blotter_b": (0.026, 0.002, "round", 0.020, "paper", "paper"),
    "outer_flange": (0.022, 0.006, "round", 0.060, "steel", "steel"),
    "decoy": (0.050, 0.013, "round", 0.226, "wheel", "abrasive"),
}
ORDER = ("inner_flange", "blotter_a", "wheel", "blotter_b", "outer_flange")  # from the collar out
SEGMENTS = {"inner_flange": 22, "blotter_a": 18, "wheel": 24, "blotter_b": 18, "outer_flange": 22, "decoy": 24}
CHIP = (5, 0.034)  # the damaged wheel: its segment 5 (75 to 90 degrees from +x toward +z) chipped to a 34 mm radius


def _homes():
    y, out = 0.0, {}
    for name in ORDER:
        t = PARTS[name][1]
        out[name] = y - t / 2
        y -= t
    return out, y


HOME, SNUG_Y = _homes()  # each part's centre y once the stack is home; the stack's outer face (the nut's seat)

# -- the nut (nut frame: origin on the axis in the face's plane at zero crush; the nut extends to -y) ---------------
PITCH = 0.0025  # m per turn (an M12 fine thread is 1.25 mm; doubled)
COEF = PITCH / 360.0  # mimic: q_thread = COEF * q_turn, the revolute leader read in degrees
RUN_DOWN = 0.00181  # the thread's free run before the nut seats: 1.81 mm (261 degrees)
WALL_Y = SNUG_Y - RUN_DOWN  # the nut's face at the thread's start
THREAD_LEN = 0.012  # the thread runs on 12 mm (it bottoms there)
FACE_R, FACE_T = 0.012, 0.003  # the washer face, 24 mm
HEX_AF, HEX_L = 0.019, 0.016  # 19 mm across flats, 16 mm long
HUB_R, HUB_L = 0.009, 0.007
SPOKE_Y = -(FACE_T + HEX_L + 0.003)  # the spokes' plane (22 mm out from the face)
SPOKES, SPOKE_R, SPOKE_TIP = 4, 0.004, 0.065
SPOKE_PHASE = math.radians(45.0)  # a spoke at 45 degrees from +x toward +z, then every 90
GRIP_R = 0.056  # where the demo grips a spoke (from the axis)
NUT_STAND_Y = -0.235  # the nut's face plane on its stand (coaxial with the spindle)
WALL_Q = WALL_Y - NUT_STAND_Y  # nut_py's upper limit: the thread's start
CAPTURE_R = 0.0025  # the nut threads on once within this of the axis ...
CAPTURE_Y = 0.001  # ... and its face within this of the thread's start
CRUSH_K = 30000.0  # N/m: the blotters' squeeze (the clamp's spring)
CRUSH_MAX = 0.0025
STAND = ((-0.006, NUT_STAND_Y - FACE_T - HEX_L + 0.002, 0.0), (0.006, NUT_STAND_Y - FACE_T - 0.002, ZA - HEX_AF / 2))

# -- clutches and the judge ----------------------------------------------------------------------------------------
SPINDLE_HOLD, SPINDLE_DAMPING = 0.03, 0.5  # N*m, N*m*s/rad: the spindle's bearings
NUT_HOLD, NUT_DAMPING = 0.12, 20.0  # the nut's free-run drag on its thread (damping high: friction capped at the hold)
NUT_PER_N = 0.008  # N*m of turning torque added per N of clamp force (face and thread friction): 0.22 N*m at 12.5 N
SLIDE_HOLD, SLIDE_DAMPING = 0.2, 5.0  # N, N*s/m: the loose nut (less than its weight, so it falls)
BUTTON_K, BUTTON_PRELOAD = 800.0, 0.002  # N/m; the spring's rest point above the top stop (1.6 N held up)

PART_ITERS = 128  # solver position iterations for the loose parts (a stack of five under load converges)

# -- the parts tray (frame; part centres in the fixture frame) -----------------------------------------------------
RAIL_R, RAIL_A, RAIL_Z = 0.003, 0.015, 0.010  # two rails along y under each row, at the slot's x +- RAIL_A
TRAY_T = 0.007
DIVIDER_H = 0.022  # dividers stand to this height, 2 mm thick, either side of each part
ROWS = {"small": -0.060, "wheels": -0.140}
SLOTS = {"outer_flange": ("small", -0.110), "blotter_a": ("small", -0.142), "inner_flange": ("small", -0.174),
         "blotter_b": ("small", -0.206), "wheel": ("wheels", -0.115), "decoy": ("wheels", -0.165)}
TRAY_Y = (-0.225, -0.090)


def nut_hold(crush):
    """The torque (N*m) it takes to turn the nut on with the stack squeezed ``crush`` m: free-run drag, then rising with
    the clamp force (CRUSH_K * crush)."""
    return NUT_HOLD + NUT_PER_N * CRUSH_K * max(0.0, crush)


def rest_z(r):
    """Centre height of a disc of radius ``r`` standing on two rails."""
    return RAIL_Z + math.sqrt((r + RAIL_R) ** 2 - RAIL_A ** 2)


def slot_pos(name):
    """A part's centre in its tray slot (fixture frame)."""
    row, y = SLOTS[name]
    return (ROWS[row], y, rest_z(PARTS[name][0]))


MASS = {"anchor": 0.5, "spindle": 0.15, "button": 0.02, "nut_bx": 0.005, "nut_bz": 0.005, "nut_by": 0.005,
        "nut_bt": 0.005, "nut": 0.08, "nut_face": 0.03}
COLORS = {"housing": (0.24, 0.32, 0.46), "guard": (0.55, 0.57, 0.60), "wheel": (0.50, 0.47, 0.42),
          "paper": (0.93, 0.88, 0.58), "oxide": (0.13, 0.13, 0.14), "steel": (0.74, 0.75, 0.78),
          "dark": (0.10, 0.10, 0.11), "red": (0.80, 0.07, 0.05), "tray": (0.36, 0.37, 0.39),
          "nut": (0.42, 0.43, 0.46), "crack": (0.05, 0.05, 0.05)}
FRICTION = {"steel": 0.5, "paper": 0.8, "abrasive": 0.7, "grip": 1.0}


# -- geometry helpers (NumPy) ----------------------------------------------------------------------------------------
def segment_angles(n):
    """Each segment's start and end angle (rad, in the part's x-z plane from +x toward +z)."""
    return [(2 * math.pi * k / n, 2 * math.pi * (k + 1) / n) for k in range(n)]


def ring_segments(name):
    """The part's convex segments: for each, its 8 corners (part frame), a ring round the bore."""
    r_out, t, bore, *_ = PARTS[name]
    n = SEGMENTS[name]
    r_in = BORE_R / math.cos(math.pi / n)  # the bore's flats clear BORE_R
    out = []
    for k, (a0, a1) in enumerate(segment_angles(n)):
        ro = CHIP[1] if (name == "decoy" and k == CHIP[0]) else r_out
        quad = []
        for a, r in ((a0, r_in), (a0, ro), (a1, ro), (a1, r_in)):
            if r == r_in and bore == "D" and math.sin(a) * r > D_CHORD:
                r = D_CHORD / math.sin(a)
            quad.append((r * math.cos(a), r * math.sin(a)))
        out.append(np.array([(x, y, z) for y in (-t / 2, t / 2) for (x, z) in quad]))
    return out


def spindle_section(n=32):
    """The spindle's cross-section (x, z about its axis): a circle with the flat on top."""
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        x, z = SPINDLE_R * math.cos(a), SPINDLE_R * math.sin(a)
        if z > FLAT_Z:
            x, z = x * FLAT_Z / z, FLAT_Z
        pts.append((x, z))
    return pts


def spoke_angles(turned):
    """The spokes' angles (rad from +x toward +z) after the nut has turned ``turned`` rad about +y (with the spindle
    at 0)."""
    return [SPOKE_PHASE + k * math.pi / 2 - turned for k in range(SPOKES)]


# -- joints at run time ----------------------------------------------------------------------------------------------
def set_limits(mech, joint, limits, state):
    """Set one joint's limits (``None`` puts back the ones it was built with). Closing a joint's limits clamps the
    articulation's default joint positions, which a reset uses, so they are put back to the start (all zero)."""
    import torch

    ids = [mech.joint_names.index(joint)]
    lim = mech.data.joint_pos_limits.torch
    orig = state.setdefault("limits", {})
    if joint not in orig:
        orig[joint] = lim[:, ids].clone()
    new = orig[joint] if limits is None else torch.tensor([[list(limits)]], dtype=lim.dtype, device=lim.device)
    mech.write_joint_position_limit_to_sim_index(limits=new.contiguous(), joint_ids=ids, warn_limit_violation=False)
    mech.data._default_joint_pos.zero_()


def lock_here(mech, joint, state, at=None):
    """Hold a joint hard where it is (or at ``at``, written there first): its limits closed on its position."""
    import torch

    i = mech.joint_names.index(joint)
    if at is not None:
        pos = torch.tensor([[float(at)]], device=mech.data.joint_pos.torch.device)
        mech.write_joint_position_to_sim_index(position=pos, joint_ids=[i])
        mech.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos), joint_ids=[i])
        q = float(at)
    else:
        q = float(mech.data.joint_pos.torch[0, i])
    set_limits(mech, joint, (q - 1e-6, q + 1e-6), state)


def set_clutch(mech, joint, damping, cap):
    import torch

    j = [mech.joint_names.index(joint)]
    dev = mech.data.joint_pos.torch.device
    mech.write_joint_damping_to_sim_index(damping=torch.full((1, 1), float(damping), device=dev), joint_ids=j)
    mech.write_joint_effort_limit_to_sim_index(limits=torch.full((1, 1), float(cap), device=dev), joint_ids=j)


# -- USD -------------------------------------------------------------------------------------------------------------
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
        self.phys = {k: self._phys(k, mu) for k, mu in FRICTION.items()}

    def _look(self, name, color):
        from pxr import Gf, Sdf, UsdShade

        mat = UsdShade.Material.Define(self.stage, f"{self.root}/Looks/{name}")
        sh = UsdShade.Shader.Define(self.stage, f"{self.root}/Looks/{name}/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        metal = 0.6 if name in ("steel", "nut", "oxide", "guard") else 0.1
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85 if name in ("wheel", "paper") else 0.4)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _phys(self, name, mu):
        from pxr import UsdPhysics, UsdShade

        mat = UsdShade.Material.Define(self.stage, f"{self.root}/Physics/{name}")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(mu)
        pm.CreateDynamicFrictionAttr(mu)
        pm.CreateRestitutionAttr(0.0)
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
            m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
            if inertia is not None:
                m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))

    def paint(self, prim, look):
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
        self.paint(c.GetPrim(), look)
        if collide:
            self.collider(c.GetPrim(), mat, offset)
        return c

    def cyl(self, path, center, radius, height, axis="Z", look=None, collide=False, mat="steel", rot_y=None,
            offset=0.001):
        """A cylinder; ``rot_y`` (degrees) turns it about the y axis through ``center``."""
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(float(radius))
        c.CreateHeightAttr(float(height))
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        if rot_y is not None:
            c.AddRotateYOp().Set(float(rot_y))
        self.paint(c.GetPrim(), look)
        if collide:
            self.collider(c.GetPrim(), mat, offset)
        return c

    def hull(self, path, pts, look=None, collide=True, mat="steel", offset=0.0005):
        """A convex mesh through ``pts`` (8 corners of a prism: 4 at one end, then the same 4 at the other)."""
        from pxr import Gf, UsdGeom, UsdPhysics, Vt

        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
        n = len(pts) // 2
        faces = [list(range(n))[::-1], list(range(n, 2 * n))]
        faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        m.CreateFaceVertexCountsAttr([len(f) for f in faces])
        m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        m.CreateDoubleSidedAttr(True)
        m.CreateSubdivisionSchemeAttr("none")
        self.paint(m.GetPrim(), look)
        if collide:
            self.collider(m.GetPrim(), mat, offset)
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        return m


def _ycyl(b, path, r, y0, y1, look, collide=False, mat="steel", xz=(0.0, ZA)):
    return b.cyl(path, (xz[0], (y0 + y1) / 2, xz[1]), r, abs(y1 - y0), "Y", look, collide, mat)


def _zcyl(b, path, r, xy, z0, z1, look, collide=False, mat="steel"):
    return b.cyl(path, (xy[0], xy[1], (z0 + z1) / 2), r, z1 - z0, "Z", look, collide, mat)


def build_frame(stage, path):
    """The fixed frame as one kinematic body at ``path`` (its prim's own transform places the fixture)."""
    from pxr import UsdGeom

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    b.rigid(prim, kinematic=True)
    b.cube(f"{path}/base", BASE, "housing")
    _ycyl(b, f"{path}/motor", MOTOR_R, *MOTOR_Y, "housing", collide=True)
    _ycyl(b, f"{path}/guard_back", GUARD_R, *GUARD_Y, "guard", collide=True)
    _ycyl(b, f"{path}/shaft_in", 0.008, COLLAR_Y[1], MOTOR_Y[0], "dark")
    b.cube(f"{path}/pedestal", PEDESTAL, "dark")
    b.cube(f"{path}/bracket", BRACKET, "dark")
    # the parts tray: a plate, two rails under each row, dividers either side of each part
    x0 = min(ROWS.values()) - 0.060
    x1 = max(ROWS.values()) + 0.035
    b.cube(f"{path}/tray", ((x0, TRAY_Y[0], 0.0), (x1, TRAY_Y[1], TRAY_T)), "tray")
    for row, x in ROWS.items():
        for side, s in (("n", -1), ("f", 1)):
            b.cyl(f"{path}/rail_{row}_{side}", (x + s * RAIL_A, sum(TRAY_Y) / 2, RAIL_Z), RAIL_R,
                  TRAY_Y[1] - TRAY_Y[0], "Y", "steel", collide=True)
    for name, (row, y) in SLOTS.items():
        t = PARTS[name][1]
        x = ROWS[row]
        for side, s in (("l", 1), ("r", -1)):
            yc = y + s * (t / 2 + 0.0015 + 0.001)
            b.cube(f"{path}/div_{name}_{side}", ((x - 0.011, yc - 0.001, TRAY_T), (x + 0.011, yc + 0.001, DIVIDER_H)),
                   "tray")
    b.cube(f"{path}/nut_stand", STAND, "dark")
    b.cube(f"{path}/nut_foot", ((-0.020, STAND[0][1] - 0.006, 0.0), (0.020, STAND[1][1] + 0.006, 0.006)), "dark")
    return prim


def _spoke(b, path, phi, y, look="dark"):
    """A spoke (rod along its radius) at angle ``phi`` from +x toward +z in the plane ``y``."""
    rc = (HUB_R - 0.001 + SPOKE_TIP) / 2
    c = (rc * math.cos(phi), y, ZA + rc * math.sin(phi))
    # a cylinder along x turned about y by -phi (USD's rotateY turns +x toward -z)
    return b.cyl(path, c, SPOKE_R, SPOKE_TIP - HUB_R + 0.001, "X", look, collide=True, mat="grip",
                 rot_y=-math.degrees(phi))


def build_mech(stage, path, world_origin, world_rot=None):
    """The spindle, its nut and the lock button as one articulation at ``path``; ``world_origin`` is where ``path``'s
    frame sits in the world (the anchor's fixed joint holds it there). Every link's frame starts at ``path``'s; the
    nut is built on its stand."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    _physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False,
                                            "physxArticulation:solverPositionIterationCount": 32,
                                            "physxArticulation:solverVelocityIterationCount": 1})
    bodies = {}
    for name, m in MASS.items():
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        b.rigid(bodies[name], mass=m, inertia=(m * 4e-4, m * 4e-4, m * 4e-4))
    # the spindle: collar and shaft (with its flat), one convex hull
    s = f"{path}/spindle"
    _ycyl(b, f"{s}/collar", COLLAR_R, *COLLAR_Y, "steel", collide=True)
    sec = spindle_section()
    pts = [(x, NOSE_Y, ZA + z) for x, z in sec] + [(x, COLLAR_Y[1], ZA + z) for x, z in sec]
    b.hull(f"{s}/shaft", pts, "steel", mat="steel")
    b.cube(f"{s}/mark", ((-0.0012, NOSE_Y - 0.0004, ZA - 0.004), (0.0012, NOSE_Y + 0.0004, ZA + 0.004)), "red",
           collide=False)  # a red line on the nose's end (shows the spindle turning)
    # the lock button's cap
    _zcyl(b, f"{path}/button/cap", BUTTON_R, BUTTON_XY, BUTTON_Z0, BUTTON_Z0 + BUTTON_H, "red", collide=True,
          mat="grip")
    # the nut, built on its stand (nut frame origin at (0, NUT_STAND_Y, ZA))
    y0 = NUT_STAND_Y
    n = f"{path}/nut"
    hex_pts = []
    for k in range(6):
        a = math.radians(30 + 60 * k)
        r = HEX_AF / 2 / math.cos(math.radians(30))
        hex_pts.append((r * math.cos(a), ZA + r * math.sin(a)))
    pts = [(x, y0 - FACE_T - HEX_L, z) for x, z in hex_pts] + [(x, y0 - FACE_T, z) for x, z in hex_pts]
    b.hull(f"{n}/hex", pts, "nut", mat="steel")
    _ycyl(b, f"{n}/hub", HUB_R, y0 + SPOKE_Y - HUB_L / 2, y0 + SPOKE_Y + HUB_L / 2, "dark", collide=True)
    for k, phi in enumerate(spoke_angles(0.0)):
        _spoke(b, f"{n}/spoke_{k}", phi, y0 + SPOKE_Y)
    _ycyl(b, f"{path}/nut_face/face", FACE_R, y0 - FACE_T, y0, "nut", collide=True, mat="steel")
    # joints
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    if world_rot is not None:
        fix.CreateLocalRot0Attr(Gf.Quatf(world_rot))
    nut0 = (0.0, y0, ZA)
    specs = (("spindle_turn", UsdPhysics.RevoluteJoint, "anchor", "spindle", "Y", (0.0, 0.0, ZA)),
             ("lock_press", UsdPhysics.PrismaticJoint, "anchor", "button", "Z", (*BUTTON_XY, BUTTON_Z0)),
             ("nut_px", UsdPhysics.PrismaticJoint, "spindle", "nut_bx", "X", nut0),
             ("nut_pz", UsdPhysics.PrismaticJoint, "nut_bx", "nut_bz", "Z", nut0),
             ("nut_py", UsdPhysics.PrismaticJoint, "nut_bz", "nut_by", "Y", nut0),
             ("nut_turn", UsdPhysics.RevoluteJoint, "nut_by", "nut_bt", "Y", nut0),
             ("nut_thread", UsdPhysics.PrismaticJoint, "nut_bt", "nut", "Y", nut0),
             ("nut_crush", UsdPhysics.PrismaticJoint, "nut", "nut_face", "Y", nut0))
    joints = {}
    for name, typ, b0, b1, axis, pt in specs:
        j = typ.Define(stage, f"{path}/{name}")
        j.CreateAxisAttr(axis)
        j.CreateBody0Rel().SetTargets([bodies[b0].GetPath()])
        j.CreateBody1Rel().SetTargets([bodies[b1].GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pt)))
        j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, pt)))
        joints[name] = j
    limits = {"spindle_turn": (-7200.0, 7200.0), "lock_press": (-BUTTON_TRAVEL, 0.0), "nut_px": (-0.15, 0.15),
              "nut_pz": (-0.06, 0.08), "nut_py": (-0.02, WALL_Q), "nut_turn": (-1.0, THREAD_LEN / COEF + 1.0),
              "nut_thread": (-0.0005, THREAD_LEN + 0.0005), "nut_crush": (-CRUSH_MAX, 0.0)}
    for name, (lo, hi) in limits.items():
        joints[name].CreateLowerLimitAttr(float(lo))
        joints[name].CreateUpperLimitAttr(float(hi))

    def drive(name, kind, damping, cap, stiffness=0.0, target=0.0):
        d = UsdPhysics.DriveAPI.Apply(joints[name].GetPrim(), kind)
        d.CreateTypeAttr("force")
        d.CreateTargetVelocityAttr(0.0)
        d.CreateTargetPositionAttr(float(target))
        d.CreateStiffnessAttr(float(stiffness))
        d.CreateDampingAttr(float(damping))
        d.CreateMaxForceAttr(float(cap))

    drive("spindle_turn", "angular", SPINDLE_DAMPING * math.pi / 180, SPINDLE_HOLD)  # USD angular drives: per degree
    drive("nut_turn", "angular", NUT_DAMPING * math.pi / 180, NUT_HOLD)
    for name in ("nut_px", "nut_pz", "nut_py"):
        drive(name, "linear", SLIDE_DAMPING, SLIDE_HOLD)
    drive("lock_press", "linear", 20.0, 20.0, BUTTON_K, BUTTON_PRELOAD)
    drive("nut_crush", "linear", 60.0, 500.0, CRUSH_K, 0.0)
    p = joints["nut_thread"].GetPrim()  # the thread: q_thread follows q_turn (NewtonMimicAPI couples on the GPU)
    p.AddAppliedSchema("NewtonMimicAPI")
    p.CreateRelationship("newton:mimicJoint").SetTargets([joints["nut_turn"].GetPath()])
    p.CreateAttribute("newton:mimicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    p.CreateAttribute("newton:mimicCoef0", Sdf.ValueTypeNames.Float).Set(0.0)
    p.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(float(COEF))
    return prim


def part_inertia(name):
    r_out, t, _, m, *_ = PARTS[name]
    r2 = r_out ** 2 + BORE_R ** 2
    return (m * (3 * r2 + t * t) / 12, m * r2 / 2, m * (3 * r2 + t * t) / 12)


def build_part(stage, path, name):
    """One loose part as a free rigid body at ``path`` (its prim's transform places it; origin at its centre, axis
    along y)."""
    from pxr import UsdGeom

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path) if stage.GetPrimAtPath(path) else UsdGeom.Xform.Define(stage, path).GetPrim()
    r_out, t, bore, m, look, mat = PARTS[name]
    b.rigid(prim, mass=m, inertia=part_inertia(name), iters=PART_ITERS)
    for k, pts in enumerate(ring_segments(name)):
        b.hull(f"{path}/seg_{k}", pts, look, mat=mat)
    if name in ("wheel", "decoy"):  # the label ring round the bore (look only), both faces
        for side, s in (("l", 1), ("r", -1)):
            b.cyl(f"{path}/label_{side}", (0.0, s * (t / 2 + 0.0002), 0.0), 0.020, 0.0004, "Y", "paper")
    if name == "decoy":
        a0, a1 = segment_angles(SEGMENTS[name])[CHIP[0]]
        a = (a0 + a1) / 2
        for side, s in (("l", 1), ("r", -1)):  # a crack from the chip toward the bore, on both faces
            c = b.cube(f"{path}/crack_{side}", ((-0.0006, s * (t / 2 + 0.0001) - 0.0003, 0.010),
                                                (0.0006, s * (t / 2 + 0.0001) + 0.0003, CHIP[1])), "crack",
                       collide=False)
            _turn_about_y(c, a - math.pi / 2)
    return prim


def _turn_about_y(gprim, angle):
    """Turn a translated, scaled cube about the part's y axis by ``angle`` (rad, +x toward +z)."""
    from pxr import UsdGeom

    x = UsdGeom.Xformable(gprim)
    ops = x.GetOrderedXformOps()
    t, s = ops[0].Get(), ops[1].Get()
    x.ClearXformOpOrder()
    x.AddRotateYOp().Set(-math.degrees(angle))
    x.AddTranslateOp(opSuffix="local").Set(t)
    x.AddScaleOp().Set(s)
