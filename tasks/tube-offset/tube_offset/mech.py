"""USD building blocks and run-time joint writers for the hand tube bender and the tube. Dimensions in m, angles in rad
unless named _deg. Everything that needs Isaac Sim is imported lazily, so the task file's scorer runs on NumPy alone."""

import math
import os

import numpy as np

ITERS, VEL_ITERS = 32, 8


def _api():
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    return Gf, Sdf, UsdGeom, UsdPhysics, UsdShade


def physx(prim, schema, **attrs):
    from pxr import Sdf

    try:
        ok = prim.ApplyAPI(schema)
    except Exception:  # PhysX schemas unregistered (a CPU-only authoring check)
        ok = False
    if not ok:
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


def _collide(prim, phys, offset=0.002):
    _, _, _, UsdPhysics, UsdShade = _api()
    UsdPhysics.CollisionAPI.Apply(prim)
    physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": float(offset), "physxCollision:restOffset": 0.0})
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def _place(xf, at, yaw=0.0, tilt=None):
    """Translate, then turn ``yaw`` about z (and optionally ``tilt`` = (axis, angle) before it)."""
    Gf, *_ = _api()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, at)))
    r = Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw))
    if tilt is not None:
        r = Gf.Rotation(Gf.Vec3d(*tilt[0]), math.degrees(tilt[1])) * r
    q = r.GetQuat()
    xf.AddOrientOp().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*map(float, q.GetImaginary()))))


def cube(stage, path, lo, hi, mat, phys=None, yaw=0.0, about=None):
    """A box from lo to hi; with ``yaw`` it is turned about ``about`` (default: its own centre)."""
    Gf, _, UsdGeom, _, UsdShade = _api()
    lo, hi = np.array(lo, float), np.array(hi, float)
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    centre = (lo + hi) / 2
    if yaw:
        o = centre if about is None else np.array(about, float)
        d = centre - o
        co, si = math.cos(yaw), math.sin(yaw)
        centre = o + np.array((co * d[0] - si * d[1], si * d[0] + co * d[1], d[2]))
    _place(c, centre, yaw)
    c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if phys is not None:
        _collide(c.GetPrim(), phys)
    return c


def cylinder(stage, path, at, r, h, axis, mat, phys=None, yaw=0.0):
    """A cylinder centred at ``at``, ``axis`` 'X', 'Y' or 'Z' in its own frame, turned ``yaw`` about z."""
    _, _, UsdGeom, _, UsdShade = _api()
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(h))
    c.CreateAxisAttr(axis)
    _place(c, at, yaw)
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if phys is not None:
        _collide(c.GetPrim(), phys)
    return c


def capsule(stage, path, at, r, h, axis, mat, phys=None, yaw=0.0):
    """A capsule: ``h`` is the straight part's length, the overall length is h + 2r."""
    _, _, UsdGeom, _, UsdShade = _api()
    c = UsdGeom.Capsule.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(h))
    c.CreateAxisAttr(axis)
    _place(c, at, yaw)
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if phys is not None:
        _collide(c.GetPrim(), phys, 0.001)
    return c


def rigid(prim, mass, ang_damp=0.5, lin_damp=0.1, depen=1.0):
    _, _, _, UsdPhysics, _ = _api()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": ITERS, "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                        "physxRigidBody:maxDepenetrationVelocity": float(depen), "physxRigidBody:angularDamping": float(ang_damp),
                                        "physxRigidBody:linearDamping": float(lin_damp)})
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))


def body(stage, path, mass, **kw):
    _, _, UsdGeom, _, _ = _api()
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    rigid(prim, mass, **kw)
    return prim


def articulation_root(prim, fixed=False, iters=ITERS):
    _, _, _, UsdPhysics, _ = _api()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False, "physxArticulation:solverPositionIterationCount": int(iters),
                                           "physxArticulation:solverVelocityIterationCount": VEL_ITERS, "physxArticulation:sleepThreshold": 0.0,
                                           "physxArticulation:stabilizationThreshold": 0.0})


def fix_to_world(stage, path, body_prim, world_origin):
    Gf, _, _, UsdPhysics, _ = _api()
    fix = UsdPhysics.FixedJoint.Define(stage, path)
    fix.CreateBody1Rel().SetTargets([body_prim.GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    return fix


def revolute(stage, path, body0, body1, at0, at1=None, axis="Z", lower=None, upper=None, stiffness=0.0, damping=0.0, rot=None, armature=None, max_force=None):
    """A revolute joint; at0/at1 are its point in each body's frame (bodies here are all authored world-aligned, so
    at1 = at0 relative to the same origin unless told). rot: (axis xyz, deg) frame rotation for both sides, so that
    the joint's local axis can be any direction."""
    Gf, _, _, UsdPhysics, _ = _api()
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateAxisAttr(axis)
    if lower is not None:  # given in rad; USD revolute limits are in degrees
        j.CreateLowerLimitAttr(math.degrees(float(lower)))
        j.CreateUpperLimitAttr(math.degrees(float(upper)))
    j.CreateBody0Rel().SetTargets([body0.GetPath()])
    j.CreateBody1Rel().SetTargets([body1.GetPath()])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, at0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, at0 if at1 is None else at1)))
    if rot is not None:
        q = Gf.Rotation(Gf.Vec3d(*rot[0]), rot[1]).GetQuat()
        qf = Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*map(float, q.GetImaginary())))
        j.CreateLocalRot0Attr(qf)
        j.CreateLocalRot1Attr(qf)
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(float(stiffness) * math.pi / 180.0)  # given per rad; USD angular drives are per degree
    d.CreateDampingAttr(float(damping) * math.pi / 180.0)
    d.CreateTargetPositionAttr(0.0)
    if max_force is not None:
        d.CreateMaxForceAttr(float(max_force))
    if armature is not None:
        physx(j.GetPrim(), "PhysxJointAPI", **{"physxJoint:armature": float(armature)})
    return j


def prismatic(stage, path, body0, body1, at, axis="X", lower=-1.0, upper=1.0, stiffness=0.0, damping=0.0):
    Gf, _, _, UsdPhysics, _ = _api()
    j = UsdPhysics.PrismaticJoint.Define(stage, path)
    j.CreateAxisAttr(axis)
    j.CreateLowerLimitAttr(float(lower))
    j.CreateUpperLimitAttr(float(upper))
    j.CreateBody0Rel().SetTargets([body0.GetPath()])
    j.CreateBody1Rel().SetTargets([body1.GetPath()])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, at)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, at)))
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(float(stiffness))
    d.CreateDampingAttr(float(damping))
    d.CreateTargetPositionAttr(0.0)
    return j


# -- run time: Coulomb hinges (a capped spring on an anchor that slips) ------------------------------------------------
def coulomb(art, store, joints, holds, k, damping, reset=False):
    """Each joint in ``joints`` gets a spring of stiffness k (N m/rad) toward its anchor, its torque capped at its hold
    (N m); the anchor slips so the spring never stretches past hold / k. So a joint stays where it is under any moment
    below its hold, and turns against its hold above it: a plastic hinge (or Coulomb friction). Written once per control
    step as the PhysX drive targets; between control steps the drive's effort limit caps the torque."""
    import torch
    import warp as wp

    names = art.joint_names
    ids = [names.index(j) for j in joints]
    dev = art.data.joint_pos.torch.device
    q = art.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(float)
    holds = np.broadcast_to(np.asarray(holds, float), q.shape)
    key = id(art)
    st = store.setdefault(key, {})
    if reset or "anchor" not in st or len(st["anchor"]) != len(ids):
        st["anchor"] = q.copy()
        st["targets"] = torch.zeros((1, len(names)), dtype=torch.float32, device=dev)
        st["k"] = None
    if st["k"] != (k, damping):
        art.write_joint_stiffness_to_sim_index(stiffness=torch.full((1, len(ids)), float(k), device=dev), joint_ids=ids)
        art.write_joint_damping_to_sim_index(damping=torch.full((1, len(ids)), float(damping), device=dev), joint_ids=ids)
        st["k"] = (k, damping)
    s = holds / k
    st["anchor"] = np.minimum(np.maximum(st["anchor"], q - s), q + s)
    for n, i in enumerate(ids):
        st["targets"][0, i] = float(st["anchor"][n])
    art.write_joint_effort_limit_to_sim_index(limits=torch.tensor(holds[None], dtype=torch.float32, device=dev), joint_ids=ids)
    t = st["targets"]
    if art.data.has_joint_ordering:
        b2u = torch.as_tensor(np.asarray(art.data.joint_ordering.backend_to_user.numpy()), device=dev).long().reshape(-1)
        t = t[:, b2u]
    art.root_view.set_dof_position_targets(wp.from_torch(t.contiguous()), art._ALL_INDICES)
    return st["anchor"]


def drives(art, store, spec):
    """Write PhysX drive targets for some joints of one articulation, once per control step. spec: {joint: (target,
    cap, stiffness, damping)}; cap is the drive's effort limit (N m), so between control steps the spring's torque never
    passes it. Stiffness and damping are written only when they change."""
    import torch
    import warp as wp

    names = art.joint_names
    dev = art.data.joint_pos.torch.device
    st = store.setdefault(("drives", id(art)), {"targets": torch.zeros((1, len(names)), dtype=torch.float32, device=dev), "kd": {}})
    ids = [names.index(j) for j in spec]
    kd = [(float(v[2]), float(v[3])) for v in spec.values()]
    if [st["kd"].get(i) for i in ids] != kd:
        art.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([[k for k, _ in kd]], device=dev), joint_ids=ids)
        art.write_joint_damping_to_sim_index(damping=torch.tensor([[d for _, d in kd]], device=dev), joint_ids=ids)
        st["kd"].update(zip(ids, kd))
    for i, v in zip(ids, spec.values()):
        st["targets"][0, i] = float(v[0])
    art.write_joint_effort_limit_to_sim_index(limits=torch.tensor([[float(v[1]) for v in spec.values()]], dtype=torch.float32, device=dev), joint_ids=ids)
    t = st["targets"]
    if art.data.has_joint_ordering:
        b2u = torch.as_tensor(np.asarray(art.data.joint_ordering.backend_to_user.numpy()), device=dev).long().reshape(-1)
        t = t[:, b2u]
    art.root_view.set_dof_position_targets(wp.from_torch(t.contiguous()), art._ALL_INDICES)


def slip(anchor, q, hold, k):
    """A Coulomb anchor's new place: it follows q so the spring never stretches past hold / k."""
    s = hold / k
    return min(max(anchor, q - s), q + s)


def push(art, body_name, force, at):
    """A dev hand: a constant force (N, world) at a world point on one body until changed (zero to stop)."""
    import torch

    i = art.body_names.index(body_name)
    dev = art.data.joint_pos.torch.device
    art.permanent_wrench_composer.set_forces_and_torques_index(forces=torch.tensor([[list(map(float, force))]], device=dev),
                                                               torques=torch.zeros((1, 1, 3), device=dev),
                                                               positions=torch.tensor([[list(map(float, at))]], device=dev), body_ids=[i], is_global=True)


# -- S0: a test handle on a pivot (spike only) -------------------------------------------------------------------------
S0 = dict(pivot=(0.26, -0.08), theta0=math.radians(70.0), z=0.040, r0=0.05, r1=0.27, d=0.022, mass=0.25)


def build_s0(stage, path, world_origin):
    """A fixed post and a horizontal handle bar (a 22 mm cushion grip from 5 to 27 cm out) turning about it."""
    root = stage.GetPrimAtPath(path)
    articulation_root(root)
    steel, blue, dark = look(stage, f"{path}/Looks/steel", (0.55, 0.57, 0.6)), look(stage, f"{path}/Looks/blue", (0.15, 0.3, 0.7)), look(stage, f"{path}/Looks/dark", (0.1, 0.1, 0.1))
    grip = surface(stage, f"{path}/Physics/grip", 0.9)
    anchor, handle = body(stage, f"{path}/anchor", 3.0), body(stage, f"{path}/handle", S0["mass"])
    px, py = S0["pivot"]
    z = S0["z"]
    cylinder(stage, f"{path}/anchor/post", (px, py, z / 2), 0.02, z - 0.012, "Z", steel)
    cylinder(stage, f"{path}/handle/hub", (px, py, z), 0.02, 0.02, "Z", blue)
    th = S0["theta0"]
    rm, L = (S0["r0"] + S0["r1"]) / 2, S0["r1"] - S0["r0"]
    cylinder(stage, f"{path}/handle/bar", (px + rm * math.cos(th), py + rm * math.sin(th), z), S0["d"] / 2, L, "X", dark, grip, yaw=th)
    cylinder(stage, f"{path}/handle/arm", (px + 0.035 * math.cos(th), py + 0.035 * math.sin(th), z), 0.008, 0.03, "X", blue, yaw=th)
    fix_to_world(stage, f"{path}/fix", anchor, world_origin)
    revolute(stage, f"{path}/sweep", anchor, handle, (px, py, z), lower=-0.5, upper=2.0)
    return root


# -- the real tool and tube --------------------------------------------------------------------------------------------
# Swagelok MS-HTB-2 hand tube bender (1/8 in OD tube, 9/16 in = 14.3 mm bend radius; manual MS-13-43) clamped by its vise
# clamp block, bend plane horizontal. Its frame sizes are not published: the sizes below are this task's estimates from
# the manual's photos (see the fabrication note), the bend radius and tube size are the maker's.
# Table frame: x from the arm bases' midpoint, y toward the left arm, z up from the table top.
R_BEND = 0.0143  # die groove centreline radius (MS-13-43 p.5: 9/16 in)
TUBE_D = 0.003175  # 1/8 in OD
TUBE_R = TUBE_D / 2
C = (0.26, -0.08)  # the bender die's axis (chosen so every handle pose is in the left arm's reach, CPU IK)
Z_T = 0.075  # the tube's centreline height above the table, in the die groove
Z_H = Z_T - 0.035  # the handles' centreline height: below the tube's plane, clear of fingers pinching the tube
ALPHA = math.radians(70.0)  # where on the die the tube leaves it toward the latch side at the start (the "0")
LONG_YAW = ALPHA - math.pi / 2  # the long handle and the latch side of the tube run this way from the die (-20 deg)
HANDLE = dict(d=0.022, r0=0.04, r1=0.26, grip0=0.16, grip1=0.26)  # cushion grip 16-26 cm from the die axis (estimate)
CLEVIS = (0.022, -0.012)  # the clevis pin on the roll support, (radial, tangential) from the die axis at the start
ROLL = dict(r=0.005, h=0.008, gap=0.0002, ahead=0.022)  # roll dies: radius, height, rest gap on the tube, 2nd die's spacing
CLEVIS_UP = math.radians(-45.0)  # the short handle swings up and away about the clevis at most this far
# The tube latch: a keeper plate beside the tube on the latch side, hinged about a horizontal axis across the tube (along
# v) at u = pivot, so it lifts in its own plane: the tube's outward push is along the hinge axis and can't open it.
# Pressing its tail (beyond the pivot) down lifts the pad; pressing the raised pad end down shuts it.
LATCH = dict(pivot=0.030, pad0=0.012, pad1=0.024, t=0.003, post=(0.016, 0.024, 0.006, 0.014, 0.018), open=math.radians(50.0), seat=math.radians(2.0), loose=math.radians(7.0))
BACKING = 0.030  # the bender die's straight groove on the latch side, from the tangent point
FLANGE = dict(over=0.0006, t=0.003, gap=float(os.environ.get("TUBE_GROOVE_PLAY", "0.0006")))  # groove lips: how far past the tube's centreline, thickness, play
# The tube: 1/8 in OD x 0.030 in wall soft-annealed copper (ASTM B280, C12200), 30 cm, marks 7.07 cm apart (5.0 cm x 1.414).
TUBE_L, MARK_GAP, ZONE = 0.300, 0.0707, 0.025
ZONE_LINKS = 20  # 2.5 mm hinge spacing within each zone
M_P = 0.38  # N m: the tube's plastic moment, Z_p x flow stress (see task.json fidelity)
E_CU = 117e9  # Pa, annealed copper's Young's modulus (handbook)
I_TUBE = math.pi / 64 * (TUBE_D ** 4 - (TUBE_D - 2 * 0.030 * 0.0254) ** 4)  # m^4
K_E = E_CU * I_TUBE / 0.0025  # N m/rad per hinge: the tube's own EI over the 2.5 mm hinge spacing (216); springback follows from it
TUBE_MASS = 0.016
LEG_MASS = 0.015  # kg for a 9 cm straight leg in the simulation (real: 4.8 g); the solver pushes lighter pinched parts through the pads
COLORS = {"steel": (0.55, 0.57, 0.60), "dark": (0.10, 0.10, 0.11), "grip": (0.10, 0.12, 0.35), "die": (0.70, 0.72, 0.75), "copper": (0.85, 0.42, 0.18),
          "mark": (0.05, 0.05, 0.05), "red": (0.85, 0.12, 0.08), "white": (0.95, 0.95, 0.92), "lane": (0.10, 0.55, 0.30), "vise": (0.20, 0.32, 0.25),
          "yellow": (0.95, 0.78, 0.10), "plate": (0.80, 0.80, 0.76)}


def polar(r, ang, z=0.0):
    return (C[0] + r * math.cos(ang), C[1] + r * math.sin(ang), z)


def uv(u, v, z=0.0):
    """A point in the latch side's frame: u from the tangent point along the latch side, v outward from the die."""
    t0 = polar(R_BEND, ALPHA)
    eu = (math.cos(LONG_YAW), math.sin(LONG_YAW))
    ev = (math.cos(ALPHA), math.sin(ALPHA))
    return (t0[0] + u * eu[0] + v * ev[0], t0[1] + u * eu[1] + v * ev[1], z)


def roller_centres(phi=0.0, clevis=0.0):
    """The two roll dies' centres (xy), with the roll support turned phi and the short handle swung clevis about its pin."""
    er, et = ALPHA + phi, ALPHA + phi + math.pi / 2
    rr = R_BEND + TUBE_R + ROLL["r"] + ROLL["gap"]
    pin = np.array(polar(CLEVIS[0], er)[:2]) + CLEVIS[1] * np.array((math.cos(et), math.sin(et)))
    out = []
    for ahead in (0.0, ROLL["ahead"]):
        p = np.array(polar(rr, er)[:2]) + ahead * np.array((math.cos(et), math.sin(et)))
        d = p - pin
        co, si = math.cos(clevis), math.sin(clevis)
        out.append(pin + np.array((co * d[0] - si * d[1], si * d[0] + co * d[1])))
    return out, pin


def grip_point(phi=0.0, clevis=0.0, r=None):
    """The handle grip point (xy) at radius r from the die axis (at the start), after phi and clevis."""
    r = HANDLE["grip0"] + 0.5 * (HANDLE["grip1"] - HANDLE["grip0"]) if r is None else r
    _, pin = roller_centres(phi, 0.0)
    p = np.array(polar(r, ALPHA + phi)[:2])
    d = p - pin
    co, si = math.cos(clevis), math.sin(clevis)
    return pin + np.array((co * d[0] - si * d[1], si * d[0] + co * d[1]))


def build_bender(stage, path, world_origin, lanes):
    """One fixed articulation: anchor (vise, clamp block, die with its groove, the straight groove on the latch side, the
    long handle, the name plate), roll support (turns about the die axis), short handle (on the clevis pin, carries the
    two roll dies), two roll dies (free rollers), tube latch (a keeper on a pin over the latch-side groove).
    lanes: the template's painted rectangles on the table, (x0, y0, x1, y1) each."""
    root = stage.GetPrimAtPath(path)
    articulation_root(root, iters=64)
    m = {k: look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}
    slick = surface(stage, f"{path}/Physics/slick", 0.15)
    grip = surface(stage, f"{path}/Physics/grip", 0.9)
    import os
    die_s = surface(stage, f"{path}/Physics/die", float(os.environ.get("TUBE_DIE_MU", "0.3")))
    B = {n: body(stage, f"{path}/{n}", mass) for n, mass in (("anchor", 5.0), ("support", 0.15), ("handle", 0.25), ("roll_1", 0.01), ("roll_2", 0.01), ("latch", 0.02))}
    a = f"{path}/anchor"
    cx, cy = C
    for i, (x0, y0, x1, y1) in enumerate(lanes):
        cube(stage, f"{a}/lane_{i}", (x0, y0, 0.0), (x1, y1, 0.0006), m["lane"])
    # the vise (a 3 in drill-press vise, drawn only) and the bender's vise clamp block in its jaws
    cube(stage, f"{a}/vise_base", (cx - 0.06, cy - 0.07, 0.0), (cx + 0.06, cy + 0.07, 0.012), m["vise"], slick)
    for s in (-1, 1):
        cube(stage, f"{a}/vise_jaw_{'lr'[s > 0]}", (cx - 0.05, cy + s * 0.012 - 0.008, 0.012), (cx + 0.05, cy + s * 0.012 + 0.008, 0.032), m["vise"], slick, yaw=0.0)
    cube(stage, f"{a}/clamp_block", (cx - 0.02, cy - 0.004, 0.014), (cx + 0.02, cy + 0.004, 0.036), m["steel"])
    cylinder(stage, f"{a}/column", (cx, cy, (0.036 + Z_T - 0.010) / 2), 0.008, Z_T - 0.010 - 0.036, "Z", m["steel"])
    # the bender die: a grooved disc, the groove's centreline at R_BEND
    zf0, zf1 = Z_T - TUBE_R - FLANGE["gap"], Z_T + TUBE_R + FLANGE["gap"]
    cylinder(stage, f"{a}/die_core", (cx, cy, Z_T), R_BEND - TUBE_R - 0.0001, zf1 - zf0, "Z", m["die"], die_s)
    # the groove's lower lip only: the tube lifts straight out once the roll dies are swung away and the latch is open;
    # above the tube, the roll dies' grooves and the latch's lip hold it down
    cylinder(stage, f"{a}/die_lo", (cx, cy, zf0 - FLANGE["t"] / 2), R_BEND + FLANGE["over"], FLANGE["t"], "Z", m["die"], die_s)
    if os.environ.get("TUBE_DIE_HI", "1") == "1":
        cylinder(stage, f"{a}/die_hi", (cx, cy, zf1 + FLANGE["t"] / 2), R_BEND + FLANGE["over"], FLANGE["t"], "Z", m["die"], die_s)  # the groove traps the wrapped arc (a round tube would roll out of an open groove)
    cylinder(stage, f"{a}/die_body", (cx, cy, (Z_T - 0.010 + zf0 - FLANGE["t"]) / 2), 0.020, zf0 - FLANGE["t"] - (Z_T - 0.010), "Z", m["die"])
    # the straight groove on the latch side, from the tangent point (u 0 to BACKING), behind and above/below the tube
    for n, (v0, v1, z0, z1) in (("back", (-0.010, -TUBE_R - 0.0001, zf0 - FLANGE["t"], zf1 + FLANGE["t"])),
                                ("back_lo", (-TUBE_R - 0.0001, FLANGE["over"], zf0 - FLANGE["t"], zf0))):
        mid = uv(BACKING / 2 + 0.002, (v0 + v1) / 2, (z0 + z1) / 2)
        cube(stage, f"{a}/{n}", (mid[0] - BACKING / 2, mid[1] - (v1 - v0) / 2, z0), (mid[0] + BACKING / 2, mid[1] + (v1 - v0) / 2, z1), m["die"], die_s, yaw=LONG_YAW)
    # the long handle (fixed; clamped, never touched) and the name plate with its degree marks
    rm = (HANDLE["r0"] + HANDLE["r1"]) / 2
    lp = polar(rm, LONG_YAW, Z_H)
    cylinder(stage, f"{a}/long_handle", lp, HANDLE["d"] / 2, HANDLE["r1"] - HANDLE["r0"], "X", m["steel"], yaw=LONG_YAW)
    gp = polar(HANDLE["grip0"] + 0.05, LONG_YAW, Z_H)
    cylinder(stage, f"{a}/long_grip", gp, HANDLE["d"] / 2 + 0.001, 0.10, "X", m["grip"], yaw=LONG_YAW)
    zp = Z_T - TUBE_R - FLANGE["gap"] - FLANGE["t"] - 0.001
    cylinder(stage, f"{a}/name_plate", (cx, cy, zp - 0.001), 0.034, 0.002, "Z", m["plate"])
    for deg in range(0, 91, 15):  # the name plate's degree marks: the roll support's 0 reaches these as it turns
        ang = ALPHA + math.radians(deg)
        w = 0.0016 if deg % 45 else 0.003
        p = polar(0.029, ang, zp + 0.0002)
        cube(stage, f"{a}/deg_{deg}", (p[0] - 0.005, p[1] - w / 2, p[2]), (p[0] + 0.005, p[1] + w / 2, p[2] + 0.0003), m["mark"], yaw=ang, about=p)
    # the roll support: a plate under the die, turning about its axis, with its 0 pointer
    s = f"{path}/support"
    cylinder(stage, f"{s}/plate", (cx, cy, Z_H + HANDLE["d"] / 2 + 0.003), 0.030, 0.004, "Z", m["steel"])
    p0 = polar(0.040, ALPHA, zp - 0.0005)
    cube(stage, f"{s}/zero", (p0[0] - 0.008, p0[1] - 0.0015, Z_H + HANDLE["d"] / 2 + 0.005), (p0[0] + 0.008, p0[1] + 0.0015, Z_H + HANDLE["d"] / 2 + 0.0056), m["red"], yaw=ALPHA, about=p0)
    # the short handle, on the clevis pin, carrying the roll dies on axles up to the tube's plane
    h = f"{path}/handle"
    rolls, pin = roller_centres()
    cylinder(stage, f"{h}/clevis", (pin[0], pin[1], Z_H), 0.007, HANDLE["d"], "Z", m["steel"])
    hp = polar(rm, ALPHA, Z_H)
    cylinder(stage, f"{h}/bar", hp, HANDLE["d"] / 2 * 0.6, HANDLE["r1"] - HANDLE["r0"], "X", m["steel"], yaw=ALPHA)
    gp = polar((HANDLE["grip0"] + HANDLE["grip1"]) / 2, ALPHA, Z_H)
    cylinder(stage, f"{h}/grip", gp, HANDLE["d"] / 2, HANDLE["grip1"] - HANDLE["grip0"], "X", m["grip"], grip, yaw=ALPHA)
    for i, rc in enumerate(rolls, 1):
        cylinder(stage, f"{h}/axle_{i}", (rc[0], rc[1], (Z_H + Z_T - ROLL["h"] / 2) / 2), 0.0025, Z_T - ROLL["h"] / 2 - Z_H, "Z", m["steel"])
        cylinder(stage, f"{path}/roll_{i}/die", (rc[0], rc[1], Z_T), ROLL["r"], ROLL["h"], "Z", m["die"], slick)
        if os.environ.get("TUBE_ROLL_LIPS", "1") == "1":
            cylinder(stage, f"{path}/roll_{i}/lip", (rc[0], rc[1], Z_T + TUBE_R + 0.0012 + 0.001), ROLL["r"] + TUBE_R + 0.0008, 0.002, "Z", m["die"], slick)  # the roll die's groove: its upper lip over the tube, 1.2 mm clear (a lip rubbing the tube's top rolls it, d9)
    # the tube latch (see LATCH): pad, bar and tail are one body hinged about v at (pivot, v0, Z_T)
    lt = f"{path}/latch"
    v0 = TUBE_R + 0.0001
    pv = v0 + LATCH["t"] / 2
    pivot = uv(LATCH["pivot"], pv, Z_T)

    def plate(name, u0, u1, va, vb, z0, z1, mat, phys):
        c = uv((u0 + u1) / 2, (va + vb) / 2, (z0 + z1) / 2)
        cube(stage, f"{lt}/{name}", (c[0] - (u1 - u0) / 2, c[1] - (vb - va) / 2, z0), (c[0] + (u1 - u0) / 2, c[1] + (vb - va) / 2, z1), mat, phys, yaw=LONG_YAW)

    plate("pad", LATCH["pad0"], LATCH["pad1"], v0, v0 + LATCH["t"], Z_T - 0.003, Z_T + 0.004, m["steel"], slick)
    plate("lip", LATCH["pad0"], LATCH["pad1"], -0.0008, v0, Z_T + TUBE_R, Z_T + TUBE_R + 0.002, m["steel"], grip)  # rests on the tube's top; its spring clamps the tube
    plate("bar", LATCH["pad1"], LATCH["pivot"] - 0.001, v0, v0 + LATCH["t"], Z_T + 0.0030, Z_T + 0.0065, m["red"], slick)
    pu0, pu1, pv0, pv1, ph = LATCH["post"]  # the keeper's red post: pushed toward +u it lifts the keeper open, back toward -u it shuts it
    plate("arm", pu0, pu1, v0 + LATCH["t"], pv1, Z_T + 0.0030, Z_T + 0.0060, m["red"], slick)
    plate("post", pu0, pu1, pv0, pv1, Z_T + 0.0030, Z_T + ph, m["red"], grip)
    cylinder(stage, f"{lt}/pin", pivot, 0.002, LATCH["t"], "X", m["steel"], yaw=ALPHA)
    fix_to_world(stage, f"{path}/fix", B["anchor"], world_origin)
    revolute(stage, f"{path}/support_turn", B["anchor"], B["support"], (cx, cy, Z_H), lower=math.radians(-2.0), upper=math.radians(100.0), stiffness=5.0, damping=0.01, max_force=0.20)
    revolute(stage, f"{path}/clevis_turn", B["support"], B["handle"], (pin[0], pin[1], Z_H), lower=CLEVIS_UP, upper=0.0, stiffness=1.6, damping=0.01, max_force=0.08)
    for i, rc in enumerate(rolls, 1):
        revolute(stage, f"{path}/roll_{i}_turn", B["handle"], B[f"roll_{i}"], (rc[0], rc[1], Z_T), damping=1e-5)
    revolute(stage, f"{path}/latch_turn", B["anchor"], B["latch"], pivot, axis="X", rot=((0.0, 0.0, 1.0), math.degrees(ALPHA)), lower=-0.06, upper=LATCH["open"] + 0.05, stiffness=1.0, damping=0.002, max_force=0.05)
    return root


def tube_links():
    """The tube's links along its own x (0 at its middle, +x toward mark A's end): (name, x0, x1). Marks at +-MARK_GAP/2;
    each mark has ZONE_LINKS short links within +-ZONE of it; the rest are straight rigid lengths."""
    a, b = MARK_GAP / 2, TUBE_L / 2
    out = [("mid", -a + ZONE, a - ZONE)]
    dl = 2 * ZONE / ZONE_LINKS
    for side, sgn in (("a", 1), ("b", -1)):
        x = a - ZONE
        for i in range(ZONE_LINKS):
            x0, x1 = x + i * dl, x + (i + 1) * dl
            out.append((f"{side}{i:02d}", sgn * x0, sgn * x1) if sgn > 0 else (f"{side}{i:02d}", sgn * x1, sgn * x0))
        out.append((f"leg_{side}", a + ZONE, b) if sgn > 0 else (f"leg_{side}", -b, -(a + ZONE)))
    return out


def tube_joints():
    """(joint name, parent, child, x): each a hinge about the vertical at the links' shared end."""
    out = []
    for side, sgn in (("a", 1), ("b", -1)):
        prev = "mid"
        names = [f"{side}{i:02d}" for i in range(ZONE_LINKS)] + [f"leg_{side}"]
        a = MARK_GAP / 2 - ZONE
        dl = 2 * ZONE / ZONE_LINKS
        for i, n in enumerate(names):
            out.append((f"h{side}{i:02d}", prev, n, sgn * (a + i * dl)))
            prev = n
    return out


def build_tube(stage, path, link_g=None, armature=1e-4):
    """The tube as one floating articulation of round links (capsules), joined by hinges about the vertical; the task
    writes each hinge's plastic anchor every step (coulomb). mid is the root link; all frames are the tube's own."""
    root = stage.GetPrimAtPath(path)
    m = {k: look(stage, f"{path}/Looks/{k}", COLORS[k]) for k in ("copper", "mark", "yellow")}
    phys = surface(stage, f"{path}/Physics/copper", float(os.environ.get("TUBE_MU", "0.8")))
    links = tube_links()
    total = sum(x1 - x0 for _, x0, x1 in links)
    B = {}
    for n, x0, x1 in links:
        mass = TUBE_MASS * (x1 - x0) / total
        B[n] = body(stage, f"{path}/{n}", mass, ang_damp=0.5, lin_damp=0.05, depen=float(os.environ.get("TUBE_DEPEN", "0.5")))
        L = x1 - x0
        capsule(stage, f"{path}/{n}/c", ((x0 + x1) / 2, 0.0, 0.0), TUBE_R, max(L - TUBE_D + 0.0005, 0.0002) if L > TUBE_D else 0.0002, "X", m["copper"], phys)
    articulation_root(B["mid"], iters=64)
    for side in (1, -1):  # the two bend marks: dark rings 2 mm wide, all the way round
        x = side * MARK_GAP / 2
        n = next(n for n, x0, x1 in links if x0 <= x <= x1 and n != "mid")
        cylinder(stage, f"{path}/{n}/mark", (x, 0.0, 0.0), TUBE_R + 0.0002, 0.002, "X", m["mark"])
    cylinder(stage, f"{path}/leg_a/dir", (TUBE_L / 2 - 0.010, 0.0, 0.0), TUBE_R + 0.0002, 0.006, "X", m["yellow"])  # end A's direction mark
    for j, p, c, x in tube_joints():
        revolute(stage, f"{path}/{j}", B[p], B[c], (x, 0.0, 0.0), stiffness=K_E, damping=0.0005, armature=armature, max_force=M_P)
    return root
