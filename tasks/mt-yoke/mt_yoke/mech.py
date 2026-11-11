"""The test bench, the AC yoke, the powder bulb and the marker pegs. Geometry constants at the top are NumPy only, so the
task file and the tests can use them without Isaac Sim; the USD builders import pxr when called.

Frames (m):
- Bench: x, y as the world's, z from the station table's top. One kinematic body: the butt-welded test plate (the weld
  cap ground flush, so the weld is a flat band), white marks at the weld's ends, the yoke's rest plate, the bulb's post
  and nozzle pad, the peg tray, and the powder patches and crack indications, which start hidden and are switched on
  from the task's note_step.
- Yoke: origin midway between its two pole faces, on their underside; +x from the handle's pole to the far pole, z up.
  Two square legs, a bridge (the coil housing) across their tops, and the handle on a short neck above the handle's
  pole. The fingers close along x, on the handle's front and back faces.
- Bulb: an articulation. Origin at the grip's centre at rest, +x the nozzle's heading in plan, z up. The root link is
  the bulb's body with its wand; the plunger plate behind it slides on a sprung prismatic joint (`squeeze`). The fingers
  close along x, on the plunger's back and the body's front: closing them squeezes the bulb, with no wrist pitch.
- Peg: origin under its base's centre. A steel magnet base, a shaft and a head, 40 mm tall.
"""

import math

import numpy as np

# -- bench ---------------------------------------------------------------------------------------------------------
PLATE = dict(x=(0.110, 0.360), y=(-0.060, 0.100), t=0.010)  # two 250 x 80 mm plates butt-welded along y = WELD_Y
PZ = PLATE["t"]  # the plate's top, above the table
WELD_Y = 0.020
WELD_X = (0.160, 0.320)  # the examined length, between the white marks
WELD_W = 0.014  # the cap, ground flush
CELL = 0.010
CELLS = np.array([(WELD_X[0] + (k + 0.5) * CELL, WELD_Y) for k in range(round((WELD_X[1] - WELD_X[0]) / CELL))])
# hidden flaws: a longitudinal (centreline) crack and a transverse crack, midpoint on a cell centre, unit direction, length
CRACKS = {
    "long": dict(mid=np.array((0.205, WELD_Y)), dir=np.array((1.0, 0.0)), len=0.020),
    "trans": dict(mid=np.array((0.285, WELD_Y)), dir=np.array((0.0, 1.0)), len=0.018),
}

# -- yoke (yoke frame) -----------------------------------------------------------------------------------------------
POLE_X = 0.050  # the pole faces' centres, at x = -POLE_X (handle pole) and +POLE_X: 100 mm apart
FOOT = 0.020  # square pole faces
LEG_H = 0.060
BRIDGE = ((-0.060, -0.014, LEG_H), (0.060, 0.014, 0.085))
NECK = ((-POLE_X - 0.008, -0.008, 0.085), (-POLE_X + 0.008, 0.008, 0.095))
HANDLE = dict(x=0.030, y=0.026, z=0.030)  # the handle block; the fingers close along x
HANDLE_C = np.array((-POLE_X, 0.0, 0.110))
LAMP = ((0.012, -0.006, 0.085), (0.026, 0.006, 0.089))
YOKE_MASS = {"legs": 0.24, "bridge": 0.20, "handle": 0.11}
YOKE_TOTAL = sum(YOKE_MASS.values())
YOKE_COM = (YOKE_MASS["legs"] * np.array((0.0, 0.0, LEG_H / 2)) + YOKE_MASS["bridge"] * np.array((0.0, 0.0, 0.0725))
            + YOKE_MASS["handle"] * HANDLE_C) / YOKE_TOTAL
YOKE_INERTIA = (6.0e-4, 1.6e-3, 1.2e-3)
FEET = np.array(((-POLE_X, 0.0, 0.0), (POLE_X, 0.0, 0.0)))  # pole face centres, yoke frame
YOKE_BOXES = {
    "leg_h": ((-POLE_X - FOOT / 2, -FOOT / 2, 0.0), (-POLE_X + FOOT / 2, FOOT / 2, LEG_H)),
    "leg_f": ((POLE_X - FOOT / 2, -FOOT / 2, 0.0), (POLE_X + FOOT / 2, FOOT / 2, LEG_H)),
    "bridge": BRIDGE,
    "neck": NECK,
    "handle": (tuple(HANDLE_C - (HANDLE["x"] / 2, HANDLE["y"] / 2, HANDLE["z"] / 2)), tuple(HANDLE_C + (HANDLE["x"] / 2, HANDLE["y"] / 2, HANDLE["z"] / 2))),
}
YOKE_HULL = (np.array((-0.060, -0.014, 0.0)), np.array((0.060, 0.014, HANDLE_C[2] + HANDLE["z"] / 2)))

# -- bulb (bulb frame) -----------------------------------------------------------------------------------------------
BODY = ((-0.010, -0.017, -0.017), (0.022, 0.017, 0.017))  # rubber bulb body (root link)
PLUNGER = ((-0.022, -0.015, -0.015), (-0.017, 0.015, 0.015))  # at joint zero (released)
SQUEEZE_TRAVEL = 0.0065  # the joint's stop: plunger 0.5 mm off the body
SQUEEZE_ON = 0.004  # squeezed this far, powder flows
SPRING_K, SPRING_TARGET, SPRING_D = 300.0, -0.010, 3.0  # N/m, m (a preload: 3 N released, 5 N at the stop), N s/m
WAND_EXIT = np.array((0.014, 0.0, -0.017))
NOZZLE = np.array((0.140, 0.0, -0.072))  # the nozzle's tip
WAND_R = 0.003
BULB_MASS = {"body": 0.045, "wand": 0.012, "plunger": 0.010}
BULB_TOTAL = sum(BULB_MASS.values())
GRIP_W = PLUNGER[0][0] - BODY[1][0]  # negative; the grip's width at rest is -GRIP_W
BULB_GRIP_W = BODY[1][0] - PLUNGER[0][0]  # 44 mm at rest
BULB_HULL = (np.array((PLUNGER[0][0], -0.017, NOZZLE[2] - WAND_R)), np.array((NOZZLE[0] + 0.004, 0.017, 0.017)))

# -- pegs (peg frame) ------------------------------------------------------------------------------------------------
PEG_BASE = dict(w=0.022, h=0.004)
PEG_SHAFT = dict(w=0.012, h=0.030)  # up to this height
PEG_HEAD = dict(w=0.018, h=0.040)  # from the shaft's top to this height
PEG_HEAD_C = np.array((0.0, 0.0, (PEG_SHAFT["h"] + PEG_HEAD["h"]) / 2))
PEG_MASS = 0.025

# -- rests (bench frame) ---------------------------------------------------------------------------------------------
YOKE_REST = dict(x=0.200, y=0.180, z=0.005, heading=-0.5 * math.pi)  # the yoke's frame on its rest plate (5 mm); handle pole at +y
YOKE_REST_PLATE = ((0.175, 0.110), (0.225, 0.250))
BULB_REST = dict(x=0.300, y=-0.250, heading=0.5 * math.pi)
BULB_PAD_T = 0.003
BULB_REST_Z = BULB_PAD_T + WAND_R + 0.0005 - NOZZLE[2]  # the grip centre above the table, the nozzle just over its pad
BULB_POST = ((-0.008, -0.014), (0.010, 0.014))  # under the body, bulb frame (x0, y0), (x1, y1)
BULB_POST_H = BULB_REST_Z + BODY[0][2]
TRAY = dict(x=(0.100, 0.220), y=(-0.220, -0.180), t=0.005)
PEG_HOME = [(0.120, -0.200), (0.160, -0.200), (0.200, -0.200)]

COLORS = {  # sRGB base colour, metallic, roughness, emissive
    "plate": ((0.30, 0.31, 0.33), 0.7, 0.6, None), "weld": ((0.55, 0.56, 0.58), 1.0, 0.35, None), "seam": ((0.12, 0.12, 0.12), 0.3, 0.8, None),
    "white": ((0.95, 0.95, 0.92), 0.0, 0.7, None), "rest": ((0.42, 0.43, 0.45), 0.9, 0.4, None), "pad": ((0.10, 0.10, 0.10), 0.0, 0.9, None),
    "tray": ((0.15, 0.30, 0.20), 0.2, 0.6, None),
    "dust": ((0.92, 0.84, 0.45), 0.0, 0.95, None), "ind": ((0.85, 0.05, 0.03), 0.0, 0.6, (1.2, 0.05, 0.02)),
    "yoke": ((0.10, 0.10, 0.11), 0.3, 0.5, None), "pole": ((0.62, 0.62, 0.64), 1.0, 0.3, None), "coil": ((0.85, 0.45, 0.08), 0.1, 0.5, None),
    "grip": ((0.05, 0.05, 0.05), 0.0, 0.7, None), "cable": ((0.04, 0.04, 0.04), 0.0, 0.7, None),
    "lamp_off": ((0.10, 0.18, 0.10), 0.0, 0.5, None), "lamp_on": ((0.30, 1.0, 0.30), 0.0, 0.3, (2.0, 8.0, 2.0)),
    "bulb": ((0.75, 0.12, 0.10), 0.0, 0.7, None), "plunger": ((0.20, 0.20, 0.22), 0.2, 0.6, None), "wand": ((0.80, 0.62, 0.25), 1.0, 0.3, None),
    "puff": ((0.95, 0.88, 0.50), 0.0, 1.0, None),
    "peg": ((0.10, 0.35, 0.85), 0.0, 0.5, None), "magnet": ((0.08, 0.08, 0.09), 0.6, 0.5, None),
}
MU = {"steel": 0.5, "grip": 1.0, "rubber": 0.9}


def _r2(a):
    return np.array(((math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a))))


# -- USD -------------------------------------------------------------------------------------------------------------
def _physx(prim, schema, **attrs):
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def _lin(c):
    """sRGB to linear, for the shader."""
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


class _Builder:
    def __init__(self, stage, root):
        self.stage, self.root = stage, root
        self.looks, self.phys = {}, {}

    def look(self, name):
        from pxr import Gf, Sdf, UsdShade

        if name not in self.looks:
            color, metal, rough, emit = COLORS[name]
            mat = UsdShade.Material.Define(self.stage, f"{self.root}/Looks/{name}")
            sh = UsdShade.Shader.Define(self.stage, f"{self.root}/Looks/{name}/shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_lin(color)))
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
            if emit is not None:
                sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emit))
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self.looks[name] = mat
        return self.looks[name]

    def surface(self, name):
        from pxr import UsdPhysics, UsdShade

        if name not in self.phys:
            mat = UsdShade.Material.Define(self.stage, f"{self.root}/Physics/{name}")
            pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            pm.CreateStaticFrictionAttr(MU[name])
            pm.CreateDynamicFrictionAttr(MU[name])
            pm.CreateRestitutionAttr(0.0)
            self.phys[name] = mat
        return self.phys[name]

    def _finish(self, g, look, collide, visible=True):
        from pxr import UsdPhysics, UsdShade

        prim = g.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.look(look))
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            _physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.001, "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.surface(collide), UsdShade.Tokens.weakerThanDescendants, "physics")
        if not visible:
            prim.GetAttribute("visibility").Set("invisible")
        return g

    @staticmethod
    def _xform(g, center, rot_y=0.0, rot_z=0.0, scale=None):
        from pxr import Gf

        g.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        if rot_z:
            g.AddRotateZOp().Set(float(rot_z))
        if rot_y:
            g.AddRotateYOp().Set(float(rot_y))
        if scale is not None:
            g.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))

    def box(self, path, lo, hi, look, collide=None, rot_y=0.0, rot_z=0.0, visible=True):
        """A box (low, high) before turning; rot_y, rot_z (degrees) turn it about its own centre."""
        from pxr import UsdGeom

        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        self._xform(c, (lo + hi) / 2, rot_y, rot_z, hi - lo)
        return self._finish(c, look, collide, visible)

    def cyl(self, path, center, r, h, look, axis="Z", rot_y=0.0, rot_z=0.0):
        from pxr import UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(float(h))
        self._xform(c, center, rot_y, rot_z)
        return self._finish(c, look, None)

    def ellipsoid(self, path, center, radii, look, rot_z=0.0):
        from pxr import UsdGeom

        s = UsdGeom.Sphere.Define(self.stage, path)
        s.CreateRadiusAttr(1.0)
        self._xform(s, center, 0.0, rot_z, radii)
        return self._finish(s, look, None)

    def rigid(self, prim, kinematic=False, mass=None, com=None, inertia=None, iters=32):
        from pxr import Gf, UsdPhysics

        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
        _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": iters, "physxRigidBody:solverVelocityIterationCount": 2,
                                             "physxRigidBody:maxDepenetrationVelocity": 0.2, "physxRigidBody:enableCCD": True})
        if mass is not None:
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr(float(mass))
            if com is not None:
                m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
            if inertia is not None:
                m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))


def build_yoke(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, mass=YOKE_TOTAL, com=YOKE_COM, inertia=YOKE_INERTIA)
    for name, (lo, hi) in YOKE_BOXES.items():
        b.box(f"{path}/c_{name}", lo, hi, "grip" if name == "handle" else "yoke", "grip" if name == "handle" else "steel")
    for k, x in enumerate((-POLE_X, POLE_X)):  # bright pole tips, the coil on the bridge
        b.box(f"{path}/v_pole{k}", (x - FOOT / 2 - 0.0005, -FOOT / 2 - 0.0005, 0.0), (x + FOOT / 2 + 0.0005, FOOT / 2 + 0.0005, 0.012), "pole")
    b.box(f"{path}/v_coil", (-0.030, -0.0145, 0.062), (0.008, 0.0145, 0.083), "coil")
    b.cyl(f"{path}/v_cable", (-POLE_X, 0.0, 0.120), 0.006, 0.012, "cable", axis="Y")  # the cable's gland, on the handle's side
    b.box(f"{path}/v_lamp", *LAMP, "lamp_off")
    lo, hi = np.array(LAMP[0]), np.array(LAMP[1])
    b.box(f"{path}/fx_lamp", lo - (0.001, 0.001, 0.0), hi + (0.001, 0.001, 0.002), "lamp_on", visible=False)
    return prim


def build_bulb(stage, path):
    from pxr import Gf, UsdGeom, UsdPhysics

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    _physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": False, "physxArticulation:solverPositionIterationCount": 32,
                                            "physxArticulation:solverVelocityIterationCount": 4})
    body = UsdGeom.Xform.Define(stage, f"{path}/body").GetPrim()
    wand_m = (WAND_EXIT + NOZZLE) / 2
    com = (BULB_MASS["body"] * (np.array(BODY[0]) + BODY[1]) / 2 + BULB_MASS["wand"] * wand_m) / (BULB_MASS["body"] + BULB_MASS["wand"])
    b.rigid(body, mass=BULB_MASS["body"] + BULB_MASS["wand"], com=com, inertia=(1.5e-5, 6.0e-5, 6.0e-5))
    plunger = UsdGeom.Xform.Define(stage, f"{path}/plunger").GetPrim()
    b.rigid(plunger, mass=BULB_MASS["plunger"], inertia=(2.0e-6, 1.5e-6, 1.5e-6))
    p = f"{path}/body"
    b.box(f"{p}/c_body", *BODY, "bulb", "rubber")
    d = NOZZLE - WAND_EXIT
    L = float(np.linalg.norm(d))
    slope = math.degrees(math.atan2(-d[2], d[0]))
    b.box(f"{p}/c_wand", wand_m - (L / 2, WAND_R, WAND_R), wand_m + (L / 2, WAND_R, WAND_R), "wand", "steel", rot_y=slope, visible=False)
    b.cyl(f"{p}/v_wand", wand_m, WAND_R, L, "wand", axis="X", rot_y=slope)
    b.cyl(f"{p}/v_nozzle", NOZZLE - 0.004 * d / L, WAND_R * 0.6, 0.010, "plunger", axis="X", rot_y=slope)
    b.ellipsoid(f"{p}/fx_puff", NOZZLE + (0.006, 0.0, -0.010), (0.012, 0.012, 0.008), "puff")
    stage.GetPrimAtPath(f"{p}/fx_puff").GetAttribute("visibility").Set("invisible")
    q = f"{path}/plunger"
    b.box(f"{q}/c_plunger", *PLUNGER, "plunger", "grip")
    b.box(f"{q}/v_bellows", (PLUNGER[1][0], -0.012, -0.012), (BODY[0][0] + 0.004, 0.012, 0.012), "bulb")
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/squeeze")
    j.CreateAxisAttr("X")
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(SQUEEZE_TRAVEL))
    j.CreateBody0Rel().SetTargets([body.GetPath()])
    j.CreateBody1Rel().SetTargets([plunger.GetPath()])
    at = Gf.Vec3f(float(PLUNGER[1][0]), 0.0, 0.0)
    j.CreateLocalPos0Attr(at)
    j.CreateLocalPos1Attr(at)
    dr = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    dr.CreateTypeAttr("force")
    dr.CreateStiffnessAttr(float(SPRING_K))
    dr.CreateDampingAttr(float(SPRING_D))
    dr.CreateTargetPositionAttr(float(SPRING_TARGET))
    dr.CreateMaxForceAttr(20.0)
    return prim


def build_peg(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, mass=PEG_MASS, com=(0.0, 0.0, 0.012), inertia=(3.0e-6, 3.0e-6, 1.5e-6))
    bw, sw, hw = PEG_BASE["w"] / 2, PEG_SHAFT["w"] / 2, PEG_HEAD["w"] / 2
    b.box(f"{path}/c_base", (-bw, -bw, 0.0), (bw, bw, PEG_BASE["h"]), "magnet", "steel")
    b.box(f"{path}/c_shaft", (-sw, -sw, PEG_BASE["h"]), (sw, sw, PEG_SHAFT["h"]), "peg", "grip")
    b.box(f"{path}/c_head", (-hw, -hw, PEG_SHAFT["h"]), (hw, hw, PEG_HEAD["h"]), "peg", "grip")
    return prim


def build_bench(stage, path):
    """The fixed bench: plate, weld, marks, rests, tray, and the hidden powder patches (dust_i) and indications (ind_<crack>)."""
    from pxr import UsdGeom

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, kinematic=True)
    (x0, x1), (y0, y1), t = PLATE["x"], PLATE["y"], PLATE["t"]
    b.box(f"{path}/plate_a", (x0, y0, 0.0), (x1, WELD_Y, t), "plate", "steel")
    b.box(f"{path}/plate_b", (x0, WELD_Y, 0.0), (x1, y1, t), "plate", "steel")
    b.box(f"{path}/weld", (x0, WELD_Y - WELD_W / 2, t), (x1, WELD_Y + WELD_W / 2, t + 0.0003), "weld")
    for k, xe in enumerate(WELD_X):  # white paint marks across the plate beside the weld's ends
        for s in (-1, 1):
            ya = WELD_Y + s * (WELD_W / 2 + 0.004)
            yb = WELD_Y + s * (WELD_W / 2 + 0.024)
            b.box(f"{path}/mark{k}_{'ab'[s > 0]}", (xe - 0.002, min(ya, yb), t), (xe + 0.002, max(ya, yb), t + 0.0004), "white")
    (rx0, ry0), (rx1, ry1) = YOKE_REST_PLATE
    b.box(f"{path}/yoke_rest", (rx0, ry0, 0.0), (rx1, ry1, YOKE_REST["z"]), "rest", "steel")
    h = YOKE_REST["heading"]
    for k, s in enumerate((-1, 1)):  # painted outlines for the pole faces
        cx, cy = YOKE_REST["x"] + s * POLE_X * math.cos(h), YOKE_REST["y"] + s * POLE_X * math.sin(h)
        b.box(f"{path}/yoke_rest_mark{k}", (cx - 0.013, cy - 0.013, YOKE_REST["z"]), (cx + 0.013, cy + 0.013, YOKE_REST["z"] + 0.0003), "white")
    bx, by, bh = BULB_REST["x"], BULB_REST["y"], BULB_REST["heading"]
    (px0, py0), (px1, py1) = BULB_POST
    c = np.array((bx, by)) + _r2(bh) @ np.array(((px0 + px1) / 2, (py0 + py1) / 2))
    b.box(f"{path}/bulb_post", (c[0] - (px1 - px0) / 2, c[1] - (py1 - py0) / 2, 0.0), (c[0] + (px1 - px0) / 2, c[1] + (py1 - py0) / 2, BULB_POST_H), "rest", "steel",
          rot_z=math.degrees(bh))
    n = np.array((bx, by)) + _r2(bh) @ np.array((NOZZLE[0] - 0.0065, 0.0))
    b.box(f"{path}/bulb_pad", (n[0] - 0.0185, n[1] - 0.015, 0.0), (n[0] + 0.0185, n[1] + 0.015, BULB_PAD_T), "pad", "rubber", rot_z=math.degrees(bh))
    (tx0, tx1), (ty0, ty1) = TRAY["x"], TRAY["y"]
    b.box(f"{path}/tray", (tx0, ty0, 0.0), (tx1, ty1, TRAY["t"]), "tray", "steel")
    # hidden until note_step shows them: a powder patch over each cell, and each crack's indication
    for i, (cx, cy) in enumerate(CELLS):
        g = b.box(f"{path}/dust_{i}", (cx - CELL / 2, cy - 0.022, t), (cx + CELL / 2, cy + 0.022, t + 0.0006), "dust")
        UsdGeom.Imageable(g.GetPrim()).MakeInvisible()
    for name, c in CRACKS.items():
        L = c["len"] + 0.004
        yaw = math.degrees(math.atan2(c["dir"][1], c["dir"][0]))
        m = c["mid"]
        g = b.box(f"{path}/ind_{name}", (m[0] - L / 2, m[1] - 0.002, t + 0.0006), (m[0] + L / 2, m[1] + 0.002, t + 0.0012), "ind", rot_z=yaw)
        UsdGeom.Imageable(g.GetPrim()).MakeInvisible()
    return prim
