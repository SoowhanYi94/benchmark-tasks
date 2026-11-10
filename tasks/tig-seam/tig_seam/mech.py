"""The welding bench, the TIG torch and the filler rod. Geometry constants at the top are NumPy only, so the task file and
the tests can use them without Isaac Sim; the USD builders import pxr when called.

Frames (m):
- Bench: x, y as the world's, z from the station table's top. One kinematic body: a steel fixture slab, the two plates
  (flat colliders; the 45 degree bevels are drawn), strap clamps (drawn), the torch's two rest posts, the rod's rest post
  and tip pad, and the bead pieces, which start hidden and are switched on from the task's note_step.
- Torch: origin at its grip block's centre, +x the way the tungsten leans (the push direction), z up. A short head with
  its axis 17.5 degrees off vertical (tungsten forward), a ceramic cup, a back cap, a horizontal handle out the back with
  the grip block on it, and a stub of power cable behind as the counterweight, so the grip is over the balance point.
- Rod: origin at its grip sleeve's centre, +x the rod's heading in plan. The 3.2 mm rod leaves the sleeve's underside
  sloping 20 degrees down. The fingers close on the sleeve's back and front faces, along the heading, so the rod can't
  pivot about the fingers in pitch.
"""

import math

import numpy as np

# -- bench ---------------------------------------------------------------------------------------------------------
SLAB = dict(x=(0.115, 0.405), y=(-0.135, 0.135), t=0.012)  # steel fixture-table top on the station table
PLATE_T = 0.006
PT = SLAB["t"] + PLATE_T  # the plates' top, above the table
BEVEL_W = 0.002  # each plate's edge is bevelled 45 degrees, 2 mm deep, so the joint shows as a 4 mm V
SEAM = ((0.315, 0.020), (0.195, 0.020), (0.195, -0.060))  # start mark, corner, end mark (plan, m)
SEG_L = 0.005
PLATE_A = ((0.195, -0.090), (0.345, 0.020))  # the inner plate (x0, y0), (x1, y1)
PLATE_B = (((0.145, 0.020), (0.345, 0.070)), ((0.145, -0.090), (0.195, 0.020)))  # the L-shaped plate, as two boxes of one part
CLAMPS = ((0.270, 0.070, "y+"), (0.310, -0.090, "y-"), (0.145, -0.050, "x-"))  # strap clamps: (x, y) of the plate edge, side

# -- torch (torch frame: origin at the grip block's centre; +x the lean) ---------------------------------------------
TILT = math.radians(17.5)  # the head's axis from vertical: a 17.5 degree push angle
STICKOUT = 0.005  # tungsten past the cup
TUNGSTEN_D = 0.0024
CUP = dict(a0=0.005, a1=0.030, r0=0.0070, r1=0.0090)  # along the head's axis from the tungsten tip; radius at each end
HEAD = dict(a0=0.030, a1=0.062, r=0.012)
BACKCAP = dict(a0=0.062, a1=0.072, r=0.008)
HEAD_A = 0.046  # the handle leaves the head at this point of its axis
HANDLE = dict(r=0.011, len=0.110)
GRIP = dict(x=0.030, y=0.036, z=0.026)  # glass-filled nylon grip block on the handle; the fingers close along x
DG = 0.045  # the grip block's centre is this far behind the head's axis
CABLE = dict(r=0.013, len=0.040)
UP = np.array((-math.sin(TILT), 0.0, math.cos(TILT)))  # up the head's axis
HEAD_C_FROM_TIP = HEAD_A * UP
TIP = np.array((DG - HEAD_C_FROM_TIP[0], 0.0, -(HEAD_C_FROM_TIP[2] + HANDLE["r"] + GRIP["z"] / 2)))  # tungsten tip in the torch frame
HEAD_C = TIP + HEAD_C_FROM_TIP  # where the handle meets the head
HANDLE_Z = HEAD_C[2]
TORCH_MASS = {"head": 0.110, "handle": 0.060, "grip": 0.030, "cable": 0.055}
TORCH_TOTAL = sum(TORCH_MASS.values())
_cable_x = HEAD_C[0] - HANDLE["len"] - CABLE["len"] / 2
_head_m = TIP + 0.035 * UP
_com = (TORCH_MASS["head"] * _head_m + TORCH_MASS["handle"] * np.array((HEAD_C[0] - HANDLE["len"] / 2, 0.0, HANDLE_Z))
        + TORCH_MASS["cable"] * np.array((_cable_x, 0.0, HANDLE_Z))) / TORCH_TOTAL
TORCH_COM = _com  # within about 1 mm of under the grip's centre: checked in the self-check
TORCH_INERTIA = (1.5e-4, 7.0e-4, 6.5e-4)
# boxes (low, high) in the torch frame that a hand mustn't be near when it has let go, and the axis-aligned colliders
TORCH_BOXES = {
    "grip": ((-GRIP["x"] / 2, -GRIP["y"] / 2, -GRIP["z"] / 2), (GRIP["x"] / 2, GRIP["y"] / 2, GRIP["z"] / 2)),
    "handle": ((HEAD_C[0] - HANDLE["len"], -HANDLE["r"], HANDLE_Z - HANDLE["r"]), (HEAD_C[0], HANDLE["r"], HANDLE_Z + HANDLE["r"])),
    "cable": ((HEAD_C[0] - HANDLE["len"] - CABLE["len"], -CABLE["r"], HANDLE_Z - CABLE["r"]), (HEAD_C[0] - HANDLE["len"], CABLE["r"], HANDLE_Z + CABLE["r"])),
}
TORCH_HULL = (np.array((_cable_x - CABLE["len"] / 2, -GRIP["y"] / 2, TIP[2])), np.array((TIP[0] + 0.003, GRIP["y"] / 2, GRIP["z"] / 2)))

# -- rod (rod frame: origin at the sleeve's centre; +x its heading) --------------------------------------------------
ROD_SLOPE = math.radians(20.0)
ROD_D = 0.0032  # 3.2 mm (1/8 in) filler, painted yellow
SLEEVE = 0.025  # nylon grip sleeve, a 25 mm cube
ROD_LEN = 0.2125  # sleeve centre to tip, along the rod
ROD_DIR = np.array((math.cos(ROD_SLOPE), 0.0, -math.sin(ROD_SLOPE)))
ROD_DROP = 0.010  # the rod is set this far below the sleeve's centre, so it leaves through the sleeve's underside, clear of the fingers
ROD_TIP = ROD_LEN * ROD_DIR - (0.0, 0.0, ROD_DROP)
ROD_BACK = -0.008 * ROD_DIR - (0.0, 0.0, ROD_DROP)  # the rod's back end, inside the sleeve
ROD_EXIT = (SLEEVE / 2 - ROD_DROP) / math.sin(ROD_SLOPE) * ROD_DIR - (0.0, 0.0, ROD_DROP)  # where it leaves the sleeve's underside
ROD_MASS = {"sleeve": 0.018, "rod": 0.014}
ROD_TOTAL = sum(ROD_MASS.values())
SLEEVE_BOX = ((-SLEEVE / 2,) * 3, (SLEEVE / 2,) * 3)

# -- rests (bench frame) ---------------------------------------------------------------------------------------------
TORCH_REST = dict(x=0.200, y=0.210, lean=math.pi)  # the grip's centre and the lean's heading when resting
TORCH_POST_H = 0.045
TORCH_REST_Z = TORCH_POST_H - TORCH_BOXES["handle"][0][2]  # the grip's centre above the table, handle on the posts
TORCH_POSTS = ((0.020, 0.040), (-0.060, -0.040))  # along the lean from the grip's centre (torch frame x)
ROD_REST = dict(x=0.120, y=-0.250, heading=0.0)  # the sleeve's centre and the rod's heading when resting
ROD_PAD_T = 0.003
ROD_REST_Z = ROD_PAD_T - ROD_TIP[2] + 0.0017  # the sleeve's centre above the table, the tip on its pad
ROD_POST_H = ROD_REST_Z - SLEEVE / 2
ROD_POST_X = (-0.010, 0.003)  # the post under the sleeve, along the heading from its centre: behind the rod's exit

COLORS = {  # sRGB base colour, metallic, roughness, emissive
    "slab": ((0.20, 0.21, 0.22), 0.8, 0.55, None), "hole": ((0.03, 0.03, 0.03), 0.0, 0.9, None),
    "plate": ((0.30, 0.31, 0.33), 0.7, 0.6, None), "bevel": ((0.78, 0.79, 0.80), 1.0, 0.25, None),
    "root": ((0.05, 0.05, 0.05), 0.3, 0.8, None), "white": ((0.95, 0.95, 0.92), 0.0, 0.7, None),
    "yellow": ((0.98, 0.82, 0.05), 0.1, 0.5, None), "clamp": ((0.12, 0.30, 0.62), 0.4, 0.45, None),
    "bolt": ((0.55, 0.56, 0.58), 1.0, 0.35, None), "post": ((0.42, 0.43, 0.45), 0.9, 0.4, None),
    "pad": ((0.10, 0.10, 0.10), 0.0, 0.9, None),
    "cup": ((0.93, 0.72, 0.68), 0.0, 0.65, None), "head": ((0.06, 0.06, 0.07), 0.0, 0.5, None),
    "collet": ((0.80, 0.55, 0.25), 1.0, 0.3, None), "tungsten": ((0.62, 0.62, 0.66), 1.0, 0.3, None),
    "grip": ((0.80, 0.20, 0.08), 0.0, 0.55, None), "cable": ((0.04, 0.04, 0.04), 0.0, 0.7, None),
    "arc": ((0.75, 0.85, 1.0), 0.0, 0.3, (3.0, 4.0, 6.0)), "soot": ((0.18, 0.17, 0.16), 0.0, 0.95, None),
    "rod": ((0.98, 0.84, 0.05), 0.4, 0.4, None), "sleeve": ((0.10, 0.25, 0.55), 0.0, 0.5, None),
    "bead": ((0.74, 0.66, 0.52), 1.0, 0.28, None), "bead_hi": ((0.62, 0.58, 0.66), 1.0, 0.25, None),
    "cold": ((0.42, 0.42, 0.40), 0.6, 0.85, None), "cut": ((0.06, 0.05, 0.05), 0.4, 0.7, None),
    "hole": ((0.0, 0.0, 0.0), 0.0, 1.0, None), "pool": ((1.0, 0.62, 0.15), 0.0, 0.4, (6.0, 2.6, 0.4)),
}
MU = {"steel": 0.5, "grip": 1.0, "rubber": 0.9}


def seam_segments():
    """Each segment's centre (plan), its direction of travel, and its leg (0 along the first leg, 1 along the second)."""
    out = []
    (x0, y0), (xc, yc), (x1, y1) = SEAM
    la, lb = math.hypot(xc - x0, yc - y0), math.hypot(x1 - xc, y1 - yc)
    da, db = np.array((xc - x0, yc - y0)) / la, np.array((x1 - xc, y1 - yc)) / lb
    n = round((la + lb) / SEG_L)
    for i in range(n):
        s = (i + 0.5) * SEG_L
        if s < la:
            out.append((np.array((x0, y0)) + s * da, da, 0))
        else:
            out.append((np.array((xc, yc)) + (s - la) * db, db, 1))
    return out


def seam_point(s):
    """The plan point s metres along the seam from the start mark, and the direction of travel there."""
    (x0, y0), (xc, yc), (x1, y1) = SEAM
    la = math.hypot(xc - x0, yc - y0)
    da = np.array((xc - x0, yc - y0)) / la
    lb = math.hypot(x1 - xc, y1 - yc)
    db = np.array((x1 - xc, y1 - yc)) / lb
    if s <= la:
        return np.array((x0, y0)) + s * da, da
    return np.array((xc, yc)) + (s - la) * db, db


SEAM_LEN = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(SEAM, SEAM[1:]))


def dist_to_seam(p, over=0.005):
    """Plan distance from p to the joint line between the marks, each end extended by ``over``."""
    p = np.asarray(p, float)[:2]
    best = 1e9
    pts = [np.array(q, float) for q in SEAM]
    for k, (a, b) in enumerate(zip(pts, pts[1:])):
        d = b - a
        L = float(np.linalg.norm(d))
        d = d / L
        lo = -over if k == 0 else 0.0
        hi = L + (over if k == len(pts) - 2 else 0.0)
        t = min(max(float(np.dot(p - a, d)), lo), hi)
        best = min(best, float(np.linalg.norm(p - (a + t * d))))
    return best


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
            self.phys[name] = mat
        return self.phys[name]

    def _finish(self, g, look, collide, offset=0.001):
        from pxr import UsdPhysics, UsdShade

        prim = g.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.look(look))
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            _physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": float(offset), "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.surface(collide), UsdShade.Tokens.weakerThanDescendants, "physics")
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

    def box(self, path, lo, hi, look, collide=None, rot_y=0.0, rot_z=0.0, offset=0.001):
        """A box (low, high) before turning; rot_y, rot_z (degrees) turn it about its own centre."""
        from pxr import UsdGeom

        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        self._xform(c, (lo + hi) / 2, rot_y, rot_z, hi - lo)
        return self._finish(c, look, collide, offset)

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

    def frustum(self, path, base, r0, r1, h, look, rot_y=0.0, n=24):
        """A closed cone frustum along its own z from base (radius r0) to h (radius r1), turned rot_y about y at the base."""
        from pxr import Gf, UsdGeom, Vt

        m = UsdGeom.Mesh.Define(self.stage, path)
        pts = [Gf.Vec3f(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z) for z, r in ((0.0, r0), (h, r1)) for k in range(n)]
        pts += [Gf.Vec3f(0, 0, 0), Gf.Vec3f(0, 0, h)]
        counts, idx = [], []
        for k in range(n):
            j = (k + 1) % n
            counts.append(4)
            idx += [k, j, n + j, n + k]
            counts.append(3)
            idx += [2 * n, j, k]
            counts.append(3)
            idx += [2 * n + 1, n + k, n + j]
        m.CreatePointsAttr(Vt.Vec3fArray(pts))
        m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        m.CreateSubdivisionSchemeAttr("none")
        self._xform(m, base, rot_y)
        return self._finish(m, look, None)

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


def _axis_box(b, path, a0, a1, half, look, collide):
    """A square box along the torch head's axis, from a0 to a1 above the tungsten tip."""
    c = TIP + (a0 + a1) / 2 * UP
    return b.box(path, c - (half, half, (a1 - a0) / 2), c + (half, half, (a1 - a0) / 2), look, collide, rot_y=-math.degrees(TILT))


def build_torch(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, mass=TORCH_TOTAL, com=TORCH_COM, inertia=TORCH_INERTIA)
    tilt = -math.degrees(TILT)
    # colliders: grip block, handle, cable stub, head, cup, tungsten
    for name, (lo, hi) in TORCH_BOXES.items():
        b.box(f"{path}/c_{name}", lo, hi, "grip" if name == "grip" else "cable", "grip" if name == "grip" else "rubber")
    stage.GetPrimAtPath(f"{path}/c_handle").GetAttribute("visibility").Set("invisible")
    stage.GetPrimAtPath(f"{path}/c_cable").GetAttribute("visibility").Set("invisible")
    for name, (a0, a1, half) in {"head": (HEAD["a0"], HEAD["a1"], HEAD["r"] * 0.85), "cup": (CUP["a0"], CUP["a1"], CUP["r1"] * 0.85),
                                 "tungsten": (0.0, CUP["a0"] + 0.004, TUNGSTEN_D / 2)}.items():
        g = _axis_box(b, f"{path}/c_{name}", a0, a1, half, "head", "steel")
        g.GetPrim().GetAttribute("visibility").Set("invisible")
    # looks: rubber-covered handle with grip rings, the head, collet body, the ceramic cup, the tungsten, back cap, cable
    hx0, hx1 = HEAD_C[0] - HANDLE["len"], HEAD_C[0]
    b.cyl(f"{path}/v_handle", ((hx0 + hx1) / 2, 0.0, HANDLE_Z), HANDLE["r"], HANDLE["len"], "head", axis="X")
    for k in range(6):
        b.cyl(f"{path}/v_ring{k}", (hx0 + 0.010 + 0.006 * k, 0.0, HANDLE_Z), HANDLE["r"] + 0.0008, 0.003, "cable", axis="X")
    b.cyl(f"{path}/v_cable", (hx0 - CABLE["len"] / 2, 0.0, HANDLE_Z), CABLE["r"], CABLE["len"], "cable", axis="X")
    b.cyl(f"{path}/v_hose", (hx0 - CABLE["len"] - 0.006, 0.0, HANDLE_Z), 0.007, 0.012, "cable", axis="X")
    b.box(f"{path}/v_grip_cap", (-GRIP["x"] / 2 + 0.002, -GRIP["y"] / 2 + 0.002, GRIP["z"] / 2), (GRIP["x"] / 2 - 0.002, GRIP["y"] / 2 - 0.002, GRIP["z"] / 2 + 0.0015), "cable")
    for name, (a0, a1, r, look) in {"v_head": (HEAD["a0"], HEAD["a1"], HEAD["r"], "head"), "v_collet": (CUP["a1"] - 0.002, HEAD["a0"] + 0.004, 0.0075, "collet"),
                                    "v_backcap": (BACKCAP["a0"], BACKCAP["a1"], BACKCAP["r"], "head")}.items():
        b.cyl(f"{path}/{name}", TIP + (a0 + a1) / 2 * UP, r, a1 - a0, look, rot_y=tilt)
    b.ellipsoid(f"{path}/v_backcap_end", TIP + BACKCAP["a1"] * UP, (BACKCAP["r"], BACKCAP["r"], 0.004), "head")
    b.frustum(f"{path}/v_cup", TIP + CUP["a0"] * UP, CUP["r0"], CUP["r1"], CUP["a1"] - CUP["a0"], "cup", rot_y=tilt)
    b.cyl(f"{path}/v_tungsten", TIP + (0.0015 + CUP["a0"] + 0.004) / 2 * UP, TUNGSTEN_D / 2, CUP["a0"] + 0.004 - 0.0015, "tungsten", rot_y=tilt)
    b.frustum(f"{path}/v_tip", TIP, 0.0002, TUNGSTEN_D / 2, 0.0015, "tungsten", rot_y=tilt, n=12)
    # switched from note_step: the arc's glow at the tip, and the soot on a contaminated cup
    b.ellipsoid(f"{path}/fx_arc", TIP - 0.0012 * UP, (0.0028, 0.0028, 0.0028), "arc")
    b.frustum(f"{path}/fx_soot", TIP + (CUP["a0"] - 0.0003) * UP, CUP["r0"] + 0.0006, CUP["r0"] + 0.0012, 0.008, "soot", rot_y=tilt)
    for n in ("fx_arc", "fx_soot"):
        stage.GetPrimAtPath(f"{path}/{n}").GetAttribute("visibility").Set("invisible")
    return prim


def build_rod(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, mass=ROD_TOTAL)
    b.box(f"{path}/c_sleeve", *SLEEVE_BOX, "sleeve", "grip")
    mid = (ROD_BACK + ROD_TIP) / 2
    L = float(np.linalg.norm(ROD_TIP - ROD_BACK))
    deg = math.degrees(ROD_SLOPE)
    g = b.box(f"{path}/c_rod", mid - (L / 2, ROD_D / 2, ROD_D / 2), mid + (L / 2, ROD_D / 2, ROD_D / 2), "rod", "steel", rot_y=deg)
    g.GetPrim().GetAttribute("visibility").Set("invisible")
    b.cyl(f"{path}/v_rod", mid, ROD_D / 2, L, "rod", axis="X", rot_y=deg)
    b.box(f"{path}/v_band", (-SLEEVE / 2 - 0.0005, -SLEEVE / 2 - 0.0005, -0.003), (SLEEVE / 2 + 0.0005, SLEEVE / 2 + 0.0005, 0.003), "white")
    return prim


def build_bench(stage, path, segs):
    """The fixed bench: slab, plates, clamps, rests and the hidden bead pieces (bead_lo_i, bead_hi_i, cold_i, cut_i, hole_i,
    pool_i for each segment)."""
    from pxr import UsdGeom

    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, kinematic=True)
    (sx0, sx1), (sy0, sy1), st = SLAB["x"], SLAB["y"], SLAB["t"]
    b.box(f"{path}/slab", (sx0, sy0, 0.0), (sx1, sy1, st), "slab", "steel")
    for i in range(6):  # the fixture table's hole grid, 50 mm pitch
        for j in range(6):
            x, y = sx0 + 0.02 + 0.05 * i, sy0 + 0.02 + 0.05 * j
            b.cyl(f"{path}/hole_{i}_{j}", (x, y, st + 0.0002), 0.007, 0.0004, "hole")
    pz0, pz1 = st, PT
    w = BEVEL_W

    def plate(name, lo, hi, seam_sides):
        """A plate's flat collider and its looks; seam_sides: the edges on the joint ('x0', 'x1', 'y0', 'y1'), drawn bevelled."""
        (x0, y0), (x1, y1) = lo, hi
        b.box(f"{path}/{name}", (x0, y0, pz0), (x1, y1, pz1 - 0.0002), "plate", "steel")
        for side in seam_sides:
            if side in ("x0", "x1"):
                xe = x0 if side == "x0" else x1
                xs = xe + (w if side == "x0" else -w)
                c = ((xe + xs) / 2, (y0 + y1) / 2, pz1 - w / 2)
                L = y1 - y0
                b.box(f"{path}/{name}_bev_{side}", (c[0] - w * 0.7071, c[1] - L / 2, c[2] - 0.0003), (c[0] + w * 0.7071, c[1] + L / 2, c[2] + 0.0003), "bevel",
                      rot_y=45.0 if side == "x0" else -45.0)
            else:
                ye = y0 if side == "y0" else y1
                ys = ye + (w if side == "y0" else -w)
                c = ((x0 + x1) / 2, (ye + ys) / 2, pz1 - w / 2)
                L = x1 - x0
                from pxr import Gf  # a y-edge bevel: a thin strip turned about x

                g = UsdGeom.Cube.Define(stage, f"{path}/{name}_bev_{side}")
                g.CreateSizeAttr(1.0)
                g.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
                g.AddRotateXOp().Set(-45.0 if side == "y0" else 45.0)
                g.AddScaleOp().Set(Gf.Vec3f(float(L), float(w * 1.414), 0.0006))
                b._finish(g, "bevel", None)

    (ax0, ay0), (ax1, ay1) = PLATE_A
    plate("plate_a", (ax0, ay0), (ax1, ay1), ("x0", "y1"))
    (b1lo, b1hi), (b2lo, b2hi) = PLATE_B
    plate("plate_b1", b1lo, b1hi, ("y0",))
    plate("plate_b2", b2lo, b2hi, ("x1",))
    # the joint's root, a dark line along the seam's full length on the plates
    (x0, y0), (xc, yc), (x1, y1) = SEAM
    b.box(f"{path}/root_a", (PLATE_B[0][0][0] + w, yc - 0.0006, PT - 0.0025), (PLATE_A[1][0], yc + 0.0006, PT - 0.0018), "root")
    b.box(f"{path}/root_b", (xc - 0.0006, PLATE_A[0][1], PT - 0.0025), (xc + 0.0006, yc, PT - 0.0018), "root")
    # paint marks beside the joint: white at the start, yellow at the end (each 30 x 6 mm, facing up)
    b.box(f"{path}/mark_start", (x0 - 0.003, y0 + 0.006, PT), (x0 + 0.003, y0 + 0.036, PT + 0.0004), "white")
    b.box(f"{path}/mark_end", (x1 - 0.036, y1 - 0.003, PT), (x1 - 0.006, y1 + 0.003, PT + 0.0004), "yellow")
    # strap clamps, drawn only: a bar over the plate edge, a stud into the slab and a heel block
    for k, (cx, cy, side) in enumerate(CLAMPS):
        d = {"y+": (0, 1), "y-": (0, -1), "x-": (-1, 0)}[side]
        u = np.array(d, float)
        v = np.array((-u[1], u[0]))
        ctr = np.array((cx, cy)) + 0.020 * u
        lo2, hi2 = ctr - 0.032 * np.abs(u) - 0.010 * np.abs(v), ctr + 0.032 * np.abs(u) + 0.010 * np.abs(v)
        b.box(f"{path}/clamp{k}_bar", (lo2[0], lo2[1], PT + 0.002), (hi2[0], hi2[1], PT + 0.010), "clamp")
        heel = np.array((cx, cy)) + 0.045 * u
        b.box(f"{path}/clamp{k}_heel", (heel[0] - 0.008, heel[1] - 0.008, st), (heel[0] + 0.008, heel[1] + 0.008, PT + 0.002), "clamp")
        stud = np.array((cx, cy)) + 0.022 * u
        b.cyl(f"{path}/clamp{k}_stud", (stud[0], stud[1], st + 0.016), 0.004, 0.032, "bolt")
        b.cyl(f"{path}/clamp{k}_nut", (stud[0], stud[1], PT + 0.013), 0.0075, 0.006, "bolt")
        b.box(f"{path}/clamp{k}_tip", (cx - 0.006 * abs(v[0]) - 0.003 * abs(u[0]) - 0.004 * u[0], cy - 0.006 * abs(v[1]) - 0.003 * abs(u[1]) - 0.004 * u[1], PT),
              (cx + 0.006 * abs(v[0]) + 0.003 * abs(u[0]) - 0.004 * u[0], cy + 0.006 * abs(v[1]) + 0.003 * abs(u[1]) - 0.004 * u[1], PT + 0.003), "clamp")
    # the torch's rest: two steel posts under the handle
    c, s = math.cos(TORCH_REST["lean"]), math.sin(TORCH_REST["lean"])
    for k, (a0, a1) in enumerate(TORCH_POSTS):
        am = (a0 + a1) / 2
        px, py = TORCH_REST["x"] + am * c, TORCH_REST["y"] + am * s
        hx, hy = abs(c) * (a1 - a0) / 2 + abs(s) * 0.018, abs(s) * (a1 - a0) / 2 + abs(c) * 0.018
        b.box(f"{path}/torch_post{k}", (px - hx, py - hy, 0.0), (px + hx, py + hy, TORCH_POST_H), "post", "steel")
    b.box(f"{path}/torch_base", (TORCH_REST["x"] - 0.075, TORCH_REST["y"] - 0.025, 0.0), (TORCH_REST["x"] + 0.055, TORCH_REST["y"] + 0.025, 0.004), "post", "steel")
    # the rod's rest: a post under the sleeve and a rubber pad under the tip
    rx, ry = ROD_REST["x"], ROD_REST["y"]
    b.box(f"{path}/rod_post", (rx + ROD_POST_X[0], ry - 0.016, 0.0), (rx + ROD_POST_X[1], ry + 0.016, ROD_POST_H), "post", "steel")
    hd = ROD_REST["heading"]
    tip = np.array((rx, ry)) + ROD_TIP[0] * np.array((math.cos(hd), math.sin(hd)))
    b.box(f"{path}/rod_pad", (tip[0] - 0.030, tip[1] - 0.015, 0.0), (tip[0] + 0.012, tip[1] + 0.015, ROD_PAD_T), "pad", "rubber")
    # the bead pieces, hidden until note_step shows them
    for i, (p, d, _) in enumerate(segs):
        yaw = math.degrees(math.atan2(d[1], d[0]))
        z = PT
        b.ellipsoid(f"{path}/bead_lo_{i}", (p[0], p[1], z), (0.0036, 0.0031, 0.0010), "bead", rot_z=yaw)
        b.ellipsoid(f"{path}/bead_hi_{i}", (p[0], p[1], z), (0.0040, 0.0036, 0.0019), "bead_hi", rot_z=yaw)
        b.ellipsoid(f"{path}/cold_{i}", (p[0], p[1], z), (0.0030, 0.0024, 0.0016), "cold", rot_z=yaw)
        b.box(f"{path}/cut_{i}", (p[0] - 0.0026, p[1] - 0.0026, z), (p[0] + 0.0026, p[1] + 0.0026, z + 0.0003), "cut", rot_z=yaw)
        b.cyl(f"{path}/hole_s{i}", (p[0], p[1], z + 0.0002), 0.0032, 0.0004, "hole")
        b.ellipsoid(f"{path}/pool_{i}", (p[0], p[1], z + 0.0003), (0.0032, 0.0030, 0.0005), "pool", rot_z=yaw)
        for n in ("bead_lo", "bead_hi", "cold", "cut", "hole_s", "pool"):
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{path}/{n}{'' if n == 'hole_s' else '_'}{i}")).MakeInvisible()
    return prim


def set_visible(stage, path, on):
    from pxr import UsdGeom

    img = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    img.MakeVisible() if on else img.MakeInvisible()
