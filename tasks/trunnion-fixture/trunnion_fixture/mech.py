"""Builds the scene's bodies into a USD stage.

The rollover fixture is one articulation: a base frame with two pillars, a cradle on a horizontal axle along y, a retaining bar sliding
along y on the cradle, a free-turning knob sleeve on the crank's pin, and a lock bolt sliding along y in the far pillar. The bolt enters a
notch in the cradle's far end plate by real contact, only when the cradle is turned over. The driver's holster and the flange's kitting post
are fixed to the base. All link frames coincide at the start (a point on the table under the axle).
(u, v, w) in comments: the cradle's own axes from the axle; they are x, y, z when the cradle is level.

Free bodies: the worm gear reducer (gearbox), its output flange, four socket head screws held in the flange, and a stubby screwdriver.
Each screw is glued (a fixed joint, standing in for the kitting step that pre-starts it) to the flange until its thread catches; then the glue is
switched off and a thread joint to the gearbox is switched on (a D6 joint: on the hole's axis, sliding along it to a target the task sets).
Every size is in metres. Pure Python at import time; pxr is imported only when a stage is built.
"""

import math

import numpy as np

# -- the fixture (cradle axes from the axle) ---------------------------------------------------------------------------
AXLE = 0.078  # above the table
GB_UV = (-0.0085, -0.0075)  # the gearbox's output axis in the cradle's (u, v): its envelope is centred on the axle
RAIL = dict(inner=0.050, wall=0.006, v0=-0.100, v1=0.050, w0=-0.034, w1=0.0435)
LEDGES = ((-0.050, -0.0465), (0.0295, 0.050))  # u ranges; under the gearbox's foot and its worm-side body, clear of the flange's square
LEDGE = dict(v0=-0.040, v1=0.025, w0=-0.034, w1=-0.029)  # the gearbox's output face J rests on w1
STOP = dict(v0=-0.0575, v1=-0.050, w1=0.033)  # the near stop plate (the bar passes over it)
PLATE = dict(half_u=0.050, half_w=0.045, v0=0.050, v1=0.060, notch_u=0.008, notch_w=-0.015)  # the far end plate; the bolt's notch toward -w
NEAR_PLATE = dict(v0=-0.106, v1=-0.100)
CRANK = dict(r=0.060, y=-0.152, half_x=0.015, half_y=0.012, half_z=0.020, v0=-0.128, v1=-0.122)  # the knob sleeve: its centre at u=-r, v=y
BAR = dict(half_u=0.0495, half_v=0.015, w0=0.0355, w1=0.0435, park=-0.075, travel=0.0675, lug_u=0.02, lug_v=0.008, lug_w=0.075)
PILLAR = dict(half_x=0.03, near=(-0.120, -0.108), far=(0.068, 0.080), near_top=0.090, far_top=0.125)
BOLT = dict(half=0.006, z=AXLE + 0.030, tip=0.066, end=0.122, travel=0.015, lug_half_x=0.02, lug_y=0.008, lug_top=0.150)
BASE = dict(h=0.006, near=(-0.126, -0.102), far=(0.062, 0.086), half_x=0.045, tie_u=(0.080, 0.090))
HOLSTER = dict(x=0.335, y=-0.20, half=0.02, h=0.040, hole=0.006)  # world x, y: a block with a 12 mm square bore
POST = dict(x=0.15, y=0.15, r=0.025, h=0.060)  # world x, y: the flange's kitting post
DRAWN_ONLY = ("axle_far", "axle_near", "pin")  # the axle stubs run in bores in the pillars; the pin in the knob's bore
ROLL_LIMIT = 195.0  # degrees: the cradle's far stop lies past turned over, so only the bolt holds it there
MASS = {"anchor": 2.0, "cradle": 0.7, "bar": 0.11, "bolt": 0.07, "knob": 0.03}
CRADLE_COM = (-0.004, -0.02, AXLE - 0.008)  # level: slightly toward the crank side, so it rests on its near stop
ROLL_DAMP, SLIDE_DAMP, KNOB_DAMP = 0.002, 5.0, 0.00002  # N m per deg/s, N per m/s, N m per deg/s
ITERS, VEL_ITERS, MU = 32, 4, 0.3
COLORS = {"steel": (0.22, 0.23, 0.25), "dark": (0.12, 0.13, 0.14), "cradle": (0.16, 0.17, 0.19), "bar": (0.10, 0.30, 0.75), "bolt": (0.85, 0.12, 0.08),
          "knob": (0.95, 0.75, 0.10), "gearbox": (0.42, 0.44, 0.47), "rib": (0.34, 0.36, 0.39), "brass": (0.75, 0.60, 0.25), "ring": (0.95, 0.45, 0.05), "flange": (0.86, 0.87, 0.89), "screw": (0.08, 0.08, 0.09),
          "handle": (0.15, 0.55, 0.20), "blade": (0.70, 0.72, 0.75), "holster": (0.25, 0.25, 0.27)}

# -- the gearbox: Motovario NMRV 030 (catalogue: I 30, G1 63, N1 29, N 55 h8, H 40, H1 40, G 55, R 57, A 54, KE 4 x M6 x 11) ----------------
# Its own frame: origin on the output axis midway between the output faces; z along the output axis, face J at -z; x from the foot (-40) to
# the top (+57); y along the worm, the input flange at +y (G 55), the end cover at -y (H1 40). Shapes are ours, from those dimensions.
GB = dict(mass=1.2, com=(0.006, 0.004, 0.0), face=0.029, spigot_r=0.0275, spigot_h=0.0025, hole_r=0.0325, hole_half=0.004, hole_depth=0.011)
HOLES = [(GB["hole_r"] * math.cos(a), GB["hole_r"] * math.sin(a)) for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4)]
# -- the output flange (FA, size 030: KQ 70, KB 6, KN 50, KC 4, KA 54.5 from the output axis) and its screws -----------------------------
# Its frame: origin at the centre of its housing face, z away from the housing. Recess for the housing's spigot: ours, 0.5 mm loose a side.
FL = dict(mass=0.13, half=0.035, plate=0.006, recess=0.003, recess_r=0.028, chamfer=0.002, chamfer_d=0.001, hole_half=0.0042, neck_r=0.027, top=0.0255, spig_r=0.025, spig_h=0.004)
SC = dict(mass=0.006, length=0.016, shank_r=0.003, head_r=0.005, head_h=0.006, socket_r=0.0034, socket_d=0.00275, proud=0.001)
SCREW_START_Z = SC["length"] - SC["proud"]  # 15 mm: the tip 1 mm proud of the flange's housing face (it rests in a tapped hole's mouth, no thread engaged)
# -- the screwdriver: Wera 367 TORX Stubby TX 30 x 25 mm, 05008861001 (maker: blade 25, handle 54, blade 5.6 mm; 81 mm overall, handle 34 mm at
#    its widest, 37 g as resellers list the maker's data). Collision: a 29 mm cylinder over the handle (the grip zone), our simplification.
DRV = dict(mass=0.037, blade_r=0.0028, blade=0.025, handle_r=0.0145, handle=0.054)


def _api():
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    return Gf, Sdf, UsdGeom, UsdPhysics, UsdShade


def physx(prim, schema, **attrs):
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def look(stage, path, color):
    Gf, Sdf, _, _, UsdShade = _api()
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(c) ** 2.2 for c in color)))  # colours are given as seen (sRGB)
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


def _collide(prim, phys):
    _, _, _, UsdPhysics, UsdShade = _api()
    UsdPhysics.CollisionAPI.Apply(prim)
    physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.001, "physxCollision:restOffset": 0.0})
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def cube(stage, path, lo, hi, mat, phys=None):
    Gf, _, UsdGeom, UsdPhysics, UsdShade = _api()
    lo, hi = np.array(lo, float), np.array(hi, float)
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
    c.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if phys is not None:
        _collide(c.GetPrim(), phys)
    return c


def hull(stage, path, pts, mat, phys=None):
    """A convex body through these points (its own convex hull: the collider and the drawn faces)."""
    from scipy.spatial import ConvexHull

    Gf, _, UsdGeom, UsdPhysics, UsdShade = _api()
    pts = np.asarray(pts, float)
    h = ConvexHull(pts)
    tris = []
    c = pts.mean(axis=0)
    for s in h.simplices:
        a, b, d = pts[s]
        if np.dot(np.cross(b - a, d - a), a - c) < 0:  # outward winding
            s = (s[0], s[2], s[1])
        tris.extend(int(i) for i in s)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    m.CreateFaceVertexCountsAttr([3] * (len(tris) // 3))
    m.CreateFaceVertexIndicesAttr(tris)
    m.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    if phys is not None:
        _collide(m.GetPrim(), phys)
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    return m


def ring(r, z, n=24, c=(0.0, 0.0), a0=0.0, a1=2 * math.pi, axis="z"):
    """Points on a circle (or arc) of radius r: about z at height z through (c), or about y (axis="y": c = (x, z), z = y)."""
    out = []
    for k in range(n + (0 if a1 - a0 >= 2 * math.pi - 1e-9 else 1)):
        a = a0 + (a1 - a0) * k / n
        if axis == "z":
            out.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a), z))
        else:
            out.append((c[0] + r * math.cos(a), z, c[1] + r * math.sin(a)))
    return out


def rigid(prim, mass, com=None, iters=ITERS):
    Gf, _, _, UsdPhysics, _ = _api()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": iters, "physxRigidBody:solverVelocityIterationCount": VEL_ITERS,
                                        "physxRigidBody:maxDepenetrationVelocity": 0.5})
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(mass)
    if com is not None:
        api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))


# -- shape lists (pure Python: the drawing script and the tests use them too) ----------------------------------------------------------------
def fixture_boxes():
    """Every box of the fixture at the start (cradle level): (body, name, lo, hi, colour), in the fixture's frame (x = u, y = v, z above
    the table). The holster and the kitting post are given in the world (they are added with the fixture's offset removed)."""
    A = AXLE
    out = []

    def add(body, name, lo, hi, col):
        out.append((body, name, tuple(map(float, lo)), tuple(map(float, hi)), col))

    def cr(name, u0, u1, v0, v1, w0, w1, col="cradle"):
        add("cradle", name, (min(u0, u1), v0, A + w0), (max(u0, u1), v1, A + w1), col)

    b, pl = BASE, PILLAR
    add("anchor", "base_near", (-b["half_x"], b["near"][0], 0.0), (b["half_x"], b["near"][1], b["h"]), "dark")
    add("anchor", "base_far", (-b["half_x"], b["far"][0], 0.0), (b["half_x"], b["far"][1], b["h"]), "dark")
    for s in (1, -1):
        add("anchor", f"tie_{'p' if s > 0 else 'm'}", (min(s * b["tie_u"][0], s * b["tie_u"][1]), b["near"][0], 0.0),
            (max(s * b["tie_u"][0], s * b["tie_u"][1]), b["far"][1], b["h"]), "dark")
    add("anchor", "near", (-pl["half_x"], pl["near"][0], b["h"]), (pl["half_x"], pl["near"][1], pl["near_top"]), "steel")
    add("anchor", "far", (-pl["half_x"], pl["far"][0], b["h"]), (pl["half_x"], pl["far"][1], BOLT["z"] - BOLT["half"]), "steel")
    add("anchor", "far_top", (-pl["half_x"], pl["far"][0], BOLT["z"] + BOLT["half"]), (pl["half_x"], pl["far"][1], pl["far_top"]), "steel")
    r = RAIL
    for s, n in ((1, "p"), (-1, "m")):
        cr(f"rail_{n}", s * r["inner"], s * (r["inner"] + r["wall"]), r["v0"], r["v1"], r["w0"], r["w1"])
    for k, (u0, u1) in enumerate(LEDGES):
        cr(f"ledge_{k}", u0, u1, LEDGE["v0"], LEDGE["v1"], LEDGE["w0"], LEDGE["w1"])
    cr("stop", -r["inner"], r["inner"], STOP["v0"], STOP["v1"], r["w0"], STOP["w1"])
    cr("near_plate", -(r["inner"] + r["wall"]), r["inner"] + r["wall"], NEAR_PLATE["v0"], NEAR_PLATE["v1"], r["w0"], r["w1"])
    pa = PLATE
    cr("plate_top", -pa["half_u"], pa["half_u"], pa["v0"], pa["v1"], pa["notch_w"], pa["half_w"])
    for s, n in ((1, "p"), (-1, "m")):
        cr(f"plate_{n}", s * pa["notch_u"], s * pa["half_u"], pa["v0"], pa["v1"], -pa["half_w"], pa["notch_w"])
    cr("axle_far", -0.008, 0.008, pa["v1"], pl["far"][0], -0.008, 0.008, "dark")
    cr("axle_near", -0.008, 0.008, pl["near"][1], NEAR_PLATE["v0"], -0.008, 0.008, "dark")
    ck = CRANK
    cr("crank", -ck["r"] - 0.008, 0.008, ck["v0"], ck["v1"], -0.008, 0.008, "dark")
    cr("pin", -ck["r"] - 0.004, -ck["r"] + 0.004, ck["y"] + ck["half_y"], ck["v0"], -0.004, 0.004, "dark")
    add("knob", "sleeve", (-ck["r"] - ck["half_x"], ck["y"] - ck["half_y"], A - ck["half_z"]), (-ck["r"] + ck["half_x"], ck["y"] + ck["half_y"], A + ck["half_z"]), "knob")
    br = BAR
    add("bar", "plate", (-br["half_u"], br["park"] - br["half_v"], A + br["w0"]), (br["half_u"], br["park"] + br["half_v"], A + br["w1"]), "bar")
    add("bar", "lug", (-br["lug_u"], br["park"] - br["half_v"] - br["lug_v"], A + br["w0"]), (br["lug_u"], br["park"] - br["half_v"], A + br["lug_w"]), "bar")
    bo = BOLT
    add("bolt", "rod", (-bo["half"], bo["tip"], bo["z"] - bo["half"]), (bo["half"], bo["end"], bo["z"] + bo["half"]), "bolt")
    add("bolt", "lug", (-bo["lug_half_x"], bo["end"] - bo["lug_y"], bo["z"] - bo["half"]), (bo["lug_half_x"], bo["end"], bo["lug_top"]), "bolt")
    return out


def station_boxes():
    """The holster (a block with a square 12 mm bore, 38 mm deep) in the world's x, y and height above the table."""
    h = HOLSTER
    x, y, s, H, q = h["x"], h["y"], h["half"], h["h"], h["hole"]
    return [("holster_floor", (x - s, y - s, 0.0), (x + s, y + s, 0.002)),
            ("holster_a", (x - s, y - s, 0.0), (x - q, y + s, H)), ("holster_b", (x + q, y - s, 0.0), (x + s, y + s, H)),
            ("holster_c", (x - q, y - s, 0.0), (x + q, y - q, H)), ("holster_d", (x - q, y + q, 0.0), (x + q, y + s, H))]


def gearbox_boxes():
    """The gearbox's box shapes in its own frame: (name, lo, hi, colour)."""
    f, hh, d = GB["face"], GB["hole_half"], GB["hole_depth"]
    c = 0.023  # the KE holes on the 65 mm circle at 45 degrees: centres at (+-23, +-23) mm
    zs = (-f, -f + d)
    out = [("body", (-0.027, -0.027, -f + d), (0.044, 0.027, f), "gearbox"), ("feet", (-0.040, -0.028, -f), (-0.027, 0.028, f), "gearbox"),
           ("inflange", (0.0, 0.045, -0.030), (0.060, 0.055, 0.030), "gearbox")]
    xs = [(-0.030, -c - hh), (-c + hh, c - hh), (c + hh, 0.044)]
    ys = [(-0.030, -c - hh), (-c - hh, -c + hh), (-c + hh, c - hh), (c - hh, c + hh), (c + hh, 0.030)]
    for j, (y0, y1) in enumerate(ys):
        if j in (1, 3):  # the strips through the holes
            for i, (x0, x1) in enumerate(xs):
                out.append((f"slab_{j}{i}", (x0, y0, zs[0]), (x1, y1, zs[1]), "gearbox"))
        else:
            out.append((f"slab_{j}", (-0.030, y0, zs[0]), (0.044, y1, zs[1]), "gearbox"))
    return out


def _clip(poly, n, c):
    """Keep the part of a convex polygon where n . p >= c (Sutherland-Hodgman, one half-plane)."""
    out = []
    for k in range(len(poly)):
        p, q = np.asarray(poly[k]), np.asarray(poly[(k + 1) % len(poly)])
        dp, dq = np.dot(n, p) - c, np.dot(n, q) - c
        if dp >= 0:
            out.append(p)
        if dp * dq < 0:
            out.append(p + (q - p) * dp / (dp - dq))
    return out


def _poly(halfplanes, H):
    poly = [np.array(v, float) for v in ((-H, -H), (H, -H), (H, H), (-H, H))]
    for n, c in halfplanes:
        poly = _clip(poly, np.asarray(n, float), c)
    return poly


def flange_pieces():
    """The flange's plate as convex prisms (flange frame): the housing-side layer (z 0 to the recess depth) round a 16-sided recess with a
    chamfered mouth, and the upper layer, both with a square clearance hole (7.6 mm) for each screw. Each piece: a list of 3-D points."""
    H, R, ch, dz, hh, top = FL["half"], FL["recess_r"], FL["chamfer"], FL["recess"], FL["hole_half"], FL["plate"]
    c = 0.023
    out = []
    n, half = 16, math.pi / 16
    for k in range(n):
        a = k * 2 * math.pi / n
        e = np.array((math.cos(a), math.sin(a)))
        side = [(-np.array((math.sin(a - half), -math.cos(a - half))), 0.0),  # between the sector's edge rays
                (np.array((math.sin(a + half), -math.cos(a + half))), 0.0)]
        if k % 4 != 2:
            parts = [[]]
        else:  # a corner sector holds a screw hole: four convex pieces round it
            sx, sy = np.sign(e[0]), np.sign(e[1])
            parts = [[((0.0, sy), sy * sy * (c + hh))], [((sx, 0.0), c + hh), ((0.0, -sy), -(c + hh))],
                     [((-sx, 0.0), -(c - hh)), ((0.0, -sy), -(c + hh)), ((0.0, sy), c - hh)],
                     [((0.0, -sy), -(c - hh)), ((-sx, 0.0), -(c + hh))]]
            parts = [[(np.asarray(n, float), cc) for n, cc in p] for p in parts]
        for extra in parts:
            pts = []
            for rr, z in ((R + ch, 0.0), (R, FL["chamfer_d"]), (R, dz)):
                for p in _poly(side + extra + [(e, rr)], H):
                    pts.append((p[0], p[1], z))
            if len(pts) >= 8:
                out.append(pts)
    xs = [(-H, -c - hh), (-c + hh, c - hh), (c + hh, H)]
    ys = [(-H, -c - hh), (-c - hh, -c + hh), (-c + hh, c - hh), (c - hh, c + hh), (c + hh, H)]
    for j, (y0, y1) in enumerate(ys):
        for x0, x1 in (xs if j in (1, 3) else [(-H, H)]):
            out.append([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (dz, top)])
    return out


def _cyl(r, z0, z1, n=24, r1=None, c=(0.0, 0.0)):
    return ring(r, z0, n, c) + ring(r if r1 is None else r1, z1, n, c)


def build_gearbox(stage, path):
    prim = stage.GetPrimAtPath(path)
    rigid(prim, GB["mass"], GB["com"])
    phys = surface(stage, f"{path}/Physics/m", MU)
    m = {k: look(stage, f"{path}/Looks/{k}", COLORS[k]) for k in ("gearbox", "ring", "dark")}
    for name, lo, hi, col in gearbox_boxes():
        cube(stage, f"{path}/{name}", lo, hi, m[col], phys)
    f, R, h = GB["face"], GB["spigot_r"], GB["spigot_h"]
    # the worm housing, along y; kept within 18 mm of the mid-plane so the 11 mm deep tapped holes stay open
    worm = [p for y in (-0.040, 0.045) for p in ring(0.018, y, 16, (0.039, 0.0), axis="y") + [(0.003, y, -0.018), (0.003, y, 0.018)]]
    hull(stage, f"{path}/worm", worm, m["gearbox"], phys)
    hull(stage, f"{path}/spigot_j", ring(R, -f + 0.0005) + ring(R, -f - h + 0.001) + ring(R - 0.0015, -f - h), m["ring"], phys)  # chamfer 1.5 mm wide, 1 mm tall
    hull(stage, f"{path}/spigot_k", ring(R, f - 0.0005) + ring(R, f + h - 0.001) + ring(R - 0.001, f + h), m["gearbox"], phys)
    for s, z in (("j", -f - h - 0.0002), ("k", f + h + 0.0002)):
        hull(stage, f"{path}/bore_{s}", _cyl(0.007, z - 0.0001, z + 0.0001), m["dark"])  # the 14 mm hollow output bore, drawn only
    gearbox_looks(stage, path, prim)
    return prim


def _rrect(x0, x1, y0, y1, r, n=6):
    """A rounded rectangle's outline (2-D points)."""
    out = []
    for cx, cy, a0 in ((x1 - r, y1 - r, 0.0), (x0 + r, y1 - r, math.pi / 2), (x0 + r, y0 + r, math.pi), (x1 - r, y0 + r, 1.5 * math.pi)):
        for k in range(n + 1):
            a = a0 + math.pi / 2 * k / n
            out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


def _prism(outline, z0, z1, axis="z"):
    pts = []
    for z in (z0, z1):
        for a, b in outline:
            pts.append((a, b, z) if axis == "z" else (a, z, b))  # axis "y": outline in (x, z)
    return pts


def gearbox_looks(stage, path, prim):
    """What the gearbox looks like (drawn only; it collides with the shapes above, which are hidden): the die-cast NMRV body with rounded
    corners and corner bosses, cooling ribs on the worm housing, the square input flange with its motor bore and bolt holes, the hollow
    output bore with its keyway, the tapped holes on face K, a breather plug. From the catalogue drawing and photos; ours."""
    from pxr import UsdGeom

    for child in prim.GetChildren():
        if child.GetTypeName() in ("Cube", "Mesh") and not child.GetName().startswith(("bore_", "spigot_")):
            UsdGeom.Imageable(child).MakeInvisible()
    L = {k: look(stage, f"{path}/Looks/v_{k}", COLORS[k]) for k in ("gearbox", "rib", "dark", "brass")}
    f = GB["face"]
    hull(stage, f"{path}/v_body", _prism(_rrect(-0.030, 0.044, -0.030, 0.030, 0.008), -f, f), L["gearbox"])
    hull(stage, f"{path}/v_feet", _prism(_rrect(-0.040, -0.026, -0.028, 0.028, 0.003), -f, f), L["gearbox"])
    worm = [p for y in (-0.040, 0.045) for p in ring(0.018, y, 16, (0.039, 0.0), axis="y") + [(0.003, y, -0.018), (0.003, y, 0.018)]]
    hull(stage, f"{path}/v_worm", worm, L["gearbox"])
    for k, y in enumerate((-0.034, -0.022, -0.010, 0.002, 0.014, 0.026, 0.038)):  # ribs round the worm housing
        hull(stage, f"{path}/v_rib{k}", [p for yy in (y - 0.001, y + 0.001) for p in ring(0.0205, yy, 16, (0.041, 0.0), axis="y")], L["rib"])
    hull(stage, f"{path}/v_inflange", _prism(_rrect(0.0, 0.060, -0.030, 0.030, 0.006), 0.045, 0.055, axis="y"), L["gearbox"])
    hull(stage, f"{path}/v_inbore", [p for yy in (0.0551, 0.0553) for p in ring(0.0115, yy, 20, (0.030, 0.0), axis="y")], L["dark"])  # motor bore
    for k, (dx, dz) in enumerate(((0.021, 0.021), (-0.021, 0.021), (-0.021, -0.021), (0.021, -0.021))):
        hull(stage, f"{path}/v_inhole{k}", [p for yy in (0.0551, 0.0553) for p in ring(0.0028, yy, 10, (0.030 + dx, dz), axis="y")], L["dark"])
    z = f + GB["spigot_h"] + 0.0003
    hull(stage, f"{path}/v_key", _prism([(-0.0025, 0.006), (0.0025, 0.006), (0.0025, 0.0093), (-0.0025, 0.0093)], z, z + 0.0002), L["dark"])
    for k, (hx, hy) in enumerate(HOLES):  # the KE tapped holes on both output faces (face J's are also real openings, under the drawn body)
        hull(stage, f"{path}/v_hole{k}", _cyl(0.003, f + 0.0001, f + 0.0003, 12, c=(hx, hy)), L["dark"])
        hull(stage, f"{path}/v_holej{k}", _cyl(0.0035, -f - 0.0003, -f - 0.0001, 12, c=(hx, hy)), L["dark"])
    hull(stage, f"{path}/v_breather", [p for x in (0.057, 0.061) for p in [(x, 0.012 + 0.004 * math.cos(a), 0.004 * math.sin(a)) for a in [k * math.pi / 3 for k in range(6)]]], L["brass"])


def build_flange(stage, path):
    prim = stage.GetPrimAtPath(path)
    rigid(prim, FL["mass"])
    phys = surface(stage, f"{path}/Physics/m", MU)
    m = look(stage, f"{path}/Looks/al", COLORS["flange"])
    for k, pts in enumerate(flange_pieces()):
        hull(stage, f"{path}/plate_{k}", pts, m, phys)
    hull(stage, f"{path}/neck", _cyl(FL["neck_r"], FL["plate"], FL["top"] - FL["spig_h"]), m, phys)
    hull(stage, f"{path}/spigot", _cyl(FL["spig_r"], FL["top"] - FL["spig_h"], FL["top"]), m, phys)
    from pxr import UsdGeom

    for k in range(len(flange_pieces())):  # drawn instead: a plate with rounded corners
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{path}/plate_{k}")).MakeInvisible()
    H = FL["half"]
    hull(stage, f"{path}/v_plate", _prism(_rrect(-H, H, -H, H, 0.008), 0.0002, FL["plate"]), m)
    dark = look(stage, f"{path}/Looks/dark", COLORS["dark"])
    hull(stage, f"{path}/v_bore", _cyl(0.007, FL["top"] + 0.0001, FL["top"] + 0.0003, 20), dark)  # the bore for the driven shaft
    for k, (hx, hy) in enumerate(HOLES):
        hull(stage, f"{path}/v_hole{k}", _cyl(0.0042, FL["plate"] + 0.0001, FL["plate"] + 0.0002, 16, c=(hx, hy)), dark)
    return prim


def build_screw(stage, path, flange_path):
    Gf, _, _, UsdPhysics, _ = _api()
    prim = stage.GetPrimAtPath(path)
    rigid(prim, SC["mass"], iters=64)
    phys = surface(stage, f"{path}/Physics/m", 0.2)
    m = look(stage, f"{path}/Looks/s", COLORS["screw"])
    L, r = SC["length"], SC["shank_r"]
    hull(stage, f"{path}/shank", ring(r, -0.001, 16) + ring(r, -L + 0.002, 16) + ring(0.001, -L, 16), m, phys)  # chamfered tip
    hr, hh, sr, sd = SC["head_r"], SC["head_h"], SC["socket_r"], SC["socket_d"]
    hull(stage, f"{path}/head_base", _cyl(hr, 0.0, hh - sd, 16), m, phys)
    for k in range(6):  # the socket's wall: six sectors round a 6.8 mm hole (a TX 30 socket, 5.6 mm, drawn round and 0.6 mm loose a side), mouth chamfered
        a0, a1 = k * math.pi / 3, (k + 1) * math.pi / 3
        pts = ring(hr, hh - sd, 4, a0=a0, a1=a1) + ring(hr, hh, 4, a0=a0, a1=a1)
        pts += ring(sr, hh - sd, 1, a0=a0, a1=a1) + ring(sr, hh - 0.0008, 1, a0=a0, a1=a1) + ring(sr + 0.0011, hh, 1, a0=a0, a1=a1)
        hull(stage, f"{path}/socket_{k}", pts, m, phys)
    return prim


def build_driver(stage, path):
    """Collision: a 5.6 mm blade with a pointed tip and a 29 mm cylinder over the handle (invisible). Drawn: the tool's own shape
    (driver_shape.py: Kraftform stubby handle, round blade, TX 30 tip)."""
    from pxr import UsdGeom

    from . import driver_shape

    prim = stage.GetPrimAtPath(path)
    rigid(prim, DRV["mass"], iters=64)
    phys = surface(stage, f"{path}/Physics/m", 0.5)
    d = DRV
    hidden = look(stage, f"{path}/Looks/c", COLORS["handle"])
    for c in (hull(stage, f"{path}/blade", ring(0.0012, 0.0, 12) + ring(d["blade_r"], 0.002, 12) + ring(d["blade_r"], d["blade"], 12), hidden, phys),
              hull(stage, f"{path}/handle", _cyl(d["handle_r"], d["blade"], d["blade"] + d["handle"], 20), hidden, phys)):
        UsdGeom.Imageable(c.GetPrim()).MakeInvisible()
    for name, pts, tris, col in driver_shape.parts():
        mesh(stage, f"{path}/look_{name}", pts, tris, look(stage, f"{path}/Looks/{name}", col))
    return prim


def mesh(stage, path, pts, tris, mat):
    """A drawn-only triangle mesh (nothing collides with it)."""
    Gf, _, UsdGeom, _, UsdShade = _api()
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    m.CreateFaceVertexCountsAttr([3] * len(tris))
    m.CreateFaceVertexIndicesAttr([int(i) for t in tris for i in t])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def build_part(stage, path, kind):
    parent = path.rsplit("/", 1)[0]
    if kind == "gearbox":
        return build_gearbox(stage, path)
    if kind == "flange":
        return build_flange(stage, path)
    if kind == "driver":
        return build_driver(stage, path)
    i = int(kind[-1])
    prim = build_screw(stage, path, f"{parent}/flange")
    _author_screw_joints(stage, path, parent, i)
    return prim


def _x_to(v):
    """(w, x, y, z) of the shortest rotation taking +x onto v."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    c = float(v[0])
    if c < -0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    axis = np.cross([1.0, 0.0, 0.0], v)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    ang = math.atan2(s, c)
    axis = axis / s * math.sin(ang / 2)
    return (math.cos(ang / 2), float(axis[0]), float(axis[1]), float(axis[2]))


def screw_offset(i):
    """Screw i's head-underside centre in the flange's frame, as it starts (its tip 1 mm proud)."""
    return (HOLES[i][0], HOLES[i][1], SCREW_START_Z)


def _author_screw_joints(stage, path, parent, i):
    """Glue (on): the screw held in the flange as it starts. Threads (off): one per tapped hole j, gearbox -> screw, along the hole's axis."""
    Gf, _, _, UsdPhysics, _ = _api()
    g = UsdPhysics.FixedJoint.Define(stage, f"{path}/glue")
    g.CreateBody0Rel().SetTargets([f"{parent}/flange"])
    g.CreateBody1Rel().SetTargets([path])
    g.CreateLocalPos0Attr(Gf.Vec3f(*map(float, screw_offset(i))))
    g.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    g.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    g.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    g.CreateJointEnabledAttr(True)
    g.CreateExcludeFromArticulationAttr(True)
    for j, (hx, hy) in enumerate(HOLES):
        jt = UsdPhysics.Joint.Define(stage, f"{path}/thread_{j}")
        jt.CreateBody0Rel().SetTargets([f"{parent}/gearbox"])
        jt.CreateBody1Rel().SetTargets([path])
        jt.CreateLocalPos0Attr(Gf.Vec3f(float(hx), float(hy), -GB["face"]))
        jt.CreateLocalRot0Attr(Gf.Quatf(*_x_to((0.0, 0.0, -1.0))))  # out of face J
        jt.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -SC["length"]))  # the tip
        jt.CreateLocalRot1Attr(Gf.Quatf(*_x_to((0.0, 0.0, 1.0))))
        jt.CreateJointEnabledAttr(False)
        jt.CreateCollisionEnabledAttr(True)
        jt.CreateExcludeFromArticulationAttr(True)
        p = jt.GetPrim()
        for dof in ("transY", "transZ", "rotY", "rotZ"):
            lim = UsdPhysics.LimitAPI.Apply(p, dof)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)  # low > high: locked
        lim = UsdPhysics.LimitAPI.Apply(p, "transX")
        lim.CreateLowAttr(-SC["length"] + FL["plate"] - 0.001)
        lim.CreateHighAttr(0.0)
        dr = UsdPhysics.DriveAPI.Apply(p, "transX")
        dr.CreateTypeAttr("force")
        dr.CreateStiffnessAttr(THREAD_K)
        dr.CreateDampingAttr(THREAD_C)
        dr.CreateTargetPositionAttr(0.0)
        dr.CreateMaxForceAttr(THREAD_MAX)
        tw = UsdPhysics.DriveAPI.Apply(p, "rotX")
        tw.CreateTypeAttr("force")
        tw.CreateStiffnessAttr(0.0)
        tw.CreateDampingAttr(0.0005)
        tw.CreateMaxForceAttr(1.0)


THREAD_K, THREAD_C, THREAD_MAX = 20000.0, 20.0, 150.0  # N/m, N s/m, N: the thread's hold along the axis (stiff: the thread does not stretch)


def build_fixture(stage, path, world_origin):
    Gf, _, UsdGeom, UsdPhysics, _ = _api()
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    physx(prim, "PhysxArticulationAPI", **{"physxArticulation:enabledSelfCollisions": True, "physxArticulation:solverPositionIterationCount": ITERS,
                                           "physxArticulation:solverVelocityIterationCount": VEL_ITERS})
    m = {k: look(stage, f"{path}/Looks/{k}", c) for k, c in COLORS.items()}
    phys = surface(stage, f"{path}/Physics/mat", MU)
    bodies = {}
    for name, mass in MASS.items():
        bodies[name] = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        rigid(bodies[name], mass, CRADLE_COM if name == "cradle" else None)
    for body, name, lo, hi, col in fixture_boxes():
        cube(stage, f"{path}/{body}/{name}", lo, hi, m[col], None if name in DRAWN_ONLY else phys)
    ox, oy = float(world_origin[0]), float(world_origin[1])
    for name, lo, hi in station_boxes():
        cube(stage, f"{path}/anchor/{name}", (lo[0] - ox, lo[1] - oy, lo[2]), (hi[0] - ox, hi[1] - oy, hi[2]), m["holster"], phys)
    hull(stage, f"{path}/anchor/post", _cyl(POST["r"], 0.0, POST["h"], 24, c=(POST["x"] - ox, POST["y"] - oy)), m["steel"], phys)
    # drawn only: round trunnion bosses on the pillars and round axle stubs (the hinge joint is the bearing)
    from pxr import UsdGeom

    for nm in ("axle_far", "axle_near"):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{path}/cradle/{nm}")).MakeInvisible()
    pl = PILLAR
    for nm, (v0, v1), body in (("boss_near", (pl["near"][1], pl["near"][1] + 0.002), "anchor"), ("boss_near_out", (pl["near"][0] - 0.0015, pl["near"][0]), "anchor"),
                               ("boss_far", (pl["far"][0] - 0.002, pl["far"][0]), "anchor"), ("boss_far_out", (pl["far"][1], pl["far"][1] + 0.004), "anchor")):
        hull(stage, f"{path}/{body}/{nm}", [p for v in (v0, v1) for p in ring(0.014, v, 24, (0.0, AXLE), axis="y")], m["dark"])
    for nm, (v0, v1) in (("shaft_near", (pl["near"][0], NEAR_PLATE["v0"])), ("shaft_far", (PLATE["v1"], pl["far"][1]))):
        hull(stage, f"{path}/cradle/{nm}", [p for v in (v0, v1) for p in ring(0.008, v, 20, (0.0, AXLE), axis="y")], m["dark"])
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([bodies["anchor"].GetPath()])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_origin)))
    P = {n: bodies[n].GetPath() for n in bodies}
    cr, bo, br = CRANK, BOLT, BAR
    _damp(_joint(stage, UsdPhysics.RevoluteJoint, f"{path}/roll", P["anchor"], P["cradle"], (0.0, 0.0, AXLE), "Y", 0.0, ROLL_LIMIT), "angular", ROLL_DAMP)
    _damp(_joint(stage, UsdPhysics.PrismaticJoint, f"{path}/bar_slide", P["cradle"], P["bar"], (0.0, 0.0, AXLE), "Y", 0.0, br["travel"]), "linear", SLIDE_DAMP)
    _damp(_joint(stage, UsdPhysics.RevoluteJoint, f"{path}/knob_spin", P["cradle"], P["knob"], (-cr["r"], cr["y"], AXLE), "Y", -720.0, 720.0), "angular", KNOB_DAMP)
    _damp(_joint(stage, UsdPhysics.PrismaticJoint, f"{path}/bolt_slide", P["anchor"], P["bolt"], (0.0, bo["tip"], bo["z"]), "Y", -bo["travel"], 0.0), "linear", SLIDE_DAMP)
    return prim


def _joint(stage, kind, path, b0, b1, at, axis, lo, hi):
    Gf, _, _, UsdPhysics, _ = _api()
    j = kind.Define(stage, path)
    j.CreateAxisAttr(axis)
    j.CreateLowerLimitAttr(float(lo))
    j.CreateUpperLimitAttr(float(hi))
    j.CreateBody0Rel().SetTargets([b0])
    j.CreateBody1Rel().SetTargets([b1])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, at)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, at)))
    return j


def _damp(joint, kind, damping):
    dr = _api()[3].DriveAPI.Apply(joint.GetPrim(), kind)
    dr.CreateTypeAttr("force")
    dr.CreateStiffnessAttr(0.0)
    dr.CreateDampingAttr(float(damping))
    dr.CreateMaxForceAttr(1000.0)


# -- run-time switches (the embodiment and the spike call these) ----------------------------------------------------------------------------
_ATTR = {}


def _attr(path, kind):
    key = (path, kind)
    if key not in _ATTR:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics

        prim = get_current_stage().GetPrimAtPath(path)
        _ATTR[key] = UsdPhysics.Joint(prim).GetJointEnabledAttr() if kind == "on" else UsdPhysics.DriveAPI(prim, "transX").GetTargetPositionAttr()
    return _ATTR[key]


def set_glue(screw_path, on):
    _attr(f"{screw_path}/glue", "on").Set(bool(on))


def set_thread(screw_path, hole, on, target=None):
    if target is not None:
        _attr(f"{screw_path}/thread_{hole}", "target").Set(float(target))
    _attr(f"{screw_path}/thread_{hole}", "on").Set(bool(on))


def set_thread_target(screw_path, hole, target):
    _attr(f"{screw_path}/thread_{hole}", "target").Set(float(target))
