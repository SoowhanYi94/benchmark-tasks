"""Writes fixture_dimensions.svg (plan and side view of the fixture at the start pose, mm) from the constants in
trunnion_fixture/mech.py and the table position in task.py. Standard library only: the constants are read with ast,
so no simulator package is imported.  Run from the task folder:  python dev/fixture_diagram.py"""
import ast
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent


def constants(path):
    def ev(n):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "dict":
            return {k.arg: ev(k.value) for k in n.keywords}
        return ast.literal_eval(n)
    out = {}
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ev(n.value)
            except (ValueError, TypeError):
                pass
    return out


M = constants(HERE / "trunnion_fixture" / "mech.py")
AX = constants(HERE / "task.py")["AX"] * 1000
AXLE = M["AXLE"] * 1000
RAIL, LEDGE, STUB, BAR, PLATE, PILLAR, BOLT, CRANK = (
    {k: v * 1000 for k, v in M[n].items()} if n != "PILLAR" else M[n] for n in ("RAIL", "LEDGE", "STUB", "BAR", "PLATE", "PILLAR", "BOLT", "CRANK"))
PILLAR = {"half_x": PILLAR["half_x"] * 1000, "near": [v * 1000 for v in PILLAR["near"]], "far": [v * 1000 for v in PILLAR["far"]],
          "near_top": PILLAR["near_top"] * 1000, "far_top": PILLAR["far_top"] * 1000}
COL = M["COLORS"]


def rgb(c):
    return "#%02x%02x%02x" % tuple(round(v * 255) for v in c)


# boxes: (name, x0, x1, y0, y1, z0, z1, colour), world mm, level cradle (x = axle x + u, z = axle z + w)
B = []
def box(n, x0, x1, y0, y1, z0, z1, c="#d4d7dc"):
    B.append((n, min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1), min(z0, z1), max(z0, z1), c))

def cb(n, u0, u1, v0, v1, w0, w1, c="#d4d7dc"):
    box(n, AX + u0, AX + u1, v0, v1, AXLE + w0, AXLE + w1, c)

pl, r, le, st, pa, br, bo, cr = PILLAR, RAIL, LEDGE, STUB, PLATE, BAR, BOLT, CRANK
steel, dark = "#b0b4ba", "#6b6e73"
# the base and its pillar rows are the literals in build_fixture (mech.py); the rest comes from the constants
box("base", AX - 70, AX + 70, -135, 100, 0, 6, "#8c9096")
box("near", AX - pl["half_x"], AX + pl["half_x"], pl["near"][0], pl["near"][1], 6, pl["near_top"], steel)
box("far", AX - pl["half_x"], AX + pl["half_x"], pl["far"][0], pl["far"][1], 6, bo["z"] - bo["half"], steel)
box("far_top", AX - pl["half_x"], AX + pl["half_x"], pl["far"][0], pl["far"][1], bo["z"] + bo["half"], pl["far_top"], steel)
for s in (1, -1):
    cb("rail", *sorted((s * r["inner"], s * (r["inner"] + r["wall"]))), r["v0"], r["v1"], r["w0"], r["w1"])
    cb("ledge", *sorted((s * le["u0"], s * r["inner"])), -le["v"], le["v"], le["w0"], le["w1"])
    for vs in (1, -1):
        cb("beam", -r["inner"], r["inner"], *sorted((vs * st["v0"], vs * st["v1"])), le["w0"], le["w1"])
        cb("stub", *sorted((s * st["u0"], s * r["inner"])), *sorted((vs * st["v0"], vs * st["v1"])), le["w1"], st["w1"], dark)
cb("plate_top", -pa["half_u"], pa["half_u"], pa["v0"], pa["v1"], pa["notch_w"], pa["half_w"])
for s in (1, -1):
    cb("plate_low", *sorted((s * pa["notch_u"], s * pa["half_u"])), pa["v0"], pa["v1"], -pa["half_w"], pa["notch_w"])
cb("axle_far", -8, 8, pa["v1"], pl["far"][0], -8, 8, dark)
cb("plate_near", -54, 54, -106, r["v0"], r["w0"], r["w1"])
cb("axle_near", -8, 8, pl["near"][1], -106, -8, 8, dark)
cb("crank", -cr["r"] - 8, 8, -128, -122, -8, 8, dark)
cb("pin", -cr["r"] - 4, -cr["r"] + 4, cr["y"] + cr["half_y"], -128, -4, 4, dark)
cb("knob", -cr["r"] - cr["half_x"], -cr["r"] + cr["half_x"], cr["y"] - cr["half_y"], cr["y"] + cr["half_y"], -cr["half_z"], cr["half_z"], rgb(COL["knob"]))
BAR_C = rgb(COL["bar"])
BOLT_C = rgb(COL["bolt"])

def bar_boxes(dy):
    return [(AX - br["half_u"], AX + br["half_u"], br["park"] + dy - br["half_v"], br["park"] + dy + br["half_v"], AXLE + br["w0"], AXLE + br["w1"]),
            (AX - br["lug_u"], AX + br["lug_u"], br["park"] + dy - br["half_v"] - br["lug_v"], br["park"] + dy - br["half_v"], AXLE + br["w0"], AXLE + br["lug_w"])]

def bolt_boxes(dy):
    return [(AX - bo["half"], AX + bo["half"], bo["tip"] + dy, bo["end"] + dy, bo["z"] - bo["half"], bo["z"] + bo["half"]),
            (AX - bo["lug_half_x"], AX + bo["lug_half_x"], bo["end"] - bo["lug_y"] + dy, bo["end"] + dy, bo["z"] - bo["half"], bo["lug_top"])]

# ---------------------------------------------------------------- drawing
S = 3.0
W = 1400
OX = 580.0           # px of y = 0 (+y to the left in both views)
PT, PB = 150.0, 150.0 + 140 * S   # plan: x = 360 at PT, x = 220 at PB
SB = 1250.0          # side view: z = 0
out = []
def f(v):
    return ("%.2f" % v).rstrip("0").rstrip(".")
def num(v):
    return f(round(v, 2))
def PX(y): return OX - y * S
def PY(x): return PB - (x - 220) * S
def SY(z): return SB - z * S
def txt(x, y, s, size=13, anchor="middle", rot=0, weight="normal", fill="#111"):
    t = f' transform="rotate({rot} {f(x)} {f(y)})"' if rot else ""
    out.append(f'<text x="{f(x)}" y="{f(y)}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{t} stroke="#fff" stroke-width="4" paint-order="stroke" stroke-linejoin="round">{s}</text>')
def line(x1, y1, x2, y2, c="#222", w=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{c}" stroke-width="{w}"{d}/>')
def rect(x0, y0, x1, y1, fill, stroke="#333", w=1, dash=None, op=0.55):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{f(min(x0,x1))}" y="{f(min(y0,y1))}" width="{f(abs(x1-x0))}" height="{f(abs(y1-y0))}" fill="{fill}" fill-opacity="{op if fill!="none" else 0}" stroke="{stroke}" stroke-width="{w}"{d}/>')
def arrow(x, y, dx, dy):
    a = 7
    px, py = -dy, dx
    out.append(f'<polygon points="{f(x)},{f(y)} {f(x-dx*a+px*2.5)},{f(y-dy*a+py*2.5)} {f(x-dx*a-px*2.5)},{f(y-dy*a-py*2.5)}" fill="#222"/>')
def hdim(xa, xb, yd, yf, label, side=0, size=13):
    """horizontal dimension between px xa, xb at px yd; extension lines from yf. side: 0 label centred above, 1 right of the line, -1 left."""
    for x in (xa, xb):
        line(x, yf, x, yd + (4 if yd > yf else -4), "#555", 0.8)
    lo, hi = sorted((xa, xb))
    if hi - lo > 22:
        arrow(lo, yd, -1, 0); arrow(hi, yd, 1, 0); line(lo, yd, hi, yd)
    else:
        arrow(lo, yd, 1, 0); arrow(hi, yd, -1, 0); line(lo - 14, yd, hi + 14, yd)
    if side == 0: txt((lo + hi) / 2, yd - 5, label, size)
    elif side == 1: txt(hi + 16, yd + 4, label, size, "start")
    else: txt(lo - 16, yd + 4, label, size, "end")
def vdim(xd, ya, yb, xf, label, side=0, size=13):
    """vertical dimension between px ya, yb at px xd; extension lines from xf. label rotated, left of the line (side 0), or horizontal above (1) / below (-1)."""
    for y in (ya, yb):
        line(xf, y, xd + (4 if xd > xf else -4), y, "#555", 0.8)
    lo, hi = sorted((ya, yb))
    if hi - lo > 22:
        arrow(xd, lo, 0, -1); arrow(xd, hi, 0, 1); line(xd, lo, xd, hi)
    else:
        arrow(xd, lo, 0, 1); arrow(xd, hi, 0, -1); line(xd, lo - 14, xd, hi + 14)
    if side == 0: txt(xd - 5, (lo + hi) / 2, label, size, rot=-90)
    elif side == 1: txt(xd, lo - 18, label, size)
    else: txt(xd, hi + 20, label, size)
def callout(x, y, tx, ty, s, anchor="start", size=13):
    line(x, y, tx, ty, "#555", 0.8)
    out.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="2" fill="#555"/>')
    txt(tx + (4 if anchor == "start" else -4), ty + 4, s, size, anchor)

def tk(v):  # a range label from two mm values
    return f"{num(min(v))} to {num(max(v))}"
def mm(v): return num(v)
def axes(x, y):  # small +x / +y triad
    arrow(x, y - 50, 0, -1); line(x, y, x, y - 50)
    arrow(x - 50, y, -1, 0); line(x, y, x - 50, y)
    txt(x + 6, y - 40, "+x", 13, "start"); txt(x - 50, y + 18, "+y", 13)

# ---- plan view
txt(30, 36, "(a) Plan view, looking down", 18, "start", weight="bold")
txt(30, 58, "x up the page, from the arm bases' midpoint (x = 0 is below the page); y to the left, from the table's centre line (+y toward the left arm)", 13, "start")
txt(30, 76, "Cradle level (start pose); parts drawn see-through; all dimensions in mm; dashed outlines show the end of each slide", 13, "start")
line(PX(0), PT - 20, PX(0), PB + 12, "#777", 0.8, "10 3 2 3")
txt(PX(0) + 6, PT - 24, "y = 0", 12, "start")
line(PX(135) - 8, PY(AX), PX(-158) + 0, PY(AX), "#777", 0.8, "10 3 2 3")
txt(PX(-137), PY(AX) - 5, f"axle x {mm(AX)}", 11, "start")
for n, x0, x1, y0, y1, z0, z1, c in B:
    rect(PX(y1), PY(x1), PX(y0), PY(x0), c)
for x0, x1, y0, y1, z0, z1 in bar_boxes(0):
    rect(PX(y1), PY(x1), PX(y0), PY(x0), BAR_C)
for x0, x1, y0, y1, z0, z1 in bar_boxes(br["travel"]):
    rect(PX(y1), PY(x1), PX(y0), PY(x0), "none", BAR_C, 1.6, "6 4")
for x0, x1, y0, y1, z0, z1 in bolt_boxes(0):
    rect(PX(y1), PY(x1), PX(y0), PY(x0), BOLT_C)
for x0, x1, y0, y1, z0, z1 in bolt_boxes(-bo["travel"]):
    rect(PX(y1), PY(x1), PX(y0), PY(x0), "none", BOLT_C, 1.6, "6 4")
# labels on the parts
txt(PX(96), PY(AX - 62), "base", 12, "start")
txt(PX(pl["near"][0]) + 13, PY(AX) + 56, "near pillar", 12, rot=-90)
txt(PX(pl["far"][1]) + 13, PY(AX) + 56, "far pillar", 12, rot=-90)
txt(PX(-30), PY(AX - r["inner"] - r["wall"] / 2) + 4, "cradle rail", 11)
txt(PX(-30), PY(AX + r["inner"] + r["wall"] / 2) + 4, "cradle rail", 11)
txt(PX(le["v"]) - 8, PY(AX + 46) + 4, "ledge", 11, "start")
txt(PX(le["v"]) - 8, PY(AX - 46) + 4, "ledge", 11, "start")
txt(PX(br["park"]), PY(AX) + 4, "(parked)", 12)
txt(PX(br["park"]), PY(AX + 15) + 4, "retaining bar", 12)
txt(PX(br["travel"] + br["park"]), PY(AX + 28), "bar closed", 12)
txt(PX(100), PY(AX) + 4, "lock bolt", 12, "start")
txt(PX(-146), PY(AX - cr["r"] + 20) - 12, "knob", 12)
txt(PX((-106 + r["v0"]) / 2) + 4, PY(AX), f"near end plate {mm(108)} x {mm(r['v0'] + 106)}", 11, rot=-90)
txt(PX((pa["v0"] + pa["v1"]) / 2) + 4, PY(AX) + 70, f"far end plate {mm(2 * pa['half_u'])} x {mm(pa['v1'] - pa['v0'])}", 11, rot=-90)
txt(30, PB + 80, f"End beams 92 x {mm(st['v1'] - st['v0'])} x {mm(le['w1'] - le['w0'])} at y &#177;{mm(st['v0'])} to &#177;{mm(st['v1'])};", 12, "start")
txt(30, PB + 97, f"corner stubs 24 x {mm(st['v1'] - st['v0'])} x {mm(st['w1'] - le['w1'])}, {mm(2 * st['v0'])} apart.", 12, "start")
txt(PX(-125), PY(AX - 8) - 24, "crank", 12, rot=-90)
# dimensions: above the plan
hdim(PX(100), PX(-135), 112, PT, f"base {mm(235)} long in y ({tk((-135, 100))})")
hdim(PX(pl["far"][1]), PX(pl["far"][0]), 134, PT, f"far pillar {mm(pl['far'][1] - pl['far'][0])} thick (y {tk(pl['far'])})", -1)
hdim(PX(pl["near"][1]), PX(pl["near"][0]), 134, PT, f"near pillar {mm(pl['near'][1] - pl['near'][0])} thick (y {tk(pl['near'])})", 1)
# below the plan
hdim(PX(bo["tip"] - bo["travel"]), PX(bo["tip"]), PB + 30, PY(AX - 6), f"bolt travel {mm(bo['travel'])}", -1)
hdim(PX(r["v1"]), PX(r["v0"]), PB + 30, PY(AX - 54), f"rail length {mm(r['v1'] - r['v0'])} (y {tk((r['v0'], r['v1']))})")
hdim(PX(br["park"]), PX(br["park"] + br["travel"]), PB + 62, PY(AX - 40), f"bar travel {mm(br['travel'])} (centre y {mm(br['park'])} to {mm(br['park'] + br['travel'])})")
hdim(PX(cr["y"]), PX(0), PB + 94, PY(AX - cr["r"] - cr["half_x"]), f"knob centre at y {mm(cr['y'])}")
# left of the plan
vdim(155, PY(AX + pl["half_x"]), PY(AX - pl["half_x"]), PX(pl["far"][1]), f"pillars {mm(2 * pl['half_x'])} wide (x {tk((AX - pl['half_x'], AX + pl['half_x']))})")
# right of the plan
vdim(1075, PY(AX + 70), PY(AX - 70), PX(-135), f"base {mm(140)} wide in x ({tk((AX - 70, AX + 70))})")
vdim(1110, PY(AX - cr["r"]), PY(AX), PX(cr["y"] - cr["half_y"]), f"knob centre x {mm(AX - cr['r'])}, {mm(cr['r'])} from the axle")
# inside the cradle
vdim(PX(52), PY(AX + r["inner"]), PY(AX - r["inner"]), PX(52), f"{mm(2 * r['inner'])} between rails")
vdim(PX(30), PY(AX + le["u0"]), PY(AX - le["u0"]), PX(30), f"{mm(2 * le['u0'])} opening")
hdim(PX(le["v"]), PX(-le["v"]), PY(AX - (r["inner"] + le["u0"]) / 2) + 14, PY(AX - (r["inner"] + le["u0"]) / 2) + 14, f"ledge {mm(2 * le['v'])} long")
axes(100, 150)

# ---- (c) the far end plate
IS = 2.0
IX, IZ = 1270.0, 330.0   # px of x = axle line, z = 30 (the plate's bottom)
def QX(x): return IX + (x - AX) * IS
def QZ(z): return IZ - (z - 30) * IS
txt(1200, 110, "(c) Far end plate,", 15, "start", weight="bold")
txt(1200, 128, "seen along +y", 15, "start", weight="bold")
x0, x1 = AX - pa["half_u"], AX + pa["half_u"]
zb, zt, zn = AXLE - pa["half_w"], AXLE + pa["half_w"], AXLE + pa["notch_w"]
nx0, nx1 = AX - pa["notch_u"], AX + pa["notch_u"]
out.append(f'<polygon points="{f(QX(x0))},{f(QZ(zb))} {f(QX(nx0))},{f(QZ(zb))} {f(QX(nx0))},{f(QZ(zn))} {f(QX(nx1))},{f(QZ(zn))} {f(QX(nx1))},{f(QZ(zb))} {f(QX(x1))},{f(QZ(zb))} {f(QX(x1))},{f(QZ(zt))} {f(QX(x0))},{f(QZ(zt))}" fill="#d4d7dc" fill-opacity="0.55" stroke="#333"/>')
rect(QX(AX - 8), QZ(AXLE + 8), QX(AX + 8), QZ(AXLE - 8), dark)
txt(QX(AX), QZ(AXLE) + 4, "axle", 11)
txt(QX(AX), QZ(zb) - 24, "notch", 12)
line(QX(AX), IZ, QX(AX), QZ(zt) - 10, "#777", 0.8, "10 3 2 3")
hdim(QX(x0), QX(x1), QZ(zb) + 28, QZ(zb), f"{mm(2 * pa['half_u'])} wide (x {tk((x0, x1))})")
hdim(QX(nx0), QX(nx1), QZ(zb) + 58, QZ(zb), f"notch {mm(2 * pa['notch_u'])} wide", 0)
vdim(QX(x0) - 26, QZ(zt), QZ(zb), QX(x0), f"{mm(2 * pa['half_w'])} tall")
vdim(QX(x1) + 24, QZ(zn), QZ(zb), QX(x1), f"notch {mm(zn - zb)} tall", 0)
txt(1200, QZ(zb) + 100, f"The plate is {mm(pa['v1'] - pa['v0'])} thick (y {tk((pa['v0'], pa['v1']))}).", 12, "start")
txt(1200, QZ(zb) + 116, f"Plate z {tk((zb, zt))}; notch z {tk((zb, zn))},", 12, "start")
txt(1200, QZ(zb) + 132, "open at the bottom edge.", 12, "start")

# ---- side view
txt(30, 735, "(b) Side view, looking along +x from the arms", 18, "start", weight="bold")
txt(30, 755, "z up from the table top; y to the left, from the table's centre line (+y toward the left arm)", 13, "start")
line(PX(0), SY(0) + 12, PX(0), SY(152), "#777", 0.8, "10 3 2 3")
txt(PX(0) + 6, SY(152) + 4, "y = 0", 12, "start")
line(PX(135), SY(AXLE), PX(-165), SY(AXLE), "#777", 0.8, "10 3 2 3")
line(PX(140), SY(0), PX(-165), SY(0), "#222", 1.5)
for n, x0, x1, y0, y1, z0, z1, c in B:
    rect(PX(y1), SY(z1), PX(y0), SY(z0), c)
for x0, x1, y0, y1, z0, z1 in bar_boxes(0):
    rect(PX(y1), SY(z1), PX(y0), SY(z0), BAR_C)
for x0, x1, y0, y1, z0, z1 in bar_boxes(br["travel"]):
    rect(PX(y1), SY(z1), PX(y0), SY(z0), "none", BAR_C, 1.6, "6 4")
for x0, x1, y0, y1, z0, z1 in bolt_boxes(0):
    rect(PX(y1), SY(z1), PX(y0), SY(z0), BOLT_C)
for x0, x1, y0, y1, z0, z1 in bolt_boxes(-bo["travel"]):
    rect(PX(y1), SY(z1), PX(y0), SY(z0), "none", BOLT_C, 1.6, "6 4")
out.append(f'<circle cx="{f(PX(0))}" cy="{f(SY(AXLE))}" r="4" fill="#111"/>')
txt(PX(80), SY(AXLE) - 5, f"axle line, z = {mm(AXLE)}", 12, "start")
# labels on the parts
txt(PX(60), SY(3) + 20, "base plate", 12)
txt(PX(pl["near"][0]) - 0, SY(30), "near pillar", 12, rot=-90)
txt(PX(pl["far"][1]) + 20, SY(30), "far pillar", 12, rot=-90)
txt(PX(0), SY(AXLE + 22) + 4, "cradle rail (nearer one drawn)", 12)
txt(PX(0), SY(le["w1"] + AXLE) + 18, "ledges, end beams, corner stubs inside", 11)
txt(PX(-102) , SY(AXLE - 36) + 4, "near end plate", 11, rot=-90)
txt(PX(63) , SY(AXLE - 40) + 4, "far end plate", 11, rot=-90)
txt(PX(cr["y"]), SY(AXLE) + 4, "knob", 12)
txt(PX(-125) + 4, SY(AXLE - 12) + 12, "crank", 11, "start")
txt(PX(br["park"]) , SY(br["lug_w"] + AXLE) - 8, "retaining bar", 12, fill=BAR_C)
txt(PX(br["park"] + br["travel"] - br["half_v"] - br["lug_v"] / 2), SY(br["lug_w"] + AXLE) - 8, "closed", 12, fill=BAR_C)
txt(PX(100), SY(bo["z"]) + 30, "lock bolt", 12, fill=BOLT_C)
# dimensions: below
hdim(PX(100), PX(-135), SB + 34, SY(0), f"base {mm(235)} (y {tk((-135, 100))}), {mm(6)} thick")
hdim(PX(pl["far"][1]), PX(pl["far"][0]), SB + 64, SY(0), f"far pillar y {tk(pl['far'])}", -1)
hdim(PX(pl["near"][1]), PX(pl["near"][0]), SB + 64, SY(0), f"near pillar y {tk(pl['near'])}", 1)
# above
hdim(PX(pa["v0"]), PX(pa["v1"]), SY(138), SY(AXLE + pa["half_w"]), f"far plate {mm(pa['v1'] - pa['v0'])} thick", -1)
hdim(PX(br["park"] + br["travel"]), PX(br["park"]), SY(142), SY(br["lug_w"] + AXLE), f"bar travel {mm(br['travel'])}", 0)
hdim(PX(bo["tip"] - bo["travel"]), PX(bo["tip"]), SY(155), SY(bo["z"] + bo["half"]), f"bolt travel {mm(bo['travel'])}", -1)
# left
vdim(160, SY(bo["z"] + bo["half"]), SY(bo["z"] - bo["half"]), PX(pl["far"][1]), f"bolt gap {mm(2 * bo['half'])} (z {tk((bo['z'] - bo['half'], bo['z'] + bo['half']))})")
vdim(125, SY(pl["far_top"]), SY(0), PX(pl["far"][1]), f"far pillar {mm(pl['far_top'])} (lower part {mm(bo['z'] - bo['half'])})")
vdim(90, SY(bo["lug_top"]), SY(0), PX(bo["end"]), f"bolt lug top {mm(bo['lug_top'])}")
# right
vdim(1075, SY(AXLE), SY(0), PX(-158), f"axle height {mm(AXLE)}")
vdim(1110, SY(pl["near_top"]), SY(0), PX(pl["near"][0]), f"near pillar {mm(pl['near_top'])}")
vdim(1145, SY(AXLE + cr["half_z"]), SY(AXLE - cr["half_z"]), PX(-158), f"knob {mm(2 * cr['half_z'])} (z {tk((AXLE - cr['half_z'], AXLE + cr['half_z']))})")
vdim(1180, SY(AXLE + r["w1"]), SY(AXLE + r["w0"]), PX(r["v0"]), f"rails {mm(r['w1'] - r['w0'])} (z {tk((AXLE + r['w0'], AXLE + r['w1']))})")
vdim(1215, SY(AXLE + pa["half_w"]), SY(AXLE - pa["half_w"]), PX(pa["v0"]), f"far plate {mm(2 * pa['half_w'])} (z {tk((AXLE - pa['half_w'], AXLE + pa['half_w']))})")

open(HERE / "fixture_dimensions.svg", "w").write(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="1400" viewBox="0 0 {W} 1400" font-family="Helvetica, Arial, sans-serif">\n'
    '<rect width="100%" height="100%" fill="#fff"/>\n' + "\n".join(out) + "\n</svg>\n")
