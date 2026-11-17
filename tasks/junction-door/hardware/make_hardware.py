"""Fabrication files for the real-trial socket, its contact spring and the plug, generated from the scene's own
dimensions in ../task.py (run from this folder: python make_hardware.py; needs numpy, trimesh, matplotlib).

Writes: socket_body.stl, socket_pad.stl, plug_body.stl (millimetres) and drawing.svg / drawing.pdf (dimensioned views,
materials, spring, tolerances, acceptance checks). Each STL is a set of closed convex solids that touch or overlap;
slicers (PrusaSlicer, Cura) merge them into one part. The cable clip is a commercial part (see the drawing).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import trimesh

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import task  # noqa: E402

MM = 1000.0
# -- socket, in mm; frame: x into the socket from its mouth, y across (centred), z up from the back plate's top -------
PLATE = task.T * MM  # the sim's back plate (2 mm); the real box's 1.2 mm sheet: the socket stands on the sheet
CAV_W, CAV_H, CAV_D = task.CAV_W * MM, task.CAV_H * MM, task.CAV_D * MM  # 19 x 15 x 16
BACK = 6.0  # back wall (sim: back = 0.006)
BLK_L, BLK_W = CAV_D + BACK, 29.0  # 22 x 29
BLK_H = (task.SOCK_TOP - task.TOP) * MM - PLATE  # 21 above the plate (23 above the bench in the sim)
Z_LO = (task.SOCK_Z - task.CAV_H / 2 - task.TOP) * MM - PLATE  # cavity floor 3.5 above the plate
Z_HI = Z_LO + CAV_H  # 18.5
LEAD = 4.0  # 45 deg lead-in on floor and sides (sim 4 mm)
LEAD_ROOF = 2.0  # 45 deg on the roof: limited by the 2.5 mm roof (sim: 4 mm plate standing proud)
PAD_X = (task.PAD.pos[0] - task.SOCK_X) * MM  # contact pad centre 13 mm in from the mouth
PAD_LX, PAD_LY = 5.8, 11.8  # pad stem (sim pad 6 x 12)
SLOT_LX, SLOT_LY = 6.2, 12.2  # roof slot: 0.2 mm clearance each side
PAD_UNDER = 11.0  # pad underside above the cavity floor at rest: 1 mm onto a 12 mm plug (sim: 1.2 mm measured squeeze)
FLANGE = (8.0, 14.0, 1.0)  # pad flange resting on the roof (its stop)
# contact spring: AISI 301 full-hard stainless shim, cantilever glued flat on the roof, pressing the pad flange
E_STEEL = 193e3  # N/mm^2
SPR_T, SPR_B = 0.10, 6.0
SPR_CLAMP = (0.0, 4.0)  # glued length on the roof, from the mouth
SPR_L = PAD_X - SPR_CLAMP[1]  # free length to the pad centre
SPR_K = E_STEEL * SPR_B * SPR_T ** 3 / (4 * SPR_L ** 3)  # N/mm
# plug
PL, PW, PH = task.PL * MM, task.PW * MM, task.PH * MM  # 44 x 15 x 12
BORE, BORE_LEN = 5.4, 20.0  # square cord channel from the rear (5 mm cord + glue)
MARK = task.SEATED_MIN * MM  # 12 mm line from the tip
# clip: placement from the scene
CLIP_C = ((task.CLIP_X0 + task.CLIP_X1) / 2, task.CLIP_Y)


def hull(pts) -> trimesh.Trimesh:
    return trimesh.convex.convex_hull(np.asarray(pts, float))


def box(x0, x1, y0, y1, z0, z1) -> trimesh.Trimesh:
    return hull([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def prism_xz(poly_xz, y0, y1) -> trimesh.Trimesh:
    return hull([(x, y, z) for x, z in poly_xz for y in (y0, y1)])


def prism_xy(poly_xy, z0, z1) -> trimesh.Trimesh:
    return hull([(x, y, z) for x, y in poly_xy for z in (z0, z1)])


def socket_body() -> list[trimesh.Trimesh]:
    w, cw = BLK_W / 2, CAV_W / 2
    sx0, sx1, sy = PAD_X - SLOT_LX / 2, PAD_X + SLOT_LX / 2, SLOT_LY / 2
    p = [
        box(CAV_D, BLK_L, -w, w, 0, BLK_H),  # back wall
        box(LEAD, CAV_D, -w, w, 0, Z_LO),  # floor
        prism_xz([(LEAD - Z_LO, 0), (LEAD, 0), (LEAD, Z_LO)], -w, w),  # floor lead-in (45 deg, meets the plate 0.5 in)
        box(LEAD, CAV_D, cw, w, Z_LO, BLK_H),  # side walls
        box(LEAD, CAV_D, -w, -cw, Z_LO, BLK_H),
        prism_xy([(0, cw + LEAD), (0, w), (LEAD, w), (LEAD, cw)], Z_LO, BLK_H),  # side lead-ins
        prism_xy([(0, -cw - LEAD), (0, -w), (LEAD, -w), (LEAD, -cw)], Z_LO, BLK_H),
        box(LEAD_ROOF, sx0, -cw, cw, Z_HI, BLK_H),  # roof, mouth side of the slot
        box(sx0, CAV_D, sy, cw, Z_HI, BLK_H),  # roof beside the slot
        box(sx0, CAV_D, -cw, -sy, Z_HI, BLK_H),
        prism_xz([(0, Z_HI + LEAD_ROOF), (0, BLK_H), (LEAD_ROOF, BLK_H), (LEAD_ROOF, Z_HI)], -cw, cw),  # roof lead-in
    ]
    return p


def socket_pad() -> list[trimesh.Trimesh]:
    stem = PAD_UNDER + Z_LO  # pad underside height above the plate
    fx, fy, fz = FLANGE
    return [box(PAD_X - PAD_LX / 2, PAD_X + PAD_LX / 2, -PAD_LY / 2, PAD_LY / 2, stem, BLK_H),
            box(PAD_X - fx / 2, PAD_X + fx / 2, -fy / 2, fy / 2, BLK_H, BLK_H + fz)]


def plug_body() -> list[trimesh.Trimesh]:
    # x from the tip (0) to the rear (44); the cord enters the rear in a square channel
    b, w, h = BORE / 2, PW / 2, PH / 2
    r0 = PL - BORE_LEN
    return [box(0, r0, -w, w, -h, h),
            box(r0, PL, b, w, -h, h), box(r0, PL, -w, -b, -h, h),
            box(r0, PL, -b, b, b, h), box(r0, PL, -b, b, -h, -b)]


def write_stl(name: str, parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    for q in parts:
        assert q.is_watertight and q.volume > 0, name
    m = trimesh.util.concatenate(parts)
    m.export(HERE / name)
    return m


# -- drawing -------------------------------------------------------------------------------------------------------
def drawing(vol_plug_mm3: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    fig = plt.figure(figsize=(16.5, 11.7))  # A3 landscape
    fig.suptitle("Junction-box task: real-trial socket, contact spring, plug and cable clip placement (all dimensions mm)",
                 fontsize=13, x=0.02, ha="left")

    def dim(ax, a, b, text, off=(0, 0), fs=7):
        ax.annotate("", a, b, arrowprops=dict(arrowstyle="<->", lw=0.6))
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, fontsize=fs, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.5))

    def setup(ax, title, xl, yl):
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.tick_params(labelsize=6)
        ax.grid(lw=0.2)

    w, cw = BLK_W / 2, CAV_W / 2
    # (1) socket side section (x-z, through the centre)
    ax = fig.add_axes([0.03, 0.52, 0.30, 0.40])
    setup(ax, "A  Socket: section on the centre line (x into the socket, z up from the back plate)", (-8, 30), (-4, 30))
    solid = dict(fc="#bbbbbb", ec="k", lw=0.8)
    for poly in ([(CAV_D, 0), (BLK_L, 0), (BLK_L, BLK_H), (CAV_D, BLK_H)],
                 [(LEAD - Z_LO, 0), (CAV_D, 0), (CAV_D, Z_LO), (LEAD, Z_LO)],
                 [(LEAD_ROOF, Z_HI), (PAD_X - SLOT_LX / 2, Z_HI), (PAD_X - SLOT_LX / 2, BLK_H), (0, BLK_H),
                  (0, Z_HI + LEAD_ROOF)],
                 [(PAD_X + SLOT_LX / 2, Z_HI), (CAV_D, Z_HI), (CAV_D, BLK_H), (PAD_X + SLOT_LX / 2, BLK_H)]):
        ax.add_patch(Polygon(poly, **solid))
    ax.add_patch(Rectangle((-8, -PLATE), 38, PLATE, fc="#dddddd", ec="k", lw=0.5))
    ax.text(-7.5, -PLATE / 2 - 0.3, "box back plate", fontsize=6, va="center")
    st = PAD_UNDER + Z_LO
    ax.add_patch(Rectangle((PAD_X - PAD_LX / 2, st), PAD_LX, BLK_H - st, fc="#e8c35a", ec="k", lw=0.8))
    ax.add_patch(Rectangle((PAD_X - FLANGE[0] / 2, BLK_H), FLANGE[0], FLANGE[2], fc="#e8c35a", ec="k", lw=0.8))
    ax.plot([SPR_CLAMP[0], SPR_CLAMP[1]], [BLK_H + 0.05, BLK_H + 0.05], color="b", lw=2.5)
    ax.plot([SPR_CLAMP[1], PAD_X + 4], [BLK_H + 0.05, BLK_H + FLANGE[2] + 0.05], color="b", lw=1.2)
    ax.text(0.5, BLK_H + 2.6, "contact spring (blue), glued 0-4", fontsize=6, color="b")
    ax.add_patch(Rectangle((-6, Z_LO + 0.4), 20, PH, fc="none", ec="r", lw=0.6, ls="--"))
    ax.text(-5.5, Z_LO + 0.4 + PH / 2, "plug entering", fontsize=6, color="r")
    dim(ax, (0, 27), (BLK_L, 27), f"{BLK_L:.0f}")
    dim(ax, (0, 24.8), (CAV_D, 24.8), f"cavity depth {CAV_D:.0f}", fs=6)
    dim(ax, (27, 0), (27, BLK_H), f"{BLK_H:.0f}")
    dim(ax, (-2.5, Z_LO), (-2.5, Z_HI), f"{CAV_H:.0f}", off=(-1.8, 0))
    dim(ax, (23.5, 0), (23.5, Z_LO), f"{Z_LO:.1f}", off=(1.8, 0))
    dim(ax, (PAD_X + 3.6, Z_LO), (PAD_X + 3.6, st), f"{PAD_UNDER:.1f}", off=(1.6, 0), fs=6)
    dim(ax, (0, 1.5), (PAD_X, 1.5), f"pad centre {PAD_X:.0f}", fs=6)
    ax.text(0.3, Z_LO - 2.3, f"45 deg lead-in x {LEAD:.0f}", fontsize=6)
    ax.text(0.3, Z_HI + 0.9, f"45 x {LEAD_ROOF:.0f}", fontsize=6)

    # (2) socket front view (y-z, looking into the mouth)
    ax = fig.add_axes([0.35, 0.52, 0.28, 0.40])
    setup(ax, "B  Socket: front view (looking into the mouth)", (-20, 20), (-4, 30))
    ax.add_patch(Rectangle((-w, 0), BLK_W, BLK_H, fc="#bbbbbb", ec="k", lw=0.8))
    ax.add_patch(Polygon([(-cw - LEAD, Z_LO), (cw + LEAD, Z_LO), (cw + LEAD, Z_HI + LEAD_ROOF),
                          (-cw - LEAD, Z_HI + LEAD_ROOF)], fc="#dddddd", ec="k", lw=0.5))
    ax.add_patch(Rectangle((-cw, Z_LO), CAV_W, CAV_H, fc="white", ec="k", lw=0.8))
    ax.add_patch(Rectangle((-PAD_LY / 2, PAD_UNDER + Z_LO), PAD_LY, Z_HI - PAD_UNDER - Z_LO, fc="#e8c35a", ec="k",
                           lw=0.6))
    ax.add_patch(Rectangle((-PW / 2, Z_LO + 0.4), PW, PH, fc="none", ec="r", lw=0.6, ls="--"))
    dim(ax, (-w, 25), (w, 25), f"{BLK_W:.0f}")
    dim(ax, (-cw, 1.6), (cw, 1.6), f"cavity {CAV_W:.0f}", fs=6)
    dim(ax, (-PAD_LY / 2, 16.0), (PAD_LY / 2, 16.0), f"pad {PAD_LY}", fs=6)
    ax.text(-19, -3, "lead-in: floor and sides 45 x 4 (mouth 27 wide); roof 45 x 2", fontsize=6)

    # (3) plug
    ax = fig.add_axes([0.66, 0.70, 0.32, 0.22])
    setup(ax, "C  Plug body: top view (tip left), cord enters the rear", (-4, 60), (-12, 12))
    ax.add_patch(Rectangle((0, -PW / 2), PL, PW, fc="#444444", ec="k", lw=0.8))
    ax.add_patch(Rectangle((PL - BORE_LEN, -BORE / 2), BORE_LEN, BORE, fc="white", ec="k", lw=0.5, ls="--"))
    ax.plot([MARK, MARK], [-PW / 2, PW / 2], color="w", lw=1.5)
    ax.plot([PL, 58], [0, 0], color="k", lw=3)
    dim(ax, (0, 10), (PL, 10), f"{PL:.0f}")
    dim(ax, (-2, -PW / 2), (-2, PW / 2), f"{PW:.0f}", off=(-1.5, 0))
    dim(ax, (0, -10), (MARK, -10), f"mark {MARK:.0f} from tip", fs=6)
    dim(ax, (PL - BORE_LEN, -9.5 + 2), (PL, -9.5 + 2), f"cord channel {BORE} sq x {BORE_LEN:.0f}", fs=6)
    ax.text(46, 5, f"height {PH:.0f}", fontsize=6)

    # (4) clip placement (plan view of the box interior)
    ax = fig.add_axes([0.64, 0.46, 0.34, 0.22])
    setup(ax, "D  Back plate, plan view (mm from the box corner)",
          (-5, 205), (-5, 155))
    ox, oy = task.X0 * MM, task.Y0 * MM
    ax.add_patch(Rectangle((0, 0), 200, 150, fc="none", ec="k", lw=0.6))
    ax.text(2, 3, "front wall (y = 0)", fontsize=5)
    ax.text(2, 143, "hinge side", fontsize=5)
    sxw = task.SOCK_X * MM - ox
    syw = task.SOCK_Y * MM - oy
    ax.add_patch(Rectangle((sxw, syw - w), BLK_L, BLK_W, fc="#bbbbbb", ec="k", lw=0.6))
    ax.text(sxw - 50, syw + w + 4, f"socket: mouth at x {sxw:.0f}, centre y {syw:.1f}; mouth faces -x (the robot)", fontsize=5.5)
    cx, cy = CLIP_C[0] * MM - ox, CLIP_C[1] * MM - oy
    L, W = task.CLIP_L * MM, task.CLIP_W * MM
    ax.add_patch(Rectangle((cx - L / 2, cy - W / 2), L, W, fc="#eee8d5", ec="k", lw=0.6))
    ax.plot([cx - 30, sxw], [cy, cy], color="k", lw=1.5)
    ax.text(cx - 20, cy - 22, f"clip: centre x {cx:.1f}, y {cy:.2f}; channel along x", fontsize=5.5)
    ax.text(cx - 20, cy - 31, f"HellermannTyton RB5 clip, {L:.0f} x {W:.0f}", fontsize=5.5)

    # (5) notes: material, spring, tolerances, checks
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.42])
    ax.axis("off")
    notes = [
        "PARTS LIST",
        "1  Socket body (socket_body.stl): PETG, printed mouth-face down or on its side, 0.2 mm layers, 100 % infill, dark grey. "
        f"{BLK_L:.0f} x {BLK_W:.0f} x {BLK_H:.0f} mm; cavity {CAV_W:.0f} x {CAV_H:.0f} x {CAV_D:.0f} deep; roof slot {SLOT_LX} x {SLOT_LY} centred {PAD_X:.0f} in from the mouth.",
        "   Fixed to the box's back plate with 3M VHB double-sided tape (or epoxy), mouth toward the robot, positioned as in view D.",
        f"2  Contact pad (socket_pad.stl): PETG, stem {PAD_LX} x {PAD_LY} mm, flange {FLANGE[0]:.0f} x {FLANGE[1]:.0f} x {FLANGE[2]:.0f} mm resting on the roof (its downward stop); "
        f"underside {PAD_UNDER} mm above the cavity floor at rest.",
        f"3  Contact spring: AISI 301 full-hard stainless shim strip {SPR_T} mm x {SPR_B:.0f} mm x 20 mm (stock shim, sold by the metre/sheet), "
        f"glued flat on the roof top from the mouth (0 to {SPR_CLAMP[1]:.0f} mm, cyanoacrylate), free over the pad flange.",
        f"   Cantilever rate k = E b t^3 / (4 L^3) = 193 GPa x {SPR_B:.0f} x {SPR_T}^3 / (4 x {SPR_L:.1f}^3) = {SPR_K:.2f} N/mm "
        "(simulation: 0.40 N/mm). The 1 mm flange pre-deflects it 1 mm: preload about 0.4 N (simulation 0.4 N).",
        "   A seated plug lifts the pad about 1 mm more: about 0.8 N on the plug (simulation: about 0.9 N, 1.2 mm squeeze measured in a run).",
        "   Why a strip, not a stock coil spring: the nearest stock rate (Century Spring 390, 3.18 mm OD, 0.28 N/mm) is 15.9 mm long free and would need a tower about",
        "   15 mm tall on the 21 mm socket, in the gripper's approach path and not in the simulation; shim strip is a stock material and its rate follows from the formula.",
        f"4  Plug body (plug_body.stl): PETG, 100 % infill, black, {PL:.0f} x {PW:.0f} x {PH:.0f} mm, about {vol_plug_mm3 * 1.27e-3:.0f} g printed; "
        f"the cord's end is glued into the {BORE} mm square rear channel ({BORE_LEN:.0f} mm deep) with hot-melt glue;",
        f"   total with glue 11-13 g (simulation 12 g). Draw a white line around it {MARK:.0f} mm from the tip. No latch, no contacts needed for scoring.",
        "5  Cable clip (commercial, as simulated): HellermannTyton RB5-PA66-NA, article 151-14059, one-piece adhesive snap-in clip, natural PA66,",
        "   L 19.0 x W 19.0 x H 6.0 mm, W2 9.7 mm, cable up to 5.0 mm (maker's product page). Stick it on the back plate centred as in view D, its cable channel along x.",
        "   The simulation models it at these sizes; its arm thickness, lip and holding force are not published (the simulation's are estimates).",
        "",
        "TOLERANCES  printed parts +/- 0.2 mm; cavity width and height +0 / +0.3 mm (measure with calipers; a 15.0 x 12.0 plug must enter freely to the pad);",
        "pad slot clearance 0.2 mm per side, pad must drop under its own weight; spring strip thickness +/- 0.01 mm; positions on the back plate +/- 1 mm.",
        "",
        "ACCEPTANCE CHECKS before trials (a 5 N push-pull gauge or a luggage scale, a ruler)",
        "1  Plug pushed in by hand seats to 16 mm (the 12 mm line hidden) with at most 3 N.   2  Pulling the seated plug straight out needs 0.7 to 1.3 N "
        "(simulation about 1 N); bend the strip or shim the flange to adjust.",
        "3  Pad lift: pushing the pad up 1 mm needs 0.6 to 1.0 N.   4  Clip: the 5 mm cord pressed in by a fingertip with at most 4 N snaps in and stays",
        "   when the door test is done by hand (record the force a thread loop under the clipped cord needs to pull it straight out; not published by the maker).",
        "",
        "Generated by make_hardware.py from the dimensions in task.py; STLs in mm, each a set of touching closed solids that a slicer merges.",
    ]
    ax.text(0, 1, "\n".join(notes), fontsize=7.3, va="top", family="monospace")
    fig.savefig(HERE / "drawing.svg")
    fig.savefig(HERE / "drawing.pdf")
    if len(sys.argv) > 1:  # optional preview PNG path (kept out of the task folder)
        fig.savefig(sys.argv[1], dpi=110)


def main() -> None:
    s = write_stl("socket_body.stl", socket_body())
    p = write_stl("socket_pad.stl", socket_pad())
    g = write_stl("plug_body.stl", plug_body())
    vol = sum(q.volume for q in plug_body())
    print(f"socket bbox {np.round(s.bounds, 2).tolist()}  pad bbox {np.round(p.bounds, 2).tolist()}  "
          f"plug bbox {np.round(g.bounds, 2).tolist()} vol {vol:.0f} mm3")
    print(f"spring free length {SPR_L:.2f} mm, rate {SPR_K:.3f} N/mm")
    drawing(vol)


if __name__ == "__main__":
    main()
