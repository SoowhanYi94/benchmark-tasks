"""Writes fixture_dimensions.svg (plan and side view at the start, mm, grid every 10 mm, labels every 50 mm) and prints the README's
fixture table, both from the shapes in trunnion_fixture/mech.py and the placement in task.py's constants. NumPy only.
Run from the task folder:  python dev/fixture_diagram.py"""
import ast
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from trunnion_fixture import mech as M  # noqa: E402

consts = {n.targets[0].elts[0].id if isinstance(n.targets[0], ast.Tuple) else n.targets[0].id: n.value
          for n in ast.parse((HERE / "task.py").read_text()).body if isinstance(n, ast.Assign)}
AX, Y0 = (ast.literal_eval(e) for e in consts["AX"].elts)  # "AX, Y0 = 0.24, -0.03"
AX, Y0 = AX * 1000, Y0 * 1000

NAMES = {"base_near": "near base plate", "base_far": "far base plate", "tie_p": "tie bar", "tie_m": "tie bar", "near": "near pillar",
         "far": "far pillar, lower", "far_top": "far pillar, upper", "rail_p": "rail", "rail_m": "rail", "ledge_0": "ledge (foot side)",
         "ledge_1": "ledge (worm side)", "stop": "near stop plate", "near_plate": "near end plate", "plate_top": "far end plate, above notch",
         "plate_p": "far end plate, beside notch", "plate_m": "far end plate, beside notch", "axle_far": "far axle stub", "axle_near": "near axle stub",
         "crank": "crank", "pin": "crank pin", "sleeve": "knob sleeve", "plate": "retaining bar", "lug": "lug", "rod": "lock bolt"}
MAT = {"steel": "steel", "dark": "dark steel", "cradle": "aluminium", "knob": "plastic", "bar": "aluminium", "bolt": "aluminium", "holster": "steel"}
MOVES = {"anchor": "fixed to the table", "cradle": "turns with the cradle, 0 to 195 deg", "knob": "turns freely on the pin",
         "bar": "slides 67.5 along +y", "bolt": "slides 15 along -y"}


def boxes():
    out = []
    for body, name, lo, hi, col in M.fixture_boxes():
        lo = (lo[0] * 1000 + AX, lo[1] * 1000 + Y0, lo[2] * 1000)
        hi = (hi[0] * 1000 + AX, hi[1] * 1000 + Y0, hi[2] * 1000)
        nm = NAMES.get(name, name) if not (body == "bolt" and name == "lug") else "bolt lug"
        nm = "bar lug" if (body == "bar" and name == "lug") else nm
        out.append((body, nm, lo, hi, M.COLORS[col], MAT.get(col, col)))
    for name, lo, hi in M.station_boxes():
        out.append(("anchor", "driver holster", tuple(v * 1000 for v in lo), tuple(v * 1000 for v in hi), M.COLORS["holster"], "steel"))
    p = M.POST
    out.append(("anchor", "flange kitting post (round)", (p["x"] * 1000 - 25, p["y"] * 1000 - 25, 0), (p["x"] * 1000 + 25, p["y"] * 1000 + 25, p["h"] * 1000),
                M.COLORS["steel"], "steel"))
    return out


def rgb(c):
    return "#%02x%02x%02x" % tuple(round(v * 255) for v in c)


def view(bxs, ia, ib, x0, y0, s, lo_a, hi_a, lo_b, hi_b, title, la, lb):
    """One orthographic view: axis ia to the right, ib up; paper origin (x0, y0) at (lo_a, lo_b)."""
    g = [f'<text x="{x0}" y="{y0 - (hi_b - lo_b) * s - 14}" font-size="14" font-weight="bold">{title}</text>']
    for v in range(int(lo_a // 10 * 10), int(hi_a) + 1, 10):
        X = x0 + (v - lo_a) * s
        g.append(f'<line x1="{X:.1f}" y1="{y0}" x2="{X:.1f}" y2="{y0 - (hi_b - lo_b) * s:.1f}" stroke="#eee" stroke-width="0.5"/>')
        if v % 50 == 0:
            g.append(f'<text x="{X:.1f}" y="{y0 + 12}" font-size="8" text-anchor="middle">{v}</text>')
    for v in range(int(lo_b // 10 * 10), int(hi_b) + 1, 10):
        Y = y0 - (v - lo_b) * s
        g.append(f'<line x1="{x0}" y1="{Y:.1f}" x2="{x0 + (hi_a - lo_a) * s:.1f}" y2="{Y:.1f}" stroke="#eee" stroke-width="0.5"/>')
        if v % 50 == 0:
            g.append(f'<text x="{x0 - 4}" y="{Y + 3:.1f}" font-size="8" text-anchor="end">{v}</text>')
    for body, nm, lo, hi, col, _ in sorted(bxs, key=lambda b: b[2][3 - ia - ib] if ia + ib < 3 else 0):
        X, Y = x0 + (lo[ia] - lo_a) * s, y0 - (hi[ib] - lo_b) * s
        g.append(f'<rect x="{X:.1f}" y="{Y:.1f}" width="{(hi[ia] - lo[ia]) * s:.1f}" height="{(hi[ib] - lo[ib]) * s:.1f}" fill="{rgb(col)}" '
                 f'fill-opacity="0.55" stroke="#222" stroke-width="0.6"><title>{nm}</title></rect>')
    g.append(f'<text x="{x0 + (hi_a - lo_a) * s / 2}" y="{y0 + 26}" font-size="10" text-anchor="middle">{la} (mm)</text>')
    g.append(f'<text x="{x0 - 30}" y="{y0 - (hi_b - lo_b) * s / 2}" font-size="10" text-anchor="middle" transform="rotate(-90 {x0 - 30} {y0 - (hi_b - lo_b) * s / 2})">{lb} (mm)</text>')
    return g


def main():
    bxs = boxes()
    s = 2.0
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="1080" font-family="sans-serif">', '<rect width="1300" height="1080" fill="white"/>']
    svg += view(bxs, 1, 0, 60, 600, s, -230, 210, 110, 370, "Plan view (looking down; cradle level, as at the start)", "y", "x")
    svg += view([b for b in bxs if b[1] not in ("driver holster", "flange kitting post (round)")], 1, 2, 60, 1040, s, -230, 210, 0, 170,
                "Side view (looking along -x, from the far side; cradle level)", "y", "z, above the table")
    svg.append('<text x="960" y="660" font-size="11">Every part is a box; hover a box for its name.</text>')
    svg.append('<text x="960" y="676" font-size="11">Numbers: README table (generated from the same code).</text>')
    svg.append("</svg>")
    (HERE / "fixture_dimensions.svg").write_text("\n".join(svg))
    print("| Part | Material | Size x * y * z (mm) | Position x / y / z (mm) | Moves or fixed |")
    print("|---|---|---|---|---|")
    for body, nm, lo, hi, col, mat in bxs:
        size = " * ".join(f"{hi[k] - lo[k]:g}" for k in range(3))
        pos = " / ".join(f"{lo[k]:g} to {hi[k]:g}" for k in range(3))
        print(f"| {nm} | {mat} | {size} | {pos} | {MOVES[body]} |")


if __name__ == "__main__":
    main()
