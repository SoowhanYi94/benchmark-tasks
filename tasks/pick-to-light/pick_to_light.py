"""On the station table: a pick-to-light rack of four parts bins stands across the far side of the bench, each holding
three identical parts (from left to right: yellow cylinders, grey blocks, orange cylinders, blue blocks). In front of
each bin are two indicator lamps; a lit lamp (bright green) means pick one part from that bin, so a bin with both lamps
lit needs two parts and a bin with both dark needs none. Now the yellow bin's two lamps are lit, the orange bin's
left lamp is lit, and the rest are dark. An order tote stands in front, between the arms. The robot must pick exactly
what the lamps show into the tote.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/pick-to-light/pick_to_light.py \\
           --instruction "pick the order shown by the lamps into the order tote: from each bin, one part for every lit green lamp in front of it, and nothing from bins whose lamps are dark" --scorer task_score ...
"""

import math

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]
LIT, DARK = (0.30, 1.00, 0.35), (0.12, 0.13, 0.12)

RACK_X = 0.34
BIN_Y = {"yellow": 0.15, "grey": 0.05, "orange": -0.05, "blue": -0.15}
BIN_IN, BIN_WALL, BIN_H, BIN_FLOOR = 0.09, 0.01, 0.03, 0.006
BIN_Z = TOP + BIN_FLOOR
PART_SLOTS = (-0.03, 0.0, 0.03)  # each bin's three parts, side by side along x
PARTS = {  # kind, size, colour per bin
    "yellow": dict(kind="cyl", size=(0.013, 0.035), mass=0.05, color=(0.95, 0.82, 0.15)),
    "grey": dict(kind="box", size=(0.025, 0.025, 0.035), mass=0.06, color=(0.55, 0.56, 0.58)),
    "orange": dict(kind="cyl", size=(0.013, 0.035), mass=0.05, color=(0.95, 0.50, 0.10)),
    "blue": dict(kind="box", size=(0.025, 0.025, 0.035), mass=0.06, color=(0.15, 0.35, 0.80)),
}
LAMPS = {"yellow": 2, "grey": 0, "orange": 1, "blue": 0}  # lit lamps in front of each bin
TOTE = dict(center=(0.18, 0.0), inner=(0.12, 0.14), wall=0.01, height=0.04, floor=0.004)
WRONG_PENALTY = 1.5
NAMES = {b: [f"{b}_part_{i}" for i in range(len(PART_SLOTS))] for b in BIN_Y}


def _box(name, cx, cy, ix, iy, w, h, f, c):
    return [
        Box(f"{name}_base", pos=(cx, cy, TOP + f / 2), size=(ix + 2 * w, iy + 2 * w, f), color=c),
        Box(f"{name}_far", pos=(cx + ix / 2 + w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=c),
        Box(f"{name}_near", pos=(cx - ix / 2 - w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=c),
        Box(f"{name}_left", pos=(cx, cy + iy / 2 + w / 2, TOP + h / 2), size=(ix, w, h), color=c),
        Box(f"{name}_right", pos=(cx, cy - iy / 2 - w / 2, TOP + h / 2), size=(ix, w, h), color=c),
    ]


def _part(name, b, x, y):
    p = PARTS[b]
    if p["kind"] == "box":
        return Box(name, pos=(x, y, BIN_Z + p["size"][2] / 2), size=p["size"], mass=p["mass"], color=p["color"])
    r, h = p["size"]
    return Cylinder(name, pos=(x, y, BIN_Z + h / 2), radius=r, height=h, mass=p["mass"], color=p["color"])


OBJECTS = [
    *[o for b, y in BIN_Y.items() for o in _box(f"bin_{b}", RACK_X, y, BIN_IN, BIN_IN, BIN_WALL, BIN_H, BIN_FLOOR, (0.30, 0.31, 0.33))],
    # two lamps in front of each bin (left one at +y), lit or dark
    *[Box(f"lamp_{b}_{k}", pos=(RACK_X - BIN_IN / 2 - BIN_WALL - 0.015, y + (0.02 if k == 0 else -0.02), TOP + 0.006),
          size=(0.015, 0.015, 0.012), color=LIT if k < LAMPS[b] else DARK) for b, y in BIN_Y.items() for k in (0, 1)],
    *_box("order_tote", *TOTE["center"], *TOTE["inner"], TOTE["wall"], TOTE["height"], TOTE["floor"], (0.20, 0.35, 0.60)),
    *[_part(n, b, RACK_X + dx, BIN_Y[b]) for b in BIN_Y for n, dx in zip(NAMES[b], PART_SLOTS)],
]


# -- scoring --------------------------------------------------------------------------------------------------------
def in_tote(name, objects):
    (cx, cy), (ix, iy) = TOTE["center"], TOTE["inner"]
    o = objects[name]
    return bool(abs(o.pos[0] - cx) <= ix / 2 and abs(o.pos[1] - cy) <= iy / 2 and o.low[2] <= TOP + TOTE["floor"] + 0.03)


def _tally(objects):
    correct = wrong = 0
    for b, names in NAMES.items():
        n = sum(in_tote(x, objects) for x in names)
        correct += min(n, LAMPS[b])
        wrong += max(0, n - LAMPS[b])
    return correct, wrong


def success(objects):
    return _tally(objects) == (sum(LAMPS.values()), 0)


def score(objects):
    """(parts picked as the lamps show - 1.5 x parts picked beyond them, from any bin) / 3, at least 0."""
    correct, wrong = _tally(objects)
    return float(max(0.0, correct - WRONG_PENALTY * wrong) / sum(LAMPS.values()))


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
Z_UP = TOP + 0.10


class _Plan:
    """Waypoints for both arms in phases: within a phase each arm moves independently, then both wait for the slower."""

    def __init__(self):
        self.wps = {a: [] for a in SIDE}
        self.t = {a: 0.0 for a in SIDE}
        self.last = {}
        for a in SIDE:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None):
        p, f, y = self.last[arm]
        p = p if pos is None else pos
        f = f if finger is None else finger
        y = y if yaw is None else yaw
        self.wps[arm].append(Waypoint(dur, p, f, label, y))
        self.last[arm] = (p, f, y)
        self.t[arm] += dur

    def sync(self):
        end = max(self.t.values())
        for a in SIDE:
            if self.t[a] < end - 1e-9:
                self.go(a, end - self.t[a], label="wait")


def _pick(plan, name, b, dx, drop):
    arm = "left" if BIN_Y[b] > 0 else "right"
    p = PARTS[b]
    width = 2 * p["size"][0] if p["kind"] == "cyl" else p["size"][1]
    grip, ready = -FINGER_PER_M * (width - 0.004), -FINGER_PER_M * (width + 0.016)
    yaw = min((k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))  # fingers along y, between parts
    x, y = RACK_X + dx, BIN_Y[b]
    gz = BIN_Z + 0.025
    plan.go(arm, 1.2, (x, y, Z_UP), ready, f"{name}: above it", yaw)
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 1.0, (x, y, gz), label="descend")
    plan.go(arm, 0.6, finger=grip, label="grip")
    plan.go(arm, 0.9, (x, y, Z_UP), label="lift")
    plan.go(arm, 1.4, (*drop, Z_UP), label="over the order tote")
    plan.go(arm, 0.8, (*drop, TOP + 0.07), label="lower into it")
    plan.go(arm, 0.5, finger=ready, label="release")
    plan.go(arm, 0.6, (*drop, Z_UP), label="up")
    plan.go(arm, 1.0, (0.18, SIDE[arm] * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
    plan.sync()


def demo():
    plan = _Plan()
    (cx, cy) = TOTE["center"]
    drops = [(cx - 0.03, cy + 0.03), (cx + 0.03, cy + 0.03), (cx, cy - 0.03)]
    k = 0
    for b, n in LAMPS.items():
        for i in range(n):  # the parts nearest the arms first
            _pick(plan, NAMES[b][i], b, PART_SLOTS[i], drops[k])
            k += 1
    return plan.wps


# -- self-check (pure Python): the plan fits the run limit and the scorer covers the obvious outcomes ----------------
if __name__ == "__main__":
    class _O:
        def __init__(self, x, y, z):
            self.pos, self.low = np.array((x, y, z + 0.015)), np.array((x - 0.01, y - 0.01, z))

    racked = {n: _O(RACK_X + dx, BIN_Y[b], BIN_Z) for b in BIN_Y for n, dx in zip(NAMES[b], PART_SLOTS)}
    tote = _O(*TOTE["center"], TOP + TOTE["floor"])
    assert score(racked) == 0.0 and not success(racked)
    done = dict(racked, yellow_part_0=tote, yellow_part_1=tote, orange_part_0=tote)
    assert score(done) == 1.0 and success(done)
    assert abs(score(dict(done, grey_part_0=tote)) - (3 - 1.5) / 3) < 1e-9  # one from a dark bin
    assert score(dict(done, orange_part_1=tote, orange_part_2=tote)) == 0.0  # emptied the orange bin
    assert score({n: tote for n in racked}) == 0.0  # everything
    t = max(sum(w.duration for w in wps) for wps in demo().values())
    assert t * 20 < 2400, t
    print(f"ok: demo {t:.1f} s ({t * 20:.0f} steps)")
