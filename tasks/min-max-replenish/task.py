"""On the station table: four blue line-side parts bins stand in a row on the far side of the bench (A to D, left to
right), each with a min/max label lying flat in front of it: a row of squares, one per part of the bin's maximum, the
first ones red up to its minimum (its reorder point) and the rest green. A bin holding no more parts than its minimum
is refilled to exactly its maximum; any other bin is left as it is. Labels: A min 1 / max 3, B min 1 / max 3,
C min 2 / max 3, D min 1 / max 2. Now A holds no parts, B 2, C 2 and D 2. A bulk tote at the front holds six spare
3/4-inch drive sockets, the bins' part. The robot must replenish the bins by their labels from the bulk tote.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/min-max-replenish/task.py \\
           --instruction "Replenish the line-side bins by their min/max labels: refill only the bins holding no more parts than their minimum (the red squares on the label) up to exactly their maximum (all the squares) with sockets from the bulk tote, and leave every other bin as it is." --scorer task_score ...
"""

import math

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Asset, Box

SIMREADY = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/SimReady/Industrial/"
SOCKET_MODEL = "sr_socket_3_4in"  # NVIDIA SimReady Socket_3_4in_A01: a 3/4-inch drive socket, 3.0 cm across, 4.7 cm tall


def _register_simready():
    """Make the SimReady model available to `Asset` under SOCKET_MODEL (the task file registers it; no simulator change)."""
    try:
        from isaaclab_arena.assets.registries import AssetRegistry
    except ImportError:  # offline tests: no Arena, nothing to register
        return

    url = SIMREADY + "Tools/Hand/Socket_3_4in_A01/sm_handTool_socket_3_4in_a01_01.usd"
    r = AssetRegistry()
    if not r.is_registered(SOCKET_MODEL, ensure_loaded=False):
        r.register(type(SOCKET_MODEL, (), {"name": SOCKET_MODEL, "tags": ["object"], "usd_path": url}), SOCKET_MODEL)


_register_simready()

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# bin: (centre x, centre y), (min, max), parts at the start
BINS = {"bin_a": (0.33, 0.21), "bin_b": (0.33, 0.07), "bin_c": (0.33, -0.07), "bin_d": (0.33, -0.21)}
LEVELS = {"bin_a": (1, 3), "bin_b": (1, 3), "bin_c": (2, 3), "bin_d": (1, 2)}
START_COUNT = {"bin_a": 0, "bin_b": 2, "bin_c": 2, "bin_d": 2}
BIN_IN, BIN_WALL, BIN_H = 0.09, 0.01, 0.03  # open-bottomed: the parts stand on the table, so nothing pops at t = 0
IN_TOL = 0.032  # a part counts when its centre is within 3.2 cm of the bin centre in x and y (inside the walls)
ON_FLOOR = 0.01  # ... and its underside is within 1 cm of the bin floor (not on another part or a wall)
SLOTS = [(-0.022, -0.022), (-0.022, 0.022), (0.022, 0.0)]  # three places in a bin
PART = (0.03, 0.03, 0.047)  # the socket: 3.0 cm across, 4.7 cm tall, chrome-vanadium steel, 90 g
BULK = dict(center=(0.17, 0.0), inner=(0.06, 0.30), wall=0.01, height=0.02)
BULK_Y = [-0.125, -0.075, -0.025, 0.025, 0.075, 0.125]
LABEL = dict(x=0.2425, depth=0.03, width=0.09, pip=0.018, pitch=0.026)

PARTS = {}
for b, n in START_COUNT.items():
    for k in range(n):
        PARTS[f"{b}_part_{k + 1}"] = (BINS[b][0] + SLOTS[k][0], BINS[b][1] + SLOTS[k][1], TOP)
for k, y in enumerate(BULK_Y):
    PARTS[f"bulk_part_{k + 1}"] = (BULK["center"][0], y, TOP)
NEED = {b: (LEVELS[b][1] - n if n <= LEVELS[b][0] else 0) for b, n in START_COUNT.items()}  # A +3, C +1
TOTAL = sum(NEED.values())


def _walls(name, cx, cy, ix, iy, w, h, c):
    return [
        Box(f"{name}_far", pos=(cx + ix / 2 + w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=c),
        Box(f"{name}_near", pos=(cx - ix / 2 - w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=c),
        Box(f"{name}_left", pos=(cx, cy + iy / 2 + w / 2, TOP + h / 2), size=(ix, w, h), color=c),
        Box(f"{name}_right", pos=(cx, cy - iy / 2 - w / 2, TOP + h / 2), size=(ix, w, h), color=c),
    ]


def _label(b):
    """A white card with one square per part of the max, left to right: red up to the min, then green."""
    lo, hi = LEVELS[b]
    y0 = BINS[b][1] + (hi - 1) * LABEL["pitch"] / 2
    card = Box(f"{b}_label", pos=(LABEL["x"], BINS[b][1], TOP + 0.001), size=(LABEL["depth"], LABEL["width"], 0.002),
               color=(0.95, 0.95, 0.95))
    pips = [Box(f"{b}_label_{k + 1}", pos=(LABEL["x"], y0 - k * LABEL["pitch"], TOP + 0.0025),
                size=(LABEL["pip"], LABEL["pip"], 0.001), color=(0.85, 0.1, 0.1) if k < lo else (0.1, 0.65, 0.2))
            for k in range(hi)]
    return [card, *pips]


OBJECTS = [
    *[o for b, (x, y) in BINS.items() for o in _walls(b, x, y, BIN_IN, BIN_IN, BIN_WALL, BIN_H, (0.20, 0.35, 0.60))],
    *[o for b in BINS for o in _label(b)],
    *_walls("bulk_tote", *BULK["center"], *BULK["inner"], BULK["wall"], BULK["height"], (0.30, 0.31, 0.33)),
    *[Asset(n, asset=SOCKET_MODEL, xy=(x, y), extent=PART[0], mass=0.09) for n, (x, y, z) in PARTS.items()],
]


# -- scoring --------------------------------------------------------------------------------------------------------
def in_bin(o, b):
    """Released in bin b: still, centre inside the walls, underside on the bin floor."""
    cx, cy = BINS[b]
    return bool(abs(o.pos[0] - cx) <= IN_TOL and abs(o.pos[1] - cy) <= IN_TOL and o.low[2] <= TOP + ON_FLOOR
                and o.resting())


def count_in(b, objects):
    return sum(1 for n in PARTS if in_bin(objects[n], b))


def tally(objects):
    """(correct, mistakes). Correct: parts added to a bin at or below its min, up to its max. Mistakes: any change
    to a bin above its min, parts past a refilled bin's max, and parts taken out of a refilled bin."""
    correct = mistakes = 0
    for b, n0 in START_COUNT.items():
        n = count_in(b, objects)
        if NEED[b]:
            correct += min(max(n - n0, 0), NEED[b])
            mistakes += max(0, n - LEVELS[b][1]) + max(0, n0 - n)
        else:
            mistakes += abs(n - n0)
    return correct, mistakes


def success(objects):
    return tally(objects) == (TOTAL, 0)


def score(objects):
    """0.25 per part correctly added (4 needed: 3 in A, 1 in C); 0 if any mistake."""
    correct, mistakes = tally(objects)
    return 0.0 if mistakes else correct / TOTAL


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
Z_UP = TOP + 0.10
DROP_H = 0.022  # let go this high above the bin floor: the finger pads clear the parts already in it


class _Plan:
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


def demo():
    """The left arm moves three sockets into bin A, the right arm one into bin C, at the same time; B and D are not
    touched."""
    plan = _Plan()
    grip, ready = -FINGER_PER_M * (PART[0] - 0.004), -FINGER_PER_M * (PART[0] + 0.012)
    g = PART[2] - 0.008
    bulk = {"left": sorted((n for n in PARTS if n.startswith("bulk") and PARTS[n][1] > 0), key=lambda n: -PARTS[n][1]),
            "right": sorted((n for n in PARTS if n.startswith("bulk") and PARTS[n][1] < 0), key=lambda n: PARTS[n][1])}
    for b, need in NEED.items():
        arm = "left" if BINS[b][1] > 0 else "right"
        yaw = min((math.pi / 2 + j * math.pi for j in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))  # along x
        for k in range(need):
            n = bulk[arm].pop(0)
            px, py, _ = PARTS[n]
            slot = SLOTS[START_COUNT[b] + k]
            qx, qy = BINS[b][0] + slot[0], BINS[b][1] + slot[1]
            plan.go(arm, 1.2, (px, py, Z_UP), ready, f"{n} to {b}: above it", yaw)
            plan.go(arm, 0.4, label="settle")
            plan.go(arm, 1.0, (px, py, TOP + g), label="descend")
            plan.go(arm, 0.5, finger=grip, label="grip")
            plan.go(arm, 0.8, (px, py, Z_UP), label="lift")
            plan.go(arm, 1.4, (qx, qy, Z_UP), label="over its bin")
            plan.go(arm, 0.4, label="settle")
            plan.go(arm, 0.9, (qx, qy, TOP + g + DROP_H), label="lower")
            plan.go(arm, 0.4, finger=ready, label="drop it in")
            plan.go(arm, 0.6, (qx, qy, Z_UP), label="up")
            plan.go(arm, 1.0, (0.18, SIDE[arm] * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
    plan.sync()
    return plan.wps
