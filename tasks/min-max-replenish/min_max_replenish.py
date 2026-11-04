"""On the station table: three line-side parts bins stand in a row on the far side of the bench (left, middle, right),
each run on a min/max rule: when a bin holds fewer than its minimum of 2 parts it is refilled up to its maximum of 3;
a bin at or above its minimum is left alone. Now the left bin holds 1 part, the middle bin 3 and the right bin none.
A bulk tote at the front holds six more of the same parts, 3/4-inch drive sockets. The robot must replenish the bins by the rule from the bulk
tote.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/min-max-replenish/min_max_replenish.py \\
           --instruction "replenish the line-side bins by their min/max rule: refill any bin holding fewer than 2 parts up to 3 parts from the bulk tote, and leave bins holding 2 or more alone" --scorer task_score ...
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
    from isaaclab_arena.assets.registries import AssetRegistry

    url = SIMREADY + "Tools/Hand/Socket_3_4in_A01/sm_handTool_socket_3_4in_a01_01.usd"
    r = AssetRegistry()
    if not r.is_registered(SOCKET_MODEL, ensure_loaded=False):
        r.register(type(SOCKET_MODEL, (), {"name": SOCKET_MODEL, "tags": ["object"], "usd_path": url}), SOCKET_MODEL)


_register_simready()

SCENE = "station"
TOP = TABLE_TOP[SCENE]

MIN, MAX = 2, 3
BINS = {"bin_left": (0.33, 0.14), "bin_middle": (0.33, 0.0), "bin_right": (0.33, -0.14)}
BIN_IN, BIN_WALL, BIN_H, BIN_FLOOR = 0.09, 0.01, 0.03, 0.0008  # floor pad under 1 mm: library models spawn on the table
BIN_Z = TOP + BIN_FLOOR
SLOTS = [(-0.022, -0.022), (-0.022, 0.022), (0.022, 0.0)]  # three places in a bin
PART = (0.03, 0.03, 0.047)  # the socket: 3.0 cm across, 4.7 cm tall, chrome-vanadium steel, 90 g
START_COUNT = {"bin_left": 1, "bin_middle": 3, "bin_right": 0}
BULK = dict(center=(0.17, 0.0), inner=(0.06, 0.30), wall=0.01, height=0.02, floor=0.0008)
BULK_Z = TOP + BULK["floor"]
BULK_Y = [-0.125, -0.075, -0.025, 0.025, 0.075, 0.125]
WRONG_PENALTY = 1.5

PARTS = {}
for b, n in START_COUNT.items():
    for k in range(n):
        PARTS[f"{b}_part_{k + 1}"] = (BINS[b][0] + SLOTS[k][0], BINS[b][1] + SLOTS[k][1], BIN_Z)
for k, y in enumerate(BULK_Y):
    PARTS[f"bulk_part_{k + 1}"] = (BULK["center"][0], y, BULK_Z)
NEED = {b: (MAX - n if n < MIN else 0) for b, n in START_COUNT.items()}  # left +2, middle 0, right +3


def _box5(name, cx, cy, ix, iy, w, h, f, c):
    return [
        Box(f"{name}_base", pos=(cx, cy, TOP + f / 2), size=(ix + 2 * w, iy + 2 * w, f), color=c),
        Box(f"{name}_far", pos=(cx + ix / 2 + w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=c),
        Box(f"{name}_near", pos=(cx - ix / 2 - w / 2, cy, TOP + h / 2), size=(w, iy + 2 * w, h), color=c),
        Box(f"{name}_left", pos=(cx, cy + iy / 2 + w / 2, TOP + h / 2), size=(ix, w, h), color=c),
        Box(f"{name}_right", pos=(cx, cy - iy / 2 - w / 2, TOP + h / 2), size=(ix, w, h), color=c),
    ]


OBJECTS = [
    *[o for b, (x, y) in BINS.items() for o in _box5(b, x, y, BIN_IN, BIN_IN, BIN_WALL, BIN_H, BIN_FLOOR, (0.20, 0.35, 0.60))],
    *_box5("bulk_tote", *BULK["center"], *BULK["inner"], BULK["wall"], BULK["height"], BULK["floor"], (0.30, 0.31, 0.33)),
    *[Asset(n, asset=SOCKET_MODEL, xy=(x, y), extent=PART[0], mass=0.09) for n, (x, y, z) in PARTS.items()],
]


# -- scoring --------------------------------------------------------------------------------------------------------
def count_in(b, objects):
    (cx, cy) = BINS[b]
    return sum(1 for n in PARTS if abs(objects[n].pos[0] - cx) <= BIN_IN / 2 and abs(objects[n].pos[1] - cy) <= BIN_IN / 2
               and objects[n].low[2] <= BIN_Z + 0.04)


def _tally(objects):
    correct = wrong = 0
    for b, n0 in START_COUNT.items():
        n = count_in(b, objects)
        target = n0 + NEED[b]
        if NEED[b]:
            correct += min(max(n - n0, 0), NEED[b])
        wrong += max(0, n - target) + max(0, n0 - n)  # overfilled, or parts taken out of any bin
    return correct, wrong


def success(objects):
    return _tally(objects) == (sum(NEED.values()), 0)


def score(objects):
    """(parts added to the bins below minimum, up to their maximum - 1.5 x parts beyond a bin's maximum or taken from
    a bin that was at its minimum or above) / 5, at least 0."""
    correct, wrong = _tally(objects)
    return float(max(0.0, correct - WRONG_PENALTY * wrong) / sum(NEED.values()))


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
    plan = _Plan()
    grip, ready = -FINGER_PER_M * (PART[0] - 0.004), -FINGER_PER_M * (PART[0] + 0.012)
    g = PART[2] - 0.008
    bulk = {"left": [n for n in PARTS if n.startswith("bulk") and PARTS[n][1] > 0][::-1],
            "right": [n for n in PARTS if n.startswith("bulk") and PARTS[n][1] < 0]}
    for b, need in NEED.items():
        for k in range(need):
            arm = "left" if BINS[b][1] > 0 else "right"
            n = bulk[arm].pop(0)
            px, py, _ = PARTS[n]
            slot = SLOTS[START_COUNT[b] + k]
            qx, qy = BINS[b][0] + slot[0], BINS[b][1] + slot[1]
            yaw = min((math.pi / 2 + j * math.pi for j in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))  # along x
            plan.go(arm, 1.2, (px, py, Z_UP), ready, f"{n} to {b}: above it", yaw)
            plan.go(arm, 0.4, label="settle")
            plan.go(arm, 1.0, (px, py, BULK_Z + g), label="descend")
            plan.go(arm, 0.5, finger=grip, label="grip")
            plan.go(arm, 0.8, (px, py, Z_UP), label="lift")
            plan.go(arm, 1.4, (qx, qy, Z_UP), label="over its bin")
            plan.go(arm, 0.4, label="settle")
            plan.go(arm, 0.9, (qx, qy, BIN_Z + g + DROP_H), label="lower")
            plan.go(arm, 0.4, finger=ready, label="drop it in")
            plan.go(arm, 0.6, (qx, qy, Z_UP), label="up")
            plan.go(arm, 1.0, (0.18, SIDE[arm] * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
            plan.sync()
    return plan.wps


# -- self-check (pure Python): the plan fits the run limit and the scorer covers the obvious outcomes ----------------
if __name__ == "__main__":
    class _O:
        def __init__(self, x, y, z):
            self.pos, self.low = np.array((x, y, z + 0.015)), np.array((x - 0.015, y - 0.015, z))

    assert NEED == {"bin_left": 2, "bin_middle": 0, "bin_right": 3}
    start = {n: _O(*xyz) for n, xyz in PARTS.items()}
    assert score(start) == 0.0 and not success(start)
    done = dict(start)
    bulk = [n for n in PARTS if n.startswith("bulk")]
    for n, b in zip(bulk, ["bin_left", "bin_left", "bin_right", "bin_right", "bin_right"]):
        done[n] = _O(*BINS[b], BIN_Z)
    assert score(done) == 1.0 and success(done)
    to_middle = dict(done, bulk_part_6=_O(*BINS["bin_middle"], BIN_Z))  # topped up a bin that was full
    assert abs(score(to_middle) - (5 - 1.5) / 5) < 1e-9
    one_each = dict(start, bulk_part_1=_O(*BINS["bin_left"], BIN_Z), bulk_part_2=_O(*BINS["bin_middle"], BIN_Z), bulk_part_3=_O(*BINS["bin_right"], BIN_Z))
    assert abs(score(one_each) - (2 - 1.5) / 5) < 1e-9
    left_part = [n for n in PARTS if n.startswith("bin_left")][0]
    robbed = dict(start, **{left_part: _O(*BINS["bin_right"], BIN_Z)})  # exploit (council): the low bin's own part moved
    assert score(robbed) == 0.0, score(robbed)
    t = max(sum(w.duration for w in wps) for wps in demo().values())
    assert t * 20 < 2400, t
    print(f"ok: demo {t:.1f} s ({t * 20:.0f} steps)")
