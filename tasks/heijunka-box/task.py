"""On the station table: a simplified single-row heijunka (production levelling) board runs across the far side of the
table: six pockets in a row, one per time slot of the shift. Six production kanbans, one per batch, wait in a tray at
the front, grouped by product as the orders came in: three for product A (red), two for B (blue), one for C (yellow).
Levelling the mix means spreading each product across the shift: the red kanbans go in every other pocket, so no two
neighbouring pockets hold the same product. The robot must put one kanban in each pocket in a levelled order.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/heijunka-box/task.py \\
           --instruction "Load the heijunka box to level the production mix: put one kanban in each of the six pockets, with the red kanbans in every other pocket so that no two neighbouring pockets hold the same colour." --scorer task_score ...
"""

import math

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
PRODUCT_COLOR = {"A": (0.80, 0.12, 0.10), "B": (0.15, 0.35, 0.80), "C": (0.95, 0.82, 0.15)}

BOX_X, POCKET, WALL, WALL_H, FLOOR = 0.34, 0.05, 0.01, 0.015, 0.004
FLOOR_Z = TOP + FLOOR
POCKET_Y = [0.15 - 0.06 * k for k in range(6)]  # slot 1 (start of shift) on the left
KANBAN = (0.03, 0.03, 0.03)  # a kanban in its holder, 40 g
KANBANS = {"kanban_a1": ("A", 0.125), "kanban_a2": ("A", 0.075), "kanban_a3": ("A", 0.025),
           "kanban_b1": ("B", -0.025), "kanban_b2": ("B", -0.075), "kanban_c1": ("C", -0.125)}  # in the tray at x = 18
TRAY_X = 0.18


def _heijunka_box():
    c = (0.30, 0.31, 0.33)
    L = 6 * POCKET + 7 * WALL
    out = [Box("heijunka_floor", pos=(BOX_X, 0.0, TOP + FLOOR / 2), size=(POCKET + 2 * WALL, L, FLOOR), color=c)]
    for k, s in (("far", 1), ("near", -1)):
        out.append(Box(f"heijunka_{k}", pos=(BOX_X + s * (POCKET / 2 + WALL / 2), 0.0, TOP + WALL_H / 2), size=(WALL, L, WALL_H), color=c))
    for k in range(7):
        y = POCKET_Y[0] + POCKET / 2 + WALL / 2 - k * (POCKET + WALL)
        out.append(Box(f"heijunka_divider_{k}", pos=(BOX_X, y, TOP + WALL_H / 2), size=(POCKET, WALL, WALL_H), color=c))
    return out


OBJECTS = [
    *_heijunka_box(),
    Box("kanban_tray", pos=(TRAY_X, 0.0, TOP + 0.002), size=(0.05, 0.30, 0.004), color=(0.55, 0.56, 0.58)),
    *[Box(n, pos=(TRAY_X, y, TOP + 0.004 + KANBAN[2] / 2), size=KANBAN, mass=0.04, color=PRODUCT_COLOR[p])
      for n, (p, y) in KANBANS.items()],
]


# -- scoring --------------------------------------------------------------------------------------------------------
def pocket_contents(k, objects):
    return [n for n in KANBANS if abs(objects[n].pos[0] - BOX_X) <= POCKET / 2 and abs(objects[n].pos[1] - POCKET_Y[k]) <= POCKET / 2
            and objects[n].low[2] <= FLOOR_Z + 0.01]


def _products(objects):
    """The product in each pocket, or None if it holds no kanban or more than one."""
    out = []
    for k in range(6):
        c = pocket_contents(k, objects)
        out.append(KANBANS[c[0]][0] if len(c) == 1 else None)
    return out


def filled(objects):
    """Pockets holding exactly one kanban."""
    return sum(p is not None for p in _products(objects))


def levelled(objects):
    """All six pockets hold one kanban each, with the red (A) ones in alternate pockets (1, 3, 5 or 2, 4, 6); the blue
    and yellow ones then fill the other three, so no two neighbours match."""
    p = _products(objects)
    return None not in p and {k for k in range(6) if p[k] == "A"} in ({0, 2, 4}, {1, 3, 5})


def success(objects):
    return levelled(objects)


def score(objects):
    """0.3 x pockets holding exactly one kanban / 6 + 0.7 if the box is levelled."""
    return float(0.3 * filled(objects) / 6 + 0.7 * levelled(objects))


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
Z_UP = TOP + 0.10
# A B A C A B, slot by slot from the left; the arms take turns
MOVES = [("left", "kanban_a1", 0), ("right", "kanban_c1", 3), ("left", "kanban_b1", 1), ("right", "kanban_a3", 4),
         ("left", "kanban_a2", 2), ("right", "kanban_b2", 5)]


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
            if not self.wps[a]:  # a straight-down grip can't hold the READY pose: wait parked instead
                self.go(a, end, (0.18, SIDE[a] * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
            elif self.t[a] < end - 1e-9:
                self.go(a, end - self.t[a], label="wait")


def demo():
    plan = _Plan()
    grip, ready = -FINGER_PER_M * (KANBAN[0] - 0.008), -FINGER_PER_M * (KANBAN[0] + 0.016)
    g = KANBAN[2] - 0.01
    for arm, n, k in MOVES:
        py, qy = KANBANS[n][1], POCKET_Y[k]
        yaw = min((math.pi / 2 + k2 * math.pi for k2 in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))  # along x
        plan.go(arm, 1.2, (TRAY_X, py, Z_UP), ready, f"{n} to slot {k + 1}: above it", yaw)
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 1.0, (TRAY_X, py, TOP + 0.004 + g), label="descend")
        plan.go(arm, 0.5, finger=grip, label="grip")
        plan.go(arm, 0.8, (TRAY_X, py, Z_UP), label="lift")
        plan.go(arm, 1.3, (BOX_X, qy, Z_UP), label="over its pocket")
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 0.9, (BOX_X, qy, TOP + WALL_H + g + 0.004), label="lower it to the pocket")
        plan.go(arm, 0.4, finger=ready, label="let it drop in")
        plan.go(arm, 0.6, (BOX_X, qy, Z_UP), label="up")
        plan.go(arm, 1.0, (0.18, SIDE[arm] * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
        plan.sync()
    return plan.wps
