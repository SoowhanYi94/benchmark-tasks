"""On the station table: a delivery trailer (an open-topped box with its door end facing the arms) stands in the middle
of the bench, its nose at the far end. Five parcels for one delivery route wait on two dock mats either side, each
coloured by the stop it goes to (two yellow, two blue, one red). A route board on the left shows the three stops in
delivery order as coloured squares, from left to right. The trailer is unloaded from the door, stop by stop.

Run it:  ROUTE_VARIANT=0 inspect-robots run --embodiment yam_arena -E task=tasks/route-loading/task.py \\
           --instruction "Load the parcels into the trailer so they can be unloaded from its door in the delivery order shown on the route board." --scorer task_score ...

ROUTE_VARIANT (0-5, default 0) picks the board's order: one of the six orders of the three colours, as in VARIANTS.
"""

import itertools
import math
import os

from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
STOP_COLOR = {"blue": (0.15, 0.35, 0.80), "red": (0.80, 0.12, 0.10), "yellow": (0.95, 0.82, 0.15)}
VARIANTS = list(itertools.permutations(("blue", "red", "yellow")))  # 0 is blue, red, yellow; 5 is yellow, red, blue
ROUTE = list(VARIANTS[int(os.environ.get("ROUTE_VARIANT", "0"))])  # delivery order, as the route board shows it

TRAILER = dict(door_x=0.14, nose_x=0.41, width=0.10, wall=0.01, height=0.05, floor=0.004)
FLOOR_Z = TOP + TRAILER["floor"]
PARCEL = (0.04, 0.06, 0.04)  # cardboard, 80 g
# name: (stop colour, start xy on a dock mat)
PARCELS = {
    "parcel_1": ("yellow", (0.20, 0.17)),
    "parcel_2": ("blue", (0.28, 0.17)),
    "parcel_3": ("red", (0.36, 0.17)),
    "parcel_4": ("blue", (0.20, -0.17)),
    "parcel_5": ("yellow", (0.28, -0.17)),
}
DOCKS = {"dock_left": (0.28, 0.17), "dock_right": (0.24, -0.17)}
BOARD = dict(center=(0.20, 0.30), size=(0.06, 0.12, 0.004))
DEEPER = 0.02  # a later stop's parcel must lie at least this much (centre to centre) nearer the nose


def _trailer():
    t, c = TRAILER, (0.62, 0.64, 0.68)
    lx, cx = t["nose_x"] - t["door_x"], (t["nose_x"] + t["door_x"]) / 2
    w, h, f, iy = t["wall"], t["height"], t["floor"], t["width"]
    return [
        Box("trailer_floor", pos=(cx, 0.0, TOP + f / 2), size=(lx, iy + 2 * w, f), color=(0.35, 0.36, 0.38)),
        Box("trailer_nose", pos=(t["nose_x"] + w / 2, 0.0, TOP + h / 2), size=(w, iy + 2 * w, h), color=c),
        Box("trailer_left_side", pos=(cx, iy / 2 + w / 2, TOP + h / 2), size=(lx, w, h), color=c),
        Box("trailer_right_side", pos=(cx, -iy / 2 - w / 2, TOP + h / 2), size=(lx, w, h), color=c),
    ]


OBJECTS = [
    *_trailer(),
    *[Box(n, pos=(*xy, TOP + 0.002), size=(0.26 if n == "dock_left" else 0.18, 0.09, 0.004), color=(0.45, 0.46, 0.48))
      for n, xy in DOCKS.items()],
    Box("route_board", pos=(*BOARD["center"], TOP + BOARD["size"][2] / 2), size=BOARD["size"], color=(0.96, 0.96, 0.93)),
    *[Box(f"route_stop_{k + 1}", pos=(BOARD["center"][0], BOARD["center"][1] + 0.035 - 0.035 * k, TOP + 0.005),
          size=(0.025, 0.025, 0.002), color=STOP_COLOR[c]) for k, c in enumerate(ROUTE)],
    *[Box(n, pos=(*xy, TOP + 0.004 + PARCEL[2] / 2), size=PARCEL, mass=0.08, color=STOP_COLOR[c]) for n, (c, xy) in PARCELS.items()],
]


# -- scoring --------------------------------------------------------------------------------------------------------
def loaded(name, objects):
    """Wholly inside the trailer (its whole footprint past the door line), still, on the floor (underside within 5 mm)."""
    t, o = TRAILER, objects[name]
    return bool(o.low[0] >= t["door_x"] and o.pos[0] <= t["nose_x"] and abs(o.pos[1]) <= t["width"] / 2
                and o.low[2] <= FLOOR_Z + 0.005 and o.resting())


def _stop(name, route=None):
    return (route or ROUTE).index(PARCELS[name][0])


def counts(name, objects, route=None):
    """Loaded, and every parcel for another stop is loaded and at least DEEPER on the correct side of it (nearer the
    nose if its stop comes later on the route, nearer the door if earlier)."""
    if not loaded(name, objects):
        return False
    x = objects[name].pos[0]
    for m in PARCELS:
        d = _stop(m, route) - _stop(name, route)
        if d and not (loaded(m, objects) and (objects[m].pos[0] - x) * (1 if d > 0 else -1) >= DEEPER):
            return False
    return True


def success(objects):
    return all(counts(n, objects) for n in PARCELS)


def score(objects, route=None):
    """(parcels that count - 2) / 3, not below 0. Chance-corrected: a blind full load counts about 2 parcels at random
    (random orders average 0.06), and 4 can't count without the fifth, so the steps are 0, 1/3 and 1."""
    c = sum(counts(n, objects, route) for n in PARCELS)
    return float(max(0.0, (c - 2) / 3))


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
PARK = {a: (0.18, s * 0.32, TOP + 0.18) for a, s in SIDE.items()}
Z_UP = TOP + 0.10  # a carried parcel clears the 5 cm trailer walls by 2 cm
SLOTS = (0.37, 0.32, 0.27, 0.22, 0.17)  # single file down the middle, nose first; 1 cm gaps, 2 cm clear of the nose


def load_plan(route=None):
    """(arm, parcel, slot x): the last stop's parcels first, at the nose; each arm takes the parcels on its side."""
    order = sorted(PARCELS, key=lambda n: -_stop(n, route))
    return [("left" if PARCELS[n][1][1] > 0 else "right", n, x) for n, x in zip(order, SLOTS)]


class _Plan:
    def __init__(self):
        self.wps = {a: [] for a in SIDE}
        self.t = {a: 0.0 for a in SIDE}
        # an arm's first waypoint (a wait, for the arm not loading first) takes it to its out-of-the-way pose, since
        # holding READY's position with the downward grip is out of reach
        self.last = {a: (PARK[a], FINGER_OPEN, 0.0) for a in SIDE}

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


def demo(route=None):
    plan = _Plan()
    w = PARCEL[1]  # gripped across its 6 cm width (fingers along y, into the 2 cm beside the side walls)
    grip, ready = -FINGER_PER_M * (w - 0.006), -FINGER_PER_M * (w + 0.014)
    release = -FINGER_PER_M * (w + 0.008)  # 6.8 cm in the 10 cm bay: 1.6 cm clear of each wall
    g = PARCEL[2] - 0.01
    for arm, n, qx in load_plan(route):
        (px, py) = PARCELS[n][1]
        yaw = min((k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))
        plan.go(arm, 1.2, (px, py, Z_UP), ready, f"{n}: above it", yaw)
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 1.0, (px, py, TOP + 0.004 + g), label="descend")
        plan.go(arm, 0.5, finger=grip, label="grip")
        plan.go(arm, 0.8, (px, py, Z_UP), label="lift")
        plan.go(arm, 1.3, (qx, 0.0, Z_UP), label="over its place in the trailer")
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 1.0, (qx, 0.0, FLOOR_Z + g + 0.004), label="set it down")
        plan.go(arm, 0.4, finger=release, label="release")
        plan.go(arm, 0.6, (qx, 0.0, Z_UP), label="up")
        plan.go(arm, 1.0, PARK[arm], FINGER_OPEN, "out of the way")
        plan.sync()  # one arm over the trailer at a time
    return plan.wps
