"""On the station table: a delivery trailer (an open-topped box with its door end facing the arms) stands in the middle
of the bench, its nose at the far end. Five parcels for one delivery route wait on two dock mats either side, each
coloured by the stop it goes to. A route board on the left shows the three stops in delivery order, from left to right:
blue, red, yellow. The trailer is unloaded from the door, so the robot must load it last stop first: the parcels for
the last stop deepest, at the nose, and those for the first stop nearest the door.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/route-loading/route_loading.py \\
           --instruction "load the parcels into the trailer for the route: the route board shows the stops in delivery order from left to right; the trailer is unloaded from the door, so the first stop's parcels must be nearest the door and the last stop's at the nose" --scorer task_score ...
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
STOP_COLOR = {"blue": (0.15, 0.35, 0.80), "red": (0.80, 0.12, 0.10), "yellow": (0.95, 0.82, 0.15)}
ROUTE = ["blue", "red", "yellow"]  # delivery order, as the route board shows it

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
    t, o = TRAILER, objects[name]
    return bool(t["door_x"] <= o.pos[0] <= t["nose_x"] and abs(o.pos[1]) <= t["width"] / 2 and o.low[2] <= FLOOR_Z + 0.02)


def in_sequence(name, objects):
    """Loaded, and every other loaded parcel for a later stop lies deeper (nearer the nose) and every one for an earlier
    stop nearer the door, by at least DEEPER."""
    if not loaded(name, objects):
        return False
    k, x = ROUTE.index(PARCELS[name][0]), objects[name].pos[0]
    for q in PARCELS:
        if q == name or not loaded(q, objects):
            continue
        kq, xq = ROUTE.index(PARCELS[q][0]), objects[q].pos[0]
        if (kq > k and xq < x + DEEPER) or (kq < k and xq > x - DEEPER):
            return False
    return True


def success(objects):
    return all(in_sequence(n, objects) for n in PARCELS)


def score(objects):
    """Parcels loaded in sequence (see in_sequence) / 5."""
    return float(sum(in_sequence(n, objects) for n in PARCELS) / len(PARCELS))


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
Z_UP = TOP + 0.10  # a carried parcel clears the 5 cm trailer walls by 2 cm
# single file down the middle of the trailer, nose first: yellow, yellow, red, blue, blue
LOAD = [("left", "parcel_1", 0.38), ("right", "parcel_5", 0.33), ("left", "parcel_3", 0.28), ("left", "parcel_2", 0.23),
        ("right", "parcel_4", 0.18)]


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
    w = PARCEL[1]  # gripped across its 6 cm width (fingers along y, into the 2 cm beside the side walls)
    grip, ready = -FINGER_PER_M * (w - 0.006), -FINGER_PER_M * (w + 0.014)
    g = PARCEL[2] - 0.01
    for arm, n, qx in LOAD:
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
        plan.go(arm, 0.4, finger=ready, label="release")
        plan.go(arm, 0.6, (qx, 0.0, Z_UP), label="up")
        plan.go(arm, 1.0, (0.18, SIDE[arm] * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
        plan.sync()  # one arm over the trailer at a time
    return plan.wps


# -- self-check (pure Python): the plan fits the run limit and the scorer covers the obvious outcomes ----------------
if __name__ == "__main__":
    class _O:
        def __init__(self, x, y, z):
            self.pos, self.low = np.array((x, y, z + 0.02)), np.array((x - 0.02, y - 0.03, z))

    start = {n: _O(*xy, TOP + 0.004) for n, (c, xy) in PARCELS.items()}
    assert score(start) == 0.0 and not success(start)
    done = dict(start, **{n: _O(x, 0.0, FLOOR_Z) for _, n, x in LOAD})
    assert score(done) == 1.0 and success(done)
    by_route = dict(start, **{n: _O(x, 0.0, FLOOR_Z) for n, x in zip(["parcel_2", "parcel_4", "parcel_3", "parcel_1", "parcel_5"],
                                                                     [0.38, 0.33, 0.28, 0.23, 0.18])})  # blue at the nose
    assert score(by_route) == 0.0  # every parcel has a wrong-stop parcel on the wrong side of it
    t = max(sum(w.duration for w in wps) for wps in demo().values())
    assert t * 20 < 2400, t
    print(f"ok: demo {t:.1f} s ({t * 20:.0f} steps)")
