"""On the station table: a hydraulic manifold block lies across the bench with four ports open on its top face, in a
row. Before its pressure test every open port must be plugged, except the test port, which is marked blue and takes
the test gauge. One port is already plugged. Plugs wait in a tray at the front, with a spare. The robot must plug the
open ports and leave the test port open.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/port-plugging/task.py \\
           --instruction "get the manifold ready for its pressure test: plug every open port, but leave the blue test port open for the gauge" --scorer task_score ...
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

MANIFOLD = dict(x=0.32, width=0.07, length=0.32, base=0.01, depth=0.02)  # ports are DEPTH deep, above a BASE-thick floor
HOLE = 0.034
FLOOR_Z = TOP + MANIFOLD["base"]
FACE_Z = FLOOR_Z + MANIFOLD["depth"]
PORTS = {"port_1": 0.105, "port_2": 0.035, "port_3": -0.035, "port_4": -0.105}  # y of each port
TEST_PORT = "port_2"
PRE_PLUGGED = "port_4"
PLUG_R, PLUG_H = 0.012, 0.03  # nylon, 20 g
PLUGS = {"plug_1": (0.18, 0.06), "plug_2": (0.18, 0.0), "plug_3": (0.18, -0.06), "plug_fitted": (MANIFOLD["x"], PORTS[PRE_PLUGGED])}
TO_PLUG = [p for p in PORTS if p not in (TEST_PORT, PRE_PLUGGED)]
WRONG_PENALTY = 1.5


def _manifold():
    m, c = MANIFOLD, (0.55, 0.57, 0.60)
    x, w, L = m["x"], m["width"], m["length"]
    side = (w - HOLE) / 2
    out = [Box("manifold_base", pos=(x, 0.0, TOP + m["base"] / 2), size=(w, L, m["base"]), color=c)]
    zc = FLOOR_Z + m["depth"] / 2
    for k, s in (("far", 1), ("near", -1)):
        out.append(Box(f"manifold_{k}_wall", pos=(x + s * (HOLE / 2 + side / 2), 0.0, zc), size=(side, L, m["depth"]), color=c))
    edges = [L / 2] + [e for y in PORTS.values() for e in (y + HOLE / 2, y - HOLE / 2)] + [-L / 2]
    for k in range(0, len(edges), 2):
        hi, lo = edges[k], edges[k + 1]
        out.append(Box(f"manifold_web_{k // 2}", pos=(x, (hi + lo) / 2, zc), size=(HOLE, hi - lo, m["depth"]), color=c))
    out.append(Box("test_port_marking", pos=(x, PORTS[TEST_PORT], FLOOR_Z + 0.0005), size=(HOLE - 0.004, HOLE - 0.004, 0.001),
                   color=(0.15, 0.35, 0.80)))
    return out


OBJECTS = [
    *_manifold(),
    Box("plug_tray", pos=(0.18, 0.0, TOP + 0.002), size=(0.05, 0.18, 0.004), color=(0.30, 0.31, 0.33)),
    *[Cylinder(n, pos=(*xy, (FLOOR_Z if n == "plug_fitted" else TOP + 0.004) + PLUG_H / 2), radius=PLUG_R, height=PLUG_H,
               mass=0.02, color=(0.95, 0.82, 0.15)) for n, xy in PLUGS.items()],
]


# -- scoring --------------------------------------------------------------------------------------------------------
def plugged(port, objects):
    """Some plug stands in the port: its centre inside the hole and its underside below the manifold face by at least
    1 cm."""
    y = PORTS[port]
    return any(abs(objects[p].pos[0] - MANIFOLD["x"]) <= HOLE / 2 and abs(objects[p].pos[1] - y) <= HOLE / 2
               and objects[p].low[2] <= FACE_Z - 0.01 for p in PLUGS)


def _tally(objects):
    f, y = objects["plug_fitted"], PORTS[PRE_PLUGGED]
    fitted_in = (abs(f.pos[0] - MANIFOLD["x"]) <= HOLE / 2 and abs(f.pos[1] - y) <= HOLE / 2 and f.low[2] <= FACE_Z - 0.01)
    return sum(plugged(p, objects) for p in TO_PLUG), int(plugged(TEST_PORT, objects)) + int(not fitted_in)


def success(objects):
    return _tally(objects) == (len(TO_PLUG), 0)


def score(objects):
    """(open ports plugged - 1.5 x (test port plugged + the fitted plug pulled)) / 2, at least 0."""
    correct, wrong = _tally(objects)
    return float(max(0.0, correct - WRONG_PENALTY * wrong) / len(TO_PLUG))


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
Z_UP = TOP + 0.10
MOVES = [("left", "plug_1", "port_1"), ("right", "plug_3", "port_3")]


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
    w = 2 * PLUG_R
    grip, ready = -FINGER_PER_M * (w - 0.008), -FINGER_PER_M * (w + 0.03)
    g = PLUG_H - 0.008  # held near its top, so the plug can go most of the way in before the fingers reach the face
    for arm, n, port in MOVES:
        (px, py) = PLUGS[n]
        qx, qy = MANIFOLD["x"], PORTS[port]
        yaw = min((math.pi / 2 + k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))  # along x
        plan.go(arm, 1.2, (px, py, Z_UP), ready, f"{n} into {port}: above it", yaw)
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 1.0, (px, py, TOP + 0.004 + g), label="descend")
        plan.go(arm, 0.5, finger=grip, label="grip")
        plan.go(arm, 0.8, (px, py, Z_UP), label="lift")
        plan.go(arm, 1.3, (qx, qy, Z_UP), label="over the port")
        plan.go(arm, 0.6, label="settle")
        plan.go(arm, 1.0, (qx, qy, FACE_Z + g + 0.004), label="tip it into the port")
        plan.go(arm, 0.4, label="settle")
        plan.go(arm, 0.4, finger=ready, label="let it drop in")
        plan.go(arm, 0.6, (qx, qy, Z_UP), label="up")
        plan.go(arm, 1.0, (0.18, SIDE[arm] * 0.32, TOP + 0.18), FINGER_OPEN, "out of the way")
        plan.sync()
    return plan.wps


# -- self-check (pure Python): the plan fits the run limit and the scorer covers the obvious outcomes ----------------
if __name__ == "__main__":
    class _O:
        def __init__(self, x, y, z):
            self.pos, self.low = np.array((x, y, z + PLUG_H / 2)), np.array((x - PLUG_R, y - PLUG_R, z))

    start = {n: _O(*xy, FLOOR_Z if n == "plug_fitted" else TOP + 0.004) for n, xy in PLUGS.items()}
    assert score(start) == 0.0 and not success(start)
    inport = lambda p: _O(MANIFOLD["x"], PORTS[p], FLOOR_Z)
    done = dict(start, plug_1=inport("port_1"), plug_3=inport("port_3"))
    assert score(done) == 1.0 and success(done)
    assert score(dict(done, plug_2=inport("port_2"))) == (2 - 1.5) / 2  # the test port plugged too
    assert score(dict(start, plug_1=inport("port_1"), plug_2=inport("port_2"), plug_3=inport("port_3"))) == (2 - 1.5) / 2
    swapped = dict(done, plug_fitted=inport("port_1"), plug_1=inport("port_4"))  # exploit (council): the fitted plug swapped out
    assert abs(score(swapped) - (2 - 1.5) / 2) < 1e-9, score(swapped)
    t = max(sum(w.duration for w in wps) for wps in demo().values())
    assert t * 20 < 1200, t
    print(f"ok: demo {t:.1f} s ({t * 20:.0f} steps)")
