"""On the station table: a pick-to-light rack of four parts bins stands across the far side of the bench, each holding
three identical parts (from left to right: yellow cylinders, grey blocks, orange cylinders, blue blocks). On a strip in
front of each bin are two indicator lamps; a lit lamp (bright green) means pick one part from that bin. One bin is lit
at a time: pressing its button dims it and lights the next, so the order can't be seen up front. In front of the strip
each bin has two push buttons that slide in toward the strip: a white confirm button (pressed after picking that bin's
parts) and a red shortage button (pressed when the lit bin is empty, instead of picking). Three orders (A, B, C) each
light three bins in turn, one of them empty; the environment variable PICK_TO_LIGHT_PATTERN picks one for a run (A by
default). An order tote stands in front, between the arms.

`yam_ptl` (pick_to_light/embodiment.py) is the stock yam_arena embodiment that also reports `left_tcp` and
`right_tcp`, so the scorer can tell whether a part has been let go, and switches the lamps as the order moves on.

Run it:  PICK_TO_LIGHT_PATTERN=A inspect-robots run --embodiment yam_ptl -E task=tasks/pick-to-light/task.py \\
           -E object_jitter=0 --instruction "Pick the order shown by the lamps into the order tote, one bin at a time: take one part for every lit green lamp from the lit bin, then press its white confirm button to light the next bin; if the lit bin is empty, press its red shortage button instead and take nothing in its place; stop when every lamp is dark." --scorer task_score ...
"""

import math
import os

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]
LIT, DARK = (0.30, 1.00, 0.35), (0.12, 0.13, 0.12)

RACK_X = 0.34
BIN_Y = {"yellow": 0.15, "grey": 0.05, "orange": -0.05, "blue": -0.15}
BIN_IN, BIN_WALL, BIN_H, BIN_FLOOR = 0.09, 0.01, 0.03, 0.006  # neighbouring bins share a wall (10 cm apart)
BIN_Z = TOP + BIN_FLOOR
PART_SLOTS = (-0.03, 0.0, 0.03)  # each bin's three parts, side by side along x
PARTS = {  # kind, size, colour per bin
    "yellow": dict(kind="cyl", size=(0.013, 0.035), mass=0.05, color=(0.95, 0.82, 0.15)),
    "grey": dict(kind="box", size=(0.025, 0.025, 0.035), mass=0.06, color=(0.55, 0.56, 0.58)),
    "orange": dict(kind="cyl", size=(0.013, 0.035), mass=0.05, color=(0.95, 0.50, 0.10)),
    "blue": dict(kind="box", size=(0.025, 0.025, 0.035), mass=0.06, color=(0.15, 0.35, 0.80)),
}
PATTERNS = {  # the order: (bin, lamps lit) in the sequence the lamps show them, and the lit bin that is empty
    "A": dict(order=(("yellow", 2), ("orange", 1), ("blue", 1)), empty="orange"),
    "B": dict(order=(("blue", 2), ("grey", 1), ("yellow", 1)), empty="grey"),
    "C": dict(order=(("orange", 1), ("grey", 2), ("blue", 2)), empty="grey"),
}
PATTERN = os.environ.get("PICK_TO_LIGHT_PATTERN", "A").upper()
ORDER, EMPTY = PATTERNS[PATTERN]["order"], PATTERNS[PATTERN]["empty"]
LAMPS = {b: dict(ORDER).get(b, 0) for b in BIN_Y}  # how many parts the order wants from each bin
LAMP_X, LAMP, STRIP_H = 0.27, 0.025, 0.01  # 2.5 cm cube lamps on a 1 cm strip, 4 cm apart (centres)
STRIP_FACE = LAMP_X - LAMP / 2  # the strip's (and lamps') face toward the arms, x = 25.75 cm
TOTE = dict(center=(0.11, 0.0), inner=(0.10, 0.14), wall=0.01, height=0.04, floor=0.004)
WRONG_PENALTY, WRONG_CAP = 2.0, 0.2
HANDS_CLEAR = 0.02  # a part is placed only when both fingertip centres are this far from it
NAMES = {b: [f"{b}_part_{i}" for i in range(len(PART_SLOTS))] for b in BIN_Y}

# Push buttons: a free plastic plunger per button, lying on the table in a fixed guide (two rails), pushed along +x
# toward the strip, which stops it after 7.5 mm. No return spring: once pushed in it stays in.
BUTTON = (0.04, 0.014, 0.014)  # plunger x, y, z
BUTTON_MASS = 0.015
BUTTON_X = STRIP_FACE - 0.0075 - BUTTON[0] / 2  # out (unpressed) centre, x = 23 cm
BUTTON_DY = {"confirm": 0.02, "shortage": -0.02}  # from the bin's centre line: confirm left, shortage right
BUTTON_COLOR = {"confirm": (0.92, 0.92, 0.90), "shortage": (0.85, 0.10, 0.10)}
RAIL = dict(len=0.03, t=0.004, h=0.008, dy=BUTTON[1] / 2 + 0.001 + 0.002)  # rails end at the strip's face
PRESS_TRAVEL = 0.004  # a button moved 4 mm or more from where it started has been pressed
BUTTONS = {f"{k}_{b}": (k, b) for b in BIN_Y for k in BUTTON_DY}


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


def _button_y(name):
    k, b = BUTTONS[name]
    return BIN_Y[b] + BUTTON_DY[k]


def _button(name):
    k, _ = BUTTONS[name]
    y, rx = _button_y(name), STRIP_FACE - RAIL["len"] / 2
    return [
        Box(name, pos=(BUTTON_X, y, TOP + BUTTON[2] / 2), size=BUTTON, mass=BUTTON_MASS, color=BUTTON_COLOR[k]),
        *[Box(f"{name}_rail_{s}", pos=(rx, y + sign * RAIL["dy"], TOP + RAIL["h"] / 2),
              size=(RAIL["len"], RAIL["t"], RAIL["h"]), color=(0.20, 0.21, 0.22)) for s, sign in (("l", 1), ("r", -1))],
    ]


PART_START = {n: _part(n, b, RACK_X + dx, BIN_Y[b]) for b in BIN_Y for n, dx in zip(NAMES[b], PART_SLOTS)}
OBJECTS = [
    *[o for b, y in BIN_Y.items() for o in _box(f"bin_{b}", RACK_X, y, BIN_IN, BIN_IN, BIN_WALL, BIN_H, BIN_FLOOR, (0.30, 0.31, 0.33))],
    Box("lamp_strip", pos=(LAMP_X, 0.0, TOP + STRIP_H / 2), size=(LAMP, 0.41, STRIP_H), color=(0.20, 0.21, 0.22)),
    # two lamps in front of each bin (left one at +y); the first bin of the order starts lit, and the yam_ptl
    # embodiment switches them as the order moves on
    *[Box(f"lamp_{b}_{k}", pos=(LAMP_X, y + (0.02 if k == 0 else -0.02), TOP + STRIP_H + LAMP / 2),
          size=(LAMP, LAMP, LAMP), color=LIT if b == ORDER[0][0] and k < ORDER[0][1] else DARK)
      for b, y in BIN_Y.items() for k in (0, 1)],
    *[o for n in BUTTONS for o in _button(n)],
    *_box("order_tote", *TOTE["center"], *TOTE["inner"], TOTE["wall"], TOTE["height"], TOTE["floor"], (0.20, 0.35, 0.60)),
    *[o for n, o in PART_START.items() if not n.startswith(EMPTY + "_")],  # the empty bin has no parts
]
START = {o.name: np.array(o.pos) for o in [*PART_START.values(), *OBJECTS] if o.mass is not None}


# -- scoring --------------------------------------------------------------------------------------------------------
def in_tote(o):
    """Centre inside the tote's walls and no more than 5 cm above its rim (held, stacked or lying: it's in the tote)."""
    (cx, cy), (ix, iy) = TOTE["center"], TOTE["inner"]
    return bool(abs(o.pos[0] - cx) <= ix / 2 and abs(o.pos[1] - cy) <= iy / 2 and o.pos[2] <= TOP + TOTE["height"] + 0.05)


def hands_clear(o, objects):
    """Both fingertip centres (from the yam_ptl embodiment) at least 2 cm from the part's bounding box: let go."""
    for arm in ("left_tcp", "right_tcp"):
        p = np.asarray(objects[arm].pos, float)
        if np.linalg.norm(p - np.clip(p, o.low, o.high)) < HANDS_CLEAR:
            return False
    return True


def placed(o, objects):
    """In the tote, let go (both fingertip centres 2 cm or more from it) and at rest, its underside within 1 cm of
    the tote's floor."""
    return bool(in_tote(o) and o.low[2] <= TOP + TOTE["floor"] + 0.01 and o.resting() and hands_clear(o, objects))


def dropped(o):
    """Off the table (fallen to the floor)."""
    return bool(o.pos[2] < TOP - 0.02)


def pushed(name, o):
    """Moved 4 mm or more from its start in any direction. Only ``in_toward_strip`` makes it a correct press."""
    return bool(np.linalg.norm(np.asarray(o.pos) - START[name]) >= PRESS_TRAVEL)


def in_toward_strip(name, o):
    """Pushed in along +x, the button's travel: pulling it out or forcing it from its rails is a wrong press."""
    return bool(o.pos[0] - START[name][0] >= PRESS_TRAVEL)


def _parts(objects, b):
    return [objects[n] for n in NAMES[b] if n in objects]


# A button press is latched for the trial, with whether it was right when it was pressed, so a press can't be undone
# (pulling a plunger back out). Only a press on the lit bin moves the order on. A part in the tote from a bin that hasn't
# been lit yet is latched too: it was taken from a dark bin, even if that bin lights later.
_PRESSED = {}
_EARLY = set()
_STEP = [0]


def active():
    """The order's current (bin, lamps lit), or None once it is finished: what the lamps show."""
    return ORDER[_STEP[0]] if _STEP[0] < len(ORDER) else None


def _latch(objects):
    lit = active()
    for b in BIN_Y:  # dark now and not done: not lit yet, or never lit
        done = {x for x, _ in ORDER[:_STEP[0]]}
        if (lit is None or b != lit[0]) and b not in done and any(map(in_tote, _parts(objects, b))):
            _EARLY.add(b)
    for n in BUTTONS:
        if n not in _PRESSED and n in objects and pushed(n, objects[n]):
            k, b = BUTTONS[n]
            lit = active()
            ok = False
            if lit is not None and b == lit[0]:
                count = sum(map(in_tote, _parts(objects, b)))
                ok = in_toward_strip(n, objects[n]) and (k == "shortage" if b == EMPTY else (k == "confirm" and count == lit[1]))
                _STEP[0] += 1
            _PRESSED[n] = ok


def reset_latch():
    """Forget everything: the embodiment calls this at every reset."""
    _PRESSED.clear()
    _EARLY.clear()
    _STEP[0] = 0


def _tally(objects):
    """(correct, wrong, out of). Correct: each part placed up to its bin's lit lamps; the lit, stocked bin's confirm,
    pressed once the tote held exactly its lit count from that bin; the lit, empty bin's shortage. Wrong: each part in the
    tote beyond its bin's lamps (so any substitute), each bin with a part in the tote before it was lit, each part on the
    floor, and every other button press (at the wrong count, or on any bin that isn't the lit one)."""
    _latch(objects)
    correct = wrong = total = 0
    for b in BIN_Y:
        parts, lit = _parts(objects, b), LAMPS[b]
        correct += min(sum(placed(o, objects) for o in parts), lit)
        wrong += max(0, sum(map(in_tote, parts)) - lit) + sum(map(dropped, parts))
        stocked, empty = lit > 0 and b != EMPTY, lit > 0 and b == EMPTY
        total += (lit + 1) if stocked else (1 if empty else 0)
        for n in (f"confirm_{b}", f"shortage_{b}"):
            if n in _PRESSED:
                correct, wrong = correct + _PRESSED[n], wrong + (not _PRESSED[n])
    return correct, wrong + len(_EARLY), total


def success(objects):
    correct, wrong, total = _tally(objects)
    return correct == total and wrong == 0


def score(objects):
    """(correct - 2 x wrong) / out of, at least 0, and at most 0.2 with anything wrong."""
    correct, wrong, total = _tally(objects)
    s = max(0.0, correct - WRONG_PENALTY * wrong) / total
    return float(min(s, WRONG_CAP) if wrong else s)


# -- scripted demo ----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
Z_UP = TOP + 0.10
PARK = 0.18, 0.32, TOP + 0.18  # x, |y|, z
PAD_TIP_PAST_TCP = 0.013
PRESS_Z = TOP + 0.004 + PAD_TIP_PAST_TCP  # closed pad tips 4 mm above the table, on the plunger's 14 mm face
PRESS_FROM_X, PRESS_TO_X = 0.185, 0.210  # TCP; the plunger's near face is at 21 cm, the stop 7.5 mm further


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


def _arm(b):
    return "left" if BIN_Y[b] > 0 else "right"


def _yaw(arm):  # yaw 0 (or a multiple of pi) closes the fingers along y
    return min((k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def _park(plan, arm):
    plan.go(arm, 1.0, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "out of the way")


def _pick(plan, name, b, dx, drop):
    arm = _arm(b)
    p = PARTS[b]
    width = 2 * p["size"][0] if p["kind"] == "cyl" else p["size"][1]
    grip, ready = -FINGER_PER_M * (width - 0.004), -FINGER_PER_M * (width + 0.016)
    # fingers closing along y, either side of the part, stay clear of its neighbours 3 cm away along x
    yaw = _yaw(arm)
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
    _park(plan, arm)
    plan.sync()


def _press(plan, button):
    """Closed fingers come down in front of the plunger and push it in toward the strip, then back off."""
    b = BUTTONS[button][1]
    arm, y = _arm(b), _button_y(button)
    plan.go(arm, 1.2, (PRESS_FROM_X, y, Z_UP), FINGER_CLOSED, f"{button}: in front of it", _yaw(arm))
    plan.go(arm, 1.0, (PRESS_FROM_X, y, PRESS_Z), label="down to the table")
    plan.go(arm, 1.0, (PRESS_TO_X, y, PRESS_Z), label="push it in")
    plan.go(arm, 0.3, label="hold")
    plan.go(arm, 0.6, (PRESS_FROM_X, y, PRESS_Z), label="back off")
    plan.go(arm, 0.6, (PRESS_FROM_X, y, Z_UP), label="up")
    _park(plan, arm)
    plan.sync()


def demo(pattern=None):
    p = PATTERNS[PATTERN if pattern is None else pattern]
    order, empty = p["order"], p["empty"]
    plan = _Plan()
    for a in SIDE:  # from READY to a top-down park pose, so an arm waiting its turn holds a pose IK reaches
        plan.go(a, 1.0, (PARK[0], SIDE[a] * PARK[1], PARK[2]), FINGER_OPEN, "park", 0.0)
    cx, cy = TOTE["center"]
    drops = [(cx - 0.025, cy + 0.03), (cx + 0.025, cy + 0.03), (cx, cy - 0.03)]
    k = 0
    for b, n in order:  # one bin at a time, as the lamps show them
        if b == empty:
            _press(plan, f"shortage_{b}")  # report the shortage, take nothing
            continue
        for i in range(n):  # the parts nearest the arms first
            _pick(plan, NAMES[b][i], b, PART_SLOTS[i], drops[k])
            k += 1
        _press(plan, f"confirm_{b}")  # after its parts are in the tote
    return plan.wps


def ik_errors(wps):
    """The largest IK position (m) and rotation (rad) error over every waypoint of both arms."""
    worst_p = worst_r = 0.0
    for arm, ws in wps.items():
        q = np.array(READY, float)
        for w in ws:
            q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q,
                                 base_pos=BASES[arm])
            worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
    return worst_p, worst_r


# -- self-check (no simulator): every pattern's demo fits the run limit and IK reaches every waypoint ---------------
if __name__ == "__main__":
    for name in PATTERNS:
        wps = demo(name)
        t = max(sum(w.duration for w in ws) for ws in wps.values())
        pe, re = ik_errors(wps)
        assert t * 20 < 2400 and pe < 1e-3 and re < 1e-2, (name, t, pe, re)
        print(f"ok: pattern {name}: demo {t:.1f} s ({t * 20:.0f} steps), {sum(map(len, wps.values()))} waypoints, "
              f"IK error <= {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg")
