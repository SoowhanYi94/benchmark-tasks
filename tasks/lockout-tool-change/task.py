"""On the station table: a small machine's tool holder stands in the middle of the bench, a square socket holding a
worn cutter (dull grey, standing upright). On the right, the machine's isolator: a red handle in a track, now at the
green ON end (a green mark beside the track); a red mark beside the far end shows OFF, and a yellow padlock stands on
its white mark on the bench beside the track. Between the isolator and the holder is the machine's control box, with a
green START button and a run lamp (dark: the machine is stopped). A new cutter (bright) stands on a tool tray on the
left, and a red scrap bin stands on the left beyond the holder.

The full lockout cycle: isolate (handle to OFF), lock (padlock dropped in the track behind the handle, where it stops
the handle going back to ON), verify zero energy (press START: the run lamp must stay off), service (worn cutter to
scrap, new cutter into the holder), then restore (padlock off the track and back on its mark, handle back to ON).

Needs the `yam_lockout` embodiment (lockout_tool_change/embodiment.py): it builds the control box, START button and
run lamp in place of the `control_box` placeholder, and reports the button and both TCPs to the scorer.

Run it:  inspect-robots run --embodiment yam_lockout -E task=tasks/lockout-tool-change/task.py \\
           --instruction "Lock out the machine, verify it is dead, change the cutter and restore it: slide the isolator handle to OFF and put the padlock in the track behind it, press the green START button and check the run lamp stays off, put the worn cutter in the scrap bin and the new cutter in the tool holder, then put the padlock back on its mark and slide the handle back to ON." --scorer task_score ...
"""

import math

import numpy as np

from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]
GREY, DARK, RED, GREEN = (0.45, 0.47, 0.50), (0.25, 0.26, 0.28), (0.80, 0.12, 0.10), (0.15, 0.60, 0.25)
YELLOW, WHITE = (0.95, 0.75, 0.10), (0.85, 0.85, 0.85)

# -- the isolator: a handle sliding along x in a track between its ON and OFF ends --------------------------------
TRACK_Y = -0.22
ON_X, OFF_X = 0.17, 0.29  # handle centre at each end
HANDLE = (0.05, 0.04, 0.04)  # x, y, z
RAIL_GAP = 0.05  # between the rails' inner faces
RAIL_LEN = OFF_X - ON_X + HANDLE[0] + 0.04
RAIL_X = ((ON_X + OFF_X - RAIL_LEN) / 2, (ON_X + OFF_X + RAIL_LEN) / 2)  # 0.125 .. 0.335
MARK_Y = TRACK_Y - (RAIL_GAP / 2 + 0.01 + 0.02)  # -0.275, beside the right rail
LOCK = (0.03, 0.03, 0.05)
LOCK_MARK = dict(center=(0.15, -0.33), size=(0.05, 0.05, 0.002))
LOCK_START = LOCK_MARK["center"]

# -- the control box: START button (sprung, 8 mm travel) and run lamp, built by the embodiment ---------------------
CONTROL = dict(center=(0.30, -0.09), size=(0.08, 0.08, 0.05))
BUTTON_R, BUTTON_PROUD, BUTTON_TRAVEL = 0.014, 0.010, 0.008
BUTTON_TOP = TOP + CONTROL["size"][2] + BUTTON_PROUD  # the cap's top at rest
PRESSED = 0.004  # pushed down this far counts as pressed
LAMP = dict(offset=(0.025, -0.025), radius=0.008)  # on the box top, at its far right corner

# -- the tool holder: a square socket (1 cm clearance a side) on a base, holding a cutter upright ------------------
HOLDER = dict(center=(0.30, 0.06), inner=0.044, wall=0.01, height=0.04, base=0.01)
HOLDER_FLOOR = TOP + HOLDER["base"]
HOLDER_ZONE = HOLDER["inner"] / 2 + HOLDER["wall"] + 0.01  # half-width of "on the machine": the holder plus 1 cm
CUTTER_R, CUTTER_H = 0.012, 0.08
NEW_START = (0.18, 0.22)  # on the tool tray
TOOL_TRAY = dict(center=(0.18, 0.22), size=(0.10, 0.10, 0.006))
SCRAP = dict(center=(0.36, 0.24), inner=(0.10, 0.10), wall=0.01, height=0.05, floor=0.004)


def _rails():
    cx = (ON_X + OFF_X) / 2
    return [
        *[Box(f"isolator_rail_{n}", pos=(cx, TRACK_Y + s * (RAIL_GAP / 2 + 0.005), TOP + 0.015), size=(RAIL_LEN, 0.01, 0.03), color=DARK)
          for n, s in (("left", 1), ("right", -1))],
        # the ON and OFF marks sit beside the track, outside the right rail (in it, the sliding handle catches on them)
        Box("isolator_on_mark", pos=(ON_X, MARK_Y, TOP + 0.001), size=(0.04, 0.03, 0.002), color=GREEN),
        Box("isolator_off_mark", pos=(OFF_X, MARK_Y, TOP + 0.001), size=(0.04, 0.03, 0.002), color=RED),
        Box("padlock_mark", pos=(*LOCK_MARK["center"], TOP + LOCK_MARK["size"][2] / 2), size=LOCK_MARK["size"], color=WHITE),
    ]


def _box5(name, center, inner, wall, height, floor, color, z0=TOP):
    (cx, cy), (ix, iy) = center, inner
    return [
        Box(f"{name}_base", pos=(cx, cy, z0 + floor / 2), size=(ix + 2 * wall, iy + 2 * wall, floor), color=color),
        Box(f"{name}_far", pos=(cx + ix / 2 + wall / 2, cy, z0 + floor + height / 2), size=(wall, iy + 2 * wall, height), color=color),
        Box(f"{name}_near", pos=(cx - ix / 2 - wall / 2, cy, z0 + floor + height / 2), size=(wall, iy + 2 * wall, height), color=color),
        Box(f"{name}_left", pos=(cx, cy + iy / 2 + wall / 2, z0 + floor + height / 2), size=(ix, wall, height), color=color),
        Box(f"{name}_right", pos=(cx, cy - iy / 2 - wall / 2, z0 + floor + height / 2), size=(ix, wall, height), color=color),
    ]


OBJECTS = [
    *_rails(),
    *_box5("holder", HOLDER["center"], (HOLDER["inner"],) * 2, HOLDER["wall"], HOLDER["height"], HOLDER["base"], GREY),
    *_box5("scrap_bin", SCRAP["center"], SCRAP["inner"], SCRAP["wall"], SCRAP["height"] - SCRAP["floor"], SCRAP["floor"], RED),
    Box("tool_tray", pos=(*TOOL_TRAY["center"], TOP + TOOL_TRAY["size"][2] / 2), size=TOOL_TRAY["size"], color=(0.20, 0.35, 0.60)),
    # placeholder: the embodiment builds the control box, the START button and the run lamp here
    Box("control_box", pos=(*CONTROL["center"], TOP), size=(0.01, 0.01, 0.01), color=DARK),
    Box("isolator_handle", pos=(ON_X, TRACK_Y, TOP + HANDLE[2] / 2), size=HANDLE, mass=0.2, color=RED),
    Box("padlock", pos=(*LOCK_START, TOP + LOCK_MARK["size"][2] + LOCK[2] / 2), size=LOCK, mass=0.15, color=YELLOW),
    Cylinder("worn_cutter", pos=(*HOLDER["center"], HOLDER_FLOOR + CUTTER_H / 2), radius=CUTTER_R, height=CUTTER_H, mass=0.15, color=DARK),
    Cylinder("new_cutter", pos=(*NEW_START, TOP + TOOL_TRAY["size"][2] + CUTTER_H / 2), radius=CUTTER_R, height=CUTTER_H, mass=0.15,
             color=(0.85, 0.87, 0.90)),
]


# -- scoring --------------------------------------------------------------------------------------------------------
def _tilt(quat_xyzw):
    """How far the part's z axis is from vertical (rad)."""
    x, y, z, w = quat_xyzw
    return math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))


_SIGNS = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)


def corners(o, size):
    """The eight corners of a part's box (a cylinder's bounding box turns with it), world frame (m)."""
    x, y, z, w = o.quat_xyzw
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.asarray(o.pos[:3], float) + (_SIGNS * np.asarray(size) / 2) @ r.T


def _moved(a, b, tol):
    """Any corner more than tol apart."""
    return bool(np.max(np.linalg.norm(a - b, axis=1)) > tol)


HANDS_CLEAR = 0.05


def released(o, objects):
    """Neither gripper's TCP within 5 cm of the part's centre."""
    return all(np.linalg.norm(np.asarray(objects[t].pos[:3]) - o.pos[:3]) > HANDS_CLEAR for t in ("left_tcp", "right_tcp"))


def handle_at(x_mark, objects):
    """The handle in the track (centre within 1 cm of its centre line), centre within 1.5 cm of the mark, upright
    within 15 degrees, underside within 1 cm of the table, still and let go."""
    o = objects["isolator_handle"]
    return bool(abs(o.pos[0] - x_mark) <= 0.015 and abs(o.pos[1] - TRACK_Y) <= 0.01 and o.low[2] <= TOP + 0.01
                and _tilt(o.quat_xyzw) <= math.radians(15) and o.resting() and released(o, objects))


def energised(objects):
    """The isolator's contacts are closed while the handle's centre is within 3 cm of ON."""
    return bool(abs(objects["isolator_handle"].pos[0] - ON_X) <= 0.03)


def _lock_in_rails(objects):
    o = objects["padlock"]
    return bool(abs(o.pos[1] - TRACK_Y) <= 0.015 and RAIL_X[0] <= o.pos[0] <= RAIL_X[1] and o.low[2] <= TOP + 0.01)


def lock_behind(objects):
    """The padlock down in the track between the ON end and the handle (its centre at least 3.5 cm short of the
    handle's), upright within 15 degrees, still and let go: there it stops the handle going back to ON."""
    o, h = objects["padlock"], objects["isolator_handle"]
    return bool(_lock_in_rails(objects) and ON_X - 0.03 <= o.pos[0] <= h.pos[0] - 0.035
                and _tilt(o.quat_xyzw) <= math.radians(15) and o.resting() and released(o, objects))


def lock_on_mark(objects):
    """The padlock back on its white mark: centre within 2 cm of the mark's, underside within 1 cm of the table,
    upright within 15 degrees, still and let go."""
    o = objects["padlock"]
    return bool(np.hypot(*(o.pos[:2] - np.array(LOCK_START))) <= 0.02 and o.low[2] <= TOP + 0.01
                and _tilt(o.quat_xyzw) <= math.radians(15) and o.resting() and released(o, objects))


def pressed(objects):
    return bool(BUTTON_TOP - objects["start_button"].pos[2] >= PRESSED)


def _in_socket(o):
    (cx, cy) = HOLDER["center"]
    return bool(abs(o.pos[0] - cx) <= HOLDER["inner"] / 2 and abs(o.pos[1] - cy) <= HOLDER["inner"] / 2)


def new_cutter_seated(objects):
    """Centre inside the socket, underside within 1 cm of its floor, upright within 10 degrees, still and let go."""
    o = objects["new_cutter"]
    return bool(_in_socket(o) and o.low[2] <= HOLDER_FLOOR + 0.01 and _tilt(o.quat_xyzw) <= math.radians(10)
                and o.resting() and released(o, objects))


def worn_in_scrap(objects):
    """Centre inside the bin's walls, underside within 2.6 cm of its floor, let go (it may rock or lie down)."""
    (cx, cy), (ix, iy) = SCRAP["center"], SCRAP["inner"]
    o = objects["worn_cutter"]
    return bool(abs(o.pos[0] - cx) <= ix / 2 and abs(o.pos[1] - cy) <= iy / 2
                and o.low[2] <= TOP + SCRAP["floor"] + 0.026 and released(o, objects))


def _on_machine(c):
    """A cutter's box corners: any within the holder's footprint plus 1 cm (at any height)."""
    (cx, cy) = HOLDER["center"]
    return bool(np.any((np.abs(c[:, 0] - cx) <= HOLDER_ZONE) & (np.abs(c[:, 1] - cy) <= HOLDER_ZONE)))


# The order matters, so the score remembers, for the whole trial: whether the machine was locked out (the padlock
# put in after the handle reached OFF), whether START was pressed while it was, and whether a cutter on the machine
# was disturbed while it wasn't both locked out and verified. None of these is ever cleared from object poses, except
# by a reset the robot can't fake (see _new_trial).
FREE = {"isolator_handle": HANDLE, "padlock": LOCK, "worn_cutter": (2 * CUTTER_R,) * 2 + (CUTTER_H,),
        "new_cutter": (2 * CUTTER_R,) * 2 + (CUTTER_H,)}
CUTTERS = ("worn_cutter", "new_cutter")
_trial = {"start": None, "last": None}


def _fresh(now):
    _trial["trials"] = _trial.get("trials", 0) + 1  # for the embodiment's debug print
    _trial.update(start=now, last=now, violated=False, locked_ever=False, verified_ever=False, verified=False,
                  lock_was_in=False, lock_ok=False, ref={n: now[n] for n in CUTTERS})


def _new_trial(objects, now):
    """The harness doesn't tell a task file when a trial starts. The first scored step starts one (its settled poses
    are the start), and so does a step on which every free part is at rest with every corner within 1 mm of its start
    after one of them jumped more than 1 cm in one step: a reset does that, and a robot putting the parts back won't
    move one 1 cm in a step and stop it dead with all four parts' corners within 1 mm of their starts. (Run one trial
    per run if in doubt.)"""
    start, last = _trial["start"], _trial["last"]
    if start is None:
        return True
    at_start = all(not _moved(now[n], start[n], 0.001) and objects[n].resting() for n in FREE)
    return at_start and any(_moved(now[n], last[n], 0.01) for n in FREE)


def locked_out(objects):
    return bool(handle_at(OFF_X, objects) and lock_behind(objects) and _trial.get("lock_ok"))


def safe(objects):
    """Locked out, and START pressed (the lamp staying off) since this lockout began."""
    return bool(locked_out(objects) and _trial.get("verified"))


def _update(objects):
    now = {n: corners(objects[n], s) for n, s in FREE.items()}
    if _new_trial(objects, now):
        _fresh(now)
    _trial["last"] = now
    # the padlock counts only if the handle was already at OFF when it went into the track
    in_rails = _lock_in_rails(objects)
    if in_rails and not _trial["lock_was_in"]:
        _trial["lock_ok"] = handle_at(OFF_X, objects)
    _trial["lock_was_in"] = in_rails
    if not in_rails:
        _trial["lock_ok"] = False
    lo = locked_out(objects)
    _trial["locked_ever"] |= lo
    _trial["verified"] = lo and (_trial["verified"] or pressed(objects))  # a new lockout must be verified again
    _trial["verified_ever"] |= _trial["verified"]
    ok = lo and _trial["verified"]
    for n in CUTTERS:
        ref = _trial["ref"][n]
        if ok:
            _trial["ref"][n] = now[n]
        elif (_on_machine(ref) or _on_machine(now[n])) and _moved(now[n], ref, 0.003):
            _trial["violated"] = True


def restored(objects):
    return lock_on_mark(objects) and handle_at(ON_X, objects)


def score(objects):
    """0.15 locked out (handle at OFF, then the padlock in the track behind it) + 0.15 START pressed while locked out
    + 0.2 worn cutter in the scrap bin + 0.3 new cutter seated in the holder + 0.2 restored with the job done (padlock
    back on its mark, handle back at ON). 0 for the whole trial once a cutter on the machine is disturbed (any corner
    of its box more than 3 mm from where it was) while the machine is not locked out and verified."""
    _update(objects)
    if _trial["violated"]:
        return 0.0
    w, n = worn_in_scrap(objects), new_cutter_seated(objects)
    return float(0.15 * _trial["locked_ever"] + 0.15 * _trial["verified_ever"] + 0.2 * w + 0.3 * n
                 + 0.2 * (w and n and _trial["verified_ever"] and restored(objects)))


def success(objects):
    return score(objects) >= 1.0 - 1e-9


# -- scripted demo ----------------------------------------------------------------------------------------------------
Z_UP = TOP + 0.15  # carry height: a held cutter (gripped 2 cm below its top) clears the holder and the bin
SIDE = {"left": 1, "right": -1}
COMFORT = {"left": -math.pi / 4, "right": math.pi / 4}
SLIDE_TRIM = 0.009  # the held handle slid 9 mm short (measured in the simulator on this track)
GRIP_HANDLE, GRIP_LOCK = 0.022, 0.03  # TCP this far above their undersides
LOCK_X = (ON_X + OFF_X) / 2 - 0.01  # where the padlock goes in the track: 6 cm short of the handle at OFF
PRESS_DEPTH = 0.012  # the demo aims 4 mm past START's 8 mm stop (counts at 4 mm): aimed at 6 mm, it went only 3.1 mm in sim, and 4.3 mm aimed at 9


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


def _yaw(arm, base=0.0):
    return min((base + k * math.pi for k in range(-2, 3)), key=lambda v: abs(v - COMFORT[arm]))


def _pick_place(plan, arm, pick, pick_z, place, place_z, label, yaw_base=0.0, z_up=Z_UP):
    yaw = _yaw(arm, yaw_base)
    (px, py), (qx, qy) = pick, place
    plan.go(arm, 1.3, (px, py, z_up), FINGER_OPEN, f"{label}: above it", yaw)
    plan.go(arm, 1.2, (px, py, pick_z), label="descend")
    plan.go(arm, 0.6, finger=FINGER_CLOSED, label="grip")
    plan.go(arm, 1.2, (px, py, z_up), label="lift")
    plan.go(arm, 1.6, (qx, qy, z_up), label="carry")
    plan.go(arm, 1.3, (qx, qy, place_z), label="lower")
    plan.go(arm, 0.6, finger=FINGER_OPEN, label="release")
    plan.go(arm, 0.8, (qx, qy, z_up), label="up")


def _slide(plan, x_from, x_to, label):
    """Right arm: grip the handle along the track (fingers clear of the rails), ease it up 2 mm, slide it."""
    g = GRIP_HANDLE
    yaw = _yaw("right", math.pi / 2)
    plan.go("right", 1.3, (x_from, TRACK_Y, TOP + 0.10), FINGER_OPEN, f"{label}: above the handle", yaw)
    plan.go("right", 1.0, (x_from, TRACK_Y, TOP + g), label="descend")
    plan.go("right", 0.6, finger=FINGER_CLOSED, label="grip")
    plan.go("right", 0.4, (x_from, TRACK_Y, TOP + g + 0.002), label="ease it off the table")
    plan.go("right", 2.0, (x_to, TRACK_Y, TOP + g + 0.002), label=label)
    plan.go("right", 0.4, (x_to, TRACK_Y, TOP + g), label="set it down")
    plan.go("right", 0.5, finger=FINGER_OPEN, label="release")
    plan.go("right", 0.6, (x_to, TRACK_Y, TOP + 0.10), label="up")


def _press_start(plan):
    """Right arm, fingers closed: the fingertips down on the START cap, pushed to its stop, held, then up."""
    (bx, by), tip = CONTROL["center"], PAD_TIP_PAST_TCP
    yaw = _yaw("right", math.pi / 2)
    plan.go("right", 1.3, (bx, by, BUTTON_TOP + 0.05), FINGER_CLOSED, "START: above it", yaw)
    plan.go("right", 0.8, (bx, by, BUTTON_TOP + tip + 0.01), label="down to the cap")
    plan.go("right", 0.6, (bx, by, BUTTON_TOP + tip - PRESS_DEPTH), label="press START")
    plan.go("right", 0.6, label="hold it in: the run lamp stays off")
    plan.go("right", 0.6, (bx, by, BUTTON_TOP + 0.05), label="let go")


OUT_RIGHT, OUT_LEFT = (0.18, -0.30, TOP + 0.20), (0.18, 0.30, TOP + 0.20)


def demo():
    plan = _Plan()
    plan.go("left", 1.3, OUT_LEFT, FINGER_OPEN, "out of the way", _yaw("left"))  # waiting at READY isn't a down grip
    # 1-2. right: isolate (handle to OFF), then lock (padlock into the track behind it)
    _slide(plan, ON_X, OFF_X + SLIDE_TRIM, "slide to OFF")
    _pick_place(plan, "right", LOCK_START, TOP + LOCK_MARK["size"][2] + GRIP_LOCK, (LOCK_X, TRACK_Y), TOP + GRIP_LOCK + 0.004,
                "padlock into the track", yaw_base=math.pi / 2, z_up=TOP + 0.10)
    # 3. right: verify zero energy
    _press_start(plan)
    plan.go("right", 1.2, OUT_RIGHT, FINGER_OPEN, "out of the way")
    plan.sync()
    # 4. left: the worn cutter to scrap, then the new one into the holder (gripped 2 cm below their tops)
    g = CUTTER_H - 0.02
    _pick_place(plan, "left", HOLDER["center"], HOLDER_FLOOR + g, SCRAP["center"], TOP + SCRAP["floor"] + g + 0.005, "worn cutter")
    _pick_place(plan, "left", NEW_START, TOP + TOOL_TRAY["size"][2] + g, HOLDER["center"], HOLDER_FLOOR + g + 0.004, "new cutter")
    plan.go("left", 1.2, OUT_LEFT, label="back")
    plan.sync()
    # 5. right: restore (padlock out of the track onto its mark, handle back to ON)
    _pick_place(plan, "right", (LOCK_X, TRACK_Y), TOP + GRIP_LOCK, LOCK_START, TOP + LOCK_MARK["size"][2] + GRIP_LOCK + 0.004,
                "padlock back to its mark", yaw_base=math.pi / 2, z_up=TOP + 0.10)
    _slide(plan, OFF_X, ON_X - SLIDE_TRIM, "slide back to ON")
    plan.go("right", 1.2, OUT_RIGHT, label="out of the way")
    plan.sync()
    return plan.wps


def ik_check():
    """Every demo waypoint solved in order from READY: (worst position error m, worst rotation error rad)."""
    from yam_arena.demo import grip_rotation

    worst_p = worst_r = 0.0
    for arm, wps in demo().items():
        q = np.array(READY, float)
        for w in wps:
            q, pe, re = ik.solve(w.pos, ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
            worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
    return worst_p, worst_r


if __name__ == "__main__":
    t = max(sum(w.duration for w in wps) for wps in demo().values())
    pe, re = ik_check()
    assert t * 20 < 2400 and pe < 0.001 and re < math.radians(1), (t, pe, re)
    print(f"ok: demo {t:.1f} s ({t * 20:.0f} steps); IK worst {pe * 1000:.3f} mm, {math.degrees(re):.3f} deg")
