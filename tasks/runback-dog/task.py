"""On the station table: a carrier on a rail has to reach its end stop, pushed by a rod from a handle at the near end. Three
capped anti-runback dogs sit on twin studs over the rail. Each has a ramp and a tooth under its cap, so a dog facing forward
lets the carrier's spring pin ride under it and stops it running back. One dog was put on backwards: its tooth stops the carrier
dead. The caps hide which way each dog faces, so the fault shows only by pushing the carrier and seeing where it stops. The
robot must lift that dog off, turn it round, seat it again, and get the carrier to its end stop. Lots A, B and C put the
backwards dog at a different station; the environment variable DOG_LOT picks one (A by default).

Run it:  DOG_LOT=A inspect-robots run --embodiment yam_dog -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Get the carrier to its end stop with all three dogs seated on their studs facing forward, then move both hands away." --scorer task_score

`yam_dog` (runback_dog/embodiment.py) spawns the rail, carrier and dogs and reports each gripper's fingertip midpoint and the
carrier's travel.
"""

import math
import os

import numpy as np

from runback_dog import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
X0 = 0.27
DOGS = ("dog_a", "dog_b", "dog_c")
LOT = os.environ.get("DOG_LOT", "A").upper()
TURNED = dict(zip("ABC", DOGS))[LOT]
SEAT_Z = TOP + M.RAIL_Z
STATION = dict(zip(DOGS, M.STATIONS))

OBJECTS = [
    # placeholders: the yam_dog embodiment spawns the rail, the carrier and the dogs (mech) in their place
    Box("rail_frame", pos=(X0, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    Box("carrier_mech", pos=(X0, 0.0, TOP), size=(0.01, 0.01, 0.01)),
    *(Box(n, pos=(X0, STATION[n], SEAT_Z + 0.001), size=(0.01, 0.01, 0.01), mass=M.DOG_MASS) for n in DOGS),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
SEAT_XY, SEAT_DZ, YAW_TOL = 0.011, 0.004, math.radians(5)
AT_STOP = 0.01  # the carrier within this of its end stop
HANDS_CLEAR = 0.08
LET_GO = 0.06  # no credit while a fingertip midpoint is this near a dog, as when it's still held
STILL_S = 1.0
_trial = {"still": 0.0}


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset."""
    _trial.update(still=0.0)


def _yaw(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _flat(q):
    x, y, z, w = (float(v) for v in q)
    return 1 - 2 * (x * x + y * y) >= math.cos(math.radians(5))


def station_of(objects, name):
    """Which station's studs it's on (the dogs are identical, so any of them), flat and resting on the rail, or None."""
    o = objects[name]
    if abs(o.pos[0] - X0) > SEAT_XY or abs(o.pos[2] - SEAT_Z) > SEAT_DZ or not _flat(o.quat_xyzw):
        return None
    return next((y for y in M.STATIONS if abs(o.pos[1] - y) <= SEAT_XY), None)


def forward(objects, name):
    return bool(station_of(objects, name) is not None and abs(_yaw(objects[name].quat_xyzw)) <= YAW_TOL)


def all_forward(objects):
    """One dog on each station's studs, all facing forward."""
    return all(forward(objects, n) for n in DOGS) and len({station_of(objects, n) for n in DOGS}) == len(DOGS)


def at_stop(objects):
    return bool(objects["carrier"].pos[0] >= M.TRAVEL - AT_STOP)


def hands_clear(objects):
    """Both fingertip midpoints at least 8 cm from the handle's whole path, the rail's dogs and the carrier."""
    lo = np.array((X0 - M.RAIL_HALF_X, M.HANDLE["y"] - 0.02, TOP))
    hi = np.array((X0 + M.RAIL_HALF_X, M.RAIL_HIGH, TOP + M.RAIL_Z + M.GRIP["z1"]))
    for arm in ("left_tcp", "right_tcp"):
        p = np.asarray(objects[arm].pos, float)
        if np.linalg.norm(p - np.clip(p, lo, hi)) < HANDS_CLEAR:
            return False
    return True


def _near_dog(objects, arm, name):
    """A fingertip midpoint within LET_GO of the dog, measured to the box around it at any yaw."""
    p, o = np.asarray(objects[arm].pos, float), objects[name].pos
    half = np.array((M.DOG_HALF_X, M.DOG_HALF_X, M.GRIP["z1"]))
    return bool(np.linalg.norm(p - np.clip(p, o - half * (1, 1, 0), o + half)) < LET_GO)


def note_step(objects, dt):
    """Called by the embodiment after every control step: the clock for parts at rest with both hands clear."""
    still = all(objects[n].resting() for n in DOGS) and hands_clear(objects)
    _trial["still"] = _trial["still"] + dt if still else 0.0


def score(objects):
    """Read at the end of the run; there's no early end. 0 unless one dog is seated on each station's studs and all face forward,
    so removing or turning a sound dog scores 0, and let go. 0.5 for that, 1.0 with the carrier also at its end stop and all at rest with
    both hands clear for the last 1 s."""
    if not all_forward(objects) or any(_near_dog(objects, a, n) for a in ("left_tcp", "right_tcp") for n in DOGS):
        return 0.0
    return 1.0 if at_stop(objects) and _trial["still"] >= STILL_S else 0.5


# -- scripted demo -----------------------------------------------------------------------------------------------------
FINGER_PER_M = 0.04695 / 0.094
PAD_TIP_PAST_TCP = 0.013
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.32, TOP + 0.18  # far enough back to be clear of the handle's path
PUSH_Z = TOP + 0.07 + PAD_TIP_PAST_TCP
BEHIND = 0.025  # TCP this far behind the handle
PUSH_PAST = 0.06  # past where it must get to: the soft arm's push grows with how far short it is
YAW_A = {"left": -math.pi / 2, "right": math.pi / 2}


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
        p = p if pos is None else tuple(float(v) for v in pos)
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


def _fingers(width):
    return -FINGER_PER_M * width


def _park(plan, arm):
    plan.go(arm, 1.2, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", 0.0)


def _push(plan, label, to_y):
    """Push the handle with the right arm's closed fingers until the carrier would be at to_y, past it, and let it rest."""
    arm = "right"
    h0 = M.HANDLE["y"] - M.HANDLE["half_y"] - BEHIND
    h1 = h0 + (to_y - M.START) + PUSH_PAST
    plan.go(arm, 1.4, (X0, h0, PUSH_Z + 0.04), FINGER_CLOSED, f"{label}: behind the handle", YAW_A[arm])
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 0.7, (X0, h0, PUSH_Z), label="down behind it")
    plan.go(arm, 5.0, (X0, h1, PUSH_Z), label=label)
    plan.go(arm, 1.0, label="hold")
    plan.go(arm, 1.2, (X0, h1 - PUSH_PAST, PUSH_Z), label="ease off")
    plan.go(arm, 0.8, (X0, h0 + (to_y - M.START) - 0.03, PUSH_Z), label="back off")
    plan.go(arm, 0.6, (X0, h0 + (to_y - M.START) - 0.03, PUSH_Z + 0.06), label="up")
    _park(plan, arm)


def demo(lot=None):
    turned = TURNED if lot is None else dict(zip("ABC", DOGS))[lot]
    stopped_at = STATION[turned]
    y = STATION[dict(zip("ABC", DOGS))[os.environ["DOG_DEMO_TURN"]]] if "DOG_DEMO_TURN" in os.environ else stopped_at  # debugging only: turn a sound dog
    plan = _Plan()
    for a in SIDE:
        _park(plan, a)
    plan.sync()
    # where it stops: the pin against the turned dog's tooth
    _push(plan, "push: the carrier stops at the backwards dog", stopped_at - M.TOOTH_Y - M.PIN_HALF)
    plan.sync()
    arm = "left" if y > 0 else "right"
    gz = M.GRIP["z1"] - 0.006
    ready, grip = _fingers(2 * M.GRIP["half"] + 0.03), _fingers(2 * M.GRIP["half"] - 0.004)
    seat = SEAT_Z + gz
    lift = seat + 0.055  # its underside over the studs and its neighbours' caps, the most the arm can reach here
    # the wrist can't turn 180 degrees in one grip: a quarter turn, set it down on the table, regrasp, another quarter turn
    y0 = {"left": -math.pi / 2, "right": 0.0}[arm]
    y1 = y0 + math.pi / 2
    ax, ay = 0.15, y  # straight back off the rail, so it never passes over a neighbour
    table, high = TOP + gz, lift
    plan.go(arm, 1.4, (X0, y, lift), ready, "above the backwards dog", y0)
    plan.go(arm, 0.4, label="settle")
    plan.go(arm, 1.0, (X0, y, seat), label="descend")
    plan.go(arm, 0.6, finger=grip, label="grip")
    plan.go(arm, 1.4, (X0, y, high), label="lift it off")
    plan.go(arm, 1.6, (ax, ay, high), label="straight back off the rail")  # no turning over the rail: its neighbours' caps are as high as the lift
    plan.go(arm, 1.2, yaw=y1, label="a quarter turn")
    plan.go(arm, 0.5, label="settle")
    plan.go(arm, 1.2, (ax, ay, table + 0.002), label="set it down")
    plan.go(arm, 0.5, finger=ready, label="let go")
    plan.go(arm, 0.8, (ax, ay, high), label="up")
    plan.go(arm, 1.2, yaw=y0, label="turn the wrist back")
    plan.go(arm, 1.0, (ax, ay, table), label="down")
    plan.go(arm, 0.6, finger=grip, label="grip again")
    plan.go(arm, 1.2, (ax, ay, high), label="lift")
    plan.go(arm, 1.2, yaw=y1, label="the other quarter turn")
    plan.go(arm, 0.5, label="settle")
    plan.go(arm, 1.8, (X0, y, high), label="straight back over the studs")
    plan.go(arm, 1.0, label="settle")
    plan.go(arm, 1.6, (X0, y, seat + 0.001), label="seat it on the studs")
    plan.go(arm, 0.5, label="settle")
    plan.go(arm, 0.6, finger=ready, label="let go")
    plan.go(arm, 0.8, (X0, y, lift), label="up")
    _park(plan, arm)
    plan.sync()
    # from behind the handle's home, wherever the first push left it
    start = M.HANDLE["y"] - M.HANDLE["half_y"] - BEHIND
    plan.go("right", 1.4, (X0, start, PUSH_Z + 0.04), FINGER_CLOSED, "behind the handle", YAW_A["right"])
    plan.go("right", 0.4, label="settle")
    plan.go("right", 0.7, (X0, start, PUSH_Z), label="down behind it")
    h1 = M.HANDLE["y"] - M.HANDLE["half_y"] - BEHIND + M.TRAVEL + PUSH_PAST
    plan.go("right", 10.0, (X0, h1, PUSH_Z), label="push it to the end stop")
    plan.go("right", 2.0, label="hold")
    plan.go("right", 1.2, (X0, h1 - PUSH_PAST - 0.02, PUSH_Z), label="ease off")
    plan.go("right", 0.6, (X0, h1 - PUSH_PAST - 0.02, PUSH_Z + 0.06), label="up")
    _park(plan, "right")
    plan.go("right", 2.0, label="still")
    plan.sync()
    return plan.wps


def ik_errors(wps):
    """Waypoints, and steps along each yaw change, to stay on one joint branch."""
    worst_p = worst_r = 0.0
    bad = []
    for arm, ws in wps.items():
        q = np.array(READY, float)
        y = 0.0
        for w in ws:
            for f in np.linspace(0.25, 1.0, 4) if abs(w.yaw - y) > 1.0 else (1.0,):
                q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(y + f * (w.yaw - y))), q, base_pos=BASES[arm])
                worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
                if pe > 1e-3 or re > 1e-2:
                    bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1)))
            y = w.yaw
    return worst_p, worst_r, bad


# -- self-check (no simulator): every lot's demo fits the run limit and IK reaches every waypoint -----------------------
if __name__ == "__main__":
    for name in "ABC":
        wps = demo(name)
        t = max(sum(w.duration for w in ws) for ws in wps.values())
        pe, re, bad = ik_errors(wps)
        print(f"lot {name}: demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg", bad[:4])
        assert t * 20 < 2400 and not bad
