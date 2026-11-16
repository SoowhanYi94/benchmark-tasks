"""The task file's runtime for the gas-strut replacement (pure Python, unit-tested): what is read each control step, the
events it latches, the inspection it applies, and the score ladder.

Read each step: the lid angle; each socket of the NEW strut on its own stud on the left side (rod end on the lower,
body stud; body end on the lid stud); whether any socket of the TAGGED (failed) strut is still on a stud; where the
tagged strut's sockets are; the grippers' points and speed.

Latched good events (never a penalty; nothing bad is latched):
  held       new strut on at both ends, grippers clear (CLEAR_M) and still (STILL_S), lid at STOP_OK or more for HOLD_S
  closed     after held: lid within CLOSED_DEG of shut, grippers clear and still for STILL_S
  inspected  after closed, applied by the task file while the grippers stay clear: each new-strut socket pulled outward
             along its stud at PULL_N for PULL_S (moves under PULL_MM, stays on); then the lid raised to its stop by a
             force on the handle (at most RAISE_N) and let go: it stays at STOP_OK or more for HOLD_S
A gripper coming back within CLEAR_M stops a phase in progress; it starts again when the conditions hold again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import gsparts as G
from strutkit import quat_matrix_xyzw

SEAT_MM = 1.0          # ball centre within 1 mm of its socket's centre
JAW_SHUT_MM = 0.4      # the jaw (clip) back within 0.4 mm of its shut stop
STOP_OK = 43.0         # deg
CLOSED_DEG = 3.0       # deg
HOLD_S, PULL_N, PULL_S, PULL_MM = 2.0, 30.0, 1.0, 1.0
RAISE_N, RAISE_S, RAISE_TO = 8.0, 4.0, 44.0
CLEAR_M, STILL_S, STILL_MPS = 0.030, 1.0, 0.01
REST_Z, OLD_PLACE_PAD = 0.035, 0.010   # tagged strut resting: socket centres at most 35 mm above the table top
NEW_SOCKETS = ("body", "lid")


@dataclass
class Reading:
    t: float
    lid_deg: float
    new_on: dict            # "body"/"lid" -> bool (new strut's socket on its own stud, left side)
    old_on: bool            # any socket of the tagged strut on any stud
    old_pts: list           # the tagged strut's two socket centres, world (m)
    new_pos: dict           # "body"/"lid" -> new strut's socket centre, world (m)
    hand_points: list       # world points of both grippers
    hand_speed: float       # fastest gripper point (m/s)


def old_resting(r: Reading) -> bool:
    """Tagged strut off its studs and lying on the table or the battens: both socket centres at most REST_Z above the
    table, neither over the enclosure, no gripper within CLEAR_M of them."""
    if r.old_on:
        return False
    pts = np.asarray(r.old_pts, float).reshape(-1, 3)
    for p in pts:
        if p[2] - G.TOP > REST_Z:
            return False
        if G.X0 - OLD_PLACE_PAD <= p[0] <= G.X1 + OLD_PLACE_PAD and abs(p[1]) <= G.HW + OLD_PLACE_PAD:
            return False
    hands = np.asarray(r.hand_points, float).reshape(-1, 3)
    return bool(np.linalg.norm(hands[:, None, :] - pts[None, :, :], axis=2).min() >= CLEAR_M)


@dataclass
class Tracker:
    phase: str = "wait"
    t0: float = 0.0
    still_since: float | None = None
    held: bool = False
    closed: bool = False
    inspected: bool = False
    pull_idx: int = 0
    pull_start: np.ndarray | None = None
    results: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def reset(self):
        self.__init__()

    def clear(self, r: Reading) -> bool:
        pts = np.asarray(r.hand_points, float).reshape(-1, 3)
        return bool(len(pts)) and min_dist_to_zone(pts, r.lid_deg) >= CLEAR_M

    def wants_force(self):
        """The test force to apply now: None, ("pull", end), or ("raise", None)."""
        if self.phase == "pull":
            return ("pull", NEW_SOCKETS[self.pull_idx])
        if self.phase == "raise":
            return ("raise", None)
        return None

    def step(self, r: Reading):
        clear = self.clear(r)
        if clear and r.hand_speed < STILL_MPS:
            self.still_since = r.t if self.still_since is None else self.still_since
        else:
            self.still_since = None
        settled = self.still_since is not None and r.t - self.still_since >= STILL_S
        both = all(r.new_on.values())
        busy = ("hold", "pull", "relax", "raise", "stay")
        if self.phase in busy and not clear:
            self._to("wait", r, "a gripper came back: phase stopped")
            return
        if self.phase == "wait":
            if not both or not settled:
                return
            if not self.held and r.lid_deg >= STOP_OK:
                self._to("hold", r)
            elif self.held and not self.closed and r.lid_deg <= CLOSED_DEG:
                self.closed = True
                self._to("wait", r, "lid closed")
            elif self.closed and not self.inspected and r.lid_deg <= CLOSED_DEG:
                self.pull_idx = 0
                self.pull_start = np.asarray(r.new_pos[NEW_SOCKETS[0]], float)
                self._to("pull", r)
        elif self.phase == "hold":
            if r.lid_deg < STOP_OK:
                self._to("wait", r, f"lid sank to {r.lid_deg:.1f} deg")
            elif r.t - self.t0 >= HOLD_S:
                self.held = True
                self._to("wait", r, "lid held itself open")
        elif self.phase == "pull":
            end = NEW_SOCKETS[self.pull_idx]
            moved = float(np.linalg.norm(np.asarray(r.new_pos[end], float) - self.pull_start)) * 1000
            if r.t - self.t0 >= PULL_S or (not r.new_on[end] and moved > 3 * PULL_MM):
                ok = moved < PULL_MM and r.new_on[end]
                self.results[f"pull_{end}"] = (round(moved, 3), ok)
                if not ok:
                    self._to("wait", r, f"{end} socket moved {moved:.2f} mm under the pull")
                else:
                    self._to("relax", r)
        elif self.phase == "relax":
            if r.t - self.t0 >= 0.5:
                self.pull_idx += 1
                if self.pull_idx < len(NEW_SOCKETS):
                    self.pull_start = np.asarray(r.new_pos[NEW_SOCKETS[self.pull_idx]], float)
                    self._to("pull", r)
                else:
                    self._to("raise", r)
        elif self.phase == "raise":
            if r.lid_deg >= RAISE_TO:
                self.results["raised_to"] = round(r.lid_deg, 2)
                self._to("stay", r)
            elif r.t - self.t0 >= RAISE_S:
                self.results["raised_to"] = round(r.lid_deg, 2)
                self._to("wait", r, f"raise reached only {r.lid_deg:.1f} deg")
        elif self.phase == "stay":
            if r.lid_deg < STOP_OK:
                self.results["stay"] = round(r.lid_deg, 2)
                self._to("wait", r, f"lid sank to {r.lid_deg:.1f} deg after the raise")
            elif r.t - self.t0 >= HOLD_S:
                self.inspected = True
                self._to("done", r, "inspection passed")

    def _to(self, phase, r, note=""):
        self.log.append((round(r.t, 2), self.phase, phase, note))
        self.phase, self.t0 = phase, r.t


def zone_points(lid_deg: float) -> np.ndarray:
    """Sample points of the lid (with handle and flanges) and both struts' stud lines, world (m)."""
    th = math.radians(lid_deg)
    pts = []
    for a in np.linspace(0.0, 0.20, 9):
        for y in np.linspace(-G.HW, G.HW, 9):
            x, z = G.lid_point(a, -0.002, th)
            pts.append((x, y, G.TOP + z))
    for y in np.linspace(-0.15, 0.15, 11):
        x, z = G.lid_point(0.18, -0.030, th)
        pts.append((x, y, G.TOP + z))
    for s in (1, -1):
        b, l = G.stud_points(s, th)
        for f in np.linspace(0, 1, 9):
            pts.append(tuple(b + f * (l - b)))
    return np.array(pts)


def min_dist_to_zone(hand_pts: np.ndarray, lid_deg: float) -> float:
    z = zone_points(lid_deg)
    d = np.linalg.norm(hand_pts[:, None, :] - z[None, :, :], axis=2)
    return float(d.min())


def ladder(r: Reading, tr: Tracker) -> float:
    """Highest rung reached; each rung needs the ones below. 0.15/0.30/0.45 are read from the state at the end; 0.60,
    0.80 and 1.0 are the latched events held, closed (after held) and inspected (after closed)."""
    if r is None or not old_resting(r):
        return 0.0
    n = sum(bool(r.new_on[e]) for e in NEW_SOCKETS)
    if n == 0:
        return 0.15
    if n == 1:
        return 0.30
    if not tr.held:
        return 0.45
    if not tr.closed:
        return 0.60
    if not tr.inspected:
        return 0.80
    return 1.0


# -- reading the sim ------------------------------------------------------------------------------------------------


def socket_centre(link_pose, xc):
    R = quat_matrix_xyzw(link_pose[3:7])
    return np.asarray(link_pose[:3], float) + R @ np.array([xc, 0.0, 0.0]), R


def socket_seated(link_pose, xc, ball, jaws_mm) -> bool:
    c, R = socket_centre(link_pose, xc)
    rel = R.T @ (np.asarray(ball, float) - c)
    return bool(np.linalg.norm(rel) * 1000 < SEAT_MM and max(jaws_mm) < JAW_SHUT_MM)
