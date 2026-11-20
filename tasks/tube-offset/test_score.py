"""Scoring checks without the simulator (NumPy only): python -m pytest -q test_score.py"""

import json
import math
import pathlib

import numpy as np
import pytest

import task as T
from tube_offset import mech as M
from yam_arena.task_file import ObjectState

HERE = pathlib.Path(__file__).parent
FAR = np.array((0.0, 0.6, 1.2))
D = math.radians


def _quat(yaw):
    return np.array((0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))


def states(q=None, origin=T.LANE_AT, yaw=T.LAID_YAW, z=M.TUBE_R, left=FAR, right=FAR, vel=0.0, sup=0.0, clev=0.0):
    """A fake simulator reading: the tube laid flat at height z (centre above the table), hinges q, hands at left/right."""
    q = {} if q is None else q
    f = T.tube_fk(q, origin, yaw)
    s = {}
    for n, (x, y, a) in f.items():
        p = np.array((x, y, T.TOP + z))
        s[f"tube/{n}"] = ObjectState(pos=p, quat_xyzw=_quat(a), lin_vel=np.array((vel, 0, 0)), ang_vel=np.zeros(3), low=p, high=p)
    for k, p in (("left_tcp", left), ("right_tcp", right)):
        s[k] = ObjectState(pos=np.asarray(p, float), quat_xyzw=_quat(0), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p)
    s["q"] = {f"tube/{j}": q.get(j, 0.0) for j in T.HINGES}
    s["q"].update({"bender/support_turn": sup, "bender/clevis_turn": clev, "bender/latch_turn": 0.0})
    return s


def run(s, steps=30, tool=True):
    """A fresh trial whose bends were all made by the bender, then `steps` control steps of the reading s."""
    T.reset_latch()
    if tool:
        _made_by(s, True)
    for _ in range(steps):
        T.note_step(s, 0.05)
    return T.score(s)


def _made_by(s, tool):
    a = np.array([s["q"][f"tube/{j}"] for j in T.HINGES])
    T.note_plastic(a, D(1.0) if tool else 0.0, 0.0 if tool else T.CLEVIS_SNAP - 0.1)


def offset(a=45.0, b=45.0, n=5):
    return T.ideal_q(D(a), D(b), n)


def test_finished_offset_in_the_bender_scores_full():
    assert run(states(offset())) == 1.0


def test_needs_one_second_of_rest():
    assert run(states(offset()), steps=20) == 0.9  # 0.95 s (rest is read between steps)
    assert run(states(offset()), steps=21) == 1.0
    T.reset_latch()
    _made_by(states(offset()), True)
    for k in range(30):  # creeping 1.5 cm/s
        T.note_step(states(offset(), origin=(T.LANE_AT[0] + 0.00075 * k, T.LANE_AT[1])), 0.05)
    assert T.score(states(offset())) == 0.9


def test_must_be_let_go():
    near = T.tube_fk(offset(), T.LANE_AT, T.LAID_YAW)["leg_a"]
    hand = np.array((near[0], near[1], T.TOP + 0.05))
    assert run(states(offset(), left=hand)) == 0.0
    assert run(states(offset(), right=hand + (0, 0, 0.02))) == 1.0  # 7 cm above the leg: let go


def test_straight_tube_scores_zero():
    assert run(states()) == 0.0


def test_one_bend():
    assert run(states(offset(45, 0))) == 0.25
    assert run(states(offset(45, 30))) == 0.25
    assert run(states(offset(0, 44))) == 0.25


def test_bend_tolerance():
    assert run(states(offset(41.2, 41.2))) == 1.0
    assert run(states(offset(40.5, 45))) == 0.25
    assert run(states(offset(49.5, 45))) == 0.25


def test_not_parallel_or_offset_wrong():
    assert run(states(offset(42, 47))) == 0.5  # ends 5 degrees apart


def test_roll_support_must_be_back_at_0():
    assert run(states(offset(), sup=D(4.9))) == 1.0
    assert run(states(offset(), sup=D(-4.9))) == 1.0
    assert run(states(offset(), sup=D(5.5))) == 0.75
    assert run(states(offset(), sup=D(45.0))) == 0.75


def test_kinks_do_not_count():
    j = T.ZONE_A[10]
    q = offset(0, 45)
    q[j] = D(45)
    assert run(states(q)) == 0.25  # B counts, the kinked A does not
    q = offset(0, 45)
    q[T.ZONE_A[9]], q[T.ZONE_A[10]] = D(22.5), D(22.5)
    assert run(states(q)) == 0.25


def test_too_wide_a_bend_does_not_count():
    assert run(states(offset(45, 45, n=12))) == 0.0  # 30 mm long: radius 38 mm, not the die's


def test_radius_reading():
    b = T.bend(states(offset()), T.ZONE_A)
    assert b[1] == 5 and abs(b[2] - 5 * T.DL / D(45)) < 1e-9
    assert T.RADIUS[0] <= b[2] <= T.RADIUS[1]


def test_bend_made_without_the_bender_does_not_count():
    s = states(offset())
    T.reset_latch()
    _made_by(s, False)
    for _ in range(30):
        T.note_step(s, 0.05)
    assert T.score(s) == 0.0


def moved(da=0, db=0, wa=1.0, wb=1.0):
    """The ideal offset with bend A's hinges moved da hinges (2.5 mm each) toward end A and bend B's db toward end B;
    w mixes the pattern moved by d with the one moved by d - 1 (w = 1: moved by d), to place a bend between hinges."""
    def zone(z, d, w):
        base = [z[i] for i in range(len(z)) if z[i] in offset()]
        out = {}
        for dd, ww in ((d, w), (d - 1, 1.0 - w)):
            for j in base:
                k = z[z.index(j) + dd]
                out[k] = out.get(k, 0.0) + ww * offset()[j]
        return out
    return {**zone(T.ZONE_A, da, wa), **zone(T.ZONE_B, db, wb)}


def test_bend_place_reading():
    for da, db in ((0, 0), (1, 0), (0, 1), (2, -2)):
        q = states(moved(da, db))["q"]
        assert abs(T.bend_place(q, 0) - da * T.DL) < 2e-4 and abs(T.bend_place(q, 1) - db * T.DL) < 2e-4
    assert T.bend_place(states()["q"], 0) is None


def test_offset_correct_but_bend_off_its_mark():
    # a correct shape whose ends still lie 5.0 +/- 0.4 cm apart and parallel: the old ladder gave it 1.0
    for q in (moved(2, 0), moved(-2, 0)):  # bend A 5 mm off mark A
        s = states(q)
        off, ang = T.offset_of(s)
        assert ang < 1e-6 and abs(off - T.OFFSET) <= T.OFFSET_TOL
        assert run(s) == 0.25
    for q in (moved(0, 2), moved(0, -2)):  # bend B 5 mm off mark B
        assert run(states(q)) == 0.25
    assert run(states(moved(2, -2))) == 0.0  # both 5 mm off, the same way along the tube: offset exact, no correct bend
    assert run(states(moved(6, -6))) == 0.0  # the whole offset 15 mm along the tube
    assert run(states(moved(1, 1))) == 1.0  # both 2.5 mm off: on their marks
    assert run(states(moved(0, 0))) == 1.0


def _at(k, target):
    """A mix of the 1- and 2-hinge moves that puts bend k (0 A, 1 B) at `target` m from its mark."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        w = (lo + hi) / 2
        q = moved(2, 0, wa=w) if k == 0 else moved(0, 2, wb=w)
        p = T.bend_place(states(q)["q"], k)
        lo, hi = (w, hi) if p < target else (lo, w)
    return q, p


def test_bend_place_boundary():
    for k in (0, 1):
        q, p = _at(k, T.MARK_TOL - 0.0002)
        assert abs(p - (T.MARK_TOL - 0.0002)) < 1e-4 and run(states(q)) == 1.0
        q, p = _at(k, T.MARK_TOL + 0.0002)
        assert abs(p - (T.MARK_TOL + 0.0002)) < 1e-4 and run(states(q)) == 0.25


def test_overbend_ruins_for_good():
    T.reset_latch()
    s = states(offset(56, 45))
    _made_by(s, True)
    T.note_step(s, 0.05)
    s = states(offset())
    for _ in range(30):
        T.note_step(s, 0.05)
    assert T.score(s) == 0.0 and T._trial["why"] == "overbent"


def test_same_way_bends_ruin_for_good():
    q = offset()
    for j in T.ZONE_B:
        q[j] = -q.get(j, 0.0)
    assert run(states(q)) == 0.0 and T._trial["why"] == "same way"


def test_start_tube_lies_in_the_groove():
    f = T.tube_fk({}, T.TUBE_START[:2], T.TUBE_START[2])
    p, h = T.tube_point(f, T.X_T0)
    assert np.linalg.norm(p - np.array(M.polar(M.R_BEND, M.ALPHA)[:2])) < 1e-9
    mark, _ = T.tube_point(f, M.MARK_GAP / 2)
    assert abs(np.linalg.norm(mark - np.array(M.C)) - math.hypot(M.R_BEND, T.WRAP_45 / 2)) < 1e-9


def test_offset_formula():
    assert abs(M.MARK_GAP - T.OFFSET * 1.414) < 0.0001
    off, ang = T.offset_of(states(offset()))
    assert ang < 1e-6 and abs(off - T.OFFSET) <= T.OFFSET_TOL


def test_force_numbers():
    D_, t = M.TUBE_D, 0.030 * 0.0254
    zp = (D_ ** 3 - (D_ - 2 * t) ** 3) / 6
    eps = 0.75 * (D_ / 2) / M.R_BEND
    mp = zp * 320e6 * eps ** 0.54
    assert abs(mp - M.M_P) < 0.01
    assert abs(M.K_E - 216.0) < 2.0  # E I / 2.5 mm for 1/8 x 0.030 in copper, E 117 GPa
    assert 0.3 < math.degrees(M.M_P / M.K_E) * 5 < 3.0  # five yielded hinges spring back about 0.5 degree


def test_text_matches_code():
    j = json.loads((HERE / "task.json").read_text())
    readme = (HERE / "README.md").read_text()
    ins = j["instruction"]
    assert ins in readme and ins in (T.__doc__ or "") + open(HERE / "task.py").read()
    sc = j["scoring"]
    for phrase in ("45 +/- 4", "55 degrees", "5.0 +/- 0.4 cm", "3 degrees", "6 cm", "1 s", "10 to 18 mm", "5 degrees", "3.5 mm", "vertex", "0.25", "0.5", "0.75", "0.9", "1.0"):
        assert phrase in sc, phrase
    for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert j.get(k)
    assert len(j["title"]) <= 120 and len(j.get("context", "")) <= 1000 and len(j["setup"]) <= 4000 and len(j["fidelity"]) <= 500


def test_demo_fits_and_reaches():
    wps = T.demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    assert t * 20 < 5000 * 0.8  # the recorded demo ran 3,891 steps; the limit leaves room
    pe, re, bad = T.ik_errors(wps)
    # the fist aims 2 cm past the latch post to push it open (a force, not a place to reach): those targets may sit
    # just beyond the left arm's reach
    assert not [b for b in bad if not (b[0] == "left" and b[2][0] >= 0.29 and b[4] < 10.0)]
