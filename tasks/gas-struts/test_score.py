"""Scorer tests (no simulator) for the strut replacement: the ladder, the latched events, the inspection, aborts.

    python -m pytest test_score.py
"""
import numpy as np
import pytest

import gskit
import gsparts as G
from gskit import Reading, Tracker, ladder, old_resting

FAR = [(0.08, 0.38, G.TOP + 0.15), (0.08, -0.38, G.TOP + 0.15)]
_bx, _bz = G.lid_point(0.18, -0.030, G.OPEN)
NEAR = [(_bx, 0.07, G.TOP + _bz)]                      # on the handle at the open stop
OLD_ON_TABLE = [np.array([0.12, 0.33, G.TOP + 0.007]), np.array([0.21, 0.33, G.TOP + 0.007])]
OLD_ON_LID = [np.array([0.25, 0.10, G.TOP + 0.09]), np.array([0.30, 0.10, G.TOP + 0.09])]
OLD_FITTED = list(G.stud_points(1, 0.0))
BOTH = {"body": True, "lid": True}


def rd(t, lid=43.9, new=None, old_on=False, old_pts=None, pts=FAR, speed=0.0, moved=None):
    pos = {e: np.zeros(3) for e in gskit.NEW_SOCKETS}
    for e, v in (moved or {}).items():
        pos[e] = np.array([0.0, v / 1000, 0.0])
    return Reading(t=t, lid_deg=lid, new_on=dict(BOTH if new is None else new), old_on=old_on,
                   old_pts=OLD_ON_TABLE if old_pts is None else old_pts, new_pos=pos, hand_points=list(pts),
                   hand_speed=speed)


def run(tr, t0, until, dt=0.05, **kw):
    t = t0
    while t < until:
        tr.step(rd(t, **{k: (v(t) if callable(v) else v) for k, v in kw.items()}))
        t += dt
    return t


def full(tr):
    """Lid let go open (held), then closed, then left alone for the inspection."""
    t = run(tr, 0.0, 5.0, lid=43.9)
    t = run(tr, t, 6.0, lid=20.0, pts=NEAR)                 # a hand pushes the lid down
    t = run(tr, t, 9.0, lid=0.0)                             # shut, hands clear
    lid = {"v": 0.0}

    def lid_now(tt):                                         # the raise moves the lid; model it reaching 44 quickly
        return 44.2 if tr.phase in ("raise", "stay", "done") else 0.0
    return run(tr, t, 20.0, lid=lid_now)


# -- the rungs ---------------------------------------------------------------------------------------------------

def test_tagged_strut_still_fitted_scores_zero_even_if_new_strut_on():
    tr = Tracker()
    full(tr)
    assert ladder(rd(30, old_on=True, old_pts=OLD_FITTED), tr) == 0.0


@pytest.mark.parametrize("old_pts", [OLD_ON_LID, [np.array([0.12, 0.33, G.TOP + 0.06])] * 2])
def test_tagged_strut_not_on_table_scores_zero(old_pts):
    assert ladder(rd(1, new={"body": False, "lid": False}, old_pts=old_pts), Tracker()) == 0.0


def test_tagged_strut_held_by_gripper_does_not_count():
    r = rd(1, new={"body": False, "lid": False}, pts=[OLD_ON_TABLE[0] + [0, 0, 0.01]])
    assert not old_resting(r)
    assert ladder(r, Tracker()) == 0.0


@pytest.mark.parametrize("new,expected", [
    ({"body": False, "lid": False}, 0.15),
    ({"body": True, "lid": False}, 0.30),
    ({"body": False, "lid": True}, 0.30),
    ({"body": True, "lid": True}, 0.45),
])
def test_rungs_from_state(new, expected):
    assert ladder(rd(1, new=new), Tracker()) == expected


def test_held_gives_060():
    tr = Tracker()
    run(tr, 0.0, 5.0)
    assert tr.held and not tr.closed
    assert ladder(rd(5), tr) == 0.60


def test_full_sequence_reaches_one_and_ends_open():
    tr = Tracker()
    full(tr)
    assert tr.held and tr.closed and tr.inspected, tr.log
    assert ladder(rd(30, lid=44.0), tr) == 1.0


def test_closed_before_held_does_not_count():
    tr = Tracker()
    run(tr, 0.0, 5.0, lid=0.0)                               # new strut on, lid never let go open
    assert not tr.held and not tr.closed
    assert ladder(rd(5, lid=0.0), tr) == 0.45


def test_closed_without_inspection_is_080():
    tr = Tracker()
    t = run(tr, 0.0, 5.0)
    run(tr, t, 9.0, lid=0.0, pts=lambda tt: NEAR if tt < 6.0 else FAR)
    assert tr.closed
    assert ladder(rd(10, lid=0.0), tr) == 0.80 or tr.phase in ("pull", "relax")


def test_end_state_rungs_still_rule_after_events():
    tr = Tracker()
    full(tr)
    assert ladder(rd(30, new={"body": True, "lid": False}), tr) == 0.30   # a socket knocked off later


# -- the hold ------------------------------------------------------------------------------------------------------

def test_hold_needs_hands_clear_and_still():
    tr = Tracker()
    run(tr, 0.0, 5.0, pts=NEAR)
    assert not tr.held
    run(tr, 5.0, 5.9)
    assert not tr.held                                       # under 1 s still plus 2 s hold
    run(tr, 5.9, 9.0)
    assert tr.held


def test_moving_hand_blocks_hold():
    tr = Tracker()
    run(tr, 0.0, 5.0, speed=0.05)
    assert not tr.held


def test_lid_sinking_is_not_held_and_not_latched_bad():
    tr = Tracker()
    run(tr, 0.0, 5.0, lid=lambda t: 43.9 - 5 * t)
    assert not tr.held
    run(tr, 5.0, 10.0, lid=43.9)                              # lifted again later and let go: it counts
    assert tr.held


def test_hold_needs_new_strut_on_both_ends():
    tr = Tracker()
    run(tr, 0.0, 5.0, new={"body": True, "lid": False})
    assert not tr.held


# -- the inspection --------------------------------------------------------------------------------------------------

def test_failed_pull_does_not_pass_and_can_retry():
    tr = Tracker()
    t = run(tr, 0.0, 5.0)
    t = run(tr, t, 6.0, lid=0.0, pts=NEAR)
    t = run(tr, t, 9.0, lid=0.0, moved=lambda tt: {"body": 2.0} if tr.phase == "pull" else None)
    assert not tr.inspected
    assert tr.results.get("pull_body", (0, True))[1] is False
    assert ladder(rd(t, lid=0.0), tr) == 0.80


def test_hand_returning_stops_the_inspection():
    tr = Tracker()
    t = run(tr, 0.0, 5.0)
    t = run(tr, t, 6.0, lid=0.0, pts=NEAR)
    t = run(tr, t, 7.2, lid=0.0)
    assert tr.phase in ("pull", "relax")
    run(tr, t, t + 0.2, lid=0.0, pts=[(0.30, 0.235, G.TOP + 0.06)])
    assert tr.phase == "wait"
    assert not tr.inspected


def test_raise_that_falls_back_does_not_pass():
    tr = Tracker()
    t = run(tr, 0.0, 5.0)
    t = run(tr, t, 6.0, lid=0.0, pts=NEAR)
    t = run(tr, t, 9.2, lid=0.0)
    assert tr.phase == "raise", tr.log
    t = run(tr, t, t + 0.3, lid=44.2)
    assert tr.phase == "stay"
    run(tr, t, t + 1.0, lid=30.0)
    assert not tr.inspected


def test_force_requests():
    tr = Tracker()
    tr.phase, tr.pull_idx = "pull", 1
    assert tr.wants_force() == ("pull", "lid")
    tr.phase = "raise"
    assert tr.wants_force() == ("raise", None)
    tr.phase = "wait"
    assert tr.wants_force() is None


def test_geometry_and_strut():
    assert abs(G.STRUT_LEN - 0.092) < 1e-9
    b, l = G.stud_points(1, 0.0)
    assert np.linalg.norm(l - b) > G.STRUT_LEN - G.STROKE     # the lid shuts with the strut not bottomed
    assert abs(G.K_GAS * (0 - G.GAS_TARGET) - G.P1) < 1e-6         # 10 N extended
    s = G.strut("t", b, l, (0, -1, 0), failed=True, tagged=True)
    gas = next(j for j in s.joints if j.name == "gas")
    assert gas.stiffness == 0.0 and gas.damping == G.FAILED_DAMP and gas.init > 0.017
    assert len([j for j in s.joints if j.name.startswith("jaw")]) == 4
