"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

os.environ["ROD_LOT"] = "B"
import task as t  # noqa: E402

M = t.M
PARTED_Y = M.JOINTS[t.PARTED]
Q = np.array((0.0, 0.0, 0.0, 1.0))


def state(pos, q=Q):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=np.array(q, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p, resting=lambda: True)


def world(plate_y=PARTED_Y, handle=0.0, gate=0.0, hand=(0.18, 0.32, t.TOP + 0.18), plate_x=t.X0, q=Q, z=None):
    return {"splice_plate": state((plate_x, plate_y, t.TOP + M.BAR_Z1 if z is None else z), q),
            "slides": state((handle, gate, 0.0)),
            "left_tcp": state(hand), "right_tcp": state((0.18, -0.32, t.TOP + 0.18))}


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def prove(**kw):
    t.note_step(world(plate_y=PARTED_Y - 0.034, handle=0.04, gate=0.028, **kw), 0.05)


def settle(w, s=1.1):
    for _ in range(round(s / 0.05)):
        t.note_step(w, 0.05)


def test_ladder():
    w = world()
    assert t.score(world(plate_y=0.3)) == 0.0
    assert t.score(w) == 0.4
    prove()
    assert t.score(world(plate_y=PARTED_Y - 0.034, handle=0.04, gate=0.028)) == 0.7
    assert t.score(w) == 0.7  # home but not yet settled
    settle(w)
    assert t.score(w) == 1.0


def test_sound_joint_scores_zero_even_if_it_moves():
    for y in M.JOINTS:
        if y != PARTED_Y:
            w = world(plate_y=y)
            t.note_step(world(plate_y=y, handle=0.04, gate=0.028), 0.05)
            settle(w)
            assert t.score(w) == 0.0


def test_no_proof_without_the_handle_or_the_gate():
    for h, g in ((0.03, 0.028), (0.04, 0.01)):
        t.note_step(world(plate_y=PARTED_Y - 0.034, handle=h, gate=g), 0.05)
        assert not t._trial["proof"]


def test_a_hand_at_the_gate_is_no_proof():
    t.note_step(world(plate_y=PARTED_Y - 0.034, handle=0.04, gate=0.028, hand=(t.X0, M.GATE["y"] - 0.028, t.TOP + 0.09)), 0.05)
    assert not t._trial["proof"]
    t.note_step(world(plate_y=PARTED_Y - 0.034, handle=0.04, gate=0.028, hand=(t.X0, 0.1, t.TOP + 0.09)), 0.05)
    assert not t._trial["proof"]


def test_the_proof_stays_after_the_handle_comes_back():
    prove()
    w = world(plate_y=PARTED_Y - 0.006)
    settle(w)
    assert t.score(w) == 1.0


def test_plate_may_ride_with_the_bars():
    assert t.plate_on(world(plate_y=PARTED_Y - 0.034, handle=0.04, gate=0.028)) == t.PARTED


def test_a_plate_wedged_outside_the_studs_doesnt_count():
    assert t.plate_on(world(plate_y=PARTED_Y - 0.02, handle=0.04, gate=0.0)) is None


def test_a_held_plate_scores_zero():
    assert t.score(world(hand=(t.X0, PARTED_Y, t.TOP + M.BAR_Z1 + 0.034))) == 0.0


def test_plate_tilted_or_off_the_line():
    assert t.plate_on(world(plate_x=t.X0 + 0.02)) is None
    assert t.plate_on(world(z=t.TOP + M.BAR_Z1 + 0.02)) is None
    s = np.sin(np.radians(20) / 2)
    assert t.plate_on(world(q=(s, 0.0, 0.0, np.cos(np.radians(20) / 2)))) is None
    assert t.plate_on(world(q=(0.0, 0.0, s, np.cos(np.radians(20) / 2)))) is None


def test_plate_reversed_end_for_end_still_seats():
    assert t.plate_on(world(q=(0.0, 0.0, 1.0, 0.0))) == t.PARTED


def test_bars_not_home_or_hands_near_hold_it_at_point_seven():
    prove()
    w = world(plate_y=PARTED_Y - 0.008, handle=0.01, gate=0.01)
    settle(w)
    assert t.score(w) == 0.7
    t._trial["still"] = 0.0
    w = world(hand=(t.X0, PARTED_Y + 0.07, t.TOP + 0.09))  # near the line, not holding the plate
    settle(w)
    assert t.score(w) == 0.7


def test_reset_forgets():
    prove()
    t.reset_latch()
    assert t.score(world()) == 0.4


def test_every_lot_plans_and_reaches():
    for lot in t.LOTS:
        wps = t.demo(lot)
        assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 2400
        assert not t.ik_errors(wps)[2]


def test_slots_fit_the_studs():
    assert M.SLOT_Y0 < M.STUD_OFF - M.STUD / 2 and M.STUD_OFF + M.STUD / 2 < M.SLOT_Y1 and M.STUD / 2 < M.SLOT_HALF_X
