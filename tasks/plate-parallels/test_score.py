"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

os.environ["PARALLELS_LOT"] = "A"
import task as t  # noqa: E402

Q = np.array((0.0, 0.0, 0.0, 1.0))
TOPZ = t.BASE_Z + t.PAR["z"]


def state(pos, q=Q, half=None, rest=True):
    p = np.array(pos, float)
    lo, hi = p.copy(), p.copy()
    if half:
        lo[2], hi[2] = p[2] - half, p[2] + half
    return NS(pos=p, quat_xyzw=np.array(q, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=lo, high=hi, resting=lambda: rest)


def world(a=0.05, b=-0.05, pins=(0.12, -0.12), pin_z=TOPZ + t.T, hand=(0.18, 0.32, t.TOP + 0.18), plate_q=Q, plate_pos=None):
    z = t.BASE_Z + t.PAR["z"] / 2
    return {"plate": state(plate_pos or (t.X0, 0.0, TOPZ), plate_q),
            "parallel_a": state((t.X0, a, z), half=t.PAR["z"] / 2), "parallel_b": state((t.X0, b, z), half=t.PAR["z"] / 2),
            "pin_a": state((t.X0, pins[0], pin_z)), "pin_b": state((t.X0, pins[1], pin_z)),
            "left_tcp": state(hand), "right_tcp": state((0.18, -0.32, t.TOP + 0.18))}


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def settled(w):
    for _ in range(22):
        t.note_step(w, 0.05)
    return t.score(w)


def test_full_score():
    assert settled(world()) == 1.0


def test_not_settled_or_still_held_is_zero():
    assert t.score(world()) == 0.0


def test_a_hand_near_holds_it_down():
    assert settled(world(hand=(t.X0, 0.0, TOPZ + 0.09))) == 0.0


def test_one_pin_is_point_two():
    w = world()
    w["pin_b"] = state((0.15, -0.28, t.TOP + 0.045))
    assert settled(w) == 0.2


def test_a_pin_stopped_by_a_parallel_doesnt_count():
    assert settled(world(pin_z=TOPZ + t.T + 0.03)) == 0.0


def test_pin_in_a_wrong_place():
    assert settled(world(pins=(0.06, -0.12))) == 0.2


def test_a_parallel_under_a_hole_scores_zero_even_if_the_pin_gets_past():
    assert t.supports(world(a=0.09, b=-0.09)) == 0
    assert settled(world(a=0.09, b=-0.09, pins=(0.13, -0.13))) == 0.0


def test_a_parallel_turned_a_quarter_doesnt_count():
    w = world()
    z = t.BASE_Z + t.PAR["z"] / 2
    w["parallel_a"] = state((t.X0, 0.05, z), (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)), half=t.PAR["z"] / 2)
    assert t.supports(w) == 1


def test_a_parallel_off_the_mat_doesnt_count():
    w = world()
    w["parallel_a"] = state((t.X0, 0.05, t.TOP + t.PAR["z"] / 2), half=t.PAR["z"] / 2)
    assert t.supports(w) == 1


def test_side_by_side_in_the_middle_fails_the_span():
    assert t.supports(world(a=0.03, b=-0.03)) == 0
    assert settled(world(a=0.03, b=-0.03)) == 0.0


def test_one_parallel_not_under_or_tipped():
    assert settled(world(a=0.14)) == 0.0
    w = world()
    w["plate"] = state((t.X0, 0.0, TOPZ), (np.sin(np.radians(2)), 0.0, 0.0, np.cos(np.radians(2))))
    assert settled(w) == 0.0


def test_plate_off_the_mat():
    assert settled(world(plate_pos=(t.X0 + 0.2, 0.0, TOPZ))) == 0.0


def test_no_single_layout_clears_every_lot():
    ys = np.arange(-0.125, 0.126, 0.001)
    clear = lambda y, hs: all(abs(y - h) >= (t.PAR["y"] + t.HOLE) / 2 for h in hs)  # noqa: E731
    for a in ys:
        for b in ys:
            assert not (a - b >= t.MIN_SPAN and all(clear(a, h) and clear(b, h) for h in t.LOTS.values()))


def test_each_lots_places_clear_its_holes_and_span():
    for k, (a, b) in t.PLACES.items():
        assert a - b >= t.MIN_SPAN
        for h in t.LOTS[k]:
            assert abs(a - h) >= (t.PAR["y"] + t.HOLE) / 2 and abs(b - h) >= (t.PAR["y"] + t.HOLE) / 2


def test_every_lot_plans_and_reaches():
    for lot in t.LOTS:
        wps = t.demo(lot)
        assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 3000
        assert not t.ik_errors(wps)[2]
