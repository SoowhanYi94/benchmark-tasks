"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

os.environ["FLANGE_LOT"] = "A"
import task as t  # noqa: E402

M = t.M
PARK = (0.12, 0.33, t.TOP + 0.18)
T = t.TARGET
Q = np.array((0.0, 0.0, 0.0, 1.0))


def state(pos, rate=0.0):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=Q, lin_vel=np.array((rate, 0.0, 0.0)), ang_vel=np.zeros(3), low=p, high=p)


def world(turn=T, ram=0.0, hand=PARK, rate=0.0):
    return {"press": state((turn, ram, t.reading(turn)), rate), "handle": state(t.handle_pos(turn)),
            "left_tcp": state(hand), "right_tcp": state((PARK[0], -PARK[1], PARK[2]))}


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def settle(w):
    for _ in range(22):
        t.note_step(w, 0.05)
    return t.score(w)


def pressed_right():
    t.note_step(world(ram=0.03), 0.05)


def test_ladder():
    assert t.score(world(turn=0.0)) == 0.0
    assert t.score(world()) == 0.5
    pressed_right()
    assert t.score(world()) == 0.85
    assert settle(world()) == 1.0


def test_a_press_anywhere_else_is_zero_for_good():
    t.note_step(world(turn=0.0, ram=0.03), 0.05)
    assert settle(world()) == 0.0
    pressed_right()
    assert settle(world()) == 0.0


def test_a_press_at_every_angle_is_zero():
    for d in range(0, 360, 30):
        t.note_step(world(turn=math.radians(d), ram=0.03), 0.05)
    assert settle(world()) == 0.0


def test_a_partial_press_counts_as_nothing():
    t.note_step(world(turn=0.0, ram=0.015), 0.05)
    assert not t._trial["bad"]


def test_the_window():
    assert t.score(world(turn=T + math.radians(14))) == 0.5
    assert t.score(world(turn=T + math.radians(16))) == 0.0


def test_turned_away_after_the_press_is_zero():
    pressed_right()
    assert settle(world(turn=T + math.radians(40))) == 0.0


def test_a_hand_on_the_handle_or_ram_is_zero():
    assert t.score(world(hand=tuple(t.handle_pos(T)))) == 0.0
    assert t.score(world(hand=tuple(t._ram_pos()))) == 0.0


def test_ram_still_down_or_a_hand_near_holds_it_at_point_five():
    pressed_right()
    assert t.score(world(ram=0.01)) == 0.5
    near = t.handle_pos(T) + (0.07, 0.0, 0.0)
    assert t.score(world(hand=tuple(near))) == 0.5


def test_a_flange_still_turning_holds_it_at_point_eight_five():
    pressed_right()
    assert settle(world(rate=math.radians(10))) == 0.85


def test_the_pointer_peaks_at_the_target_and_is_flat_past_sixty():
    assert t.reading(T) == pytest.approx(M.DIAL["travel"])
    assert t.reading(T + math.radians(30)) == pytest.approx(M.DIAL["travel"] / 2)
    assert t.reading(T + math.radians(61)) == 0.0


def test_the_start_reads_zero_for_every_lot():
    for deg in t.LOTS.values():
        assert t._off(0.0, math.radians(deg)) >= t.PEAK_WIDTH - 1e-9


def test_the_ram_stops_above_the_flange():
    assert M.RAM["z0"] - M.RAM["travel"] >= M.Z_FLANGE1 + 0.01 - 1e-9


def test_reset_forgets():
    t.note_step(world(turn=0.0, ram=0.03), 0.05)
    t.reset_latch()
    assert t.score(world()) == 0.5


def test_every_lot_plans_and_reaches():
    for lot in t.LOTS:
        wps = t.demo(lot)
        assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 2400
        assert not t.ik_errors(wps)[2]
