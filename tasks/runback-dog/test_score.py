"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

os.environ["DOG_LOT"] = "B"
import task as t  # noqa: E402

M = t.M
Q = np.array((0.0, 0.0, 0.0, 1.0))
TURNED = np.array((0.0, 0.0, 1.0, 0.0))


def state(pos, q=Q, rest=True):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=np.array(q, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p, resting=lambda: rest)


def world(carrier=M.TRAVEL, hand=(0.12, 0.32, t.TOP + 0.18), **dogs):
    w = {n: state((t.X0, t.STATION[n], t.SEAT_Z)) for n in t.DOGS}
    w.update(dogs)
    w["carrier"] = state((carrier, 0.0, 0.0))
    w["left_tcp"], w["right_tcp"] = state(hand), state((0.12, -0.32, t.TOP + 0.18))
    return w


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def settled(w):
    for _ in range(22):
        t.note_step(w, 0.05)
    return t.score(w)


def test_full_score():
    assert settled(world()) == 1.0


def test_all_forward_but_carrier_short():
    assert settled(world(carrier=0.15)) == 0.5
    t.reset_latch()
    assert t.score(world()) == 0.5  # at the stop, not yet settled


def test_a_backwards_dog_scores_zero_wherever_the_carrier_is():
    assert settled(world(dog_b=state((t.X0, 0.0, t.SEAT_Z), TURNED))) == 0.0


def test_a_dog_taken_off_scores_zero():
    assert settled(world(dog_a=state((0.15, -0.12, t.TOP)))) == 0.0


def test_a_dog_on_the_wrong_studs_scores_zero():
    assert settled(world(dog_a=state((t.X0, 0.0, t.SEAT_Z)))) == 0.0


def test_identical_dogs_swapped_still_count():
    w = world(dog_a=state((t.X0, t.STATION["dog_b"], t.SEAT_Z)), dog_b=state((t.X0, t.STATION["dog_a"], t.SEAT_Z)))
    assert settled(w) == 1.0


def test_gripping_a_caps_end_is_still_held():
    assert settled(world(hand=(t.X0 + 0.04, 0.0, t.SEAT_Z + 0.055))) == 0.0


def test_play_on_the_studs_cant_jam_the_carrier():
    play = M.BACK_INNER - (M.STUD_X + M.STUD / 2)
    assert play + M.DOG_HALF_Y * math.sin(t.YAW_TOL) < M.LEG_INNER - M.CARRIER["half_x"]


def test_a_dog_nudged_to_the_end_of_its_slot_still_counts():
    assert settled(world(dog_a=state((t.X0, t.STATION["dog_a"] + 0.01, t.SEAT_Z)))) == 1.0


def test_a_dog_turned_a_quarter_or_tipped():
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert not t.forward(world(dog_a=state((t.X0, t.STATION["dog_a"], t.SEAT_Z), q)), "dog_a")
    s = math.sin(math.radians(8) / 2)
    assert not t.forward(world(dog_a=state((t.X0, t.STATION["dog_a"], t.SEAT_Z), (s, 0.0, 0.0, math.cos(math.radians(8) / 2)))), "dog_a")


def test_a_hand_near_holds_it_down():
    assert settled(world(hand=(t.X0 + 0.12, 0.0, t.TOP + 0.08))) == 0.5


def test_a_held_dog_scores_zero():
    assert settled(world(hand=(t.X0, 0.0, t.SEAT_Z + M.GRIP["z1"] - 0.006))) == 0.0


def test_handle_stops_short_of_the_first_dog_wherever_it_sits():
    assert M.HANDLE["y"] + M.TRAVEL + M.HANDLE["half_y"] < M.STATIONS[0] - M.DOG_HALF_Y - (M.SLOT_HALF_Y - M.STUD / 2)


def test_last_dog_at_the_end_of_its_slot_cant_stop_the_carrier():
    assert M.STATIONS[-1] + (M.SLOT_HALF_Y - M.STUD / 2) + M.TOOTH_Y + M.PIN_HALF <= M.STOP


def test_the_pin_clears_the_rail_pressed_past_the_tooth():
    pressed = M.PIN_TOP - (M.TIP_Z + M.RAIL_Z)
    assert pressed < M.PIN_DROP and M.PIN_BOTTOM - pressed > M.RAIL_Z


def test_pin_passes_under_the_cap_and_is_caught_by_the_tooth():
    pin_top = M.PIN_TOP - M.RAIL_Z
    assert M.TIP_Z < pin_top < M.CAP_Z0
    assert pin_top - M.PIN_DROP < M.TIP_Z


def test_every_lot_plans_and_reaches():
    for lot in "ABC":
        wps = t.demo(lot)
        assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 2400
        assert not t.ik_errors(wps)[2]
