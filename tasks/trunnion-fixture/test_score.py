"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t

Q = np.array((0.0, 0.0, 0.0, 1.0))
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}
FLIP = np.array((0.0, 1.0, 0.0, 0.0))  # 180 degrees about y: the cradle turned over
OVER = math.pi


def state(pos, rate=(0.0, 0.0, 0.0), q=Q, ang=(0.0, 0.0, 0.0)):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=np.array(q, float), lin_vel=np.array(rate, float), ang_vel=np.array(ang, float), low=p, high=p)


def roty(th):
    return np.array((0.0, math.sin(th / 2), 0.0, math.cos(th / 2)))


def in_cradle(th, u=0.0, v=0.0, w=None, q=None):
    """The housing's state, sitting in the cradle at this roll, at these offsets in the cradle's axes."""
    w = (t.M.LEDGE["w1"] if th < math.pi / 2 else t.SEAT_W[1] - 0.002) if w is None else w
    c, s = math.cos(th), math.sin(th)
    return state((t.AX + u * c + w * s, v, t.AXLE - u * s + w * c), q=roty(th) if q is None else q)


def pad_state(h, x=0.0, y=0.0, z=None):
    """The pad on the housing's recess floor, offset in the housing's own frame."""
    R = t._rot(h.quat_xyzw)
    z = t.PART["depth"] if z is None else z
    return state(h.pos + R @ np.array((x, y, z)))


def world(part=True, bar=True, roll=OVER, bolt=True, pad=True, **over):
    w = {a: state(p) for a, p in PARK.items()}
    w["fixture"] = state((roll, t.M.BAR["travel"] if bar else 0.0, t.M.BOLT["travel"] if bolt else 0.0))
    w["housing"] = in_cradle(roll) if part else state((*t.PART_START, t.TOP))
    w["pad"] = pad_state(w["housing"]) if pad else state((*t.PAD_START, t.TOP))
    for k, v in over.items():
        w[k] = v if isinstance(v, NS) else state(v)
    return w


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def run(w, n=22):
    for _ in range(n):
        t.note_step(w, 0.05)
    return t.score(w)


def test_start_scores_zero():
    assert run(world(False, False, 0.0, False, False)) == 0.0


def test_full_marks():
    assert run(world()) == 1.0


def test_needs_a_second_of_rest():
    assert run(world(), n=5) == 0.9


def test_part_in_level():
    assert run(world(True, False, 0.0, False, False)) == 0.2


def test_bar_closed_level():
    assert run(world(True, True, 0.0, False, False)) == 0.4


def test_turned_over():
    assert run(world(True, True, OVER, False, False)) == 0.6


def test_bolt_home():
    assert run(world(True, True, OVER, True, False)) == 0.8


def test_part_lost_is_zero_for_good():
    w = world(True, True, OVER, False, False)
    t.note_step(w, 0.05)
    t.note_step(world(False, True, OVER, False, False), 0.05)
    assert t._trial["bad"] and run(world()) == 0.0


def test_turning_with_the_bar_open_is_zero_for_good():
    t.note_step(world(True, False, math.radians(40), False, False), 0.05)
    assert t._trial["bad"] and run(world()) == 0.0


def test_turning_empty_is_zero_for_good():
    t.note_step(world(False, True, math.radians(40), False, False), 0.05)
    assert t._trial["bad"] and run(world()) == 0.0


def test_a_small_lift_without_the_bar_is_not_turning():
    t.note_step(world(True, False, math.radians(20), False, False), 0.05)
    assert not t._trial["bad"]


def test_pad_before_the_bolt_is_zero_for_good():
    t.note_step(world(True, True, OVER, False, True), 0.05)
    assert t._trial["bad"] and run(world()) == 0.0


def test_bolt_after_the_pad_can_not_recover():
    run(world(True, True, OVER, False, True), n=1)
    assert run(world()) == 0.0


def test_opening_the_bar_once_turned_is_zero_for_good():
    t.note_step(world(True, False, OVER, True, False), 0.05)
    assert t._trial["bad"]


def test_housing_in_upside_down_earns_nothing():
    w = world(True, False, 0.0, False, False)
    w["housing"] = in_cradle(0.0, q=FLIP)
    assert t.part_tilt(w) > 3.0 and run(w) == 0.0


def test_upside_down_load_can_not_reach_full_marks():
    w = world()
    w["housing"] = in_cradle(OVER, q=Q)  # flipped by hand beforehand: its recess faces the bar
    assert run(w) == 0.0


def test_housing_not_in_the_pocket_earns_nothing():
    w = world(True, False, 0.0, False, False)
    w["housing"] = in_cradle(0.0, u=0.03)
    assert run(w) == 0.0


def test_housing_on_the_table_with_pad_in_it_earns_nothing():
    w = world(False, False, 0.0, False, False)
    w["pad"] = pad_state(w["housing"])
    assert t.score(w) == 0.0


def test_held_housing_does_not_count():
    w = world(True, False, 0.0, False, False)
    h = w["housing"].pos
    w["left_tcp"] = state((h[0], h[1], h[2] + t.PART_GZ))
    assert run(w) == 0.0


def test_finger_at_the_housings_end_does_not_count():
    w = world(True, False, 0.0, False, False)
    h = w["housing"].pos
    w["left_tcp"] = state((h[0], h[1] + t.PART["half_y"] + 0.047, h[2] + t.PART_GZ))  # an open hand with one finger on the end face
    assert run(w) == 0.0


def test_held_pad_does_not_count():
    w = world()
    w["left_tcp"] = state(w["pad"].pos + (0.0, 0.0, t.PAD_GZ))
    assert run(w) == 0.8


def test_hand_on_the_knob_stops_short():
    w = world()
    w["right_tcp"] = state(t.knob_at(OVER))
    assert run(w) == 0.9


def test_hand_on_the_bolt_stops_short():
    w = world()
    w["left_tcp"] = state((t.AX, t.M.BOLT["end"] - t.M.BOLT["travel"], t.TOP + t.M.BOLT["lug_top"]))
    assert run(w) == 0.9


def test_moving_parts_stop_short():
    w = world()
    w["pad"].ang_vel = np.array((1.0, 0.0, 0.0))
    assert run(w) == 0.9


def test_bolt_pulled_back_loses_credit():
    assert run(world(True, True, OVER, False, False)) == 0.6


def test_short_of_turned_over_loses_credit():
    assert run(world(True, True, OVER - math.radians(10), False, False)) == 0.4


def test_part_off_the_table_is_zero_for_good():
    t.note_step({**world(), "pad": state((0.2, 0.9, 0.05))}, 0.05)
    assert t._trial["bad"] and run(world()) == 0.0


def test_pad_offset_is_judged_in_the_housings_frame():
    w = world()
    w["housing"] = in_cradle(OVER, q=FLIP)
    w["pad"] = pad_state(w["housing"], x=0.006)
    assert t.pad_in(w)
    w["pad"] = pad_state(w["housing"], x=0.012)
    assert not t.pad_in(w)
    # the same offset in a housing yawed in the cradle's play
    th = math.radians(5)
    q = np.array((math.sin(th / 2) * 0.0, math.cos(th / 2), math.sin(th / 2), 0.0))
    w["housing"].quat_xyzw = q
    w["pad"] = pad_state(w["housing"], y=0.006)
    assert t.pad_in(w)


def test_pad_lying_on_its_side_does_not_count():
    w = world()
    w["pad"].quat_xyzw = np.array((math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)))
    assert not t.pad_in(w)


def test_masses_suit_the_material():
    p = t.PART
    part = (4 * p["half_x"] * p["half_y"] * p["h"] - 4 * p["rec_x"] * p["rec_y"] * p["depth"]) * 2700
    pad = 8 * t.PAD["half_x"] * t.PAD["half_y"] * t.PAD["h"] / 2 * 1410
    assert abs(part - p["mass"]) < 0.01 and abs(pad - t.PAD["mass"]) < 0.005


def test_plan_fits_and_reaches():
    wps = t.demo()
    assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 3000
    assert not t.ik_errors(wps)[2]
