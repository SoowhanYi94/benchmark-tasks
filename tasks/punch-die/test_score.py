"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

os.environ["PUNCH_LOT"] = "A"
import task as t  # noqa: E402

Q = np.array((0.0, 0.0, 0.0, 1.0))
PARK = {"left_tcp": (0.12, 0.33, t.PARK[2]), "right_tcp": (0.12, -0.33, t.PARK[2])}
WANT, WRONG = "die_plus", "die_mid"


def state(pos, rate=0.0, q=Q):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=np.array(q, float), lin_vel=np.array((rate, 0.0, 0.0)), ang_vel=np.zeros(3), low=p, high=p)


def world(press=None, old="rack", ram=0.0, **moved):
    """press: the die in the press, home. Others stay in their row."""
    w = {a: state(p) for a, p in PARK.items()}
    w["ram"] = state((ram, 0.0, 0.0))
    w["guard"] = state(t.GUARD_HOME)
    w["die_old"] = state((t.RACK[0] - t.PLATE_X, t.RACK[1], t.TOP + t.RACK_T) if old == "rack" else (t.CX, t.CY, t.BED_Z))
    for n in t.PLACES:
        w[n] = state((t.CX, t.CY, t.BED_Z) if n == press else (*t.START[n], t.TOP))
    for k, v in moved.items():
        w[k] = v if isinstance(v, NS) else state(v)
    return w


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def run(w, n=22):
    for _ in range(n):
        t.note_step(w, 0.05)
    return t.score(w)


def stroke_right():
    t.note_step(world(WANT, ram=0.03), 0.05)


def test_start_scores_zero():
    assert run(world(old="press")) == 0.0


def test_old_die_racked():
    assert run(world()) == 0.1


def test_old_die_racked_still_held_is_zero():
    assert run(world(left_tcp=(t.RACK[0] - t.PLATE_X + t.GX, t.RACK[1], t.TOP + 0.04))) == 0.0


def test_right_die_home():
    assert run(world(WANT)) == 0.5


def test_right_die_home_still_held():
    assert run(world(WANT, right_tcp=(t.CX + t.GX, t.CY, t.BED_Z + t.GZ))) == 0.1


def test_wrong_die_home_unstroked():
    assert run(world(WRONG)) == 0.1


def test_full_marks():
    stroke_right()
    assert run(world(WANT)) == 1.0


def test_needs_a_second_of_rest():
    stroke_right()
    assert run(world(WANT), n=5) == 0.8


def test_hands_on_pad_stops_short():
    stroke_right()
    pad = (t.CX, 0.12, t.TOP + t.M.PAD["z1"] + 0.016)
    assert run(world(WANT, left_tcp=pad)) == 0.5


def test_ram_held_down_stops_short():
    stroke_right()
    assert run(world(WANT, ram=0.02)) == 0.5


def test_empty_stroke_with_the_guard_on_is_free():
    assert run(world(ram=0.03), n=1) == 0.1
    assert run(world()) == 0.1


def aside():
    return (*t.GUARD_ASIDE, t.TOP)


def test_guard_off_scores_short_of_full():
    assert run(world(WANT, guard=aside())) == 0.5


def test_stroke_with_guard_off_is_zero_for_good():
    t.note_step(world(WANT, ram=0.03, guard=aside()), 0.05)
    assert run(world(WANT)) == 0.0


def test_stroke_then_guard_taken_off_stops_short():
    stroke_right()
    assert run(world(WANT, guard=aside())) == 0.5


def test_guard_still_held_at_the_stroke_does_not_count():
    grip = (t.GUARD_HOME[0], t.GUARD_HOME[1], t.GUARD_HOME[2] + 0.03)
    t.note_step(world(WANT, ram=0.03, left_tcp=grip), 0.05)
    assert t._trial["bad"]


def test_pressing_part_way_with_guard_off_is_allowed():
    t.note_step(world(ram=0.02, guard=aside()), 0.05)
    assert not t._trial["bad"]


@pytest.mark.parametrize("die", ["die_mid", "die_minus"])
def test_stroke_on_a_wrong_die_is_zero_for_good(die):
    t.note_step(world(die, ram=0.03), 0.05)
    assert t._trial["bad"]
    assert run(world(WANT)) == 0.0


def test_stroke_on_old_die_is_zero_for_good():
    t.note_step(world(old="press", ram=0.03), 0.05)
    assert run(world(WANT)) == 0.0


def test_partial_insertion_crash():
    t.note_step(world(ram=0.03, die_mid=(t.CX - 0.03, t.CY, t.BED_Z)), 0.05)
    assert t._trial["bad"]


def test_right_die_part_way_in_under_a_lowered_punch_crashes():
    t.note_step(world(ram=0.03, die_plus=(t.CX - 0.03, t.CY, t.BED_Z)), 0.05)
    assert t._trial["bad"]


def test_light_contact_above_the_die_is_no_crash():
    t.note_step(world(WRONG, ram=0.02), 0.05)
    assert not t._trial["bad"]


@pytest.mark.parametrize("dy", [-0.004, 0.004])
def test_right_die_clears_at_the_edge_of_its_play(dy):
    t.note_step(world(ram=0.03, die_plus=(t.CX, t.CY + dy, t.BED_Z)), 0.05)
    assert not t._trial["bad"]


@pytest.mark.parametrize("dy", [-0.004, 0.0, 0.004])
@pytest.mark.parametrize("die", ["die_mid", "die_minus"])
def test_wrong_die_hits_anywhere_in_its_play(die, dy):
    t.note_step(world(ram=0.03, **{die: (t.CX, t.CY + dy, t.BED_Z)}), 0.05)
    assert t._trial["bad"]


def test_stroke_with_die_held_does_not_count():
    t.note_step(world(WANT, ram=0.03, right_tcp=(t.CX + t.GX, t.CY, t.BED_Z + t.GZ)), 0.05)
    assert run(world(WANT)) == 0.5


def test_stroke_then_die_pulled_loses_it():
    stroke_right()
    assert run(world(die_plus=(t.STAGE_X, t.CY, t.BED_Z))) == 0.1


def test_die_turned_round_does_not_count():
    q = (0.0, 0.0, 1.0, 0.0)  # turned, the -y pocket lies where the +y one should
    assert run(world(die_minus=state((t.CX, t.CY, t.BED_Z), q=q))) == 0.1


def test_die_off_the_table_is_zero():
    stroke_right()
    assert run(world(WANT, die_mid=(0.2, 0.9, 0.05))) == 0.0


def test_lots_are_balanced_and_each_die_differs():
    assert set(t.LOTS.values()) == set(t.PLACES)
    assert len(set(t.PLACES.values())) == 3


def test_wrong_die_overlaps_the_punch_by_a_margin():
    pitch = min(abs(a - b) for a in t.PLACES.values() for b in t.PLACES.values() if a != b)
    assert pitch - t.POCKET - t.M.PUNCH["half"] - 0.004 >= 0.01  # still 1 cm of metal under it at the far edge of its play


def test_guard_hides_every_punch_place_from_above():
    assert t.GUARD["hy"] >= max(abs(y) for y in t.PLACES.values()) + t.M.PUNCH["half"]
    assert t.GUARD["hx"] >= t.M.PUNCH["half"] + t.GUARD_XY  # however it sits within its tolerance


def test_masses_suit_the_material():
    al = 2700
    plate = (t.HX1 - t.HX0) * 2 * t.HY * t.T
    assert abs((plate - (2 * t.POCKET) ** 2 * t.T) * al + 0.01 - t.DIE_MASS) < 0.015
    assert abs(plate * al + 0.01 - t.OLD_MASS) < 0.015


@pytest.mark.parametrize("ram", [0.0, 0.03])
def test_a_die_carried_over_the_punch_is_no_crash(ram):
    t.note_step(world(ram=ram, die_mid=(t.CX, t.CY, t.TOP + 0.12)), 0.05)
    assert not t._trial["bad"]


def test_a_die_lowered_onto_the_punch_is_a_crash():
    t.note_step(world(die_mid=(t.CX, t.CY, t.TOP + t.M.PUNCH["tip"] - 0.002)), 0.05)
    assert t._trial["bad"]


def test_rack_is_judged_at_the_plate_centre():
    x = t.RACK[0] - t.PLATE_X + 0.05
    assert run(world(die_old=(x, t.RACK[1], t.TOP + t.RACK_T))) == 0.0
