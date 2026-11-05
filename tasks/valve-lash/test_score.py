"""The score must credit nothing unless the screw was set on the blade, then the lash within 2.0 +- 0.3 mm, then the locknut seated, then the blade out and the hands
clear, and nothing for a lash outside the tolerance, a nut seated on a wrong lash, a blade left in the gap (however it
is turned), a hand still at the screw, or a screw or nut forced off its axis."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t
from valve_lash import fixture as F

UP = np.array((0.0, 0.0, 0.0, 1.0))


def state(p, q=UP):
    p = np.asarray(p, float)
    return SimpleNamespace(pos=p, quat_xyzw=np.asarray(q, float), low=p, high=p)


def yaw_q(a):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


HOME_Y = t.Y0 - 0.008 - F.BLADE_Y[0]  # gauge origin with the blade in the gap
IN_GAP = state((t.X0, HOME_Y, t.TOP + F.STEM_TOP))
ON_TABLE = state((*t.GAUGE_START, t.TOP))
FAR = ((0.16, 0.30, t.TOP + 0.2), (0.16, -0.30, t.TOP + 0.2))


def world(lash=0.002, nut=0.0, gauge=ON_TABLE, hands=FAR, screw_xy=None, screw_q=UP, nut_q=UP, used=True):
    """lash: the screw tip above the stem; nut: the nut's underside above the rocker (m); used: the embodiment's
    blade_used latch."""
    sxy = screw_xy or (t.X0, t.Y0)
    return {"screw": state((*sxy, t.TOP + lash - F.GAP0), screw_q),
            "nut": state((t.X0, t.Y0, t.TOP + nut - F.NUT_LIFT0), nut_q),
            "gauge_root": gauge, "left_tcp": state(hands[0]), "right_tcp": state(hands[1]),
            "blade_used": state((float(used), 0.0, 0.0))}


def test_start_scores_0():
    assert t.score(world(lash=F.GAP0, nut=F.NUT_LIFT0)) == 0


def test_done_scores_1():
    w = world()
    assert t.score(w) == pytest.approx(1.0) and t.success(w)


@pytest.mark.parametrize("lash", [0.0017, 0.0020, 0.0023])
def test_tolerance_edges_count(lash):
    assert t.score(world(lash=lash)) == pytest.approx(1.0)


@pytest.mark.parametrize("lash", [0.0, 0.0016, 0.0024, F.GAP0])
def test_lash_out_of_tolerance_scores_0(lash):
    w = world(lash=lash)  # even with the nut seated, the blade out and the hands clear
    assert t.score(w) == 0 and not t.success(w)


def test_ladder():
    assert t.score(world(nut=0.001, gauge=IN_GAP, hands=((t.X0, HOME_Y, t.TOP + 0.09), FAR[1]))) == pytest.approx(0.4)
    assert t.score(world(nut=0.001)) == pytest.approx(0.4)  # lash set, nut still loose
    assert t.score(world(nut=0.00025)) == pytest.approx(0.4)  # nut 0.25 mm short of the rocker
    assert t.score(world(gauge=IN_GAP)) == pytest.approx(0.7)  # locked, blade left in
    assert t.score(world(nut=0.0002)) == pytest.approx(1.0)
    assert t.score(world(nut=-0.0004)) == pytest.approx(1.0)  # pressed 0.4 mm into the rocker under load
    assert t.score(world(nut=-0.0006)) == pytest.approx(0.4)  # deeper than contact gives: not a seated nut


@pytest.mark.parametrize("lash", [0.002, 0.0019, 0.0022])
def test_without_the_blade_scores_0(lash):
    # counting turns from the fixed start, or bottoming out on the stem and backing off, never touched the blade
    w = world(lash=lash, used=False)
    assert t.score(w) == 0 and not t.success(w)


def test_blade_touched_when_screw_rests_on_the_blade():
    assert t.blade_touched(world(lash=0.0020, gauge=IN_GAP))
    assert t.blade_touched(world(lash=0.0018, gauge=IN_GAP))  # tip pressed 0.2 mm into it under load


@pytest.mark.parametrize("kw", [
    dict(lash=0.0024, gauge=IN_GAP),  # blade in, but the tip 0.4 mm above it
    dict(lash=0.0020),  # right lash, blade on the table
    dict(lash=0.0045, gauge=state((t.X0, HOME_Y, t.TOP + F.STEM_TOP + 0.0025))),  # blade held up near the tip
])
def test_blade_not_touched(kw):
    assert not t.blade_touched(world(**kw))


def test_blade_in_gap_is_detected():
    assert t.blade_in_gap(world(gauge=IN_GAP)) and not t.blade_clear(world(gauge=IN_GAP))


@pytest.mark.parametrize("gauge", [
    state((t.X0, HOME_Y + 0.020, t.TOP + F.STEM_TOP)),  # pulled back 2 cm: its tip 1.2 cm from the axis
    state((t.X0 + 0.07, t.Y0 + 0.04, t.TOP + F.STEM_TOP), yaw_q(math.radians(-60))),  # turned, blade across the axis
    state((t.X0 + 0.004, HOME_Y, t.TOP + F.STEM_TOP + 0.003)),  # resting under the rocker, above the tip
])
def test_blade_near_the_axis_is_not_clear(gauge):
    w = world(gauge=gauge)
    assert not t.blade_clear(w) and t.score(w) == pytest.approx(0.7)


def test_blade_pulled_well_out_is_clear():
    assert t.blade_clear(world(gauge=state((t.X0, HOME_Y + 0.03, t.TOP + F.STEM_TOP))))


@pytest.mark.parametrize("hand", [
    (t.X0 + F.TBAR_R, t.Y0, t.TOP + 0.13),  # on a T-bar pin
    (t.X0, t.Y0 - F.SPOKE_R, t.TOP + 0.11),  # on a locknut pin
    (t.X0, t.Y0 + 0.11, t.TOP + 0.30),  # high above, but within 12 cm horizontally
])
def test_hand_near_the_screw_is_not_clear(hand):
    w = world(hands=(FAR[0], hand))
    assert not t.hands_clear(w) and t.score(w) == pytest.approx(0.7)


@pytest.mark.parametrize("kw", [
    dict(screw_xy=(t.X0 + 0.002, t.Y0)),  # screw pushed off its axis
    dict(screw_q=(math.sin(math.radians(2.5) / 2), 0.0, 0.0, math.cos(math.radians(2.5) / 2))),  # screw tilted
    dict(nut_q=(0.0, math.sin(math.radians(3) / 2), 0.0, math.cos(math.radians(3) / 2))),  # nut tilted
])
def test_forced_mechanism_scores_0(kw):
    assert t.score(world(**kw)) == 0


def test_screw_and_nut_turned_any_way_round_still_count():
    assert t.score(world(screw_q=yaw_q(2.0), nut_q=yaw_q(-1.0))) == pytest.approx(1.0)


def test_instruction_is_one_sentence():
    s = t.INSTRUCTION
    assert s[0].isupper() and s.endswith(".") and s.count(". ") == 0
