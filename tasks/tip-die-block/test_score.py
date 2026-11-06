"""The scorer against every outcome in the scoring rules, the tip and slide arithmetic, and fixed plans. No simulator."""

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t

PARKED = {"left_tcp": np.array((0.18, 0.32, t.TOP + 0.18)), "right_tcp": np.array((0.18, -0.32, t.TOP + 0.18))}
RX = lambda deg: t._rx(deg)  # noqa: E731


def block(deg, y=0.0, moving=False, dz=0.0):
    return SimpleNamespace(pos=np.array((t.X0, y, t.BLOCK_Z + dz)), quat_xyzw=np.array(RX(deg)), resting=lambda: not moving)


def state(b, **kw):
    return {"die_block": b, **{k: SimpleNamespace(pos=v) for k, v in PARKED.items()}, **kw}


def turned(start_deg, s):
    """The block after one tip with a hand on its ``s`` side: over its far bottom edge, a quarter turn about x."""
    return start_deg + s * 90.0


@pytest.fixture(params=list(t.LOTS))
def lot(request, monkeypatch):
    monkeypatch.setenv("TIP_LOT", request.param)
    importlib.reload(t)
    yield request.param
    monkeypatch.delenv("TIP_LOT")
    importlib.reload(t)


def _final(name, tips):
    deg = t.LOTS[name]["rot"]
    for _, s in tips:
        deg = turned(deg, s)
    return deg


def test_marking_red_is_body_z():
    assert t.red_up_cos(block(0)) == pytest.approx(1.0)
    assert t.red_up_cos(block(180)) == pytest.approx(-1.0)
    assert t._rot(RX(-90))[1, 2] == pytest.approx(1.0)  # A: red on +y
    assert t._rot(RX(90))[1, 2] == pytest.approx(-1.0)  # B: red on -y


def test_a_tip_turns_the_right_way():
    """A hand on +y pushes toward -y: the top goes to -y, so +y comes up. Turning about x by +90 sends +y up."""
    R = t._rot(RX(90))
    assert (R @ (0, 1, 0))[2] == pytest.approx(1.0)


@pytest.mark.parametrize("name", list(t.LOTS))
def test_each_lots_tips_bring_red_up(name):
    assert t.red_up(block(_final(name, t.LOTS[name]["tips"])))


def test_doing_nothing_scores_zero(lot):
    assert t.score(state(block(t.LOTS[lot]["rot"]))) == 0.0


def test_full_score(lot):
    s = state(block(0.0, y=0.12))
    assert t.score(s) == 1.0 and t.success(s)


@pytest.mark.parametrize("deg", [9.0, -9.0])
def test_within_ten_degrees_is_up(deg):
    assert t.score(state(block(deg))) == 1.0


@pytest.mark.parametrize("deg", [11.0, -11.0, 45.0, 90.0, 180.0])
def test_beyond_ten_degrees_is_not(deg):
    assert t.score(state(block(deg))) == 0.0


def test_moving_is_half():
    assert t.score(state(block(0.0, moving=True))) == 0.5


def test_off_the_mat_is_half():
    assert t.score(state(block(0.0, y=t.LANE_Y_HALF + 0.02))) == 0.5


def test_held_up_on_something_is_half():
    assert t.score(state(block(0.0, dz=0.03))) == 0.5


@pytest.mark.parametrize("arm", ["left_tcp", "right_tcp"])
def test_hand_near_is_half(arm):
    near = np.array((t.X0, 0.0, t.BLOCK_Z + t.SIDE / 2 + t.HANDS_CLEAR - 0.005))
    assert t.score(state(block(0.0), **{arm: SimpleNamespace(pos=near)})) == 0.5


def test_hand_just_clear_is_full():
    far = np.array((t.X0, 0.0, t.BLOCK_Z + t.SIDE / 2 + t.HANDS_CLEAR + 0.005))
    assert t.score(state(block(0.0), left_tcp=SimpleNamespace(pos=far))) == 1.0


# -- a plan that doesn't look -------------------------------------------------------------------------------------------
def test_fixed_plans_match_only_one_lot():
    """Every fixed sequence of up to four tips, either way, across the lots."""
    plans = [()]
    for _ in range(4):
        plans += [p + (s,) for p in plans for s in (1, -1) if len(p) < 4]
    best = 0.0
    for plan in set(plans):
        total = 0.0
        for lot in t.LOTS.values():
            deg = lot["rot"] + sum(s * 90.0 for s in plan)
            total += 1.0 if t.red_up(block(deg)) else 0.0
        best = max(best, total / len(t.LOTS))
    assert best == pytest.approx(1 / 3)


# -- the physics the task relies on -------------------------------------------------------------------------------------
def test_too_wide_to_pick_up():
    assert t.SIDE > 0.094


def test_high_push_tips_low_push_slides():
    g, mu = 9.81, 1.0  # the mat
    tip = lambda h: t.BLOCK_MASS * g * (t.SIDE / 2) / h  # noqa: E731
    slide = mu * t.BLOCK_MASS * g
    assert tip(t.PUSH_H) < 0.7 * slide  # the demo's push tips it, with margin
    assert tip(0.04) > slide  # a low push only slides it


def test_demo_push_goes_past_the_balance_point():
    a = math.radians(45)
    s = (t.PUSH_H - t.SIDE * math.sin(a)) / math.cos(a)
    y = t.SIDE * math.cos(a) - s * math.sin(a)
    assert t.TRAVEL > t.SIDE - y + 0.01


def test_lane_holds_the_block_square():
    assert t.LANE_X_HALF - t.SIDE / 2 == pytest.approx(0.005)


def test_lane_is_long_enough_for_two_tips_either_way():
    assert 2 * t.SIDE + t.SIDE / 2 < t.LANE_Y_HALF


def test_curbs_are_below_the_push():
    assert t.CURB_H < t.PUSH_H


# -- the demo -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(t.LOTS))
def test_demo_fits_run_limit_and_ik_reaches_every_waypoint(name):
    wps = t.demo(name)
    assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 1800
    pe, re = t.ik_errors(wps)
    assert pe < 1e-3 and re < 1e-2


@pytest.mark.parametrize("name", list(t.LOTS))
def test_demo_ends_clear_of_the_block(name):
    wps = t.demo(name)
    deg = _final(name, t.LOTS[name]["tips"])
    y = -sum(s * t.SIDE for _, s in t.LOTS[name]["tips"])
    s = {"die_block": block(deg, y=y), **{f"{a}_tcp": SimpleNamespace(pos=np.array(wps[a][-1].pos)) for a in wps}}
    assert t.score(s) == 1.0
