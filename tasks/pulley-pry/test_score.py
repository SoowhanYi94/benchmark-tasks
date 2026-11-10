"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t
from pulley_pry import mech as M

Q = np.array((0.0, 0.0, 0.0, 1.0))
FLIP = np.array((0.0, 0.0, 1.0, 0.0))  # turned 180 degrees about z
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}
Z = t.TOP + M.ZS


def state(pos, quat=Q, v=(0.0, 0.0, 0.0)):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=np.asarray(quat, float), lin_vel=np.array(v, float), ang_vel=np.zeros(3), low=p, high=p)


def in_v(quat=Q):
    """The pulley's frame pose that stands it in the V-block, rim middle on the V's centre."""
    R = t._rot(quat)
    c = np.array((t.VBLOCK[0], t.VBLOCK[1], t.TOP + t.V_Z))
    return state(c - R @ np.array((M.PULLEY["rim_w"] / 2, 0.0, 0.0)), quat)


def world(travel=0.0, pulley=None, **over):
    w = {a: state(p) for a, p in PARK.items()}
    w["pulley"] = pulley if pulley is not None else state((t.XP + travel, 0.0, Z))
    w["bar_l"] = state((t.RACK_X, t.RACK_TIP, t.TOP + 0.046))
    w["bar_r"] = state((t.RACK_X, -t.RACK_TIP, t.TOP + 0.046))
    w["seat"] = state((1.0, 0.0, 0.0))
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
    assert run(world()) == 0.0


@pytest.mark.parametrize("mm,want", [(2.0, 0.0), (3.5, 0.25), (6.5, 0.5), (8.5, 0.75), (30.0, 0.75)])
def test_rungs(mm, want):
    assert run(world(mm / 1000)) == want


def test_full_marks():
    run(world(0.009))
    assert run(world(pulley=in_v())) == 1.0


def test_reversed_pulley_counts():
    run(world(0.009))
    assert run(world(pulley=in_v(FLIP))) == 1.0


def test_pushed_further_on_scores_zero():
    assert run(world(-0.002)) == 0.0


def test_not_still():
    run(world(0.009))
    p = in_v()
    p.lin_vel = np.array((0.0, 0.05, 0.0))
    assert run(world(pulley=p)) == 0.75


def test_still_needs_a_full_second():
    moving = state((t.XP + 0.03, 0.0, Z), v=(0.05, 0.0, 0.0))
    run(world(pulley=moving))
    assert run(world(pulley=in_v()), n=19) == 0.75


def test_held_in_v_block_is_not_let_go():
    run(world(0.009))
    p = in_v()
    hub = p.pos + (t.HUB_GRIP, 0.0, 0.0)
    assert run(world(pulley=p, right_tcp=hub)) == 0.75


def test_let_go_by_finger_geometry():
    """Let go means both fingertip midpoints 6 cm or more from the pulley's box: an open finger sits 4.7 cm out."""
    run(world(0.009))
    p = in_v()
    top = M.PULLEY["r_o"]
    assert run(world(pulley=p, right_tcp=p.pos + (0.02, 0.0, top + 0.05))) == 0.75
    assert run(world(pulley=p, right_tcp=p.pos + (0.02, 0.0, top + 0.07))) == 1.0


def test_hand_on_seized_pulley_latches():
    hub = (t.XP + t.HUB_GRIP, 0.0, Z)
    run(world(0.002, right_tcp=hub))
    assert run(world(0.009)) == 0.0
    assert run(world(pulley=in_v())) == 0.0


def test_touch_limit_is_5_cm():
    """The open fingers reach 4.7 cm past the fingertip midpoint, so a hand 4.5 cm above the seized pulley's box counts."""
    above = (t.XP + t.HUB_GRIP, 0.0, Z + t.P["r_o"])
    assert t.TOUCH == 0.05
    run(world(0.002, right_tcp=above + np.array((0.0, 0.0, 0.045))))
    assert run(world(pulley=in_v())) == 0.0
    t.reset_latch()
    run(world(0.002, right_tcp=above + np.array((0.0, 0.0, 0.055))))
    assert run(world(0.004)) == 0.25


def test_hand_on_loose_pulley_is_fine():
    run(world(0.009))
    hub = (t.XP + 0.009 + t.HUB_GRIP, 0.0, Z)
    assert run(world(0.012, right_tcp=hub)) == 0.75


def test_hands_on_bars_are_fine():
    run(world(0.004, left_tcp=t._arc("left", 0.3) + (Z,), right_tcp=t._arc("right", 0.3) + (Z,)))
    assert t.score(world(0.004)) == 0.25


def test_off_table_latches():
    run(world(0.009))
    run(world(0.009, bar_l=(0.5, 0.5, t.TOP - 0.2)))
    assert run(world(pulley=in_v())) == 0.0


def test_pulley_off_table_scores_zero():
    run(world(0.009))
    assert run(world(pulley=state((0.6, 0.0, t.TOP - 0.1)))) == 0.0


def test_pulley_dropped_on_bench():
    run(world(0.009))
    lying = state((0.38, -0.05, t.TOP + 0.03), (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)))
    assert run(world(pulley=lying)) == 0.75


def test_tilted_in_v_block_fails():
    run(world(0.009))
    a = math.radians(15)
    assert run(world(pulley=in_v((0.0, 0.0, math.sin(a / 2), math.cos(a / 2))))) == 0.75


def test_v_offset_judged_in_pulley_frame():
    """The rim's middle is found through the pulley's own axis: the reversed pulley sits a rim width further along x."""
    run(world(0.009))
    p = in_v(FLIP)
    p.pos = p.pos + (M.PULLEY["rim_w"], 0.0, 0.0)  # its back face put where an unturned one's would be
    assert run(world(pulley=p)) == 0.75


def test_masses_agree_with_material():
    P, B = M.PULLEY, M.BAR
    groove = 2 * 0.5 * 0.010 * 0.008 * 2 * math.pi * (P["r_o"] - 0.004)
    vol = (math.pi * (P["r_o"] ** 2 - P["r_rim"] ** 2) * P["rim_w"] - groove + math.pi * (P["r_rim"] ** 2 - P["r_hub"] ** 2) * P["web"]
           + math.pi * (P["r_hub"] ** 2 - P["bore"] ** 2) * P["hub_l"])
    assert abs(P["mass"] / vol - 7200) / 7200 < 0.1  # grey cast iron
    assert abs(B["mass"] / (B["t"] * B["h"] * B["len"]) - 7850) / 7850 < 0.1  # steel, with its rubber sleeve


def test_lever_ratio_and_budget():
    """At the demo's grip, each bar turns a 5 N hand push into about 40 N at its tip: two bars beat the drag."""
    ratio = (M.BOSS - 0.004 + t.GRIP_AT - M.W + M.BEVEL) / (M.W - M.BEVEL - M.BOSS - 0.003)
    assert 2 * 5.0 * ratio > t.DRAG
