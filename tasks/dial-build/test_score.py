"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t

Q = np.array((0.0, 0.0, 0.0, 1.0))
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}
TURN = math.radians(100.0)
BLOCKS, PEGS = ("block_a", "block_b"), ("peg_a", "peg_b")


def state(pos, rate=(0.0, 0.0, 0.0)):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=Q, lin_vel=np.array(rate, float), ang_vel=np.zeros(3), low=p, high=p)


def world(blocks=(True, True), pegs=(True, True), stopped=True, pad=0.0, rate=None, turn=TURN, **over):
    w = {a: state(p) for a, p in PARK.items()}
    r = (0.0 if stopped else t.RATE) if rate is None else rate
    w["dial"] = state((turn, pad, 1.0 if stopped else 0.0), (r, 0.0, 0.0))
    for k in (0, 1):
        c, _ = t.pocket_at(turn, k)
        bn, pn = BLOCKS[k], PEGS[k]
        w[bn] = state(c if blocks[k] else (*t.BLOCK_START[bn], t.TOP))
        w[pn] = state(c + (0.0, 0.0, t.BLOCK["h"]) if pegs[k] else (*t.PEG_START[pn], t.TOP + t.RACK_H))
    for k, v in over.items():
        w[k] = v if isinstance(v, NS) else state(v)
    return w


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()
    t._trial["moving"] = True


def run(w, n=22):
    for _ in range(n):
        t.note_step(w, 0.05)
    return t.score(w)


def test_start_scores_zero():
    assert run(world((False, False), (False, False), stopped=False)) == 0.0


def test_one_block():
    assert run(world((True, False), (False, False), stopped=False)) == 0.2


def test_block_and_peg():
    assert run(world((True, False), (True, False), stopped=False)) == 0.5


def test_both_blocks_one_peg():
    assert run(world((True, True), (True, False), stopped=False)) == 0.7


def test_built_but_turning_stops_short():
    assert run(world(stopped=False)) == 0.9


def test_full_marks():
    assert run(world()) == 1.0


def test_needs_a_second_of_rest():
    assert run(world(), n=5) == 0.9


def test_peg_without_its_block_earns_nothing():
    assert run(world((False, True), (True, False), stopped=False)) == 0.2


def test_a_block_held_in_the_pocket_does_not_count():
    c, _ = t.pocket_at(TURN, 0)
    assert run(world((True, False), (False, False), stopped=False, left_tcp=tuple(c + (0.0, 0.0, t.BLOCK_GZ)))) == 0.0


def test_finger_still_touching_a_block_does_not_count():
    c, _ = t.pocket_at(TURN, 0)
    near = tuple(c + (t.BLOCK["half"] + 0.047, 0.0, t.BLOCK_GZ))  # an open hand with one finger on the block
    assert run(world((True, False), (False, False), stopped=False, left_tcp=near)) == 0.0


def test_block_out_of_its_pocket_loses_credit():
    assert run(world((False, False), (False, False), stopped=False, block_a=(0.27, 0.0, t.TOP + t.RIM))) == 0.0


def test_part_off_the_table_is_zero_for_good():
    t.note_step(world(block_a=(0.2, 0.9, 0.05)), 0.05)
    assert run(world()) == 0.0


def test_stopping_early_is_zero_for_good():
    t.note_step(world((True, False), (False, False), stopped=False, pad=0.02), 0.05)
    assert t._trial["bad"]
    assert run(world()) == 0.0


def test_holding_the_dial_back_is_zero_for_good():
    w = world((True, False), (False, False), stopped=False, rate=0.0)
    for _ in range(25):
        t.note_step(w, 0.05)
    assert t._trial["bad"]


def test_a_brief_jolt_is_not_holding_it_back():
    w = world((True, False), (False, False), stopped=False, rate=-0.1)
    for _ in range(10):
        t.note_step(w, 0.05)
    assert not t._trial["bad"]
    t.note_step(world((True, False), (False, False), stopped=False), 0.05)
    assert t._trial["slow"] == 0.0


def test_pressing_the_pad_once_built_is_fine():
    t.note_step(world(stopped=False, pad=0.02), 0.05)
    assert not t._trial["bad"]


def test_hand_on_the_pad_stops_short():
    assert run(world(left_tcp=(*t.PAD_AT, t.TOP + t.M.PAD["z1"]))) == 0.9


def test_masses_suit_the_material():
    b = (4 * t.BLOCK["half"] ** 2 - 4 * t.BLOCK["bore"] ** 2) * t.BLOCK["h"] * 2700
    p = (4 * t.PEG["head"] ** 2 * t.PEG["head_h"] + 4 * t.PEG["shaft"] ** 2 * t.PEG["shaft_l"]) * 2700
    assert abs(b - t.BLOCK["mass"]) < 0.01 and abs(p - t.PEG["mass"]) < 0.01


def test_arms_keep_clear_when_both_work():
    a, _ = t.pocket_at(t.RATE * t.t_at(0, 270.0), 0)
    b, _ = t.pocket_at(t.RATE * t.t_at(1, 90.0), 1)
    assert np.linalg.norm(a[:2] - b[:2]) >= 0.15


@pytest.mark.parametrize("deg", [0.0, 30.0, 45.0, 100.0, 225.0])
def test_a_peg_in_the_bores_corner_counts_at_any_dial_angle(deg):
    turn = math.radians(deg)
    w = world(turn=turn)
    # the block turned with the dial, the peg off-centre along its sides by the bore's play
    th = t.FIXTURES[0] + turn - math.pi / 2
    q = np.array((0.0, 0.0, math.sin(th / 2), math.cos(th / 2)))
    c, s = math.cos(th), math.sin(th)
    off = np.array((c * 0.007 - s * 0.007, s * 0.007 + c * 0.007, t.BLOCK["h"]))
    w["block_a"] = NS(pos=w["block_a"].pos, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=w["block_a"].pos, high=w["block_a"].pos)
    w["peg_a"] = state(w["block_a"].pos + off)
    assert t.peg_seated(w, 0)


def test_pressing_the_pad_with_a_hand_still_on_a_part_is_zero_for_good():
    c, _ = t.pocket_at(TURN, 0)
    t.note_step(world(stopped=False, pad=0.02, right_tcp=tuple(c + (0.0, 0.0, t.BLOCK["h"] + 0.02))), 0.05)
    assert t._trial["bad"]


def swapped(blocks=True, pegs=True):
    w = world()
    if blocks:
        w["block_a"], w["block_b"] = w["block_b"], w["block_a"]
    if pegs:
        w["peg_a"], w["peg_b"] = w["peg_b"], w["peg_a"]
    return w


@pytest.mark.parametrize("blocks,pegs", [(True, True), (True, False), (False, True)])
def test_any_block_and_peg_counts_in_any_place(blocks, pegs):
    assert run(swapped(blocks, pegs)) == 1.0


def test_pressing_the_pad_after_a_swapped_build_is_fine():
    w = swapped()
    w["dial"] = state((TURN, 0.02, 0.0), (t.RATE, 0.0, 0.0))
    t.note_step(w, 0.05)
    assert not t._trial["bad"]


def test_two_blocks_in_one_pocket_can_not_count_twice():
    c, _ = t.pocket_at(TURN, 0)
    assert run(world((True, False), (False, False), stopped=False, block_b=c + (0.0, 0.0, t.BLOCK["h"]))) == 0.2


def test_let_go_check_follows_a_turned_block():
    turn = math.radians(45.0) - t.FIXTURES[0] + math.pi / 2
    w = world((True, False), (False, False), stopped=False, turn=turn)
    th = math.radians(45.0)
    w["block_a"].quat_xyzw = np.array((0.0, 0.0, math.sin(th / 2), math.cos(th / 2)))
    c = w["block_a"].pos
    corner = c + np.array((math.cos(th) * (t.BLOCK["half"] + 0.05), math.sin(th) * (t.BLOCK["half"] + 0.05), t.BLOCK_GZ))
    w["left_tcp"] = state(corner)
    assert not t.block_seated(w, 0)
