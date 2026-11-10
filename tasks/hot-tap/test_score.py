"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t

M = t.M
Q = np.array((0.0, 0.0, 0.0, 1.0))
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}
SEAT = (t.X0, t.SEAT_Y, t.TOP + t.SEAT_Z)
STAND = (*t.STAND, t.TOP + t.STAND_Z)
BLANK_ON = (t.X0, t.BLANK_Y, t.TOP + M.WALL_TOP)
BLANK_RACK = (*t.RACK, t.TOP + t.RACK_H)
FED = 0.052


def state(pos, vel=(0.0, 0.0, 0.0), q=Q):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3), low=p, high=p)


def world(machine=SEAT, feed=0.0, gate=0.0, plug=0.0, blank=BLANK_RACK, **over):
    w = {a: state(p) for a, p in PARK.items()}
    w["machine"] = machine if isinstance(machine, NS) else state(machine)
    w["feed"] = state((feed, 0.0, 0.0))
    w["line"] = state((gate, plug, 0.0))
    w["blank"] = blank if isinstance(blank, NS) else state(blank)
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


def cut_it():
    """A real cut: the machine seated, gate open, bar nose on the plug, plug pressed in, held."""
    nose_face = M.PLUG["face"] - 0.006
    feed = t.SEAT_Y + M.BAR["nose"] - nose_face
    run(world(gate=M.GATE["travel"], plug=0.006, feed=feed), n=14)
    assert t._trial["cut"]


def done(**over):
    kw = dict(machine=STAND, gate=0.0, blank=BLANK_ON)
    kw.update(over)
    return world(**kw)


def test_start_scores_zero():
    assert run(world(machine=STAND)) == 0.0


def test_full_marks():
    cut_it()
    assert run(done()) == 1.0


def test_needs_a_second_of_rest():
    cut_it()
    t._trial["still"] = 0.0
    assert run(done(), n=5) == 0.9


def test_machine_seated_alone():
    assert run(world()) == 0.1


def test_seated_but_held_is_nothing():
    grip = (t.X0, t.SEAT_Y + M.BLOCK["y_at"], t.TOP + t.SEAT_Z + t.GRIP_LOCAL_Z)
    assert run(world(left_tcp=grip)) == 0.0


def test_cut_gate_open_machine_seated():
    cut_it()
    assert run(world(gate=M.GATE["travel"])) == 0.3


def test_cut_shut_machine_on():
    cut_it()
    assert run(world(gate=0.0)) == 0.6


def test_cut_shut_machine_off_no_blank():
    cut_it()
    assert run(world(machine=STAND, gate=0.0)) == 0.8


def test_blank_placed_hand_still_on_it():
    cut_it()
    assert run(done(right_tcp=(t.X0, t.BLANK_Y, t.TOP + M.WALL_TOP + t.KNOB_Z))) == 0.8


def test_finger_still_touching_machine_does_not_count():
    cut_it()
    near = (t.STAND[0] + M.BODY["x"] + 0.047, t.STAND[1], t.TOP + t.STAND_Z + 0.06)  # an open hand with one finger on the housing
    assert run(world(machine=STAND, gate=0.0, left_tcp=near)) == 0.3


def test_hand_on_gate_post_blocks_full_marks():
    cut_it()
    assert run(done(right_tcp=(t.X0, t.POST_Y, t.TOP + t.GPOST_Z))) < 1.0


def test_machine_off_with_gate_open_after_cut_is_zero_for_good():
    cut_it()
    t.note_step(world(machine=(t.X0, t.SEAT_Y, t.TOP + 0.10), gate=M.GATE["travel"]), 0.05)
    assert t._trial["spray"]
    assert run(done()) == 0.0  # even when the scene's put right afterwards


def test_machine_off_with_gate_shut_after_cut_is_fine():
    cut_it()
    t.note_step(world(machine=(t.X0, t.SEAT_Y, t.TOP + 0.10), gate=0.0), 0.05)
    assert not t._trial["spray"]


def test_machine_off_before_any_cut_no_spray():
    t.note_step(world(machine=(t.X0, t.SEAT_Y, t.TOP + 0.10), gate=M.GATE["travel"]), 0.05)
    assert not t._trial["spray"]


def test_finger_pressing_the_plug_with_bar_retracted_is_no_cut():
    run(world(gate=M.GATE["travel"], plug=0.006, feed=0.0), n=20)
    assert not t._trial["cut"]


def test_cut_needs_the_dwell():
    nose_face = M.PLUG["face"] - 0.006
    feed = t.SEAT_Y + M.BAR["nose"] - nose_face
    run(world(gate=M.GATE["travel"], plug=0.006, feed=feed), n=5)
    assert not t._trial["cut"]


def test_cut_needs_the_machine_seated():
    nose_face = M.PLUG["face"] - 0.006
    feed = t.SEAT_Y + M.BAR["nose"] - nose_face
    run(world(machine=(t.X0, t.SEAT_Y, t.TOP + 0.04), gate=M.GATE["travel"], plug=0.006, feed=feed), n=20)
    assert not t._trial["cut"]


def test_cut_stays_cut():
    cut_it()
    run(world(), n=5)
    assert t._trial["cut"]


def test_machine_gone_from_saddle_off_stand_loses_credit():
    cut_it()
    assert run(world(machine=(0.15, 0.3, t.TOP + 0.002), gate=0.0)) == 0.3


def test_part_off_table_is_zero_for_good():
    t.note_step(world(blank=(0.2, 0.9, 0.05)), 0.05)
    assert run(world()) == 0.0


def test_blank_tilted_is_not_seated():
    cut_it()
    a = math.radians(15)
    w = done(blank=state(BLANK_ON, q=np.array((math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)))))
    assert not t.blank_on(w)


@pytest.mark.parametrize("axis", [0, 1])
def test_blank_off_centre_is_not_seated(axis):
    cut_it()
    p = np.array(BLANK_ON, float)
    p[axis] += 0.008
    assert not t.blank_on(done(blank=p))


def test_gate_not_quite_shut_does_not_count():
    assert not t.gate_shut(world(gate=0.006))


def test_let_go_check_follows_a_yawed_machine():
    th = math.radians(45.0)
    q = np.array((0.0, 0.0, math.sin(th / 2), math.cos(th / 2)))
    corner = np.array(STAND) + (0.0, 0.0, 0.06) + np.array((math.cos(th) * (M.BODY["x"] + 0.05), math.sin(th) * (M.BODY["x"] + 0.05), 0.0))
    w = world(machine=state(STAND, q=q), left_tcp=corner)
    assert not t._machine_clear(w)


def test_masses_suit_the_material():
    al = 2700.0
    wall = 0.003
    B = M.BODY
    shell = (2 * B["x"] * 2 * B["y"] * B["z"] - (2 * B["x"] - 2 * wall) * (2 * B["y"] - 2 * wall) * (B["z"] - 2 * wall)) * al
    grip_shell = 2 * ((2 * M.BLOCK["x"] + 2 * M.BLOCK["y"]) * 0.04) * 0.0025 * al  # a thin-walled grip block
    assert abs(shell + grip_shell - M.MASS_MACHINE["body"]) < 0.06
    bar_tube = 4 * 0.02 * 0.002 * (M.BAR["tail"] - M.BAR["nose"]) * al
    post = 4 * 0.024 * 0.0025 * 0.045 * al
    assert abs(bar_tube + post - M.MASS_MACHINE["bar"]) < 0.04
    b = t.BLANK
    v = 2 * b["flange_x"] * 2 * b["flange_y"] * b["flange_t"] + 2 * b["stem_x"] * 2 * b["stem_y"] * b["stem_l"] + 2 * b["knob_x"] * 2 * b["knob_y"] * b["knob_h"]
    assert abs(v * al - b["mass"]) < 0.015


def test_machine_light_enough_and_balance_in_reach():
    assert sum(M.MASS_MACHINE.values()) <= 0.5
    assert t.X0 <= 0.35


def test_demo_fits_run_limit():
    wps = t.demo()
    assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 3000
