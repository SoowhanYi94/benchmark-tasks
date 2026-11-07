"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t

PARK = {"left_tcp": (0.12, 0.40, t.PARK[2]), "right_tcp": (0.12, -0.40, t.PARK[2])}
ENDS = (t.FIRST_END, t.SECOND_END, t.THIRD_END, t.FOURTH_END)
SEAT_BAR = t.M.GROOVE["h"] + t.M.COVER_T


def part(angle, z=0.0, dr=0.0, skew=0.0, rate=(0.0, 0.0, 0.0), tilt=0.0):
    """A part at this angle in the ring's frame, with its own x axis along the radius plus `skew`."""
    a = math.radians(angle)
    r = t.R + dr
    p = np.array((t.CX + r * math.cos(a), t.CY + r * math.sin(a), t.TOP + t.M.FLOOR + z))
    y = math.radians(angle + skew) / 2
    q = np.array((0.0, 0.0, math.sin(y), math.cos(y)))
    if tilt:
        h = math.radians(tilt) / 2
        qx, qw = math.sin(h), math.cos(h)  # about own x, after the yaw
        q = np.array((q[3] * qx, q[2] * qx, q[2] * qw, q[3] * qw))
    return NS(pos=p, quat_xyzw=q, lin_vel=np.array(rate, float), ang_vel=np.zeros(3), low=p, high=p)


def start(n):
    y = t.START_YAW / 2
    p = np.array((*t.SEG_START[n], t.TOP))
    return NS(pos=p, quat_xyzw=np.array((0.0, 0.0, math.sin(y), math.cos(y))), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p)


def bar_start():
    y = t.START_YAW / 2
    p = np.array((*t.BAR_START, t.TOP + t.BAR["dog_h"]))
    return NS(pos=p, quat_xyzw=np.array((0.0, 0.0, math.sin(y), math.cos(y))), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p)


def tcp(xyz):
    p = np.array(xyz, float)
    return NS(pos=p, quat_xyzw=np.array((0.0, 0.0, 0.0, 1.0)), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p)


def world(n=4, on=True, **over):
    w = {a: tcp(p) for a, p in PARK.items()}
    for k, name in enumerate(t.SEGMENTS):
        w[name] = part(ENDS[k]) if k < n else start(name)
    w["bar"] = part(t.GATE, SEAT_BAR) if on else bar_start()
    w.update(over)
    return w


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def run(w, n=22):
    for _ in range(n):
        t.note_step(w, 0.05)
    return t.score(w)


def test_start_scores_zero():
    assert run(world(0, False)) == 0.0


def test_full_marks():
    assert run(world()) == 1.0


@pytest.mark.parametrize("n,s", [(1, 0.2), (2, 0.4), (3, 0.6), (4, 0.8)])
def test_each_rung(n, s):
    assert run(world(n, False)) == s


def test_bar_on_without_all_four_adds_nothing():
    assert run(world(3)) == 0.6


def test_needs_a_second_of_rest():
    assert run(world(), n=5) == 0.9


def test_a_moving_part_stops_short():
    assert run(world(seg_a=part(t.FIRST_END, rate=(0.05, 0.0, 0.0)))) == 0.9


def test_segment_in_the_gate_earns_nothing():
    assert run(world(1, False, seg_a=part(t.GATE))) == 0.0


def test_segment_partly_in_the_gate_earns_nothing():
    assert run(world(1, False, seg_a=part(t.EXIT_FACE - t.HALF + 4.0))) == 0.0


def test_segment_just_clear_of_the_gate_counts():
    assert run(world(1, False, seg_a=part(t.EXIT_FACE - t.HALF))) == 0.2


def test_segment_on_the_stop_counts_and_not_past_it():
    assert run(world(1, False, seg_a=part(t.STOP_FACE + t.HALF))) == 0.2
    t.reset_latch()
    assert run(world(1, False, seg_a=part(t.STOP_FACE + t.HALF - 5.0))) == 0.0


def test_segment_on_the_cover_earns_nothing():
    assert run(world(1, False, seg_a=part(t.FIRST_END, z=t.M.GROOVE["h"] + t.M.COVER_T))) == 0.0


def test_segment_riding_on_the_step_earns_nothing():
    assert run(world(1, False, seg_a=part(t.STEP, z=t.M.STEP_H))) == 0.0


def test_tilted_segment_earns_nothing():
    assert run(world(1, False, seg_a=part(t.FIRST_END, tilt=8.0))) == 0.0


def test_out_of_the_groove_radially_earns_nothing():
    assert run(world(1, False, seg_a=part(t.FIRST_END, dr=0.01))) == 0.0


@pytest.mark.parametrize("angle", [t.FIRST_END, t.SECOND_END, t.THIRD_END, t.FOURTH_END])
def test_skew_is_judged_against_the_radius_at_any_angle(angle):
    assert run(world(1, False, seg_a=part(angle, skew=5.0))) == 0.2
    t.reset_latch()
    assert run(world(1, False, seg_a=part(angle, skew=20.0))) == 0.0


def test_segment_held_does_not_count():
    assert run(world(1, False, left_tcp=tcp(t._at(t.FIRST_END, t.M.FLOOR + t.SEG_GZ)))) == 0.0


def test_open_finger_on_a_segment_does_not_count():
    a = math.radians(t.FIRST_END)
    near = (t.CX + (t.R + 0.012 + 0.047) * math.cos(a), t.CY + (t.R + 0.012 + 0.047) * math.sin(a), t.TOP + t.M.FLOOR + 0.03)
    assert run(world(1, False, right_tcp=tcp(near))) == 0.0


def test_let_go_box_follows_the_segment_round():
    # a hand 7 cm past the segment's end, along its own arc direction, is clear at any angle; 5 cm isn't
    for angle in ENDS:
        a = math.radians(angle)
        for gap, want in ((0.07, 0.2), (0.05, 0.0)):
            d = t.SEG_BOX[1][1] + gap
            c = np.array((t.CX + t.R * math.cos(a), t.CY + t.R * math.sin(a), t.TOP + t.M.FLOOR + 0.03))
            h = c + d * np.array((-math.sin(a), math.cos(a), 0.0))
            t.reset_latch()
            assert run(world(1, False, seg_a=part(angle), right_tcp=tcp(h))) == want


def test_a_part_leaving_loses_credit():
    assert run(world(1, False)) == 0.2
    t.reset_latch()
    assert run(world(0, False)) == 0.0


def test_off_the_table_is_zero_for_good():
    w = world()
    away = world(seg_a=tcp((0.2, 0.9, 0.05)))
    t.note_step(away, 0.05)
    assert t._trial["bad"]
    assert run(w) == 0.0


def test_bar_off_the_table_is_zero_for_good():
    t.note_step(world(bar=tcp((0.2, -0.9, 0.05))), 0.05)
    assert run(world()) == 0.0


def test_bar_held_stops_short():
    w = world(right_tcp=tcp(t._at(t.GATE, SEAT_BAR + t.M.FLOOR + t.BAR_GZ)))
    assert not t.bar_seated(w) and run(w) < 0.9


def test_bar_off_the_gate_does_not_count():
    assert run(world(bar=part(t.GATE + 12.0, SEAT_BAR))) == 0.8


def test_bar_propped_on_a_knob_does_not_count():
    assert run(world(bar=part(t.GATE, SEAT_BAR + 0.01))) == 0.8


def test_wrong_order_the_bar_first_can_not_pass():
    # the bar shut with the last segment still out: nothing for the bar, whichever segment is missing
    for missing in t.SEGMENTS:
        t.reset_latch()
        w = world()
        w[missing] = start(missing)
        assert run(w) == 0.6


@pytest.mark.parametrize("order", [(1, 0, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1)])
def test_segments_are_interchangeable(order):
    w = world()
    w.update({n: part(ENDS[order[k]]) for k, n in enumerate(t.SEGMENTS)})
    assert run(w) == 1.0


def test_two_in_one_place_can_not_count_twice():
    w = world(1, False, seg_b=part(t.FIRST_END + 10.0))
    assert run(w) == 0.2


def test_every_end_is_under_the_cover_with_margin():
    for e in ENDS:
        assert t.under_cover(e)
    assert ENDS[1] + t.HALF < t.STEP - t.STEP_HALF and ENDS[2] - t.HALF > t.STEP + t.STEP_HALF


def test_groove_only_holds_two_either_side_of_the_step():
    assert (t.EXIT_FACE - (t.STEP + t.STEP_HALF)) // t.SEG_DEG == 2
    assert ((t.STEP - t.STEP_HALF) - t.STOP_FACE) // t.SEG_DEG == 2


def _box_volume(boxes):
    return sum(np.prod(b[1]) for b in boxes.values())


def test_masses_suit_the_material():
    # facets overlap a little on their inner edges, so judge the true shapes
    arc = math.pi * (t.SEG["r1"] ** 2 - t.SEG["r0"] ** 2) * t.SEG_DEG / 360 * t.SEG["h"]
    seg = arc + t.SEG["stem"] ** 2 * (t.SEG["stem_top"] - t.SEG["h"]) + t.SEG["knob"] ** 2 * (t.SEG["knob_top"] - t.SEG["stem_top"])
    plate = math.pi * (t.BAR["r1"] ** 2 - t.BAR["r0"] ** 2) * 2 * t.BAR["half_deg"] / 360 * t.BAR["t"]
    dogs = 2 * (t.BAR["dog_r1"] - t.BAR["dog_r0"]) * math.radians(2 * t.BAR["dog_half"]) * t.R * t.BAR["dog_h"]
    bar = plate + dogs + t.SEG["stem"] ** 2 * (t.BAR["stem_top"] - t.BAR["t"]) + t.BAR["knob"] ** 2 * (t.BAR["knob_top"] - t.BAR["stem_top"])
    assert abs(seg * t.AL_DENSITY - t.SEG["mass"]) < 0.01
    assert abs(bar * t.AL_DENSITY - t.BAR["mass"]) < 0.01


def test_segment_boxes_fit_the_groove_and_stem_the_slot():
    g, s, sl = t.M.GROOVE, t.SEG, t.M.SLOT
    assert s["r0"] - g["r0"] >= 0.005 and g["r1"] - s["r1"] >= 0.005
    assert (t.R - s["stem"] / 2) - sl["r0"] >= 0.005 - 1e-9 and sl["r1"] - (t.R + s["stem"] / 2) >= 0.005 - 1e-9
    assert sl["r1"] - sl["r0"] >= 0.022


def test_four_segments_leave_play():
    assert 4 * t.SEG_DEG + 2 * t.GATE_HALF + 2 * t.STEP_HALF < 360.0
    assert t.SEG_DEG == 66.0
