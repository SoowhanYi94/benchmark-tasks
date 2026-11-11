"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import json
import math
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t
from mt_yoke import mech as M

DT = 0.05
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}
HERE = os.path.dirname(os.path.abspath(__file__))


def quat_z(a):
    return np.array((0.0, 0.0, math.sin(a / 2), math.cos(a / 2)))


def quat_tilt(a, b):
    """Turned b about z, then tilted a about x."""
    qz = quat_z(b)
    qx = np.array((math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)))
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    return np.array((w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2))


def state(pos, yaw=0.0, v=0.0, q=None, gap=0.094):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=quat_z(yaw) if q is None else q, lin_vel=np.array((v, 0.0, 0.0)), ang_vel=np.zeros(3),
              low=p - (gap / 2, 0, 0), high=p + (gap / 2, 0, 0))


def yoke_rest():
    return state((M.YOKE_REST["x"], M.YOKE_REST["y"], t.TOP + M.YOKE_REST["z"]), M.YOKE_REST["heading"])


def bulb_rest():
    return state((M.BULB_REST["x"], M.BULB_REST["y"], t.TOP + M.BULB_REST_Z), M.BULB_REST["heading"])


def world(**over):
    w = {a: state(p) for a, p in PARK.items()}
    w["yoke"], w["bulb"] = yoke_rest(), bulb_rest()
    for n, (x, y) in zip(t.PEGS, M.PEG_HOME):
        w[n] = state((x, y, t.TOP + M.TRAY["t"]))
    w["squeeze"] = state((0.0, 0.0, 0.0))
    w.update(over)
    return w


def yoke_at(w, xy, heading, lift=0.0, hand="left_tcp", closed=True, tilt=0.0):
    o = np.array((xy[0], xy[1], t.PLATE_Z + lift))
    w["yoke"] = state(o, heading, q=quat_tilt(tilt, heading) if tilt else None)
    if hand:
        w[hand] = state(o + t._rz(heading) @ M.HANDLE_C + (0, 0, t.GRIP_DZ), gap=M.HANDLE["x"] if closed else 0.06)


def bulb_at(w, tip_xy, heading, height=0.025, sq=0.006, hand="right_tcp"):
    tip = np.array((tip_xy[0], tip_xy[1], t.PLATE_Z + height))
    o = tip - t._rz(heading) @ M.NOZZLE
    w["bulb"] = state(o, heading)
    w["squeeze"] = state((sq, 0.0, 0.0))
    if hand:
        w[hand] = state(o + (0, 0, t.GRIP_DZ), gap=M.BULB_GRIP_W - sq)


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def steps(w, n=12):
    for _ in range(n):
        t.note_step(w, DT)


def do_pass(which, live=True, bulb=True, bulb_hand="right_tcp", yoke_hand="left_tcp", dust_first=False):
    places = t.PASS_A if which == "A" else t.PASS_B
    head, bhead = (t.HEAD_YA, t.HEAD_BA) if which == "A" else (t.HEAD_YB, t.HEAD_BB)
    w = world()
    for c in places:
        yoke_at(w, (c, M.WELD_Y), head, hand=yoke_hand, closed=live)
        if bulb:
            bulb_at(w, (c, M.WELD_Y), bhead, hand=bulb_hand)
        steps(w)
        w["squeeze"] = state((0.0, 0.0, 0.0))
        yoke_at(w, (c, M.WELD_Y), head, lift=0.015, hand=yoke_hand, closed=live)
        steps(w, 2)
    return w


def full_exam():
    do_pass("A")
    do_pass("B")


def place_pegs(w, targets):
    for n, xy in zip(t.PEGS, targets):
        w[n] = state((xy[0], xy[1], t.PLATE_Z))
    return w


def finish(w, n=30):
    steps(w, n)
    return t.score(w)


def good_end(targets=None):
    targets = [M.CRACKS["long"]["mid"], M.CRACKS["trans"]["mid"]] if targets is None else targets
    return place_pegs(world(), targets)


# -- the full outcome ladder ------------------------------------------------------------------------------------------
def test_untouched_scores_zero():
    assert finish(world()) == 0.0


def test_full_exam_both_marked_rested_is_one():
    full_exam()
    assert t._trial["A"].all() and t._trial["B"].all() and all(t._trial["found"].values())
    assert finish(good_end()) == 1.0


def test_one_crack_marked_is_partial():
    full_exam()
    assert finish(good_end([M.CRACKS["long"]["mid"]])) == t.SCORE_ONE


def test_both_marked_tools_not_rested():
    full_exam()
    w = good_end()
    w["bulb"] = state((0.2, -0.15, t.TOP + 0.08))
    assert finish(w) == t.SCORE_TWO


def test_both_marked_hand_on_tool():
    full_exam()
    w = good_end()
    w["left_tcp"] = state(np.asarray(w["yoke"].pos) + t._rz(M.YOKE_REST["heading"]) @ M.HANDLE_C)
    assert finish(w) == t.SCORE_TWO


def test_needs_one_second_still():
    full_exam()
    w = good_end()
    steps(w, 10)
    assert t.score(w) == t.SCORE_TWO
    w["peg0"].lin_vel = np.array((0.05, 0, 0))
    steps(w, 30)
    assert t.score(w) == t.SCORE_TWO


# -- the shortcuts the judges named -----------------------------------------------------------------------------------
def test_dust_first_then_yoke_alone_scores_zero():
    """One arm at a time: the whole weld dusted with the yoke on its rest, then the yoke stepped through both passes."""
    w = world()
    for c in t.PASS_A:
        bulb_at(w, (c, M.WELD_Y), t.HEAD_BB)
        steps(w)
    assert t._trial["dust"].all() and t._trial["live_steps"] == 0
    do_pass("A", bulb=False)
    do_pass("B", bulb=False)
    assert not t._trial["A"].any() and not any(t._trial["found"].values())
    assert finish(good_end()) == 0.0


def test_powder_while_trigger_released_counts_nothing():
    do_pass("A", live=False)
    do_pass("B", live=False)
    assert not t._trial["A"].any() and t._trial["dry_steps"] > 0
    assert finish(good_end()) == 0.0


def test_one_direction_only_scores_zero():
    do_pass("A")
    assert t._trial["found"]["long"] and not t._trial["found"]["trans"]
    assert finish(good_end([M.CRACKS["long"]["mid"]])) == 0.0
    t.reset_latch()
    do_pass("B")
    assert t._trial["found"]["trans"] and not t._trial["found"]["long"]
    assert finish(good_end([M.CRACKS["trans"]["mid"]])) == 0.0


def test_wrong_peg_scores_zero():
    full_exam()
    assert finish(good_end([M.CRACKS["long"]["mid"], t.WRONG])) == 0.0
    t.reset_latch()
    full_exam()
    assert finish(good_end([M.CRACKS["long"]["mid"], M.CRACKS["trans"]["mid"], t.WRONG])) == 0.0


def test_peg_off_by_more_than_10mm_is_stray():
    full_exam()
    off = M.CRACKS["trans"]["mid"] + (0.0, 0.012)
    assert finish(good_end([M.CRACKS["long"]["mid"], off])) == 0.0


def test_no_peg_scores_zero():
    full_exam()
    assert finish(world()) == 0.0


def test_one_hand_on_both_tools_counts_nothing():
    """The bulb in the hand that holds the yoke (or in no hand) gives no live powder."""
    do_pass("A", bulb_hand="left_tcp")
    do_pass("B", bulb_hand=None)
    assert not t._trial["A"].any() and not t._trial["B"].any()


def test_yoke_not_held_counts_nothing():
    do_pass("A", yoke_hand=None)
    assert not t._trial["A"].any()


# -- the field and the powder's geometry ------------------------------------------------------------------------------
def test_field_needs_both_poles_down():
    w = world()
    yoke_at(w, (0.22, M.WELD_Y), t.HEAD_YA)
    assert t.field_on(w)
    yoke_at(w, (0.22, M.WELD_Y), t.HEAD_YA, lift=t.GATE + 0.001)
    assert not t.field_on(w)
    yoke_at(w, (0.22, M.WELD_Y), t.HEAD_YA, tilt=math.radians(8.0))
    assert not t.field_on(w)
    yoke_at(w, (0.22, M.WELD_Y), t.HEAD_YA, closed=False)
    assert not t.field_on(w)
    yoke_at(w, (M.PLATE["x"][1] - 0.005, M.WELD_Y), t.HEAD_YA)  # a pole off the plate's edge
    assert not t.field_on(w)


def test_crack_shows_only_across_the_field():
    w = world()
    mid = M.CRACKS["long"]["mid"]
    yoke_at(w, mid, t.HEAD_YB)  # field along the weld, parallel to the longitudinal crack
    bulb_at(w, mid, t.HEAD_BB)
    steps(w)
    assert not t._trial["found"]["long"]
    yoke_at(w, mid, t.HEAD_YB, lift=0.02)
    steps(w, 2)
    yoke_at(w, mid, t.HEAD_YA)
    bulb_at(w, mid, t.HEAD_BA)
    steps(w)
    assert t._trial["found"]["long"]
    assert "ind_long" in t.shown() and "fx_lamp" in t.shown() and "fx_puff" in t.shown()


def test_crack_needs_the_zone():
    w = world()
    mid = M.CRACKS["long"]["mid"]
    yoke_at(w, mid + (0.03, 0.0), t.HEAD_YA)  # the pole line 30 mm past the crack: outside the zone
    bulb_at(w, mid + (0.015, 0.0), t.HEAD_BA)
    steps(w)
    assert not t._trial["found"]["long"]


def test_skewed_yoke_counts_no_direction():
    w = world()
    yoke_at(w, (0.24, M.WELD_Y), t.HEAD_YA + math.radians(35))
    bulb_at(w, (0.24, M.WELD_Y), t.HEAD_BA)
    steps(w)
    assert t.direction(w) is None and not t._trial["A"].any() and not t._trial["B"].any()


def test_nozzle_too_high_or_not_squeezed_gives_no_powder():
    w = world()
    yoke_at(w, (0.24, M.WELD_Y), t.HEAD_YA)
    bulb_at(w, (0.24, M.WELD_Y), t.HEAD_BA, height=0.08)
    steps(w)
    bulb_at(w, (0.24, M.WELD_Y), t.HEAD_BA, sq=M.SQUEEZE_ON - 0.001)
    steps(w)
    assert not t._trial["dust"].any() and not t._trial["A"].any()


def test_powder_lands_within_35mm():
    w = world()
    yoke_at(w, (0.24, M.WELD_Y), t.HEAD_YA)
    bulb_at(w, (0.24, M.WELD_Y), t.HEAD_BA)
    steps(w, 1)
    d = np.abs(t.CELLS[:, 0] - 0.24)
    assert (t._trial["dust"] == (d <= t.R_POWDER)).all()
    assert (t._trial["A"] == (d <= t.ZONE_N)).all()


def test_smearing_latches_zero():
    w = world()
    for k in range(10):  # slide the live yoke 2 mm a step along the weld
        yoke_at(w, (0.20 + 0.002 * k, M.WELD_Y), t.HEAD_YA)
        bulb_at(w, (0.20 + 0.002 * k, M.WELD_Y), t.HEAD_BA)
        steps(w, 1)
    assert t._trial["smear"]
    yoke_at(w, (0.22, M.WELD_Y), t.HEAD_YA, lift=0.02)
    steps(w)
    full_exam()
    assert finish(good_end()) == 0.0


def test_lift_between_placements_is_not_smearing():
    full_exam()
    assert not t._trial["smear"]


# -- pegs --------------------------------------------------------------------------------------------------------------
def test_peg_on_an_unfound_crack_is_stray():
    do_pass("A")
    do_pass("B")
    t._trial["found"]["trans"] = False  # as if the transverse crack never showed
    assert finish(good_end()) == 0.0


def test_pegs_set_before_the_exam_dont_mark():
    """Order is enforced: pegs set on the known crack spots first, then the weld examined, scores 0."""
    w = good_end()
    steps(w, 4)
    full_exam()
    assert finish(good_end()) == 0.0


def test_one_crack_marked_then_the_other_examined_is_fine():
    do_pass("A")
    found_first = [c for c, v in t._trial["found"].items() if v]
    assert len(found_first) == 1
    steps(good_end([M.CRACKS[found_first[0]]["mid"]]), 4)
    do_pass("B")
    assert finish(good_end()) == 1.0


def test_hand_near_a_peg_isnt_clear():
    full_exam()
    w = good_end()
    w["right_tcp"] = state(t._world(w["peg1"], M.PEG_HEAD_C) + (0.045, 0, 0))
    assert finish(w) == t.SCORE_TWO


def test_peg_still_in_hand_or_toppled_doesnt_mark():
    full_exam()
    w = good_end()
    w["right_tcp"] = state(t._world(w["peg1"], M.PEG_HEAD_C))
    assert finish(w) == 0.0
    w = good_end()
    w["peg1"].quat_xyzw = quat_tilt(math.radians(30), 0.0)
    assert finish(w) == 0.0


def test_pegs_left_in_tray_arent_stray():
    full_exam()
    w = place_pegs(world(), [M.CRACKS["long"]["mid"], M.CRACKS["trans"]["mid"]])
    assert t.marking(w)[1] is False and t.score(w) in (t.SCORE_TWO, 1.0)


# -- faults and reset --------------------------------------------------------------------------------------------------
def test_tool_off_table_is_zero():
    full_exam()
    w = good_end()
    w["yoke"] = state((0.5, 0.5, t.TOP - 0.2))
    assert finish(w) == 0.0


def test_reset_clears_everything():
    full_exam()
    t._trial["smear"] = True
    t.reset_latch()
    assert not t._trial["A"].any() and not t._trial["smear"] and not any(t._trial["found"].values())


def test_plan_examines_every_cell_with_margin():
    """Each placement's examined band (and the powder's reach) covers every cell with 5 mm or more to spare."""
    for places, half in ((t.PASS_A, t.ZONE_N), (t.PASS_B, min(t.ZONE_T, t.R_POWDER))):
        for x in t.CELLS[:, 0]:
            assert min(abs(x - c) for c in places) <= half - 0.005 + 1e-9


def test_task_json_text():
    j = json.load(open(os.path.join(HERE, "task.json")))
    assert len(j["fidelity"]) <= 500
    assert j["instruction"] in open(os.path.join(HERE, "task.py")).read()
    assert j["instruction"] in open(os.path.join(HERE, "README.md")).read()
    for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert j[k]
