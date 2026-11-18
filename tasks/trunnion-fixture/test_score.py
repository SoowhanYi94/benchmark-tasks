"""Scoring and screw-rule checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t

M = t.M
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}
OVER = math.pi


def quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array((x, y, z, w))


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))


def state(pos, R=np.eye(3), lin=(0, 0, 0), ang=(0, 0, 0)):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=quat(R), lin_vel=np.array(lin, float), ang_vel=np.array(ang, float), low=p, high=p)


def world(roll=OVER, bar=True, bolt=True, flange=True, depth=None, gb_w=None):
    """The scene with the gearbox in the cradle at this roll; the flange seated on face J (if turned) or on its post; the screws at
    `depth` mm run down from the start (None: at the start; 9: seated)."""
    w = {a: state(p) for a, p in PARK.items()}
    w["fixture"] = state((roll, M.BAR["travel"] if bar else 0.0, M.BOLT["travel"] if bolt else 0.0))
    Rc = Ry(roll)
    gw = (0.0 if roll < math.pi / 2 else 0.004) if gb_w is None else gb_w
    w["gearbox"] = state(t._axle() + Rc @ np.array((M.GB_UV[0], M.GB_UV[1], gw)), Rc)
    if flange:
        Rf = Rc @ Ry(math.pi)
        fp = w["gearbox"].pos + Rc @ np.array((0.0, 0.0, -M.GB["face"]))
        w["flange"] = state(fp, Rf)
    else:
        Rf = np.eye(3)
        w["flange"] = state(t.FL_START)
    run = 0.0 if depth is None else depth / 1000
    for i in range(4):
        o = np.array(M.screw_offset(i)) - (0.0, 0.0, run)
        w[f"screw{i}"] = state(w["flange"].pos + Rf @ o, Rf)
    w["driver"] = state(t.DRV_START)
    return w


def drive(w, i, angles):
    """The driver standing square in screw i's socket, at these angles about its axis (deg), one control step each."""
    s = w[f"screw{i}"]
    R = t._rot(s.quat_xyzw)
    floor = s.pos + R @ np.array((0.0, 0.0, M.SC["head_h"] - M.SC["socket_d"]))
    for a in angles:
        w["driver"] = state(floor, R @ Rz(math.radians(a)))
        t.note_step(w, 0.05)
    w["driver"] = state(t.DRV_START)


def seat_all(w, order=range(4)):
    """Catch each screw and run it down by the rule (the test moves the screw as the thread joint would)."""
    for i in order:
        drive(w, i, [0, -10])
        s = t._trial["screws"][i]
        assert s["on"]
        for k in range(1, 9):
            drive(w, i, [-10 - 90 * (k - 1), -10 - 90 * k])
        w[f"screw{i}"] = world(depth=9)[f"screw{i}"]
        drive(w, i, [-730, -750, -770])  # turned on 40 deg against the snug torque
    t.take_actions()


def finish(w, n=60):
    """Rest long enough for the lift test (scored as if the flange holds) and the 1 s after it."""
    for _ in range(n):
        t.note_step(w, 0.05)
    return t.score(w)


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def test_start_scores_zero():
    assert finish(world(0.0, bar=False, bolt=False, flange=False), 5) == 0.0


def test_retained():
    assert finish(world(0.0, bolt=False, flange=False), 5) == 0.25


def test_turned_without_bolt_stays_at_retained():
    assert finish(world(bolt=False, flange=False), 5) == 0.25


def test_turned_and_locked():
    assert finish(world(flange=False), 5) == 0.5


def test_flange_seated():
    assert finish(world(), 5) == 0.75


def test_flange_held_does_not_count():
    w = world()
    w["left_tcp"] = state(w["flange"].pos + (0.0, 0.0, t.FL_TCP))
    assert finish(w, 5) == 0.5


def test_flange_proud_on_its_screw_tips_does_not_count():
    w = world()
    w["flange"].pos = w["flange"].pos + (0.0, 0.0, M.SC["proud"])
    assert not t.flange_on(w) and finish(w, 5) == 0.5


def test_flange_off_centre_does_not_count():
    w = world()
    w["flange"].pos = w["flange"].pos + (0.0015, 0.0, 0.0)
    assert not t.flange_on(w)


def test_full_marks_after_the_lift_test():
    w = world()
    seat_all(w)
    sc = finish(w)
    assert t._trial["lift_ok"] and sc == 1.0


def test_lift_test_runs_and_force_is_applied():
    w = world()
    seat_all(w)
    forces = []
    for _ in range(60):
        t.note_step(w, 0.05)
        forces.append(t.lift_force())
    assert max(forces) == t.LIFT_N and forces[-1] == 0.0


def test_lift_that_raises_the_flange_fails():
    w = world()
    seat_all(w)
    for _ in range(25):  # rest 1 s, the test starts
        t.note_step(w, 0.05)
    assert t._trial["lift"] is not None
    w["flange"] = state(w["flange"].pos + (0.0, 0.0, 0.001), t._rot(w["flange"].quat_xyzw))
    t.note_step(w, 0.05)
    while t._trial["lift"] is not None:
        t.note_step(w, 0.05)
    assert not t._trial["lift_ok"] and t.score(w) < 1.0


def test_two_diagonal_screws_score_09():
    w = world()
    seat_all(w, (0, 2))
    assert finish(w, 5) == 0.9


def test_two_adjacent_screws_score_075():
    w = world()
    seat_all(w, (0, 1))
    assert finish(w, 5) == 0.75


def test_screws_not_run_score_075():
    assert finish(world(), 40) == 0.75


def test_needs_a_second_of_rest_and_the_test():
    w = world()
    seat_all(w)
    assert finish(w, 5) == 0.9


def test_screw_run_with_the_bolt_out_is_not_credited():
    w = world(bolt=False)
    seat_all(w)
    w["fixture"] = state((OVER, M.BAR["travel"], M.BOLT["travel"]))  # bolted afterwards
    assert all(s["unlocked"] for s in t._trial["screws"]) and finish(w) == 0.75


def test_screw_does_not_catch_outside_a_hole():
    w = world(flange=False)  # the flange on its post: no hole under the tips
    drive(w, 0, [0, -10, -40])
    assert not t._trial["screws"][0]["on"] and not t.take_actions()


def test_screw_does_not_catch_on_a_loosening_turn():
    w = world()
    drive(w, 0, [0, 10, 40])
    assert not t._trial["screws"][0]["on"]


def test_catch_switches_glue_off_and_thread_on_at_the_depth():
    w = world()
    drive(w, 0, [0, -10])
    acts = t.take_actions()
    assert ("glue", 0, False) in acts
    th = [a for a in acts if a[0] == "thread"][0]
    assert th[3] is True and abs(th[4] + M.SC["proud"]) < 1e-6


def test_thread_runs_pitch_per_turn_and_stops_at_the_seat():
    w = world()
    drive(w, 0, [0, -10])
    t.take_actions()
    drive(w, 0, [-10, -100, -190, -280, -370])
    s = t._trial["screws"][0]
    assert abs(s["target"] - (-M.SC["proud"] - t.PITCH)) < 1e-6
    drive(w, 0, [-370 - 30 * k for k in range(1, 110)])
    assert abs(s["target"] - (t.SEAT_TRAVEL - t.OVERRUN)) < 1e-9


def test_driver_merely_near_does_not_turn():
    w = world()
    s = w["screw0"]
    R = t._rot(s.quat_xyzw)
    floor = s.pos + R @ np.array((0.0, 0.0, M.SC["head_h"] - M.SC["socket_d"]))
    for a, off in ((0, 0.003), (-20, 0.003), (-60, 0.003)):
        w["driver"] = state(floor + (off, 0.0, 0.0), R @ Rz(math.radians(a)))
        t.note_step(w, 0.05)
    for a in (0, -20, -60):  # above the socket
        w["driver"] = state(floor + (0.0, 0.0, 0.003), R @ Rz(math.radians(a)))
        t.note_step(w, 0.05)
    assert not t._trial["screws"][0]["on"]


def test_driver_tilted_does_not_turn():
    w = world()
    s = w["screw0"]
    R = t._rot(s.quat_xyzw)
    floor = s.pos + R @ np.array((0.0, 0.0, M.SC["head_h"] - M.SC["socket_d"]))
    for a in (0, -20, -60):
        w["driver"] = state(floor, R @ Ry(math.radians(12)) @ Rz(math.radians(a)))
        t.note_step(w, 0.05)
    assert not t._trial["screws"][0]["on"]


def test_turning_with_the_bar_open_is_zero_for_good():
    t.note_step(world(math.radians(40), bar=False, bolt=False, flange=False), 0.05)
    assert t._trial["bad"] and finish(world()) == 0.0


def test_small_lift_without_the_bar_is_not_turning():
    t.note_step(world(math.radians(20), bar=False, bolt=False, flange=False), 0.05)
    assert not t._trial["bad"]


def test_gearbox_lost_is_zero_for_good():
    w = world(flange=False)
    w["gearbox"] = state((0.2, 0.0, t.TOP - 0.02))
    t.note_step(w, 0.05)
    assert t._trial["bad"] and finish(world()) == 0.0


def test_flange_off_the_table_is_zero_for_good():
    w = world(flange=False)
    w["flange"] = state((0.2, 0.9, 0.3))
    t.note_step(w, 0.05)
    assert t._trial["bad"]


def test_gearbox_resting_on_the_bar_counts_and_one_out_of_place_does_not():
    assert t.gb_in(world(gb_w=0.004)) and not t.gb_in(world(gb_w=0.009))
    w = world()
    w["gearbox"].pos = w["gearbox"].pos + (0.007, 0.0, 0.0)
    assert not t.gb_in(w)


def test_flange_on_face_k_earns_nothing_for_the_flange():
    w = world(0.0, bolt=False)
    Rc = np.eye(3)
    w["flange"] = state(w["gearbox"].pos + (0.0, 0.0, M.BAR["w1"]), Rc)  # on the closed bar over face K, the cradle level
    assert not t.flange_on(w) and finish(w, 5) == 0.25


def test_hand_on_the_knob_or_the_bolt_or_driver_stops_short():
    for spot in ("knob", "bolt", "driver"):
        t.reset_latch()
        w = world()
        seat_all(w)
        p = {"knob": t.knob_at(OVER, 0.0), "bolt": t._bolt_lug(w), "driver": w["driver"].pos + (0.0, 0.0, 0.05)}[spot]
        w["right_tcp"] = state(p)
        assert finish(w) == 0.9, spot


def test_moving_parts_stop_short():
    w = world()
    seat_all(w)
    w["gearbox"].ang_vel = np.array((1.0, 0.0, 0.0))
    assert finish(w) == 0.9


def test_bolt_pulled_back_loses_credit():
    w = world()
    seat_all(w)
    w["fixture"] = state((OVER, M.BAR["travel"], 0.0))
    assert finish(w) == 0.25


def test_text_numbers_match_the_code():
    assert t.LIFT_N == 10.0 and t.SEAT_GAP == 0.0005 and t.PITCH == 0.001 and t.STILL_S == 1.0
    assert t.T_RUN == 0.15 and t.T_SNUG == 0.5 and abs(t.SNUG_TURN - math.radians(15)) < 1e-12 and t.RUN_STEPS == 9600
    assert abs(t.SEAT_TRAVEL + 0.010) < 1e-12 and t.BOLT_HOME == 0.012 and abs(t.OVER - math.radians(6)) < 1e-12


def test_each_screw_takes_its_real_turns():
    """Tip 1 mm into the hole's mouth, head 9 mm over the flange at the start; 10 mm of thread in the 11 mm hole when seated: 9 turns of
    M6 x 1. The demo turns 18 x 210 deg = 10.5 turns."""
    run = -t.SEAT_TRAVEL - M.SC["proud"]
    assert abs(M.SC["proud"] - 0.001) < 1e-12 and abs(run / t.PITCH - 9) < 1e-9 and abs(-t.SEAT_TRAVEL - 0.010) < 1e-12
    assert abs(M.GB["hole_depth"] - 0.011) < 1e-12
    assert t.STROKES == 18 and t.STROKES * (t.STROKE_HI - t.STROKE_LO) / (2 * math.pi) >= (run + t.OVERRUN) / t.PITCH + 0.5


def test_seated_but_not_snug_is_not_down():
    w = world()
    drive(w, 0, [0, -10])
    w["screw0"] = world(depth=9)["screw0"]
    drive(w, 0, [-10, -15])  # only 5 deg on once seated
    s = t._trial["screws"][0]
    assert t.head_down(w, 0) and not t.screw_seated(w, 0)
    drive(w, 0, [-15, -25, -35])
    assert t.screw_seated(w, 0) and abs(s["snug"] - math.radians(25)) < 1e-6


def test_thread_resists_the_driver_both_ways():
    w = world()
    drive(w, 0, [0, -10])
    s = w["screw0"]
    R = t._rot(s.quat_xyzw)
    ax = R[:, 2]
    floor = s.pos + R @ np.array((0.0, 0.0, M.SC["head_h"] - M.SC["socket_d"]))
    for sign in (1, -1):
        w["driver"] = state(floor, R, ang=sign * 2.0 * ax)
        tq = t.driver_torque(w)
        assert abs(tq @ ax + sign * t.T_RUN) < 1e-9
    w["driver"] = state(floor, R, ang=0.0 * ax)
    assert np.allclose(t.driver_torque(w), 0.0)
    w["screw0"] = world(depth=9)["screw0"]
    s = w["screw0"]
    floor = s.pos + R @ np.array((0.0, 0.0, M.SC["head_h"] - M.SC["socket_d"]))
    w["driver"] = state(floor, R, ang=2.0 * ax)
    assert abs(t.driver_torque(w) @ ax + t.T_SNUG) < 1e-9
    w["driver"] = state(t.DRV_START, ang=2.0 * ax)
    assert np.allclose(t.driver_torque(w), 0.0)


def test_masses_are_the_stated_ones():
    assert M.GB["mass"] == 1.2 and M.FL["mass"] == 0.13 and M.SC["mass"] == 0.006 and M.DRV["mass"] == 0.037


def test_plan_fits_and_reaches():
    wps = t.demo()
    assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < t.RUN_STEPS
    assert not t.ik_errors(wps)[2]


def test_flange_set_down_unturned_does_not_seat():
    """Turned 35 deg as it starts on its post, its screw tips rest on face J and it stands 1 mm proud: no credit for the flange."""
    w = world()
    Rf = t._rot(w["flange"].quat_xyzw) @ Rz(t.FL_YAW)
    w["flange"] = state(w["flange"].pos + t._rot(w["flange"].quat_xyzw) @ np.array((0.0, 0.0, M.SC["proud"])), Rf)
    assert not t.flange_on(w) and finish(w, 5) == 0.5


def test_flange_starts_turned_and_its_screws_with_it():
    assert abs(t.FL_YAW - math.radians(35)) < 1e-12
    for i, s in enumerate(t.SCREWS):
        off = np.subtract(t.START[s], t.FL_START)
        assert abs(math.atan2(off[1], off[0]) - math.atan2(M.HOLES[i][1], M.HOLES[i][0]) - t.FL_YAW) < 1e-9
        assert t.START_Q[s] == t.START_Q["flange"]


def test_run_limit_is_9600_steps():
    import json
    assert t.RUN_STEPS == 9600 and json.load(open("task.json"))["run_limit"].startswith("9,600 steps")


def test_catch_needs_the_tip_inside_the_drawn_hole_mouth():
    for (dx, dy), ok in (((0.0019, 0.0019), True), ((0.0021, 0.0), False), ((0.0, -0.0021), False)):
        t.reset_latch()
        w = world()
        Rg = t._rot(w["gearbox"].quat_xyzw)
        w["screw0"].pos = w["screw0"].pos + Rg @ np.array((dx, dy, 0.0))
        drive(w, 0, [0, -10])
        assert t._trial["screws"][0]["on"] is ok
