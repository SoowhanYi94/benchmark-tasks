"""The score must credit each part in place from the collar out in the right order, then the nut threaded on and
seated, then the judge's braked spin and its tug, held with both hands clear; and must give nothing (or stop the
credit) for a part out of order, missing, tilted or short of home, the damaged wheel on the spindle, a wheel cracked by
overtightening (even if backed off again), a nut not threaded on, a hand or the lock still on, or a failed or
unfinished test. Also checks the inner flange's D key, and the nut's rising torque."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t
from wheel_mount import fixture as F

UP = np.array((0.0, 0.0, 0.0, 1.0))
FAR = ((0.16, 0.30, t.TOP + 0.2), (0.16, -0.30, t.TOP + 0.2))
GOOD_CRUSH = 120 * t.DEG


def state(p, q=UP):
    p = np.asarray(p, float)
    return SimpleNamespace(pos=p, quat_xyzw=np.asarray(q, float), low=p, high=p, lin_vel=np.zeros(3))


def q_about(axis, a):
    v = np.zeros(3)
    v["xyz".index(axis)] = math.sin(a / 2)
    return (*v, math.cos(a / 2))


def on_spindle(y, dz=-0.0015):
    return (t.X0, t.Y0 + y, t.TOP + F.ZA + dz)


def in_tray(name):
    return t._w(F.slot_pos(name))


def world(places=None, crush=GOOD_CRUSH, engaged=True, cracked=False, hands=FAR, button=0.0, judge=3,
          quats=None):
    """Everything done: the stack home, the nut 120 degrees past snug, hands clear, the judge passed. ``places``
    overrides parts' centres (world), ``quats`` their orientations."""
    pos = {n: on_spindle(F.HOME[n]) for n in F.ORDER}
    pos["decoy"] = in_tray("decoy")
    pos.update(places or {})
    quats = quats or {}
    w = {f"{n}_root": state(pos[n], quats.get(n, UP)) for n in F.PARTS}
    w.update({"spindle": state((0.0, button, 0.0)), "nut": state((0.0, F.WALL_Q if engaged else 0.0, 0.0)),
              "thread": state((0.0, 0.0, crush)), "latch": state((float(engaged), float(cracked), 0.0)),
              "judge": state((float(judge), 0.0, 0.0)), "left_tcp": state(hands[0]), "right_tcp": state(hands[1])})
    return w


def start():
    return world({n: in_tray(n) for n in F.ORDER}, crush=0.0, engaged=False, judge=0)


def test_start_scores_0():
    assert t.score(start()) == 0 and t.prefix(start()) == 0


def test_done_scores_1():
    w = world()
    assert t.score(w) == pytest.approx(1.0) and t.success(w) and t.judge_ready(w)


@pytest.mark.parametrize("k", range(6))
def test_each_part_in_order_adds_0_1(k):
    places = {n: in_tray(n) for n in F.ORDER[k:]}
    w = world(places, crush=0.0, engaged=False, judge=0)
    assert t.prefix(w) == k and t.score(w) == pytest.approx(0.1 * k)


def test_parts_on_but_not_pushed_home_score_only_up_to_the_gap():
    places = {"blotter_b": on_spindle(F.HOME["blotter_b"] - 0.011), "outer_flange": on_spindle(-0.048)}
    w = world(places, crush=0.0, engaged=False, judge=0)
    assert t.prefix(w) == 3 and t.score(w) == pytest.approx(0.3)


@pytest.mark.parametrize("dy", [-0.0015, 0.0015])
def test_home_tolerance_edges_count(dy):
    assert t.prefix(world({"wheel": on_spindle(F.HOME["wheel"] + dy)})) == 5


@pytest.mark.parametrize("dy", [-0.0016, 0.0016])
def test_beyond_home_tolerance_stops_the_count(dy):
    w = world({"wheel": on_spindle(F.HOME["wheel"] + dy)})
    assert t.prefix(w) == 2 and t.score(w) == pytest.approx(0.2)


def test_blotter_missing_caps_at_the_parts_before_it():
    # no first blotter: everything after it sits 2 mm closer to the collar
    places = {"blotter_a": in_tray("blotter_a")}
    for n in ("wheel", "blotter_b", "outer_flange"):
        places[n] = on_spindle(F.HOME[n] + 0.002)
    w = world(places)  # even with the nut tight, hands clear and a passed test
    assert t.prefix(w) == 1 and t.score(w) == pytest.approx(0.1) and not t.success(w)


def test_flanges_swapped_scores_0():
    w = world({"inner_flange": on_spindle(F.HOME["outer_flange"]), "outer_flange": on_spindle(F.HOME["inner_flange"])})
    assert t.prefix(w) == 0 and t.score(w) == 0


def test_blotter_and_wheel_swapped_cap():
    w = world({"blotter_a": on_spindle(F.HOME["wheel"]), "wheel": on_spindle(F.HOME["blotter_a"])})
    assert t.prefix(w) == 1 and t.score(w) == pytest.approx(0.1)


def test_the_two_blotters_are_interchangeable_only_by_name_position():
    # blotter_a and blotter_b are identical parts; either in either place must count
    w = world({"blotter_a": on_spindle(F.HOME["blotter_b"]), "blotter_b": on_spindle(F.HOME["blotter_a"])})
    assert t.score(w) == pytest.approx(1.0)


def test_damaged_wheel_on_the_spindle_scores_0():
    w = world({"wheel": in_tray("wheel"), "decoy": on_spindle(F.HOME["wheel"])})
    assert t.score(w) == 0 and not t.success(w)
    w = world({"decoy": on_spindle(-0.050)})  # even as an extra, the good wheel in place
    assert t.score(w) == 0


def test_damaged_wheel_off_the_spindle_but_held_near_it_is_not_on():
    w = world({"decoy": (t.X0, t.Y0 - 0.080, t.TOP + F.ZA)})  # beyond the nose's end
    assert not t.decoy_on(w) and t.score(w) == pytest.approx(1.0)


def test_part_beside_the_spindle_is_not_on_it():
    w = world({"wheel": (t.X0 + 0.004, t.Y0 + F.HOME["wheel"], t.TOP + F.ZA)})
    assert t.prefix(w) == 2


@pytest.mark.parametrize("name,deg,ok", [("wheel", 4.9, True), ("wheel", 5.1, False), ("blotter_a", 7.9, True),
                                         ("blotter_a", 8.1, False)])
def test_tilt_limit(name, deg, ok):
    w = world(quats={name: q_about("x", math.radians(deg))})
    assert (t.prefix(w) == 5) == ok


def test_turn_about_the_spindle_is_not_tilt():
    w = world(quats={n: q_about("y", 2.0) for n in F.ORDER if n != "inner_flange"})  # the keyed flange has its own test
    assert t.score(w) == pytest.approx(1.0)


def test_nut_not_threaded_on_scores_only_the_stack():
    w = world(engaged=False, crush=0.0, judge=0)
    assert t.score(w) == pytest.approx(0.5)


def test_nut_threaded_but_short_of_the_flange():
    assert t.score(world(crush=0.0, judge=0)) == pytest.approx(0.5)


def test_crush_without_the_nut_threaded_on_counts_nothing():
    assert t.score(world(engaged=False, judge=0)) == pytest.approx(0.5)


@pytest.mark.parametrize("judge,expect", [(0, 0.6), (1, 0.6), (4, 0.6), (2, 0.8), (5, 0.8), (3, 1.0)])
def test_judge_outcomes(judge, expect):
    # 4: a loose nut lets the wheel slip in the braked spin; 5: the clamp gives in the tug
    w = world(judge=judge)
    assert t.judge_ready(w) and t.score(w) == pytest.approx(expect) and t.success(w) == (judge == 3)


@pytest.mark.parametrize("crush", [t.SEATED, 60 * t.DEG, 239 * t.DEG])
def test_any_seated_uncracked_nut_is_judged(crush):
    # tightness is judged by the spin and the tug, not by a count of degrees
    assert t.judge_ready(world(crush=crush)) and t.score(world(crush=crush)) == pytest.approx(1.0)


def test_crush_below_seated_is_not_seated():
    w = world(crush=t.SEATED * 0.9)
    assert not t.seated(w) and not t.judge_ready(w) and t.score(w) == pytest.approx(0.5)


@pytest.mark.parametrize("crush", [GOOD_CRUSH, 0.0])
def test_cracked_wheel_scores_0_even_backed_off(crush):
    assert t.score(world(crush=crush, cracked=True)) == 0


def test_partial_runs_never_beat_a_correct_one():
    assert t.score(world(cracked=True)) < t.score(world({"outer_flange": in_tray("outer_flange")}, judge=0))


def test_a_passed_judge_needs_judge_ready_still():
    # the test passed, then the damaged wheel was hung on as well, or the nut cracked the wheel
    assert t.score(world(judge=3, places={"decoy": on_spindle(-0.050)})) == 0
    assert t.score(world(judge=3, cracked=True)) == 0


@pytest.mark.parametrize("hand", [
    (t.X0, t.Y0 - 0.05, t.TOP + F.ZA + 0.06),  # on a spoke
    (t.X0 - 0.05, t.Y0 - 0.02, t.TOP + F.ZA + 0.05),  # at the wheel's top
    (t.X0, t.Y0 - 0.06, t.TOP + F.ZA + 0.079),  # just inside 8 cm of the nose
])
def test_hand_on_the_stack_or_nut(hand):
    w = world(hands=(FAR[0], hand))
    assert not t.hands_clear(w) and t.score(w) == pytest.approx(0.6) and not t.success(w)


def test_hand_on_the_lock_button():
    btn = (t.X0 + F.BUTTON_XY[0], t.Y0 + F.BUTTON_XY[1], t.TOP + F.BUTTON_Z0 + F.BUTTON_H + 0.01)
    w = world(hands=(btn, FAR[1]), button=0.003)
    assert not t.hands_clear(w) and t.score(w) == pytest.approx(0.6)


def test_button_held_down_without_a_hand_blocks_the_test():
    w = world(button=0.001)
    assert not t.judge_ready(w) and t.score(w) == pytest.approx(0.6)


def test_hands_8_cm_out_are_clear():
    w = world(hands=(FAR[0], (t.X0, t.Y0 - 0.03, t.TOP + F.ZA + 0.081)))
    assert t.hands_clear(w) and t.score(w) == pytest.approx(1.0)


def test_scores_are_one_of_the_documented_levels():
    levels = {round(v, 2) for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)}
    for w in (start(), world(), world(crush=0.0), world(crush=30 * t.DEG, judge=4), world(judge=5),
              world(cracked=True), world(judge=1)):
        assert round(t.score(w), 2) in levels


def test_demo_plan_reaches_every_step():
    wps = t.demo()
    labels = [w.label for w in wps["right"]]
    assert sum("slide on" == lab for lab in labels) == 5 and any("thread" in lab for lab in labels)
    assert any("press it down" == w.label for w in wps["left"])
    assert sum(w.duration for w in wps["right"]) * 20 < 3000


def test_inner_flange_keys_only_with_its_flat_on_the_spindle_flat():
    assert F.FLAT_Z < F.D_CHORD < F.SPINDLE_R

    def fits(deg):  # can the spindle's section sit inside the D bore, at any offset the clearance allows?
        a = math.radians(deg)
        pts = np.array([(x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a))
                        for x, z in F.spindle_section(360)])
        for dx in np.arange(-0.002, 0.002, 0.0001):
            for dz in np.arange(-0.002, 0.002, 0.0001):
                p = pts + (dx, dz)
                if np.all(np.hypot(p[:, 0], p[:, 1]) <= F.BORE_R) and np.all(p[:, 1] <= F.D_CHORD):
                    return True
        return False

    assert fits(0) and fits(5) and fits(-5)  # slides on lined up (and a little off)
    for deg in (90, -90):  # turned sideways, the spindle's 12 mm doesn't fit under the 11.9 mm D
        assert not fits(deg)
    # the D bore's segments clear the spindle when lined up: every inner vertex outside the spindle's section
    for seg in F.ring_segments("inner_flange"):
        for x, _, z in seg[[0, 3]]:
            assert z >= F.FLAT_Z or math.hypot(x, z) > F.SPINDLE_R
            assert z <= F.D_CHORD + 1e-9


def test_round_bores_do_not_key():
    for seg in F.ring_segments("outer_flange"):
        for x, _, z in seg[[0, 3]]:
            assert math.hypot(x, z) >= F.BORE_R - 1e-9


def test_thread_start_is_fixed():
    assert F.RUN_DOWN == pytest.approx(0.00181) and t.TURN > F.RUN_DOWN / F.PITCH * 360


def test_nut_gets_harder_to_turn_as_it_squeezes_the_stack():
    free = F.nut_hold(0.0)
    assert F.nut_hold(-0.001) == free
    assert F.nut_hold(60 * t.DEG) > 1.7 * free and F.nut_hold(120 * t.DEG) > 2.5 * free
    assert F.nut_hold(240 * t.DEG) < 0.6  # still within a hand's turn on the spoke


def test_crack_leaves_room_past_the_demo():
    assert t.TURN - F.RUN_DOWN / F.PITCH * 360 < math.degrees(t.CRACK_TURN) - 60


def test_decoy_has_its_chip_and_no_tag():
    assert not hasattr(F, "TAG")
    radii = [max(math.hypot(x, z) for x, _, z in seg) for seg in F.ring_segments("decoy")]
    assert min(radii) < F.PARTS["decoy"][0] - 0.012


def test_inner_flange_flat_down_is_not_keyed():
    w = world()
    assert t.keyed(w)
    w["inner_flange_root"] = state(w["inner_flange_root"].pos, (0.0, 1.0, 0.0, 0.0))  # turned 180 degrees about y
    assert not t.keyed(w) and t.prefix(w) == 0


def test_inner_flange_stays_keyed_as_the_spindle_turns_with_it():
    w = world()
    th = 1.5
    w["spindle"] = state((th, 0.0, 0.0))
    w["inner_flange_root"] = state(w["inner_flange_root"].pos, (0.0, math.sin(th / 2), 0.0, math.cos(th / 2)))
    assert t.keyed(w)


def test_damaged_wheel_tried_stays_zero():
    w = world()
    w["latch"] = state((1.0, 0.0, 1.0))
    assert t.score(w) == 0.0 and not t.judge_ready(w)


def test_a_part_still_held_or_moving_does_not_fill_its_place():
    w = world()
    w["right_tcp"] = state(w["outer_flange_root"].pos)
    assert not t.in_place(w, "outer_flange") and t.prefix(w) < 5
    w = world()
    w["outer_flange_root"].lin_vel = np.array((0.0, 0.5, 0.0))
    assert t.prefix(w) == 5  # the nut is on: a part may wobble as the judge spins it
    w = world(engaged=False)
    w["outer_flange_root"].lin_vel = np.array((0.0, 0.5, 0.0))
    assert t.prefix(w) == 4


def test_seated_needs_the_stack_home():
    w = world()
    p = w["outer_flange_root"].pos.copy()
    p[1] -= 0.001
    w["outer_flange_root"].pos = p
    assert not t.seated(w)
