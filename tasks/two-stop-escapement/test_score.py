"""The scorer against every outcome in the scoring rules, the geometry the mechanism relies on, and blind plans across
all the lots. No simulator."""

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t
from two_stop_escapement import mech as M

PARKED = {"left_tcp": np.array((0.18, 0.32, t.TOP + 0.18)), "right_tcp": np.array((0.18, -0.32, t.TOP + 0.18))}
BIN_AT = np.array((t.BIN["center"][0], t.BIN["center"][1] - 0.05, t.TOP + 0.01))


def obj(pos, moving=False):
    return SimpleNamespace(pos=np.asarray(pos, float), resting=lambda: not moving)


def joints(pin=0.0, lever=0.0):
    return SimpleNamespace(pos=np.array((pin, lever, 0.0)))


def state(n, red_at=None, **kw):
    """``n`` parts in the bin and the rest queued up against the lower stop, all settled; ``kw`` replaces any object."""
    s = {**{k: obj(v) for k, v in PARKED.items()}}
    for i, name in enumerate(t.NAMES):
        s[name] = obj(BIN_AT + (0, 0.01 * i, 0)) if i < n else obj(t.world(M.part_y(i - n), 0.0005))
    s = {**s, "stop_joints": joints(), **kw}
    t.reset_latch()
    t.note_step({**s, "stop_joints": joints(pin=-0.015)}, 0.05)  # the parts went by with the pin down
    t.note_step(s, 1.0)
    return s


@pytest.fixture(params=list(t.LOTS))
def lot(request, monkeypatch):
    monkeypatch.setenv("ESCAPEMENT_LOT", request.param)
    importlib.reload(t)
    yield request.param
    monkeypatch.delenv("ESCAPEMENT_LOT")
    importlib.reload(t)


def test_lots_put_red_second_to_fifth():
    assert sorted(t.LOTS.values()) == [1, 2, 3, 4]


def test_exact_release_scores_full(lot):
    assert t.score(state(t.AHEAD)) == 1.0


def test_part_going_by_without_the_pin_down_scores_zero(lot):
    s = state(t.AHEAD)
    t.reset_latch()
    t.note_step(s, 1.0)  # pin up as they went by
    assert t._trial["bad"] and t.score(s) == 0.0


def test_red_part_put_back_stays_out(lot):
    s = state(t.AHEAD)
    gone = {**s, t.RED_NAME: obj(BIN_AT)}
    t.note_step({**gone, "stop_joints": joints(pin=-0.015)}, 0.05)
    t.note_step(s, 1.0)
    assert t.red_held(s) and t.score(s) == 0.0


def test_reset_clears_the_latches(lot):
    s = state(t.AHEAD)
    t.reset_latch()
    t.note_step(s, 1.0)
    assert t._trial["bad"]
    assert t.score(state(t.AHEAD)) == 1.0


@pytest.mark.parametrize("pin,lever", [(-0.01, 0.0), (0.0, 5.0)])
def test_a_stop_still_pressed_is_not_full_score(lot, pin, lever):
    s = state(t.AHEAD, stop_joints=joints(pin, lever))
    assert t.score(s) < 1.0


def test_a_part_gripped_over_the_bin_is_not_full_score(lot):
    s = state(t.AHEAD)
    s["left_tcp"] = obj(s[t.NAMES[0]].pos)
    assert t.score(s) < 1.0


def test_part_at_the_chute_exit_is_not_in_the_bin():
    assert not t.in_bin(obj(t.world(-0.006, 0.0)))


def test_too_many_with_red_held_scores_zero(monkeypatch):
    monkeypatch.setenv("ESCAPEMENT_LOT", "A")
    importlib.reload(t)
    s = state(1)
    s[t.NAMES[2]] = obj(BIN_AT)
    t._trial["red_out"] = False
    assert t.released(s) == 2 and t.score(s) == 0.0
    monkeypatch.delenv("ESCAPEMENT_LOT")
    importlib.reload(t)


def test_only_the_pin_fits_the_gap():
    gap = M.PITCH - (M.BODY[1][1] - M.BODY[0][1])
    assert M.UPPER_T < gap < M.LOWER_T


def test_no_early_end_for_a_blind_one_at_a_time_plan():
    assert not hasattr(t, "success")


def test_one_still_frame_is_not_enough():
    s = state(t.AHEAD)
    t._trial["still"] = 0.05
    assert t.score(s) < 1.0


def test_nothing_done_scores_zero(lot):
    assert t.score(state(0)) == 0.0


def test_red_released_scores_zero(lot):
    assert t.score(state(t.AHEAD + 1)) == 0.0


def test_whole_queue_dumped_scores_zero(lot):
    assert t.score(state(6)) == 0.0


def test_one_short_earns_only_with_a_part_out(lot):
    want = t.ONE_SHORT if t.AHEAD >= 2 else 0.0
    assert t.score(state(t.AHEAD - 1)) == want


def test_more_than_one_short_scores_zero():
    assert t.AHEAD == 1 or t.score(state(0)) == 0.0


def test_two_short_scores_zero(monkeypatch):
    monkeypatch.setenv("ESCAPEMENT_LOT", "D")
    importlib.reload(t)
    assert t.score(state(2)) == 0.0
    monkeypatch.delenv("ESCAPEMENT_LOT")
    importlib.reload(t)


@pytest.mark.parametrize("arm", ["left_tcp", "right_tcp"])
@pytest.mark.parametrize("knob", ["U", "L"])
def test_a_hand_on_a_knob_is_not_full_score(lot, arm, knob):
    k = t.KNOB_U_WORLD if knob == "U" else t.KNOB_L_WORLD
    assert t.score(state(t.AHEAD, **{arm: obj(k + (0, 0, 0.05))})) == t.ONE_SHORT


def test_hand_just_clear_counts(lot):
    assert t.score(state(t.AHEAD, left_tcp=obj(t.KNOB_U_WORLD + (0, 0, t.HANDS_CLEAR + 0.001)))) == 1.0


def test_a_moving_part_is_not_released(lot):
    s = state(t.AHEAD)
    s[t.NAMES[0]] = obj(s[t.NAMES[0]].pos, moving=True)
    t.note_step(s, 0.05)
    assert t.score(s) < 1.0


def test_a_moving_red_part_is_not_full_score(lot):
    s = state(t.AHEAD)
    s[t.RED_NAME] = obj(s[t.RED_NAME].pos, moving=True)
    t.note_step(s, 0.05)
    assert t.score(s) < 1.0


def test_part_on_the_table_is_not_released(lot):
    s = state(t.AHEAD)
    s[t.NAMES[0]] = obj((t.BIN["center"][0] + 0.2, t.BIN["center"][1], t.TOP + 0.01))
    assert t.score(s) < 1.0


def test_part_on_the_floor_is_not_released(lot):
    s = state(t.AHEAD)
    s[t.NAMES[0]] = obj((t.BIN["center"][0], t.BIN["center"][1], 0.02))
    assert t.released(s) == t.AHEAD - 1 and not t.in_bin(s[t.NAMES[0]])


def test_red_past_the_stop_is_out(lot):
    s = state(t.AHEAD, **{t.RED_NAME: obj(t.world(M.STOP_FACE + 0.01, 0.0005))})
    assert not t.red_held(s) and t.score(s) == 0.0


def test_red_off_the_chute_is_out(lot):
    s = state(t.AHEAD, **{t.RED_NAME: obj((0.5, 0.3, t.TOP))})
    assert t.score(s) == 0.0


def test_red_anywhere_along_the_queue_is_held():
    assert all(t.red_held({t.RED_NAME: obj(t.world(M.part_y(i), 0.0005))}) for i in range(6))


# -- a plan that never reads the queue ---------------------------------------------------------------------------------
def _fixed_plan_mean(m):
    """The mean over the lots of releasing ``m`` parts every time."""
    total = 0.0
    for ahead in t.LOTS.values():
        if m > ahead:
            continue  # the red part went out
        total += 1.0 if m == ahead else t.ONE_SHORT if m >= 1 and m >= ahead - 1 else 0.0
    return total / len(t.LOTS)


def test_best_blind_plan_is_low():
    best = max(_fixed_plan_mean(m) for m in range(7))
    assert best == pytest.approx((1 + t.ONE_SHORT) / 4) and best < 0.3


# -- the mechanism's geometry ------------------------------------------------------------------------------------------
def test_blocks_fit_the_chute_and_cover():
    assert M.BODY[1][0] < M.INNER_HALF and M.BODY[1][2] < M.COVER_Z


def test_a_part_cannot_come_out_through_the_slot():
    assert M.SLOT_HALF < M.BODY[1][0]


def test_stops_fit_the_slot():
    assert M.STOP_W / 2 < M.SLOT_HALF


def test_noses_pass_under_the_closed_stops_and_blocks_do_not():
    assert M.NOSE[1][2] < M.TIP_Z < M.BODY[1][2]


def test_pitch_leaves_a_gap_the_pin_fits():
    gap = M.PITCH - (M.BODY[1][1] - M.BODY[0][1])
    assert gap - M.UPPER_T == pytest.approx(0.003)
    rear, front = M.part_y(0) + M.BODY[1][1], M.part_y(1) + M.BODY[0][1]
    assert rear < M.UPPER_Y - M.UPPER_T / 2 and M.UPPER_Y + M.UPPER_T / 2 < front


def test_pin_at_rest_clears_the_blocks():
    assert M.UPPER_REST_TIP > M.BODY[1][2]


def test_open_lower_stop_clears_the_blocks_and_barely_moves_uphill():
    a = math.radians(M.LEVER_MAX)
    dy, dz = M.STOP_FACE - M.PIVOT[0], M.TIP_Z - M.PIVOT[1]
    y = dy * math.cos(a) - dz * math.sin(a)
    z = dy * math.sin(a) + dz * math.cos(a)
    assert M.PIVOT[1] + z > M.BODY[1][2] + 0.002  # lifts clear of a 25 mm block
    assert y - dy < 0.001  # at most 1 mm uphill: it can't jam the first block into the held queue


def test_knob_pads_are_clear_of_the_chute_and_the_parts():
    assert M.KNOB_X - M.KNOB_H / 2 > M.INNER_HALF + M.WALL


def test_queue_fits_the_chute():
    assert M.part_y(5) + M.BODY[1][1] < M.LENGTH


def test_slope_beats_friction():
    assert math.tan(M.TILT) > 2 * M.MU_SLIDE * 0.7


def test_chute_exit_clears_the_bin_wall():
    assert t.CHUTE_LOW[2] - M.WALL > t.TOP + t.BIN["height"]


def test_four_parts_fit_past_the_apron_in_a_line():
    assert t.BIN["inner"][1] - t.BIN["apron_run"] + M.PITCH > 4 * M.PITCH


def test_apron_starts_under_the_exit_and_below_it():
    assert t.CHUTE_LOW[2] - M.WALL > t.TOP + t.BIN["apron_top"]
    assert t.BIN["center"][1] + t.BIN["inner"][1] / 2 >= t.CHUTE_LOW[1] - 0.006


# -- the demo ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(t.LOTS))
def test_demo_fits_run_limit_and_ik_reaches_every_waypoint(name):
    wps = t.demo(name)
    assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 2400
    pe, re = t.ik_errors(wps)
    assert pe < 1e-3 and re < 1e-2


@pytest.mark.parametrize("name", list(t.LOTS))
def test_demo_presses_as_many_cycles_as_parts_ahead(name):
    labels = [w.label for w in t.demo(name)["right"]]
    assert labels.count("open the lower stop") == t.LOTS[name]


def test_demo_never_opens_the_lower_stop_without_the_upper_held():
    """Walk both arms' timelines: at each lower-stop press the left arm must already be down on the upper stop."""
    wps = t.demo("D")
    def timeline(arm):
        out, now = [], 0.0
        for w in wps[arm]:
            out.append((now, now + w.duration, w.label))
            now += w.duration
        return out
    left = timeline("left")
    for start, _, label in timeline("right"):
        if label != "open the lower stop":
            continue
        at = next(lab for s, e, lab in left if s <= start + 1e-9 < e)
        assert at == "settle" or at == "wait"
        i = [lab for s, e, lab in left if e <= start + 1e-9]
        assert "hold the upper stop down" in [x for x in i if x != "wait"][-2:]


def test_demo_ends_with_hands_clear():
    for name, wps in ((n, t.demo(n)) for n in t.LOTS):
        for arm in wps:
            end = np.array(wps[arm][-1].pos)
            assert min(np.linalg.norm(end - k) for k in (t.KNOB_U_WORLD, t.KNOB_L_WORLD)) > t.HANDS_CLEAR, name


