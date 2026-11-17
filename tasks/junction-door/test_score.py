"""Scorer tests on synthetic states (no simulator): every rung, every cap, the door test, the unintended strategies.

    python -m pytest -q test_score.py      (from this folder, with yam_arena importable)
"""

import math

import numpy as np
import pytest

import task
from cablekit import CableState
from bodykit import BodyState

DT = 0.05


def m(**kw):
    """One step's measurements; defaults: nothing done, door open, hands clear, still."""
    base = {"t": 0.0, "door": 90.0, "bolt": 0.0, "depth": -1.0, "clipped": False, "outside": False, "hands": True,
            "still": True}
    base.update(kw)
    return base


DONE = dict(door=0.0, bolt=0.0, depth=0.014, clipped=True)  # the robot's finished work


def run(tr, seq):
    """Feed (duration, measurements) pieces at 20 Hz; returns the commands issued."""
    cmds = []
    for dur, kw in seq:
        for _ in range(max(1, round(dur / DT))):
            tr.update(m(t=tr.t + DT, **kw))
            cmds.append(tr._test_command(tr.t - tr.t_test) if tr.phase == "test" else None)
    return cmds


def finished_robot_phase():
    """Plugged, clipped, door swung shut and latched, hands clear, still 1 s: the door test has started."""
    tr = task.Tracker()
    run(tr, [(1.0, {}), (1.0, dict(depth=0.014)), (1.0, dict(depth=0.014, clipped=True)),
             (1.0, dict(depth=0.014, clipped=True, door=40.0)), (1.2, DONE)])
    assert tr.phase == "test"
    return tr


def passed_door_test(tr):
    run(tr, [(0.4, DONE), (2.5, dict(DONE, door=120.0)), (2.0, dict(DONE, door=60.0)), (3.0, DONE)])
    assert tr.phase == "done"
    run(tr, [(1.1, DONE)])  # the final one still second after the test
    return tr


def through_door_test(tr):
    """The door test passes; stops the moment the test's timeline ends (phase done, no still time yet)."""
    while tr.phase != "done":
        dt = tr.t - tr.t_test
        door = 120.0 if 0.4 <= dt < 3.4 else 60.0 if 3.4 <= dt < 5.0 else 0.0
        tr.update(m(t=tr.t + DT, **dict(DONE, door=door)))
    return tr


# -- the ladder ---------------------------------------------------------------------------------------------------
def test_nothing_done_scores_zero():
    tr = task.Tracker()
    run(tr, [(3.0, {})])
    assert tr.score() == 0.0


def test_plug_seated():
    tr = task.Tracker()
    run(tr, [(1.0, dict(depth=0.014))])
    assert tr.score() == 0.2


def test_plug_half_in_is_not_seated():
    tr = task.Tracker()
    run(tr, [(1.0, dict(depth=0.011, clipped=True))])
    assert tr.score() == 0.0


def test_plug_and_clip():
    tr = task.Tracker()
    run(tr, [(1.0, dict(depth=0.014, clipped=True))])
    assert tr.score() == 0.4


def test_latched():
    tr = task.Tracker()
    run(tr, [(0.5, dict(depth=0.014, clipped=True)), (0.5, DONE)])
    assert tr.score() == 0.6 and tr.phase == "robot"


def test_hand_over_needs_one_still_second():
    tr = task.Tracker()
    run(tr, [(0.5, dict(depth=0.014, clipped=True)), (0.9, dict(DONE, still=False)), (0.9, DONE)])
    assert tr.phase == "robot"
    run(tr, [(0.2, DONE)])
    assert tr.phase == "test" and tr.score() == 0.6


def test_door_test_commands_in_order():
    tr = finished_robot_phase()
    cmds = run(tr, [(7.5, DONE)])
    kinds = [c["hinge"] for c in cmds if c]
    assert kinds[0] == "friction" and "lead" in kinds and kinds[-1] == "friction"
    leads = [c["target"] for c in cmds if c and c["hinge"] == "lead"]
    assert max(leads) >= task.STOP and min(leads) < 0  # opens past the stop, then is led past shut


def test_opened_to_the_stop_scores_08():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (2.5, dict(DONE, door=120.0))])
    assert tr.score() == 0.8


def test_full_score_and_success():
    tr = passed_door_test(finished_robot_phase())
    assert tr.phase == "done" and tr.score() == 1.0 and tr.success()


# -- the final one-second stillness after the door test (review #3) ---------------------------------------------------
def test_no_full_score_the_moment_the_test_ends():
    tr = through_door_test(finished_robot_phase())
    assert tr.phase == "done" and tr.score() == 0.8 and not tr.success()


def test_full_score_only_after_one_still_second():
    tr = through_door_test(finished_robot_phase())
    run(tr, [(0.9, DONE)])
    assert tr.score() == 0.8 and not tr.success()
    run(tr, [(0.15, DONE)])
    assert tr.score() == 1.0 and tr.success()


def test_still_moving_after_the_test_stays_at_08():
    tr = through_door_test(finished_robot_phase())
    run(tr, [(3.0, dict(DONE, still=False))])
    assert tr.score() == 0.8 and not tr.success()


def test_movement_after_the_test_restarts_the_still_second():
    tr = through_door_test(finished_robot_phase())
    run(tr, [(0.8, DONE), (0.1, dict(DONE, still=False)), (0.8, DONE)])
    assert tr.score() == 0.8
    run(tr, [(0.3, DONE)])
    assert tr.score() == 1.0


def test_full_score_lost_if_the_door_springs_open_at_the_end():
    tr = passed_door_test(finished_robot_phase())
    assert tr.score() == 1.0
    run(tr, [(0.5, dict(DONE, door=3.0))])
    assert tr.score() == 0.8 and not tr.success()


# -- the hinge loop: too short, trapped ------------------------------------------------------------------------------
def test_too_short_loop_pulls_the_plug_in_the_door_test():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (1.0, dict(DONE, door=100.0)), (1.5, dict(DONE, door=118.0, depth=0.004)),
             (5.0, dict(DONE, depth=0.004))])
    assert tr.score() == 0.6 and not tr.success()


def test_too_short_loop_stops_the_door_short_of_its_stop():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (3.0, dict(DONE, door=104.0)), (5.0, DONE)])
    assert tr.score() == 0.6


def test_cord_pulled_out_of_the_clip_in_the_door_test():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (2.5, dict(DONE, door=120.0, clipped=False)), (5.0, DONE)])
    assert tr.score() == 0.6


def test_loop_trapped_when_the_test_shuts_the_door():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (2.5, dict(DONE, door=120.0)), (2.0, dict(DONE, door=5.0, outside=True)), (3.0, DONE)])
    assert tr.score() == 0.0 and not tr.success()


def test_loop_trapped_while_the_test_opens_the_door_is_zero():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (0.2, dict(DONE, door=5.0, outside=True)), (2.5, dict(DONE, door=120.0)), (5.0, DONE)])
    assert tr.score() == 0.0 and not tr.success()


def test_trapped_cord_and_plug_pulled_in_the_same_door_test_is_zero():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (2.5, dict(DONE, door=120.0)),
             (2.0, dict(DONE, door=5.0, outside=True, depth=0.004)), (3.0, dict(DONE, depth=0.004))])
    assert tr.test_unseated and tr.score() == 0.0


def test_cord_across_the_rim_when_the_robot_shuts_the_door_is_zero_for_good():
    tr = task.Tracker()
    run(tr, [(1.0, dict(depth=0.014, clipped=True)), (0.5, dict(DONE, door=5.0, outside=True)), (3.0, DONE)])
    assert tr.pinched and tr.score() == 0.0
    run(tr, [(5.0, DONE)])
    assert tr.phase == "robot" and tr.score() == 0.0


def test_cord_outside_with_the_door_open_is_fine():
    tr = task.Tracker()
    run(tr, [(1.0, dict(outside=True, door=60.0)), (1.0, dict(depth=0.014, clipped=True))])
    assert not tr.pinched and tr.score() == 0.4


# -- unintended strategies --------------------------------------------------------------------------------------------
def test_door_shut_before_the_plug_is_zero_even_if_plugged_later():
    tr = task.Tracker()
    run(tr, [(1.0, dict(door=3.0)), (1.0, dict(door=60.0)), (3.0, DONE)])
    assert tr.shut_before_plug and tr.score() == 0.0 and tr.phase == "robot"


def test_plug_held_in_by_a_gripper_scores_zero():
    tr = task.Tracker()
    run(tr, [(2.0, dict(depth=0.014, clipped=True, hands=False))])
    assert tr.score() == 0.0


def test_door_held_shut_by_a_hand_scores_zero_and_starts_no_test():
    tr = task.Tracker()
    run(tr, [(1.0, dict(depth=0.014, clipped=True)), (3.0, dict(DONE, hands=False))])
    assert tr.score() == 0.0 and tr.phase == "robot"


def test_door_shut_but_not_latched_is_not_credited():
    tr = task.Tracker()
    run(tr, [(1.0, dict(depth=0.014, clipped=True)), (3.0, dict(DONE, door=3.0))])  # resting on the bolt
    assert tr.score() == 0.4 and tr.phase == "robot"
    run(tr, [(3.0, dict(DONE, bolt=-0.003))])  # bolt pushed back, hook not caught
    assert tr.score() == 0.4 and tr.phase == "robot"


def test_cord_left_out_of_the_clip_gets_no_clip_credit():
    tr = task.Tracker()
    run(tr, [(1.0, dict(depth=0.014)), (3.0, dict(DONE, clipped=False))])
    assert tr.score() == 0.2 and tr.phase == "robot"


def test_plug_pulled_out_after_seating_caps_at_02():
    tr = task.Tracker()
    run(tr, [(1.0, dict(depth=0.014)), (0.2, dict(depth=0.005)), (3.0, DONE)])
    assert tr.unseated and tr.score() == 0.2 and tr.phase == "robot"


def test_plug_pulled_out_by_the_door_on_the_way_open_is_a_failed_test_at_06():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (1.0, dict(DONE, door=80.0)), (1.5, dict(DONE, door=91.0, depth=0.004)),
             (5.0, dict(DONE, depth=0.004))])
    assert tr.test_unseated and not tr.unseated and tr.phase == "done" and tr.score() == 0.6


def test_plug_pulled_out_by_the_door_after_115_deg_is_a_failed_test_at_08():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (2.5, dict(DONE, door=120.0)), (2.0, dict(DONE, door=60.0, depth=0.004)),
             (3.0, dict(DONE, depth=0.004))])
    assert tr.test_unseated and not tr.unseated and tr.score() == 0.8 and not tr.success()


def test_plug_stays_in_through_the_door_test_scores_10():
    tr = passed_door_test(finished_robot_phase())
    assert not tr.test_unseated and not tr.unseated and tr.score() == 1.0


def test_hand_in_the_way_during_the_door_test_fails_it():
    tr = finished_robot_phase()
    run(tr, [(0.4, DONE), (2.5, dict(DONE, door=120.0, hands=False)), (5.0, DONE)])
    assert tr.score() == 0.6 and not tr.success()


def test_violation_flags_are_never_cleared():
    tr = task.Tracker()
    run(tr, [(1.0, dict(depth=0.014)), (0.1, dict(depth=0.0)), (1.0, dict(depth=0.014))])
    assert tr.unseated
    run(tr, [(10.0, DONE)])
    assert tr.unseated and tr.score() == 0.2


# -- measurements ------------------------------------------------------------------------------------------------------
def _qx(deg):
    return np.array(task._q_x(deg))


def _body(quat=(0, 0, 0, 1), pos=(0, 0, 0)):
    z = np.zeros(3)
    return BodyState(pos=np.asarray(pos, float), quat_xyzw=np.asarray(quat, float), lin_vel=z, ang_vel=z, low=z,
                     high=z)


@pytest.mark.parametrize("deg", [0.0, 0.5, 14.0, 90.0, 120.0])
def test_door_angle(deg):
    assert task.door_angle(_body(_qx(-deg))) == pytest.approx(deg, abs=1e-6)


def _cord(points, plug, end_dir):
    z = np.zeros(3)
    pts = np.asarray(points, float)
    return CableState(pos=pts.mean(0), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=z, ang_vel=z, low=pts.min(0),
                      high=pts.max(0), points=pts, start_plug=None, end_plug=np.asarray(plug, float),
                      start_dir=np.array([0, -1.0, 0]), end_dir=np.asarray(end_dir, float), seg_len=0.015)


def _plugged(depth, dy=0.0, dz=0.0, angle_deg=0.0):
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a), 0.0])
    tip = np.array([task.SOCK_X + depth, task.SOCK_Y + dy, task.SOCK_Z + dz])
    plug = tip - d * task.PL / 2
    pts = [plug - d * (task.PL / 2 + 0.0075 + 0.015 * k) for k in range(5)][::-1]
    return _cord(pts, plug, d)


def test_plug_depth():
    assert task.plug_depth(_plugged(0.014)) == pytest.approx(0.014)
    assert task.plug_depth(_plugged(0.014, dy=0.006)) == -1.0  # beside the socket
    assert task.plug_depth(_plugged(0.014, dz=0.008)) == -1.0  # on top of it
    assert task.plug_depth(_plugged(0.014, angle_deg=30)) == -1.0  # skewed


def test_in_clip():
    xs = np.linspace(task.CLIP_X0 - 0.03, task.CLIP_X1 + 0.03, 30)
    seated = np.c_[xs, np.full_like(xs, task.CLIP_Y), np.full_like(xs, task.CLIP_FLOOR + 0.0025)]
    assert task.in_clip(seated)
    assert not task.in_clip(seated + [0, 0, 0.004])  # resting on the clip's mouth
    assert not task.in_clip(seated + [0, 0.006, 0])  # beside it
    over = seated.copy()
    over[:, 2] += 0.03
    assert not task.in_clip(over)



def test_in_clip_needs_the_cord_under_the_lip():
    """RB5 clip: clipped only when the cord's centre line is no higher than the lip's edge (3.9 mm above the floor)."""
    xs = np.linspace(task.CLIP_X0 - 0.03, task.CLIP_X1 + 0.03, 30)
    at = np.c_[xs, np.full_like(xs, task.CLIP_Y), np.full_like(xs, task.CLIP_FLOOR + task.LIP_Z)]
    assert task.in_clip(at)
    assert not task.in_clip(at + [0, 0, 0.0005])  # on the lip, not yet snapped under it
    assert task.CLIP_X1 - task.CLIP_X0 == pytest.approx(0.019)  # the RB5's 19 mm length

def test_outside_box():
    inside = np.array([[0.25, 0.0, task.TOP + 0.03], [0.30, -0.05, task.TOP + 0.01]])
    assert not task.outside_box(inside)
    assert task.outside_box(np.vstack([inside, [0.25, task.Y0 - 0.002, task.TOP + 0.081]]))  # over the front rim
    assert task.outside_box(np.vstack([inside, [0.25, -0.02, task.TOP + task.H + 0.001]]))  # under the shut door


def test_hands_clear():
    assert task.hands_clear([np.array(task.PARK["left"]), np.array(task.PARK["right"])])
    assert not task.hands_clear([np.array([0.27, -0.02, task.TOP + 0.05])])  # in the box
    assert not task.hands_clear([np.array([0.27, 0.10, task.TOP + 0.15])])  # where the door swings


def test_friction_hinge_sticks_then_slips():
    slip = task.FRICTION / task.FRICTION_K
    assert task.friction_anchor(90.0, 90.0 + 0.5 * slip) == 90.0
    assert task.friction_anchor(90.0, 80.0) == pytest.approx(80.0 + slip)
    assert task.friction_anchor(10.0, 30.0) == pytest.approx(30.0 - slip)
