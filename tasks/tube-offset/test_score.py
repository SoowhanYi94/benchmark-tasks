"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t

M = t.M
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}
BENT = t.BENT
Z = t.BED_TOP
ON = (0.25, 0.15, Z)  # a place well inside the bed, whichever way it lies


def yawq(a):
    return np.array((0.0, 0.0, math.sin(a / 2), math.cos(a / 2)))


def state(pos, quat=None, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0)):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=yawq(0.0) if quat is None else quat, lin_vel=np.array(lin, float), ang_vel=np.array(ang, float), low=p, high=p)


def tube(at=None, psi=0.0, q1=0.0, q2=0.0):
    """The tube with its middle segment here, heading psi, bends q1 and q2: the segments' origins all sit on their own hinges' frame, as in the sim."""
    at = np.array(at if at is not None else (t.START[0], t.START[1], Z), float)
    return {"tube_mid": state(at, yawq(psi)), "tube_nose": state(at + _pivot(psi, -1) - _turned(psi + q1, -1), yawq(psi + q1)),
            "tube_tail": state(at + _pivot(psi, 1) - _turned(psi + q2, 1), yawq(psi + q2)), "tube_q": state((q1, q2, 0.0))}


def _pivot(psi, s):
    return np.array((-math.sin(psi) * s * t.L_M / 2, math.cos(psi) * s * t.L_M / 2, 0.0))


_turned = _pivot  # a segment's own origin is on the middle's, so its hinge sits L_M/2 along its own +y


def world(at=None, psi=0.0, q1=0.0, q2=0.0, phi=0.0, **over):
    w = {a: state(p) for a, p in PARK.items()}
    w.update(tube(at, psi, q1, q2))
    w["handle"] = state((phi, 0.0, 0.0))
    for k, v in over.items():
        w[k] = v if isinstance(v, NS) else state(v)
    return w


def finished(q=-BENT, q2=None):
    """The finished tube lying on the template: its long ends along y, in the lanes."""
    q2 = q if q2 is None else q2
    return world((*t.GRIP_AT, Z), psi=math.pi + abs(q), q1=q, q2=q2)


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def run(w, n=22):
    for _ in range(n):
        t.note_step(w, 0.05)
    return t.score(w)


def test_start_scores_zero():
    assert run(world()) == 0.0


def test_full_marks():
    assert run(finished()) == 1.0


def test_needs_a_second_of_rest():
    assert run(finished(), n=5) == 0.9


def test_moving_tube_is_not_at_rest():
    w = finished()
    w["tube_mid"].lin_vel = np.array((0.05, 0.0, 0.0))
    assert run(w) == 0.9


def test_one_good_bend_is_a_quarter():
    assert run(world(ON, q1=-BENT)) == 0.25


def test_either_band_first():
    assert run(world(ON, q2=-BENT)) == 0.25


def test_both_bends_not_parallel_is_a_half():
    assert run(world(ON, q1=-math.radians(40.0), q2=-math.radians(50.5))) == 0.5


def test_both_bends_parallel_off_the_template():
    assert run(world(ON, q1=-BENT, q2=-BENT)) == 0.75


def test_turned_end_for_end_on_the_template_still_counts():
    w = world((*t.GRIP_AT, Z), psi=abs(BENT), q1=-BENT, q2=-BENT)
    assert run(w) == 1.0


def test_shifted_off_a_lane_loses_the_template():
    assert run(world((t.GRIP_AT[0] + 0.03, t.GRIP_AT[1], Z), psi=math.pi + BENT, q1=-BENT, q2=-BENT)) == 0.75


def test_a_bend_short_of_tolerance_earns_nothing():
    assert run(world(ON, q1=-math.radians(38.0))) == 0.0


def test_a_bend_past_tolerance_earns_nothing():
    assert run(world(ON, q1=-math.radians(52.0))) == 0.0


def test_a_hand_still_on_it_scores_nothing():
    w = finished()
    w["left_tcp"] = state((*t.GRIP_AT, t.GRIP_Z))
    assert run(w) == 0.0


def test_an_open_finger_touching_still_counts_as_holding():
    w = finished()
    c = t.GRIP_AT
    # a finger on the block, in the middle segment's own frame: 4.7 cm out from the fingertip midpoint, across the block
    R = t._rot(w["tube_mid"].quat_xyzw)
    w["left_tcp"] = state(np.array((*c, t.GRIP_Z)) + R @ np.array((M.TUBE["w"] / 2 + 0.047, 0.0, 0.0)))
    assert run(w) == 0.0


def test_let_go_is_judged_in_the_segments_own_frame():
    # 6.5 cm out from the nose's long side, measured across its own width while it lies diagonal
    w = world(psi=math.radians(45.0))
    n = w["tube_nose"]
    R = t._rot(n.quat_xyzw)
    w["left_tcp"] = state(n.pos + R @ np.array((M.TUBE["w"] / 2 + 0.065, -t.L_M / 2 - t.L_N / 2, 0.01)))
    assert t.let_go(w)
    w["left_tcp"] = state(n.pos + R @ np.array((M.TUBE["w"] / 2 + 0.05, -t.L_M / 2 - t.L_N / 2, 0.01)))
    assert not t.let_go(w)


def test_overbending_is_zero_for_good():
    t.note_step(world(q1=-math.radians(56.0)), 0.05)
    assert t._trial["bad"]
    assert run(finished()) == 0.0


def test_same_way_bends_are_zero_for_good():
    t._trial.update(sign=[-1, 1], peak=[BENT + t.SPRINGBACK, BENT + t.SPRINGBACK])
    assert run(world(q1=-BENT, q2=BENT)) == 0.0
    assert t._trial["why"] == "same way"


def test_opposite_signs_are_not_parallel():
    assert not t.parallel(world(q1=-BENT, q2=BENT))


def test_leaving_the_bed_is_zero_for_good():
    t.note_step(world(at=(0.2, 0.60, Z)), 0.05)
    assert t._trial["bad"]
    assert run(finished()) == 0.0


def test_falling_off_is_zero_for_good():
    t.note_step(world(at=(0.2, 0.2, t.TOP - 0.05)), 0.05)
    assert t._trial["bad"]


def test_going_back_does_not_clear_a_violation():
    t.note_step(world(q1=-math.radians(56.0)), 0.05)
    t.reset_latch()  # only the embodiment's reset does
    assert not t._trial["bad"]


def at_mark(k=0, psi=0.0):
    """The tube fed in with band k at the former's mark."""
    s = -1 if k == 0 else 1
    j = np.array((*M.BAND_AT, Z))
    return np.array(j - _pivot(psi, s))


def test_band_at_reads_the_hinge():
    w = world(at_mark(0))
    assert np.allclose(t.band_at(w, 0), M.BAND_AT)


def contact_phi(**w):
    """The handle's sweep at which the shoe first presses what swings about band 0, by scanning."""
    for deg in np.arange(0.0, 145.0, 0.25):
        g = t.shoe_gap(world(phi=math.radians(deg), **w), 0, -1)
        if g is not None and g <= t.TOUCH:
            return math.radians(deg)
    raise AssertionError("the shoe never reaches it")


def press(phi, dphi=math.radians(1.0), **w):
    """Two steps with the handle advancing to this sweep: the first only sets where it was."""
    t.ratchet(world(phi=phi - dphi, **w), 0.05)
    return t.ratchet(world(phi=phi, **w), 0.05)


def test_locked_where_it_lies_with_the_handle_home():
    lims = t.ratchet(world(at_mark(0), phi=0.0), 0.05)
    assert all(abs(a) < 1e-9 and abs(b) < 1e-9 for a, b in lims)


def test_pressing_shoe_opens_only_the_band_at_the_mark():
    lims = press(contact_phi(at=at_mark(0)), at=at_mark(0))
    assert lims[0][0] <= -t.STEP + 1e-9 and abs(lims[1][0]) < 1e-9 and abs(lims[1][1]) < 1e-9


def test_a_bend_advances_by_the_step_only():
    w = dict(at=at_mark(0))
    lims = press(contact_phi(**w), **w)
    assert abs(lims[0][0] + t.STEP) < 1e-9


def test_away_from_the_mark_the_handle_bends_nothing():
    at = at_mark(0) + (0.0, 0.03, 0.0)
    lims = press(contact_phi(at=at), at=at)
    assert all(abs(a) < 1e-9 and abs(b) < 1e-9 for a, b in lims)


def test_a_parked_handle_clear_of_the_tube_lets_no_hand_bend_it():
    """The review's shortcut: handle parked at the demo's sweep, shoe clear, a hand swinging the rest round."""
    psi = math.radians(30.0)
    for deg in range(0, 60, 2):
        psi = math.radians(deg)
        w = world(at_mark(0, psi), psi=psi, q1=-psi, phi=t.SWEEP)
        w["tube_nose"].quat_xyzw = yawq(0.0)
        lims = t.ratchet(w, 0.05)
    assert t._trial["peak"] == [0.0, 0.0]
    assert abs(lims[0][0]) < 1e-9 and abs(lims[0][1]) < 1e-9


def test_a_handle_parked_touching_does_not_advance_it():
    w = dict(at=at_mark(0))
    phi = contact_phi(**w)
    t.ratchet(world(phi=phi, **w), 0.05)
    lims = t.ratchet(world(phi=phi, **w), 0.05)
    assert abs(lims[0][0]) < 1e-9 and t._trial["peak"][0] == 0.0


def test_a_handle_advancing_clear_of_the_tube_does_not_open_it():
    lims = press(math.radians(5.0), at=at_mark(0))
    assert abs(lims[0][0]) < 1e-9 and abs(lims[0][1]) < 1e-9


def test_a_shoe_past_the_segment_does_not_count():
    assert t.shoe_gap(world(at_mark(0), phi=math.radians(150.0)), 0, 1) is None


def bent(deg):
    """The tube fed in and bent this far: the nose straight in the guide, the rest swung round."""
    psi = math.radians(deg)
    return dict(at=at_mark(0, psi), psi=psi, q1=-psi)


def test_bending_lifts_the_peak_for_good():
    w = bent(30.0)
    t._trial.update(sign=[-1, 0], on=[True, False])
    press(contact_phi(**w), **w)
    assert abs(t._trial["peak"][0] - math.radians(30.0)) < 1e-9


def test_springback_is_three_degrees_and_one_way():
    t._trial.update(sign=[-1, 0], peak=[math.radians(48.0), 0.0], hi=[math.radians(48.0), 0.0])
    for _ in range(40):
        lims = t.ratchet(world(at_mark(0) + (0.0, 0.2, 0.0), q1=-math.radians(47.0), phi=0.0), 0.05)
    assert abs(lims[0][0] + math.radians(45.0)) < 1e-6 and abs(lims[0][1] + math.radians(45.0)) < 1e-6


def test_a_finished_band_stays_locked_at_the_mark():
    t._trial.update(sign=[-1, 0], peak=[math.radians(48.0), 0.0], hi=[math.radians(48.0), 0.0])
    w = bent(45.0)
    t.ratchet(world(phi=0.0, **w), 0.05)
    assert t._trial["done"][0]
    for _ in range(40):
        lims = press(contact_phi(**w), **w)
    assert abs(lims[0][0] + math.radians(45.0)) < 1e-6


def test_an_underbent_band_can_take_another_sweep():
    t._trial.update(sign=[-1, 0], peak=[math.radians(30.0), 0.0], hi=[math.radians(30.0), 0.0])
    w = bent(25.0)
    t.ratchet(world(phi=0.0, **w), 0.05)
    assert not t._trial["done"][0]
    lims = press(contact_phi(**w), **w)
    assert abs(lims[0][0] + math.radians(25.0) + t.STEP) < 1e-9


def test_an_end_overhanging_the_bed_is_zero_for_good():
    # the block's well inside, but the nose runs past the bed's far edge
    w = world(at=(M.BED["x1"] - 0.10, 0.2, Z), psi=math.pi / 2)
    assert t.BOXES and not t.on_bed(w)
    t.note_step(w, 0.05)
    assert t._trial["bad"]


def test_the_start_and_the_template_lie_on_the_bed():
    assert t.on_bed(world(psi=t.START[2])) and t.on_bed(finished())


def test_the_demos_sweep_ends_short_of_the_stop():
    assert max(t.SWEEP, t.SWEEP_2) < math.radians(M.HANDLE["stop"]) - math.radians(20.0)


def test_masses_suit_the_material():
    area = math.pi * (0.02 ** 2 - 0.0176 ** 2) / 4
    assert abs(area * (M.TUBE["n"] + M.TUBE["m"] + M.TUBE["t"]) * 7850 - 0.195) < 0.002
    assert abs(M.MASS["nose"] - area * M.TUBE["n"] * 7850) < 0.001
    assert abs(M.MASS["mid"] - area * M.TUBE["m"] * 7850 - M.TUBE["w"] * M.GRIP["along"] * M.GRIP["h"] * 2700) < 0.001


def test_target_offset_matches_the_text():
    assert abs(t.TARGET_OFFSET - 0.099) < 0.001
