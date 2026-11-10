"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t
from tig_seam import mech as M

DT = 0.05
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}
LA, LB = math.pi, 1.5 * math.pi


def quat_z(a):
    return np.array((0.0, 0.0, math.sin(a / 2), math.cos(a / 2)))


def state(pos, yaw=0.0, v=0.0):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=quat_z(yaw), lin_vel=np.array((v, 0.0, 0.0)), ang_vel=np.zeros(3), low=p, high=p)


def rest_world(**over):
    w = {a: state(p) for a, p in PARK.items()}
    w["torch"] = state((M.TORCH_REST["x"], M.TORCH_REST["y"], t.TOP + M.TORCH_REST_Z), M.TORCH_REST["lean"])
    w["rod"] = state((M.ROD_REST["x"], M.ROD_REST["y"], t.TOP + M.ROD_REST_Z), M.ROD_REST["heading"])
    w.update(over)
    return w


def working(tip, lean, rtip, heading, torch_hand="left_tcp", rod_hand="right_tcp"):
    """Both tools in hand: the tungsten tip at ``tip`` leaning along ``lean``, the rod tip at ``rtip`` with its heading."""
    g = np.asarray(tip, float) - t._rz(lean) @ M.TIP
    s = np.asarray(rtip, float) - t._rz(heading) @ M.ROD_TIP
    w = {a: state(p) for a, p in PARK.items()}
    w["torch"], w["rod"] = state(g, lean), state(s, heading)
    if torch_hand:
        w[torch_hand] = state(g + (0, 0, t.GRIP_DZ))
    if rod_hand:
        w[rod_hand] = state(s + (0, 0, t.SLEEVE_DZ))
    return w


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def weld(speed=0.003, dab=1.5, arc=0.0035, ahead=0.0055, rod_hand="right_tcp", lift=0.010, stop=None, rod=True, lat=0.0, dip_at=None,
         rod_into_tungsten_at=None):
    """Run the ideal process step by step through note_step: dwell 2.7 s at the start, travel, turn at the corner, crater 1 s."""
    la = 0.12
    track = [(0.0, LA)] * 54
    s = 0.0
    while s < la:
        s = min(la, s + speed * DT)
        track.append((s, LA))
    track += [(la, LA + (LB - LA) * k / 30) for k in range(1, 31)]
    while s < M.SEAM_LEN:
        s = min(M.SEAM_LEN, s + speed * DT)
        track.append((s, LB))
    track += [(M.SEAM_LEN, LB)] * 20
    clock = -2.3
    for k, (s, lean) in enumerate(track):
        if stop is not None and s >= stop:
            break
        p, _ = M.seam_point(s)
        n = np.array((-math.sin(lean), math.cos(lean)))
        tip = np.array((*(p + lat * n), t.PLATE_Z + arc))
        if dip_at is not None and abs(s - dip_at) < 0.001:
            tip[2] = t.PLATE_Z + 0.0002
        u = np.array((math.cos(lean), math.sin(lean), 0.0))
        heading = t.HEAD_A if lean < LA + 0.25 * math.pi else t.HEAD_B
        clock += DT
        ph = (clock % dab) / dab if clock >= 0 else 0.9
        down = 0.25 <= ph < 0.5
        rtip = tip + ahead * u
        rtip[2] = t.PLATE_Z + (0.002 if down else lift)
        if rod_into_tungsten_at is not None and abs(s - rod_into_tungsten_at) < 0.001:
            rtip = tip + 0.001 * u
        w = working(tip, lean, rtip if rod else rtip + (0, 0, 0.2), heading, rod_hand=rod_hand)
        t.note_step(w, DT)
    return w


def settle(n=30, **over):
    w = rest_world(**over)
    for _ in range(n):
        t.note_step(w, DT)
    return t.score(w)


def test_start_scores_zero():
    assert settle() == 0.0


def test_full_marks():
    weld()
    assert t.good_segments().all() and not t._trial["dirty"] and settle() == 1.0


def test_needs_a_second_of_rest():
    weld()
    assert settle(n=10) == 0.9


def test_full_seam_tools_not_back_scores_point_nine():
    w = weld()
    for _ in range(30):
        t.note_step(w, DT)
    assert t.score(w) == 0.9


def test_a_tool_left_off_its_rest():
    weld()
    rod = state((M.ROD_REST["x"] + 0.03, M.ROD_REST["y"], t.TOP + M.ROD_REST_Z), 0.0)
    assert settle(rod=rod) == 0.9


def test_a_hand_still_on_the_torch():
    weld()
    g = (M.TORCH_REST["x"], M.TORCH_REST["y"], t.TOP + M.TORCH_REST_Z)
    assert settle(left_tcp=state(np.add(g, (0.0, 0.0, 0.05)))) == 0.9  # 5 cm over the grip: fingers on it


def test_let_go_by_finger_geometry():
    weld()
    g = (M.TORCH_REST["x"], M.TORCH_REST["y"], t.TOP + M.TORCH_REST_Z + M.GRIP["z"] / 2)
    assert settle(left_tcp=state(np.add(g, (0.0, 0.0, 0.065)))) == 1.0


def test_half_the_seam():
    weld(stop=0.1)
    sc = settle()
    assert 0.35 <= sc <= 0.5 and sc == round(0.9 * t.good_segments().sum() / t.N, 3)


def test_torch_alone_scores_zero():
    weld(rod=False)
    assert settle() == 0.0
    assert t.faults().count("undercut") >= 38  # fused without filler: undercut all along


def test_rod_in_the_torch_hand_scores_zero():
    weld(rod_hand="left_tcp")
    assert settle() == 0.0


def test_rod_not_held_scores_zero():
    """Cold wire laid in the joint: the rod's tip is in the zone but no hand holds it."""
    weld(rod_hand=None)
    assert settle() == 0.0


def test_rod_held_down_without_dabbing_feeds_once():
    weld(dab=1e9, lift=0.002)  # the tip stays down the whole way: one dab at most
    assert t._trial["dabs"] <= 1 and settle() <= 0.05


def test_rod_behind_the_torch_feeds_nothing():
    weld(ahead=-0.006)
    assert settle() == 0.0


def test_rod_too_far_ahead_feeds_nothing():
    weld(ahead=0.014)
    assert settle() == 0.0


def test_too_fast_is_lack_of_fusion():
    weld(speed=0.008)
    f = t.faults()
    assert f.count("cold") >= 12 and f.count("good") <= 4 and settle() < 0.1


def test_too_slow_burns_through_and_caps():
    weld(speed=0.0012)
    assert t._trial["burnt"].any() and settle() <= 0.5


def test_off_the_line_has_no_arc():
    weld(lat=0.006)  # 6 mm to one side: only where the corner turn swings the tip back over the line
    assert settle() <= 0.05


def test_arc_too_long():
    weld(arc=0.0065)
    assert settle() == 0.0


def test_tungsten_dip_caps_and_stops_the_weld():
    weld(dip_at=0.06)
    assert t._trial["dirty"]
    sc = settle()
    assert 0.0 < sc <= 0.5 and t.good_segments().sum() <= 13


def test_rod_touching_the_tungsten_caps():
    weld(rod_into_tungsten_at=0.15)
    assert t._trial["dirty"] and settle() == 0.5


def test_contamination_stays_latched():
    weld(dip_at=0.199)
    assert t._trial["dirty"] and settle() == 0.5
    t.note_step(rest_world(), DT)
    assert t._trial["dirty"]


def test_burn_stays_latched_after_rest():
    weld(speed=0.0012)
    settle()
    assert t._trial["burnt"].any()


def test_a_tool_off_the_table_scores_zero():
    weld()
    rod = state((0.5, -0.6, t.TOP - 0.3), 0.0)
    w = rest_world(rod=rod)
    t.note_step(w, DT)
    assert settle() == 0.0


def test_hands_swapped_still_counts():
    """Either arm may hold either tool, as long as they're different hands."""
    weld(rod_hand="right_tcp")
    a = t.good_segments().sum()
    t.reset_latch()
    # swap: the rod in the left hand, torch in the right, mirrored roles
    clock, s = -2.3, 0.0
    for k in range(54 + 40 * 20):
        s = min(0.12, max(0.0, (k - 54) * 0.003 * DT))
        p, _ = M.seam_point(s)
        tip = np.array((*p, t.PLATE_Z + 0.0035))
        clock += DT
        ph = (clock % 1.5) / 1.5 if clock >= 0 else 0.9
        rtip = tip + 0.0055 * np.array((-1.0, 0.0, 0.0))
        rtip[2] = t.PLATE_Z + (0.002 if 0.25 <= ph < 0.5 else 0.01)
        t.note_step(working(tip, LA, rtip, t.HEAD_A, torch_hand="right_tcp", rod_hand="left_tcp"), DT)
    assert a == t.N and t.good_segments()[:20].all()


def test_offsets_in_the_tools_own_frames():
    """Turning the torch about its tip doesn't move the tip; the lean follows the turn."""
    tip = np.array((0.25, 0.02, t.PLATE_Z + 0.003))
    for lean in (LA, LB, 0.3):
        w = working(tip, lean, tip + (0.0, 0.0, 0.05), 0.0)
        ws = t.weld_state(w)
        assert np.allclose(ws["tip"], tip, atol=1e-9) and np.allclose(ws["u"], (math.cos(lean), math.sin(lean)), atol=1e-9)


def test_masses_suit_the_materials():
    steel, nylon = 7850.0, 1150.0
    rod = steel * math.pi * (M.ROD_D / 2) ** 2 * M.ROD_LEN
    assert abs(rod - M.ROD_MASS["rod"]) / rod < 0.15
    assert abs(nylon * M.SLEEVE ** 3 - M.ROD_MASS["sleeve"]) / M.ROD_MASS["sleeve"] < 0.2
    assert abs(nylon * M.GRIP["x"] * M.GRIP["y"] * M.GRIP["z"] - M.TORCH_MASS["grip"]) / M.TORCH_MASS["grip"] < 0.25
    assert 0.2 <= M.TORCH_TOTAL <= 0.35  # a small air-cooled torch head and handle with a stub of cable


def test_one_gripper_cannot_put_the_rod_tip_in_the_zone():
    """With both grips in one hand (centres within the 94 mm opening), the rod tip is far from the zone ahead of the tungsten."""
    reach_rod = float(np.hypot(M.ROD_TIP[0], M.ROD_TIP[1]))
    reach_torch = float(np.hypot(M.TIP[0], M.TIP[1]))
    assert reach_rod - reach_torch - 0.094 > 0.04
    assert (-M.ROD_TIP[2]) - (-M.TIP[2]) > 0.0  # and the sleeve rides lower than the torch's grip with both tips down


def test_rod_leaves_the_sleeve_underside_clear_of_its_post():
    assert M.ROD_EXIT[2] == pytest.approx(-M.SLEEVE / 2)
    assert abs(M.ROD_EXIT[0]) < M.SLEEVE / 2  # out through the bottom face, not the front face the finger presses
    assert M.ROD_POST_X[1] < M.ROD_EXIT[0] - M.ROD_D / 2
    assert M.ROD_REST["heading"] == 0.0  # so ROD_POST_X runs along the heading


def test_rod_grip_closes_along_the_heading():
    """Yaw 0 closes the fingers along y, yaw pi/2 along x: the rod's fingers must close along its heading (no pitch hinge)."""
    for h in (0.0, t.HEAD_A, t.HEAD_B):
        _, yaw = t.rod_tcp(np.zeros(3), h)
        close = np.array((-math.sin(yaw), math.cos(yaw)))
        assert abs(float(np.dot(close, (math.cos(h), math.sin(h))))) == pytest.approx(1.0)
