"""The score must give 0.5 for the straightedge resting, released and aligned, across both supports on the plate, plus
0.25 for each support within 1 cm of a Bessel point, and 0 for anything else, including the exploits found in review
(the bar held still in the grippers, laid yawed or rolled, or with both supports on one side of its middle)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t

_SIGNS = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)


def _q(axis, a):
    s = math.sin(a / 2)
    return np.array((*(s * np.asarray(axis, float)), math.cos(a / 2)))


def _mul(a, b):
    (ax, ay, az, aw), (bx, by, bz, bw) = a, b
    return np.array((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz))


def box(x, y, z, size, yaw=0.0, pitch=0.0, roll=0.0, moving=False):
    """A box centred at (x, y, z): turned yaw about z, its y axis tilted pitch (about x), rolled about its y axis."""
    q = _mul(_mul(_q((0, 0, 1), yaw), _q((1, 0, 0), pitch)), _q((0, 1, 0), roll))
    o = SimpleNamespace(pos=np.array((x, y, z)), quat_xyzw=q, resting=lambda: not moving)
    c = o.pos + (_SIGNS * np.asarray(size) / 2) @ t._rot(q).T
    o.low, o.high = c.min(axis=0), c.max(axis=0)
    return o


SEAT = t.PLATE_Z + t.SUPPORT[2]  # the supports' tops
PX = t.PLATE["center"][0]


def sup(y, x=PX, **kw):
    return box(x, y, t.PLATE_Z + t.SUPPORT[2] / 2, t.SUPPORT, **kw)


def bar(y=0.0, x=PX, gap=0.0, **kw):
    """The straightedge with its underside gap above the supports' tops."""
    return box(x, y, SEAT + gap + t.BAR[2] / 2, t.BAR, **kw)


def tcp(x, y, z):
    return SimpleNamespace(pos=np.array((x, y, z)))


AWAY = dict(left_tcp=tcp(0.18, 0.32, t.TOP + 0.18), right_tcp=tcp(0.18, -0.32, t.TOP + 0.18))


def done(**kw):
    s = dict(straightedge=bar(), support_a=sup(t.SUPPORT_Y), support_b=sup(-t.SUPPORT_Y), **AWAY)
    return dict(s, **kw)


def test_start_scores_0():
    s = dict(straightedge=box(*t.BAR_START, t.TOP + 0.003 + t.BAR[2] / 2, t.BAR),
             **{n: box(*xy, t.TOP + t.SUPPORT[2] / 2, t.SUPPORT) for n, xy in t.SUPPORT_START.items()}, **AWAY)
    assert t.score(s) == 0.0 and not t.success(s)


def test_full_score():
    assert t.score(done()) == 1.0 and t.success(done())


def test_bessel_spacing_is_rounded_6_61_and_16_78():
    assert round(100 * t.BESSEL * t.BAR_L, 2) == 6.61 and round(200 * t.SUPPORT_Y, 2) == 16.78


@pytest.mark.parametrize("off", [0.0095, -0.0095])
def test_within_1cm_counts(off):
    assert t.score(done(support_a=sup(t.SUPPORT_Y + off), support_b=sup(-t.SUPPORT_Y - off))) == 1.0


def test_wrong_spacing_scores_half():
    assert t.score(done(support_a=sup(0.04), support_b=sup(-0.04))) == 0.5


def test_bar_off_centre_scores_half():
    assert t.score(done(straightedge=bar(0.03))) == 0.5


def test_one_support_off_scores_three_quarters():
    assert t.score(done(support_b=sup(-t.SUPPORT_Y - 0.02))) == 0.75


def test_turned_end_for_end_counts():
    assert t.score(done(straightedge=bar(yaw=math.pi))) == 1.0


def test_bar_on_plate_scores_0():
    assert t.score(done(straightedge=box(PX, 0.0, t.PLATE_Z + t.BAR[2] / 2, t.BAR))) == 0.0


def test_bar_moving_scores_0():
    assert t.score(done(straightedge=bar(moving=True))) == 0.0


def test_support_knocked_over_scores_0():
    fallen = box(PX, t.SUPPORT_Y, t.PLATE_Z + t.SUPPORT[0] / 2, t.SUPPORT, roll=math.pi / 2, pitch=math.pi / 2)
    assert t.score(done(support_a=fallen)) == 0.0


def test_support_off_plate_scores_0():
    assert t.score(done(support_a=box(0.18, t.SUPPORT_Y, t.TOP + t.SUPPORT[2] / 2, t.SUPPORT))) == 0.0


def test_support_not_under_bar_scores_0():
    assert t.score(done(support_a=sup(t.SUPPORT_Y, x=PX + 0.03))) == 0.0


def test_bar_tilted_3_degrees_scores_0():
    assert t.score(done(straightedge=bar(pitch=math.radians(3)))) == 0.0


# -- exploits found in review -----------------------------------------------------------------------------------------
def test_held_hovering_3_5mm_scores_0():
    assert t.score(done(straightedge=bar(gap=0.0035))) == 0.0


def test_seated_within_1_5mm_counts():
    assert t.score(done(straightedge=bar(gap=0.0014))) == 1.0


@pytest.mark.parametrize("arm", ["left_tcp", "right_tcp"])
def test_held_in_a_gripper_scores_0(arm):
    """Resting on the supports but a TCP 2 cm above the bar's top: not let go."""
    assert t.score(done(**{arm: tcp(PX, 0.12, SEAT + t.BAR[2] + 0.02)})) == 0.0


def test_hands_just_clear_count():
    assert t.score(done(left_tcp=tcp(PX, 0.12, SEAT + t.BAR[2] + 0.035))) == 1.0


@pytest.mark.parametrize("deg", [10, 19, -10])
def test_yawed_bar_scores_0(deg):
    """Yawed 10-19 degrees, the supports still match its Bessel points along world y."""
    y = t.SUPPORT_Y * math.cos(math.radians(deg))
    assert t.score(done(straightedge=bar(yaw=math.radians(deg)), support_a=sup(y), support_b=sup(-y))) == 0.0


def test_yawed_4_degrees_counts():
    assert t.score(done(straightedge=bar(yaw=math.radians(4)))) == 1.0


def test_rolled_bar_scores_0():
    assert t.score(done(straightedge=bar(roll=math.radians(10)))) == 0.0


def test_rolled_onto_another_face_counts():
    assert t.score(done(straightedge=bar(roll=math.pi / 2))) == 1.0


def test_supports_on_one_side_of_middle_score_0():
    assert t.score(done(support_a=sup(0.10), support_b=sup(0.02))) == 0.0


# -- the scripted demo ------------------------------------------------------------------------------------------------
def test_demo_fits_run_limit_and_ik_reaches_every_waypoint():
    steps, mm, deg = t.check_demo()
    assert steps < 2400 and mm < 1.0 and deg < 1.0, (steps, mm, deg)


def test_demo_ends_hands_off_the_bar():
    ends = {f"{a}_tcp": tcp(*wps[-1].pos) for a, wps in t.demo().items()}
    assert t.hands_off(done(**ends))


def test_demo_lays_supports_and_bar_at_full_score():
    """Where the demo puts the supports and the bar (centred over the plate) scores 1.0."""
    plan = t.demo()
    ys = [w.pos[1] for a in plan for w in plan[a] if w.label == "stand it on the plate"]
    assert sorted(ys) == pytest.approx([-t.SUPPORT_Y, t.SUPPORT_Y])
    assert t.score(done(support_a=sup(ys[0]), support_b=sup(ys[1]))) == 1.0
