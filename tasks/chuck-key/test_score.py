"""The score must credit the bar in the jaws at the marked stick-out, all three jaws on it, the judge's pull held with
hands clear and the key back in its holder, and cap any trial that leaves the key in or on the chuck."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t
from chuck_key import lathe as L


def state(p, q=(0.0, 0.0, 0.0, 1.0)):
    p = np.asarray(p, float)
    return SimpleNamespace(pos=p, quat_xyzw=np.asarray(q, float), low=p, high=p)


def rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return (*(a * math.sin(angle / 2)), math.cos(angle / 2))


def bar(stick=t.STICK_OUT, dx=0.0, dz=0.0, tilt=0.0, flip=False):
    """The bar at a stick-out, its axis through (X0 + dx, AXIS_Z + dz) at the jaws' mid-plane, tilted about x."""
    d = np.array((0.0, math.cos(tilt), math.sin(tilt)))
    mid = np.array((t.X0 + dx, t.JAW_MID_Y, t.AXIS_Z + dz))
    centre_y = t.Y0 - stick + L.BAR_LEN / 2 * d[1]
    centre = mid + d * (centre_y - mid[1]) / d[1]
    q = rot((1, 0, 0), tilt + (math.pi if flip else 0.0))
    return state(centre, q)


CLOSED = L.BAR_R - 0.0002  # each jaw face 0.2 mm into a centred bar (the give compressed)
FAR = ((0.16, 0.30, t.TOP + 0.2), (0.16, -0.30, t.TOP + 0.2))
KEY_HOME = state((*t.HOLDER, t.KEY_HOME_Z), rot((0, 0, 1), -math.pi / 2))
KEY_IN_CHUCK = state(t.SOCKET, rot((0, 0, 1), 0.3))


def world(b=None, jaws=(CLOSED,) * 3, hands=FAR, pull=(30.0, 0.0, 1.0), key=KEY_HOME, pinion=(12.5, 0.0, 0.0)):
    return {"bar": b or bar(), "jaws": state(jaws), "pinion": state(pinion), "left_tcp": state(hands[0]),
            "right_tcp": state(hands[1]), "pull_test": state(pull), "key": key}


OPEN = (L.R_OPEN,) * 3


def test_full_score():
    w = world()
    assert t.score(w) == pytest.approx(1.0) and t.success(w)


def test_flipped_bar_quaternion_reads_the_same():
    assert t.score(world(b=bar(flip=True))) == pytest.approx(1.0)


def test_ladder():
    resting = bar(dz=-t.REST_DROP)
    assert t.score(world(b=state(t.BAR_START), jaws=OPEN, pull=(0, 0, 0))) == 0  # still on the V-block
    assert t.score(world(b=resting, jaws=OPEN, pull=(0, 0, 0))) == pytest.approx(0.2)  # in the open jaws
    assert t.score(world(pull=(0, 0, 0))) == pytest.approx(0.4)  # clamped, not yet pulled
    assert t.score(world(pull=(30.0, 0.0, 0.5))) == pytest.approx(0.4)  # pull held only 0.5 s
    assert t.score(world(pull=(0.0, 0.0012, 0.0))) == pytest.approx(0.4)  # failed pull
    off = state((0.15, -0.15, t.TOP + 0.005), rot((0, 1, 0), math.pi / 2))
    assert t.score(world(key=off)) == pytest.approx(0.7)  # locked, key lying on the table


def test_key_left_in_chuck_is_capped():
    assert t.score(world(key=KEY_IN_CHUCK)) == pytest.approx(t.CAP)
    assert not t.success(world(key=KEY_IN_CHUCK))
    lying = state((t.X0 + 0.01, t.Y0 + 0.04, t.SOCKET[2] + 0.006), rot((1, 0, 0), math.pi / 2))
    assert t.score(world(key=lying)) == pytest.approx(t.CAP)  # resting on top of the chuck
    in_jaws_area = state((t.X0, t.Y0 - 0.005, t.AXIS_Z + 0.04), rot((0, 1, 0), 1.2))
    assert t.score(world(key=in_jaws_area)) == pytest.approx(t.CAP)


def test_key_on_table_next_to_lathe_is_not_on_chuck():
    beside = state((t.X0 + 0.07, t.Y0 + 0.01, t.TOP + 0.005), rot((0, 1, 0), math.pi / 2))
    assert not t.key_on_chuck(world(key=beside))


def test_violation_never_outscores_a_correct_partial_run():
    capped = t.score(world(key=KEY_IN_CHUCK))
    partial = t.score(world(b=bar(dz=-t.REST_DROP), jaws=OPEN, pull=(0, 0, 0)))
    assert capped <= partial


def test_key_home_only_counts_once_the_bar_holds():
    assert t.score(world(pull=(0, 0, 0))) == pytest.approx(0.4)  # key never left the holder: no key credit


@pytest.mark.parametrize("key", [
    state((t.HOLDER[0] + 0.005, t.HOLDER[1], t.KEY_HOME_Z)),  # 5 mm off the hole's centre
    state((*t.HOLDER, t.KEY_HOME_Z + 0.01)),  # not down in the hole
    state((*t.HOLDER, t.KEY_HOME_Z), rot((1, 0, 0), math.radians(14))),  # leaning
])
def test_key_not_home(key):
    assert t.score(world(key=key)) == pytest.approx(0.7)


def test_hand_still_on_key_is_not_home():
    tbar = (t.HOLDER[0], t.HOLDER[1] + t.GRIP_R, t.KEY_HOME_Z + L.TBAR_Z)
    assert t.score(world(hands=(tbar, FAR[1]))) == pytest.approx(0.7)


@pytest.mark.parametrize("b", [
    bar(stick=t.STICK_OUT + 0.004),  # 54 mm out
    bar(stick=t.STICK_OUT - 0.004),  # pushed in too far
    bar(tilt=math.radians(4)),  # cocked in the jaws
    bar(dx=0.012),  # beside the jaws
])
def test_not_inserted(b):
    assert t.score(world(b=b, jaws=OPEN, pull=(0, 0, 0))) == 0
    assert t.score(world(b=b)) == 0


def test_two_jaws_on_it_is_not_clamped():
    w = world(jaws=(CLOSED, CLOSED, CLOSED + 0.001), pull=(0, 0, 0))
    assert not t.clamped(w) and t.score(w) == pytest.approx(0.2)


def test_off_centre_bar_is_not_clamped():
    w = world(b=bar(dz=-0.001), pull=(0, 0, 0))
    assert not t.clamped(w)


def test_slightly_tilted_clamp_is_not_clamped():
    assert not t.clamped(world(b=bar(tilt=math.radians(1.5))))


def test_hand_on_bar_blocks_the_pull_credit():
    on_bar = (t.X0, t.Y0 - t.STICK_OUT + 0.02, t.AXIS_Z + L.BAR_R + 0.05)  # 5 cm above the bar
    w = world(hands=(FAR[0], on_bar))
    assert not t.hands_off(w) and not t.pull_ready(w) and t.score(w) == pytest.approx(0.4)


def test_hand_near_the_far_end_of_the_bar_counts():
    near_end = (t.X0, t.Y0 - t.STICK_OUT - 0.04, t.AXIS_Z)  # 4 cm beyond the outer end
    assert not t.hands_off(world(hands=(FAR[0], near_end)))


def test_pinched_jaws_without_bar_score_nothing():
    w = world(b=state(t.BAR_START), jaws=(0.005,) * 3)
    assert t.score(w) == 0


def test_no_pull_while_the_key_is_in_the_chuck():
    assert not t.pull_ready(world(key=KEY_IN_CHUCK, pull=(0, 0, 0)))
    assert t.pull_ready(world(pull=(0, 0, 0)))


@pytest.mark.parametrize("key", [
    state((t.X0 + 0.025, t.Y0 + 0.012, t.AXIS_Z - 0.02), rot((1, 0, 0), math.pi / 2)),  # lying on a lower jaw
    state((t.X0, t.Y0 + 0.03, t.AXIS_Z - 0.005), rot((1, 0, 0), math.pi / 2)),  # pushed into the bore
    state((t.X0 + 0.02, t.Y0 - 0.03, t.AXIS_Z + L.BAR_R + 0.006), rot((0, 1, 0), math.pi / 2)),  # on the bar
    state((t.X0 - 0.03, t.Y0 + 0.02, t.AXIS_Z + 0.01), rot((0, 1, 0), 1.0)),  # leaning on the chuck's side
])
def test_key_cap_cannot_be_dodged(key):
    w = world(key=key)
    assert t.key_on_chuck(w) and t.score(w) == pytest.approx(t.CAP) and not t.pull_ready(w)


def test_key_held_high_above_the_chuck_is_not_capped():
    high = state((t.X0, t.SOCKET[1], t.SOCKET[2] + 0.06))
    assert not t.key_on_chuck(world(key=high))


def test_tight_is_the_pull_alone_whatever_the_key_turn():
    # jaws on the bar and the pull held: tight, with the key turned barely at all, or far
    for angle in (0.0, 1.0, 40.0):
        assert t.score(world(pinion=(angle, 0.0, 0.0))) == pytest.approx(1.0)


def test_failed_pull_is_not_tight():
    assert t.score(world(pull=(0.0, 0.0012, 0.0))) == pytest.approx(0.4)  # the bar slipped, pull stopped
    assert t.score(world(pull=(30.0, 0.0012, 1.0))) == pytest.approx(0.4)  # slipped 1.2 mm
