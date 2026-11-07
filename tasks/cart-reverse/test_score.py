"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t
from cart_reverse import mech as M

Q0 = np.array(t.CART_QUAT)
UP = np.array((0.0, 0.0, 0.0, 1.0))
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}


def state(pos, quat=UP, vel=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0)):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float), ang_vel=np.array(ang, float), low=p, high=p)


def yawq(a):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def chock_seat():
    """Against the wheel nearer the robot, on the hitch side, with the cart square in bay B."""
    return (t.B_AT[0] - M.WHEEL["y"], t.B_AT[1] - t.WR - t.CHOCK["half"] - 0.004, t.TOP)


def world(cart=t.B_AT, yaw=t.HEADING0, chock=None, hinge=0.0, vel=(0.0, 0.0, 0.0), **over):
    w = {a: state(p) for a, p in PARK.items()}
    w["cart"] = state((*cart, t.TOP), yawq(yaw), vel)
    w["hitch"] = state((hinge, 0.0, 0.0))
    w["chock"] = chock if isinstance(chock, NS) else state(chock_seat() if chock is None else chock)
    for k, v in over.items():
        w[k] = v if isinstance(v, NS) else state(v)
    return w


START = dict(cart=t.A_AT, chock=(*t.CHOCK_START, t.TOP))
ASIDE = (t.CHOCK_SPOT[0], t.CHOCK_SPOT[1], t.TOP)


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()
    t._trial["chock_off"] = True  # most tests look at a later moment in the trial


def run(w, n=22):
    for _ in range(n):
        t.note_step(w, 0.05)
    return t.score(w)


def test_start_scores_zero():
    t._trial["chock_off"] = False
    assert run(world(**START)) == 0.0


def test_pulling_the_wrong_way_earns_nothing():
    assert run(world(cart=(t.A_AT[0], t.A_AT[1] - 0.07), chock=ASIDE)) == 0.0
    t.reset_latch()
    t._trial["chock_off"] = True
    assert run(world(cart=(t.A_AT[0] + 0.07, t.A_AT[1]), chock=ASIDE)) == 0.0


def test_toward_the_bay_earns_credit_from_6_cm():
    assert run(world(cart=(t.A_AT[0], t.A_AT[1] + 0.055), chock=ASIDE)) == 0.0
    t.reset_latch()
    t._trial["chock_off"] = True
    assert run(world(cart=(t.A_AT[0], t.A_AT[1] + 0.065), chock=ASIDE)) == 0.2


def test_full_marks():
    assert run(world()) == 1.0


def test_needs_a_second_of_rest():
    assert run(world(), n=5) == 0.9


def test_rungs():
    assert run(world(cart=(0.22, 0.12), chock=ASIDE)) == 0.2
    t.reset_latch()
    assert run(world(cart=(0.30, 0.20), chock=ASIDE)) == 0.4
    t.reset_latch()
    assert run(world(chock=ASIDE)) == 0.7


def test_chock_on_either_wheel():
    x, y, z = chock_seat()
    assert run(world(chock=(x + 2 * M.WHEEL["y"], y, z))) == 1.0


def test_chock_behind_the_wheel_does_not_count():
    x, y, z = chock_seat()
    assert run(world(chock=(x, y + 2 * (t.WR + t.CHOCK["half"]), z))) == 0.7


def test_chock_judged_in_the_carts_own_frame():
    a = math.radians(9)  # cart yawed in the bay, chock placed square to the world: off by more than the chock's own band along the cart
    c = np.array(t.B_AT)
    v = np.array((t.WR + t.CHOCK["half"], M.WHEEL["y"]))
    h = t.HEADING0 + a
    p = c + v[0] * np.array((math.cos(h), math.sin(h))) + v[1] * np.array((-math.sin(h), math.cos(h)))
    assert run(world(yaw=h, chock=(p[0], p[1], t.TOP))) == 1.0


def test_tipped_chock_does_not_count():
    a = math.radians(20)
    assert run(world(chock=state(chock_seat(), (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))))) == 0.7


def test_nose_in_is_zero_for_good():
    assert run(world(yaw=-t.HEADING0)) == 0.0


def test_yawed_or_off_line_does_not_count():
    assert run(world(yaw=t.HEADING0 + math.radians(12), chock=ASIDE)) == 0.4
    t.reset_latch()
    assert run(world(cart=(t.B_AT[0] + 0.03, t.B_AT[1]), chock=ASIDE)) == 0.4
    t.reset_latch()
    assert run(world(cart=(t.B_AT[0], t.B_AT[1] - 0.04), chock=ASIDE)) == 0.4


def test_held_bar_does_not_count():
    w0 = world(chock=ASIDE)
    h = t.handle_at(w0)
    assert run(world(chock=ASIDE, left_tcp=tuple(h))) == 0.4
    t.reset_latch()
    assert run(world(chock=ASIDE, left_tcp=tuple(h + (0.047 + M.HANDLE["half_y"], 0.0, 0.0)))) == 0.4  # one open finger on it


def test_chock_still_in_hand_does_not_count():
    x, y, z = chock_seat()
    assert run(world(left_tcp=(x, y, z + t.CHOCK_GZ))) == 0.7


def test_jackknife_latches():
    t.note_step(world(cart=(0.25, 0.1), chock=ASIDE, hinge=math.radians(71)), 0.05)
    assert t._trial["why"] == "jackknife"
    assert run(world()) == 0.0


def test_tip_and_lift_latch():
    a = math.radians(7)
    q = np.array((math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)))
    w = world()
    w["cart"].quat_xyzw = q
    t.note_step(w, 0.05)
    assert t._trial["bad"]
    assert run(world()) == 0.0
    t.reset_latch()
    w = world()
    w["cart"].pos[2] += 0.02
    t.note_step(w, 0.05)
    assert run(world()) == 0.0


def test_side_skid_latches():
    # the cart's sides along y run along x: step it sideways, along the axle, 5 mm a step
    for k in range(10):
        t.note_step(world(cart=(0.25 + 0.005 * k, 0.1), chock=ASIDE), 0.05)
    assert t._trial["why"] == "skid"
    assert run(world()) == 0.0


def test_rolling_is_not_a_skid():
    for k in range(40):
        t.note_step(world(cart=(0.25, 0.1 + 0.005 * k), chock=ASIDE), 0.05)
    assert not t._trial["bad"]


def _hand_in_part_frame(x, y, z, cart=(0.25, 0.1)):
    """A world point from a point in the cart's frame; the cart's +x is world -y and its +y is world +x."""
    return (cart[0] + y, cart[1] - x, t.TOP + z)


def _shifted(hand, dy=0.01):
    """A hand at this spot while the cart rolls this far along its line."""
    w = world(cart=(0.25, 0.1), chock=ASIDE)
    w["left_tcp"] = state(hand)
    t.note_step(w, 0.05)
    w2 = world(cart=(0.25, 0.1 + dy), chock=ASIDE)
    w2["left_tcp"] = state(hand)
    t.note_step(w2, 0.05)
    return t._trial["why"]


@pytest.mark.parametrize("name,x,y,z", [
    ("nose grip", 0.125, 0.0, 0.045),
    ("wheel pinch", 0.0, M.WHEEL["y"], 0.03),
    ("rod", t.WB + 0.04, 0.0, 0.05),
    ("open finger on the tote, midpoint 6.5 cm off axis", 0.04, 0.065, 0.06),
    ("tote top", 0.04, 0.0, 0.09),
])
def test_any_part_but_the_handle_latches(name, x, y, z):
    assert _shifted(_hand_in_part_frame(x, y, z)) == "hand on the cart", name


def test_slow_crawl_with_a_hand_on_the_nose_latches():
    hand = _hand_in_part_frame(0.125, 0.0, 0.045)
    for k in range(12):
        w = world(cart=(0.25, 0.1 + 0.0005 * k), chock=ASIDE)  # 1 cm/s
        w["left_tcp"] = state(hand)
        t.note_step(w, 0.05)
    assert t._trial["why"] == "hand on the cart"


def test_turning_on_the_spot_with_a_hand_near_latches():
    w = world(cart=(0.25, 0.1), chock=ASIDE)
    w["left_tcp"] = state(_hand_in_part_frame(0.125, 0.0, 0.045))
    t.note_step(w, 0.05)
    w2 = world(cart=(0.25, 0.1), yaw=t.HEADING0 + math.radians(4), chock=ASIDE)
    w2["left_tcp"] = w["left_tcp"]
    t.note_step(w2, 0.05)
    assert t._trial["why"] == "hand on the cart"


def test_a_hand_on_the_handle_is_allowed_while_it_moves():
    for k in range(40):
        w = world(cart=(0.25, 0.1 + 0.005 * k), chock=ASIDE)
        w["right_tcp"] = state(t.handle_at(w) + (0.006, 0.006, 0.01))
        t.note_step(w, 0.05)
    assert not t._trial["bad"]


def test_a_hand_on_a_swung_handle_is_allowed():
    for k in range(10):
        w = world(cart=(0.25, 0.1 + 0.005 * k), chock=ASIDE, hinge=math.radians(65))
        w["right_tcp"] = state(t.handle_at(w))
        t.note_step(w, 0.05)
    assert not t._trial["bad"]


def test_hand_by_the_cart_at_rest_is_allowed():
    w = world(chock=ASIDE)
    w["left_tcp"] = state((t.B_AT[0], t.B_AT[1] - 0.04, t.TOP + 0.09))
    for _ in range(40):
        t.note_step(w, 0.05)
    assert not t._trial["bad"]


def test_hand_clear_of_the_cart_while_it_moves_is_allowed():
    assert _shifted(_hand_in_part_frame(0.04, 0.18, 0.06)) == ""


def test_pushing_against_the_chock_latches():
    t._trial["chock_off"] = False
    t.note_step(world(cart=(t.A_AT[0], t.A_AT[1] + 0.02), chock=(*t.CHOCK_START, t.TOP)), 0.05)
    assert t._trial["why"] == "pushed against the chock"
    assert run(world()) == 0.0


def test_shoving_the_chock_along_with_the_cart_latches():
    t._trial["chock_off"] = False
    y = t.A_AT[1] + 0.10
    t.note_step(world(cart=(t.A_AT[0], y), chock=(t.CHOCK_START[0], t.CHOCK_START[1] + 0.10, t.TOP)), 0.05)
    assert t._trial["why"] == "pushed against the chock"


def test_chock_carried_back_over_its_old_spot_is_fine():
    t._trial["chock_off"] = False
    t.note_step(world(cart=t.A_AT, chock=(*t.CHOCK_START, t.TOP + 0.05)), 0.05)  # lifted off
    assert t._trial["chock_off"]
    t._trial["last"] = None  # the test jumps the cart; that isn't a skid
    t.note_step(world(cart=(0.33, 0.31), chock=(*t.CHOCK_START, t.TOP)), 0.05)
    assert not t._trial["bad"]


def _body_hand(x, y, z):
    return state(_hand_in_part_frame(x, y, z, cart=t.B_AT))


@pytest.mark.parametrize("name,x,y,z", [("tote", 0.04, 0.0, 0.09), ("rod by the hitch", t.WB + 0.01, 0.0, 0.05), ("nose", t.WB, 0.0, 0.045), ("wheel", 0.0, M.WHEEL["y"], 0.05)])
def test_a_hand_left_on_the_cart_earns_no_park_or_full_marks(name, x, y, z):
    assert run(world(left_tcp=_body_hand(x, y, z))) < 0.7, name


def test_a_hand_left_on_the_cart_loses_the_clock():
    w = world()
    run(w)
    assert t._trial["still"] >= t.STILL_S
    t.note_step(world(left_tcp=_body_hand(0.04, 0.0, 0.09)), 0.05)
    assert t._trial["still"] == 0.0


def _chocking(dy, hand_on_chock=True, n=4):
    """Cart parked short, hand on the chock against the wheel; the cart rolls dy during it."""
    y = t.B_AT[1] - 0.02
    cx = t.B_AT[0] - M.WHEEL["y"]
    for k in range(n + 1):
        cy = y + dy * k / n
        ch = (cx, cy - t.WR - t.CHOCK["half"], t.TOP)
        h = (ch[0], ch[1], t.TOP + t.CHOCK_GZ) if hand_on_chock else (cx + 0.02, cy, t.TOP + 0.06)
        t.note_step(world(cart=(t.B_AT[0], cy), chock=ch, left_tcp=h), 0.05)
    return ch, cy


def test_pressing_the_chock_on_the_wheel_is_not_a_hand_on_the_cart():
    ch, cy = _chocking(0.006)
    assert not t._trial["bad"]
    assert run(world(cart=(t.B_AT[0], cy), chock=ch)) == 1.0


def test_rolling_the_cart_far_with_the_chock_latches():
    _chocking(0.025)
    assert t._trial["why"] == "moved the cart with the chock"


def test_a_hand_by_the_wheel_without_the_chock_still_latches():
    _chocking(0.006, hand_on_chock=False)
    assert t._trial["why"] == "hand on the cart"


def test_a_push_in_steps_with_retreats_latches():
    far = (0.05, 0.40, t.TOP + 0.2)
    for k in range(3):  # 4 mm with the hand on the tote, then the hand away
        y = 0.10 + 0.004 * k
        for dy, h in ((0.0, None), (0.004, None), (0.004, far)):
            w = world(cart=(0.25, y + dy), chock=ASIDE, left_tcp=_hand_in_part_frame(0.04, 0.0, 0.09, cart=(0.25, y)) if h is None else h)
            t.note_step(w, 0.05)
    assert t._trial["why"] == "hand on the cart"


def test_a_cart_turned_round_latches():
    w = world(cart=(0.25, 0.1), chock=ASIDE, yaw=t.HEADING0 + math.radians(76))
    t.note_step(w, 0.05)
    assert t._trial["why"] == "turned round"
    assert run(world()) == 0.0


def test_nose_within_the_limit_only_rolls_away_from_the_bay():
    h = t.HEADING0 + t.TURNED_ROUND
    assert math.sin(h) < 0  # nose still has a -y part, so a forward roll loses ground


def test_chock_slid_clear_along_the_table_counts_as_taken():
    t._trial["chock_off"] = False
    x, y = t.A_AT[0] - M.WHEEL["y"], t.A_AT[1] + 0.081
    t.note_step(world(cart=t.A_AT, chock=(x, y, t.TOP)), 0.05)
    assert t._trial["chock_off"]


def test_chock_nudged_is_not_taken():
    t._trial["chock_off"] = False
    x, y = t.A_AT[0] - M.WHEEL["y"], t.A_AT[1] + 0.07
    t.note_step(world(cart=t.A_AT, chock=(x, y, t.TOP)), 0.05)
    assert not t._trial["chock_off"]


def test_off_the_table_latches():
    t.note_step(world(chock=(0.3, 0.9, 0.05)), 0.05)
    assert t._trial["why"] == "off the table"
    assert run(world()) == 0.0


def test_moving_cart_scores_short_of_full():
    assert run(world(vel=(0.0, 0.02, 0.0))) == 0.9


def test_masses_agree_with_material():
    al = 2700.0
    vol = (2 * t.CHOCK["half"]) ** 2 * t.CHOCK["h"] + (2 * t.CHOCK["post"]) ** 2 * (t.CHOCK["post_h"] - t.CHOCK["h"])
    assert abs(t.CHOCK["mass"] / (vol * al) - 1) < 0.1
    assert t.CHOCK_G > t.FINGER_PER_M * -0.094 and t.CHOCK_R > -0.04695
    # the deck is a folded sheet tray, the tote a thin plastic box: solid they'd weigh about 470 g and 260 g
    assert 0.2 < M.MASS["chassis"] < 0.73
    rod = (2 * 0.006) ** 2 * M.BAR * 2700.0  # aluminium, plus a hollow plastic handle
    assert rod < M.MASS["bar"] < rod + 0.035
    tyre = math.pi * M.WHEEL["r"] ** 2 * M.WHEEL["w"] * 1100.0  # rubber
    assert 0.5 * M.WHEEL["mass"] < tyre < 1.5 * M.WHEEL["mass"]
