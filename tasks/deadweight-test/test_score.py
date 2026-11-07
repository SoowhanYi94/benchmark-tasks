"""Scoring checks with no simulator: python -m pytest -q test_score.py"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t

Q = np.array((0.0, 0.0, 0.0, 1.0))
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}
H = 0.029
UP = t.GEOM["lever_up"]


def state(pos, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0), q=Q):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=np.array(q, float), lin_vel=np.array(lin, float), ang_vel=np.array(ang, float), low=p, high=p)


def on_stem(layer, h=H, ang=(0.0, 0.0, 0.0), q=Q):
    return state((t.PX, t.PY, t.TOP + t.PLATEN + h + layer * t.H_W), ang=ang, q=q)


def world(h=H, lever=UP, knob=0.0, spin=0.0, a=None, b=None, **over):
    w = {k: state(p) for k, p in PARK.items()}
    w["rig"] = state((h, lever, knob), ang=(0.0, 0.0, spin))
    w["weight_a"] = a if a is not None else on_stem(0, h, (0.0, 0.0, spin))
    w["weight_b"] = b if b is not None else on_stem(1, h, (0.0, 0.0, spin))
    for k, v in over.items():
        w[k] = v if isinstance(v, NS) else state(v)
    return w


def at_rack(n):
    return state((*t.RACK[n], t.TOP + t.RACK_H))


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def steps(w, n=22):
    for _ in range(n):
        t.note_step(w, 0.05)
    return t.score(w)


def pumped(strokes=5):
    """Full strokes: the oil rises by real lever travel, only on the way down."""
    t.note_step(world(h=0.0, lever=UP), 0.05)
    for _ in range(strokes):
        t.note_step(world(h=0.0, lever=t.GEOM["lever_down"]), 0.05)
        t.note_step(world(h=0.0, lever=UP), 0.05)
    t.note_step(world(h=H), 0.05)


def spun(knob=0.0):
    steps(world(spin=4.0, knob=knob), 40)


def set_oil_true(w):
    t._trial["oil"] = t.FULL_N / t.GEOM["oil_k"] + w["rig"].pos[0]


def test_start_scores_zero():
    assert steps(world(h=0.0, a=at_rack("weight_a"), b=at_rack("weight_b"))) == 0.0


def test_one_stroke_raises_the_oil_a_stroke():
    pumped(1)
    assert t._trial["oil"] == pytest.approx(t.STROKE_RISE)


def test_upstrokes_add_nothing():
    w = world(h=0.0, lever=UP)
    t.note_step(w, 0.05)
    for _ in range(3):
        t.note_step(world(h=0.0, lever=UP + 0.5), 0.05)
        t.note_step(w, 0.05)
    assert t._trial["oil"] < 0.001


def test_one_weight():
    assert steps(world(h=0.0, b=at_rack("weight_b"))) == 0.15


def test_both_weights_unfloated():
    assert steps(world(h=0.0)) == 0.3


def test_in_band_without_spin():
    pumped()
    assert steps(world()) == 0.5


def test_out_of_band_stops_at_both_weights():
    assert steps(world(h=t.GEOM["band"][0] - 0.002)) == 0.3
    assert steps(world(h=t.GEOM["band"][1] + 0.002)) == 0.3


def test_spun():
    pumped()
    spun()
    set_oil_true(world())
    assert steps(world()) == 0.75


def test_full_marks():
    pumped()
    set_oil_true(world())
    spun()
    steps(world(spin=3.5, knob=t.KNOB_TURN), 2)
    assert steps(world(knob=t.KNOB_TURN, spin=3.0)) == 1.0


def test_swapped_weights_score_the_same():
    pumped()
    set_oil_true(world())
    w = world(spin=4.0, a=on_stem(1, ang=(0.0, 0.0, 4.0)), b=on_stem(0, ang=(0.0, 0.0, 4.0)))
    steps(w, 40)
    steps(world(spin=3.5, knob=t.KNOB_TURN, a=on_stem(1, ang=(0.0, 0.0, 3.5)), b=on_stem(0, ang=(0.0, 0.0, 3.5))), 2)
    assert steps(world(knob=t.KNOB_TURN, spin=3.0, a=on_stem(1, ang=(0.0, 0.0, 3.0)), b=on_stem(0, ang=(0.0, 0.0, 3.0)))) == 1.0


def test_needs_a_second_of_rest():
    pumped()
    set_oil_true(world())
    spun()
    steps(world(spin=3.5, knob=t.KNOB_TURN), 2)
    t.note_step(world(knob=t.KNOB_TURN, spin=3.0, left_tcp=(t.PX, t.PY, t.TOP + 0.1)), 0.05)
    assert steps(world(knob=t.KNOB_TURN, spin=3.0), 5) == 0.9


def test_a_slow_spin_frees_nothing():
    pumped()
    steps(world(spin=t.SPIN - 0.1), 40)
    assert t._trial["freed"] is False and t._trial["stick"] == t.STICK


def test_only_the_rotor_turning_frees_nothing():
    pumped()
    steps(world(spin=4.0, a=on_stem(0, ang=(0.0, 0.0, 0.0)), b=on_stem(1, ang=(0.0, 0.0, 0.0))), 40)
    assert t._trial["freed"] is False


def test_spinning_on_the_stop_frees_nothing():
    steps(world(h=0.0, spin=4.0), 40)
    assert t._trial["freed"] is False


def test_a_null_on_the_stuck_needle_falls_off_when_spun():
    pumped()
    set_oil_true(world())
    k = (-t.START_ERROR - t.STICK) / t.TRIM_GAIN
    steps(world(knob=k), 3)
    assert abs(t.needle_error(world(knob=k))) <= t.NEEDLE_TOL  # looks right, stuck
    spun(k)
    assert t.needle_error(world(knob=k)) < -t.NEEDLE_TOL and steps(world(knob=k)) == 0.75


def test_right_turn_made_before_the_spin_is_not_a_null():
    pumped()
    set_oil_true(world())
    steps(world(knob=t.KNOB_TURN), 3)
    spun(t.KNOB_TURN)
    assert abs(t.needle_error(world(knob=t.KNOB_TURN))) <= t.NEEDLE_TOL and steps(world(knob=t.KNOB_TURN)) == 0.75


def test_right_turn_made_after_the_spin_died_is_not_a_null():
    pumped()
    set_oil_true(world())
    spun()
    steps(world(), 5)
    assert steps(world(knob=t.KNOB_TURN)) == 0.75


def test_moving_the_piston_after_the_spin_undoes_it():
    pumped()
    set_oil_true(world())
    spun()
    steps(world(spin=3.5, knob=t.KNOB_TURN), 2)
    t.note_step(world(h=H + 0.004, knob=t.KNOB_TURN), 0.05)
    assert t._trial["freed"] is False and steps(world(h=H + 0.004, knob=t.KNOB_TURN)) == 0.5


def test_needle_reads_the_real_load():
    pumped()
    set_oil_true(world())
    spun()
    w = world(knob=t.KNOB_TURN)
    assert abs(t.needle_error(w)) < 0.01
    t._trial["oil"] -= t.WEIGHT["mass"] * t.GRAVITY / t.GEOM["oil_k"]  # one weight off carries less
    assert t.needle_error(w) < -t.NEEDLE_TOL


def test_over_pump_is_zero_for_good():
    t.note_step(world(h=t.OVER_PUMP), 0.05)
    assert steps(world()) == 0.0


def test_weight_off_the_table_is_zero_for_good():
    t.note_step(world(a=state((0.2, 0.9, 0.05))), 0.05)
    assert steps(world()) == 0.0


def test_weight_leaving_loses_credit():
    assert steps(world(h=0.0, a=state((0.2, -0.2, t.TOP)))) == 0.15


def test_weight_beside_the_stem_earns_nothing():
    assert steps(world(h=0.0, a=state((t.PX, t.PY + 0.03, t.TOP + t.PLATEN)), b=at_rack("weight_b"))) == 0.0


def test_slot_turned_is_still_judged_in_the_weights_frame():
    """Turned a half turn, the stem's still in the slot; pushed 3 cm along the weight's own -x it has left the slot's closed end."""
    half = (0.0, 0.0, 1.0, 0.0)
    assert t._on_stem(world(a=on_stem(0, h=H, q=half)), "weight_a")
    moved = state((t.PX + 0.03, t.PY, t.TOP + t.PLATEN + H), q=half)
    assert t._on_stem(world(a=moved), "weight_a")
    moved = state((t.PX - 0.03, t.PY, t.TOP + t.PLATEN + H), q=half)
    assert not t._on_stem(world(a=moved), "weight_a")


def test_tilted_weight_earns_nothing():
    s = math.sin(math.radians(4)), math.cos(math.radians(4))
    assert t._on_stem(world(a=on_stem(0, q=(s[0], 0.0, 0.0, s[1]))), "weight_a") is False


def test_held_weight_does_not_count():
    above = (t.PX, t.PY, t.TOP + t.PLATEN + t.H_W + 0.013)
    assert steps(world(h=0.0, b=at_rack("weight_b"), right_tcp=above)) == 0.0


def test_open_finger_on_a_weight_does_not_count():
    near = (t.PX + t.R_W + 0.047, t.PY, t.TOP + t.PLATEN + 0.008)
    assert steps(world(h=0.0, b=at_rack("weight_b"), right_tcp=near)) == 0.0


def test_hand_on_the_knob_stops_short():
    pumped()
    set_oil_true(world())
    spun()
    steps(world(spin=3.5, knob=t.KNOB_TURN), 2)
    k = (*t.GEOM["knob_at"], t.TOP + 0.075)
    assert steps(world(knob=t.KNOB_TURN, left_tcp=k)) == 0.9


def test_lever_held_down_stops_short():
    pumped()
    set_oil_true(world())
    spun()
    steps(world(spin=3.5, knob=t.KNOB_TURN), 2)
    assert t.score(world(knob=t.KNOB_TURN, lever=t.GEOM["lever_up"] + 10.0)) == 0.9


def test_masses_agree_with_brass():
    assert t.WEIGHT["mass"] == pytest.approx(0.324, abs=0.002)
    assert 0.25 < t.WEIGHT["mass"] < 0.4


def test_band_is_not_at_a_stop_and_fits_one_stroke_only():
    sink = t.FULL_N / t.GEOM["oil_k"]
    b0, b1 = t.GEOM["band"]
    inside = [n for n in range(1, 12) if b0 <= n * t.STROKE_RISE - sink <= b1]
    assert inside == [5] and b1 < t.OVER_PUMP < t.GEOM["top_stop"] and b0 > 0.0


def test_loading_after_pumping_sinks_out_of_the_band():
    b0 = t.GEOM["band"][0]
    one = (t.GEOM["piston_mass"] + t.GEOM["rotor_mass"] + t.WEIGHT["mass"]) * t.GRAVITY / t.GEOM["oil_k"]
    sink = t.FULL_N / t.GEOM["oil_k"]
    assert b0 <= 4 * t.STROKE_RISE - one and 4 * t.STROKE_RISE - sink < b0  # in band with one weight, out with two
