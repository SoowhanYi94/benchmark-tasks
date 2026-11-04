"""The score must credit the full lockout cycle in order (lock, verify, swap, restore) and give 0 for the whole trial
once a cutter on the machine is disturbed while it isn't locked out and verified, whatever happens afterwards."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t

CUT = (2 * t.CUTTER_R, 2 * t.CUTTER_R, t.CUTTER_H)
FAR_L, FAR_R = np.array(t.OUT_LEFT), np.array(t.OUT_RIGHT)


def part(x, y, z, size, yaw=0.0, roll=0.0, moving=False):
    """A part whose underside is at height z (before rolling), turned yaw and rolled about x."""
    q = np.array((math.sin(roll / 2) * math.cos(yaw / 2), -math.sin(roll / 2) * math.sin(yaw / 2),
                  math.cos(roll / 2) * math.sin(yaw / 2), math.cos(roll / 2) * math.cos(yaw / 2)))
    o = SimpleNamespace(pos=np.array((x, y, z + size[2] / 2)), quat_xyzw=q, resting=lambda: not moving)
    c = t.corners(o, size)
    o.low, o.high = c.min(axis=0), c.max(axis=0)
    return o


def point(p):
    p = np.asarray(p, float)
    return SimpleNamespace(pos=p, low=p, high=p, quat_xyzw=np.array((0, 0, 0, 1.0)), resting=lambda: True)


HC = t.HOLDER["center"]
BIN = (*t.SCRAP["center"], t.TOP + t.SCRAP["floor"])
MARK_Z = t.TOP + t.LOCK_MARK["size"][2]


def handle(x, **kw):
    return part(x, t.TRACK_Y, t.TOP, t.HANDLE, **kw)


def lock(x=t.LOCK_X, y=t.TRACK_Y, z=t.TOP, **kw):
    return part(x, y, z, t.LOCK, **kw)


def button(depth=0.0):
    return point((*t.CONTROL["center"], t.BUTTON_TOP - depth))


def state(**kw):
    s = {"isolator_handle": handle(t.ON_X), "padlock": lock(*t.LOCK_START, MARK_Z),
         "worn_cutter": part(*HC, t.HOLDER_FLOOR, CUT), "new_cutter": part(*t.NEW_START, t.TOP + t.TOOL_TRAY["size"][2], CUT),
         "start_button": button(), "left_tcp": point(FAR_L), "right_tcp": point(FAR_R)}
    return dict(s, **kw)


OFF = dict(isolator_handle=handle(t.OFF_X))
LOCKED = dict(OFF, padlock=lock())
SWAPPED = dict(worn_cutter=part(*BIN, CUT), new_cutter=part(*HC, t.HOLDER_FLOOR, CUT))


def run(*states):
    """Score a trial step by step; return the last score."""
    return [t.score(s) for s in states][-1]


@pytest.fixture(autouse=True)
def fresh_process():
    t._trial.clear()
    t._trial.update(start=None, last=None)


def lock_steps():
    return [state(), state(**OFF), state(**OFF, padlock=lock(z=t.TOP + 0.03)), state(**LOCKED)]


def verify_steps():
    return [state(**LOCKED, start_button=button(0.006)), state(**LOCKED)]


def swap_steps():
    lifted = part(*HC, t.HOLDER_FLOOR + 0.06, CUT)
    return [state(**LOCKED, worn_cutter=lifted), state(**LOCKED, worn_cutter=part(*BIN, CUT)), state(**LOCKED, **SWAPPED)]


def restore_steps():
    return [state(**OFF, **SWAPPED, padlock=lock(z=t.TOP + 0.05)), state(**OFF, **SWAPPED, padlock=lock(*t.LOCK_START, MARK_Z)),
            state(**SWAPPED, padlock=lock(*t.LOCK_START, MARK_Z))]


def full():
    return [*lock_steps(), *verify_steps(), *swap_steps(), *restore_steps()]


def approx(a, b):
    return abs(a - b) < 1e-9


# -- every outcome -----------------------------------------------------------------------------------------------
def test_nothing_done():
    assert run(state()) == 0.0 and not t.success(state())


def test_full_cycle():
    assert approx(run(*full()), 1.0) and t.success(full()[-1])


def test_partial_credit_in_order():
    assert approx(run(*lock_steps()), 0.15)
    t._trial.update(start=None, last=None)
    assert approx(run(*lock_steps(), *verify_steps()), 0.3)
    t._trial.update(start=None, last=None)
    assert approx(run(*lock_steps(), *verify_steps(), *swap_steps()[:2]), 0.5)
    t._trial.update(start=None, last=None)
    assert approx(run(*lock_steps(), *verify_steps(), *swap_steps()), 0.8)


def test_new_cutter_only():
    s = [*lock_steps(), *verify_steps(), state(**LOCKED, new_cutter=part(*t.NEW_START, t.TOP + 0.1, CUT))]
    s += [state(**LOCKED, worn_cutter=part(0.25, 0.15, t.TOP, CUT), new_cutter=part(*HC, t.HOLDER_FLOOR, CUT))]
    assert approx(run(*s), 0.6)


def test_left_locked_after_swap_keeps_lock_credit():
    assert approx(run(*lock_steps(), *verify_steps(), *swap_steps()), 0.8)


def test_padlock_removed_but_handle_left_off():
    assert approx(run(*full()[:-1]), 0.8)


def test_handle_on_but_padlock_not_on_its_mark():
    s = full()
    s[-1] = state(**SWAPPED, padlock=lock(0.10, -0.30, t.TOP))
    assert approx(run(*s), 0.8)


def slide(x0, x1, **kw):
    """The handle slid in 5 mm steps (a robot can't jump it there in one step)."""
    n = int(round(abs(x1 - x0) / 0.005))
    return [state(isolator_handle=handle(x0 + (x1 - x0) * k / n), **kw) for k in range(1, n + 1)]


def test_restore_without_swap_earns_nothing_more():
    s = [*lock_steps(), *verify_steps(), state(**OFF, padlock=lock(*t.LOCK_START, MARK_Z)),
         *slide(t.OFF_X, t.ON_X, padlock=lock(*t.LOCK_START, MARK_Z))]
    assert approx(run(*s), 0.3)


# -- the order and the latch ---------------------------------------------------------------------------------------
def test_service_before_lockout_is_zero_even_after_doing_it_right():
    early = dict(worn_cutter=part(*BIN, CUT))
    s = [state(), state(worn_cutter=part(*HC, t.HOLDER_FLOOR + 0.05, CUT)), state(**early),
         state(**OFF, **early), state(**LOCKED, **early), state(**LOCKED, **early, start_button=button(0.006)),
         state(**LOCKED, **SWAPPED), *restore_steps()]
    assert run(*s) == 0.0


def test_touch_after_lock_before_verify_is_zero():
    s = [*lock_steps(), state(**LOCKED, worn_cutter=part(*HC, t.HOLDER_FLOOR + 0.05, CUT)), *verify_steps(), *swap_steps(),
         *restore_steps()]
    assert run(*s) == 0.0


def test_fake_reset_does_not_clear_the_flag():
    """The council's exploit: lift the cutter early, put it back exactly where it was (no 1 cm jump), redo it right."""
    up = [state(worn_cutter=part(*HC, t.HOLDER_FLOOR + h, CUT)) for h in (0.004, 0.008, 0.004)]
    assert run(state(), *up, state(), *full()[1:]) == 0.0


def test_real_reset_clears_the_flag():
    run(state(), state(worn_cutter=part(*HC, t.HOLDER_FLOOR + 0.05, CUT)), state(worn_cutter=part(*BIN, CUT)))
    assert t._trial["violated"]
    assert approx(run(state(), *full()[1:]), 1.0)  # everything back at its start in one step: a reset


def test_twist_tilt_or_small_lift_counts_as_touching():
    for w in (part(*HC, t.HOLDER_FLOOR, CUT, yaw=math.radians(15)), part(*HC, t.HOLDER_FLOOR, CUT, roll=math.radians(5)),
              part(*HC, t.HOLDER_FLOOR + 0.004, CUT), part(HC[0] + 0.004, HC[1], t.HOLDER_FLOOR, CUT)):
        t._trial.update(start=None, last=None)
        assert run(state(), state(worn_cutter=w), *full()[1:]) == 0.0


def test_settling_jitter_is_not_touching():
    assert approx(run(state(), state(worn_cutter=part(HC[0] + 0.001, HC[1], t.HOLDER_FLOOR, CUT)), *full()[1:]), 1.0)


def test_new_cutter_into_the_machine_before_lockout_is_zero():
    assert run(state(), state(new_cutter=part(*HC, t.HOLDER_FLOOR + 0.12, CUT)), *full()[1:]) == 0.0


def test_new_cutter_moved_on_its_tray_is_fine():
    assert approx(run(state(), state(new_cutter=part(0.20, 0.22, t.TOP + 0.006, CUT)), *full()[1:]), 1.0)


def test_touching_cutter_after_restore_is_zero():
    s = full()
    assert run(*s, state(**dict(SWAPPED, new_cutter=part(*HC, t.HOLDER_FLOOR + 0.03, CUT)))) == 0.0


def test_unlock_mid_swap_then_touch_is_zero():
    s = [*lock_steps(), *verify_steps(), state(**LOCKED, worn_cutter=part(*BIN, CUT)),
         state(**OFF, worn_cutter=part(*BIN, CUT), padlock=lock(*t.LOCK_START, MARK_Z)),
         state(**OFF, worn_cutter=part(*BIN, CUT), new_cutter=part(*HC, t.HOLDER_FLOOR, CUT), padlock=lock(*t.LOCK_START, MARK_Z))]
    assert run(*s) == 0.0


def test_relock_must_be_verified_again():
    s = [*lock_steps(), *verify_steps(), state(**OFF, padlock=lock(z=t.TOP + 0.05)), state(**LOCKED),
         state(**LOCKED, worn_cutter=part(*HC, t.HOLDER_FLOOR + 0.05, CUT))]
    assert run(*s) == 0.0


def test_press_before_lockout_does_not_verify():
    s = [state(), state(start_button=button(0.006)), state(), *lock_steps()[1:],
         state(**LOCKED, worn_cutter=part(*HC, t.HOLDER_FLOOR + 0.05, CUT))]
    assert run(*s) == 0.0


def test_shallow_press_does_not_verify():
    assert approx(run(*lock_steps(), state(**LOCKED, start_button=button(0.003)), state(**LOCKED)), 0.15)


def test_press_while_live_lights_the_lamp_but_is_not_a_violation():
    assert t.energised(state()) and not t.energised(state(**OFF))
    assert approx(run(state(start_button=button(0.006)), *full()[1:]), 1.0)


# -- the lockout itself ------------------------------------------------------------------------------------------
def test_padlock_in_before_handle_at_off_does_not_lock():
    s = [state(), state(padlock=lock(0.145)), state(isolator_handle=handle(t.OFF_X), padlock=lock(0.145))]
    assert run(*s, state(**OFF, padlock=lock(0.145), start_button=button(0.006))) == 0.0


def test_padlock_against_the_handle_does_not_count():
    assert run(state(), state(**OFF), state(**OFF, padlock=lock(t.OFF_X - 0.03))) == 0.0


def test_handle_held_up_or_on_the_rails_is_not_off():
    for h in (part(t.OFF_X, t.TRACK_Y, t.TOP + 0.10, t.HANDLE), part(t.OFF_X, t.TRACK_Y, t.TOP + 0.03, t.HANDLE, yaw=math.pi / 2),
              part(t.OFF_X, t.TRACK_Y + 0.015, t.TOP, t.HANDLE)):
        t._trial.update(start=None, last=None)
        assert run(state(), state(isolator_handle=h), state(isolator_handle=h, padlock=lock())) == 0.0


def test_held_items_do_not_count():
    at_lock = point(lock().pos)
    assert run(state(), state(**OFF), state(**LOCKED, right_tcp=at_lock)) == 0.0
    t._trial.update(start=None, last=None)
    s = [*lock_steps(), *verify_steps(), *swap_steps()[:-1], state(**LOCKED, **SWAPPED, left_tcp=point(SWAPPED["new_cutter"].pos))]
    assert approx(run(*s), 0.5)
    t._trial.update(start=None, last=None)
    s = [*lock_steps(), *verify_steps(), state(**LOCKED, worn_cutter=part(*BIN, CUT), left_tcp=point(part(*BIN, CUT).pos))]
    assert approx(run(*s), 0.3)


def test_cutter_tipped_in_the_holder_is_not_seated():
    tipped = dict(SWAPPED, new_cutter=part(*HC, t.HOLDER_FLOOR, CUT, roll=math.radians(15)))
    s = [*lock_steps(), *verify_steps(), *swap_steps()[:-1], state(**LOCKED, **tipped),
         state(**OFF, **tipped, padlock=lock(*t.LOCK_START, MARK_Z)), state(**tipped, padlock=lock(*t.LOCK_START, MARK_Z))]
    assert approx(run(*s), 0.5)


# -- the demo --------------------------------------------------------------------------------------------------------
def test_demo_fits_and_every_waypoint_solves():
    total = max(sum(w.duration for w in wps) for wps in t.demo().values())
    pe, re = t.ik_check()
    assert total * 20 < 2400 and pe < 0.001 and re < math.radians(1)
