"""The score must credit only parts released in the tote as the lamps show, the stocked bin's confirm pressed after its
picks and the empty bin's shortage button, and penalise extra or substitute parts, parts on the floor and every other
button press (early confirms, button mashing), in every lamp pattern."""
import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t

TOTE_FLOOR = t.TOP + t.TOTE["floor"]
CX, CY = t.TOTE["center"]


def obj(pos, moving=False, half=(0.013, 0.013, 0.0175)):
    pos = np.array(pos, float)
    return SimpleNamespace(pos=pos, low=pos - np.array(half), resting=lambda: not moving)


def part(x, y, z, moving=False):
    """A 3.5 cm tall part whose underside is at height z."""
    return obj((x, y, z + 0.0175), moving)


def tote(dz=0.0, **kw):
    return part(CX, CY, TOTE_FLOOR + dz, **kw)


def start():
    """The scene as a reset leaves it: parts in their bins (none in the empty bin), every button out."""
    s = {n: obj(p) for n, p in t.START.items() if n in t.BUTTONS or not n.startswith(t.EMPTY + "_")}
    return s


def press(s, *buttons, dx=0.006):
    s = dict(s)
    for b in buttons:
        s[b] = obj(t.START[b] + (dx, 0, 0))
    return s


def run(*states):
    """Score a sequence of steps of one episode; returns (score, success) at the last."""
    t.reset_latch()
    for s in states:
        out = t.score(s), t.success(s)
    return out


@pytest.fixture(params=sorted(t.PATTERNS))
def pat(request, monkeypatch):
    p = t.PATTERNS[request.param]
    monkeypatch.setattr(t, "LAMPS", p["lamps"])
    monkeypatch.setattr(t, "EMPTY", p["empty"])
    return p


def stocked(p):
    return next(b for b, n in p["lamps"].items() if n and b != p["empty"])


def dark(p):
    return next(b for b, n in p["lamps"].items() if not n)


def picked(p, extra=()):
    """The stocked bin's lit count released on the tote's floor (plus any named extras)."""
    s = start()
    b = stocked(p)
    for n in [*t.NAMES[b][: p["lamps"][b]], *extra]:
        s[n] = tote()
    return s


def total(p):
    return p["lamps"][stocked(p)] + 2


def done(p):
    """The correct episode: shortage, picks, then confirm."""
    s0 = start()
    s1 = press(s0, f"shortage_{p['empty']}")
    s2 = press(picked(p), f"shortage_{p['empty']}")
    return [s0, s1, s2, press(s2, f"confirm_{stocked(p)}")]


def test_patterns():
    for p in t.PATTERNS.values():
        assert sum(p["lamps"].values()) == 3 and p["lamps"][p["empty"]] > 0
        assert sum(1 for b, n in p["lamps"].items() if n and b != p["empty"]) == 1
    assert len({stocked(p) for p in t.PATTERNS.values()}) == 3  # each pattern picks from a different bin


def test_nothing_done(pat):
    assert run(start()) == (0.0, False)


def test_full_score(pat):
    assert run(*done(pat)) == (1.0, True)


def test_any_order_of_shortage_and_picks(pat):
    s = press(picked(pat), f"confirm_{stocked(pat)}")
    assert run(start(), picked(pat), s, press(s, f"shortage_{pat['empty']}")) == (1.0, True)


def test_picks_only_like_plain_kitting(pat):
    n = pat["lamps"][stocked(pat)]
    assert run(start(), picked(pat))[0] == pytest.approx(n / total(pat))


def test_shortage_only(pat):
    assert run(start(), press(start(), f"shortage_{pat['empty']}")) == (pytest.approx(1 / total(pat)), False)


def test_picks_and_confirm_without_shortage(pat):
    s = picked(pat)
    sc, ok = run(start(), s, press(s, f"confirm_{stocked(pat)}"))
    assert sc == pytest.approx((total(pat) - 1) / total(pat)) and not ok


def test_one_part_short(pat):
    b = stocked(pat)
    if pat["lamps"][b] < 2:
        pytest.skip("one-part order")
    s = start()
    s[t.NAMES[b][0]] = tote()
    assert run(start(), s) == (pytest.approx(1 / total(pat)), False)


def test_confirm_before_picking_is_wrong(pat):
    b = stocked(pat)
    s = press(start(), f"confirm_{b}", f"shortage_{pat['empty']}")
    picks = press(picked(pat), f"confirm_{b}", f"shortage_{pat['empty']}")
    sc, ok = run(start(), s, picks)
    assert sc <= t.WRONG_CAP and not ok


def test_confirm_after_a_short_pick_is_wrong(pat):
    b = stocked(pat)
    if pat["lamps"][b] < 2:
        pytest.skip("one-part order")
    s = start()
    s[t.NAMES[b][0]] = tote()
    s = press(s, f"confirm_{b}")
    full = press(picked(pat), f"confirm_{b}", f"shortage_{pat['empty']}")
    assert run(start(), s, full)[0] <= t.WRONG_CAP


def test_substitute_for_the_empty_bin(pat):
    """Taking a part from another bin in place of the empty one is wrong, from a dark bin or the stocked bin."""
    for extra in (t.NAMES[dark(pat)][0], t.NAMES[stocked(pat)][2]):
        last = press(picked(pat, [extra]), f"shortage_{pat['empty']}", f"confirm_{stocked(pat)}")
        sc, ok = run(start(), last)
        assert sc <= t.WRONG_CAP and not ok


def test_substitute_without_shortage(pat):
    s = press(picked(pat, [t.NAMES[dark(pat)][0]]), f"confirm_{stocked(pat)}")
    assert run(start(), s)[0] <= t.WRONG_CAP


@pytest.mark.parametrize("which", ["confirm_empty", "shortage_stocked", "confirm_dark", "shortage_dark"])
def test_each_wrong_button(pat, which):
    k, kind = which.split("_")
    b = {"empty": pat["empty"], "stocked": stocked(pat), "dark": dark(pat)}[kind]
    last = done(pat)[-1]
    sc, ok = run(*done(pat), press(last, f"{k}_{b}"))
    assert sc <= t.WRONG_CAP and not ok


def test_mashing_every_button(pat):
    every = list(t.BUTTONS)
    assert run(start(), press(start(), *every))[0] == 0.0
    assert run(start(), picked(pat), press(picked(pat), *every))[0] == 0.0
    assert run(*done(pat), press(done(pat)[-1], *every))[0] == 0.0


def test_wrong_never_beats_a_clean_partial(pat):
    last = press(done(pat)[-1], f"confirm_{dark(pat)}")
    assert run(*done(pat), last)[0] < 1 / total(pat)


def test_unpressing_does_not_clear_a_wrong_press(pat):
    """Pulling a wrongly pressed plunger back out to where it started: the press stays counted."""
    states = done(pat)
    wrong = press(states[-1], f"confirm_{dark(pat)}")
    assert run(*states, wrong, states[-1]) == (t.WRONG_CAP, False)


def test_button_forced_out_of_its_rails_counts_as_pressed(pat):
    last = dict(done(pat)[-1])
    b = f"shortage_{dark(pat)}"
    last[b] = obj(t.START[b] + (0, 0.02, 0.03))  # lifted out and set aside
    assert run(*done(pat), last)[0] <= t.WRONG_CAP


def test_small_nudge_is_not_a_press(pat):
    s = press(done(pat)[-1], f"confirm_{dark(pat)}", dx=0.002)
    assert run(*done(pat), s) == (1.0, True)


def test_reset_clears_the_latch(pat):
    run(*done(pat))
    assert t.score(start()) == 0.0 and not t._PRESSED


def test_extra_from_stocked_bin(pat):
    s = press(picked(pat, [t.NAMES[stocked(pat)][2]]), f"shortage_{pat['empty']}")
    assert run(start(), s)[0] <= t.WRONG_CAP


def test_extra_stacked_on_a_part_counts(pat):
    s = done(pat)[-1]
    s = dict(s, **{t.NAMES[dark(pat)][0]: tote(dz=0.037)})
    assert run(*done(pat), s) == (t.WRONG_CAP, False)


def test_extra_held_over_tote_counts(pat):
    s = dict(done(pat)[-1], **{t.NAMES[dark(pat)][0]: tote(dz=0.06, moving=True)})
    assert run(*done(pat), s)[0] == t.WRONG_CAP


def test_part_far_above_tote_not_counted(pat):
    s = dict(done(pat)[-1], **{t.NAMES[dark(pat)][0]: tote(dz=0.15, moving=True)})
    assert run(*done(pat), s) == (1.0, True)


def test_held_part_not_credited(pat):
    """A part still in the gripper over the tote floor, or still moving, isn't placed."""
    b = stocked(pat)
    for held in (tote(dz=0.02), tote(moving=True)):
        s = dict(done(pat)[-1], **{t.NAMES[b][0]: held})
        assert run(*done(pat), s) == (pytest.approx((total(pat) - 1) / total(pat)), False)


def test_part_on_the_floor_costs(pat):
    s = dict(done(pat)[-1], **{t.NAMES[dark(pat)][2]: part(0.5, 0.5, t.TOP - 0.7475)})
    assert run(*done(pat), s) == (t.WRONG_CAP, False)


def test_part_knocked_onto_table_is_free(pat):
    s = dict(done(pat)[-1], **{t.NAMES[dark(pat)][2]: part(0.30, 0.25, t.TOP)})
    assert run(*done(pat), s) == (1.0, True)


def test_part_just_outside_tote_not_counted(pat):
    s = start()
    s[t.NAMES[stocked(pat)][0]] = part(CX + t.TOTE["inner"][0] / 2 + 0.005, CY, t.TOP)
    assert run(start(), s)[0] == 0.0


def test_emptying_the_rack(pat):
    s = {n: (tote() if n not in t.BUTTONS else o) for n, o in start().items()}
    assert run(start(), s)[0] == 0.0


def test_score_bounds(pat):
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = {n: (tote(dz=rng.choice([0.0, 0.037])) if n not in t.BUTTONS and rng.random() < 0.4 else o)
             for n, o in start().items()}
        s = press(s, *[b for b in t.BUTTONS if rng.random() < 0.3])
        assert 0.0 <= run(start(), s)[0] <= 1.0


def test_scene_matches_pattern(monkeypatch):
    """The lit lamps are the pattern's, and the empty bin has no parts (the scene is built at import: rebuild it)."""
    for name, p in t.PATTERNS.items():
        monkeypatch.setenv("PICK_TO_LIGHT_PATTERN", name.lower())
        m = importlib.reload(t)
        lit = {o.name for o in m.OBJECTS if o.name.startswith("lamp_") and tuple(o.color) == m.LIT}
        assert lit == {f"lamp_{b}_{k}" for b, n in p["lamps"].items() for k in range(n)}
        names = {o.name for o in m.OBJECTS}
        assert not any(n in names for n in m.NAMES[p["empty"]])
        assert all(n in names for b in m.BIN_Y if b != p["empty"] for n in m.NAMES[b])
        assert set(m.BUTTONS) <= names
    monkeypatch.delenv("PICK_TO_LIGHT_PATTERN")
    importlib.reload(t)


def test_buttons_clear_of_each_other_and_the_tote():
    ys = sorted(t._button_y(n) for n in t.BUTTONS)
    assert min(np.diff(ys)) >= 0.04 - 1e-9
    assert t.BUTTON_X + t.BUTTON[0] / 2 + 0.0075 == pytest.approx(t.STRIP_FACE)  # 7.5 mm travel to the stop
    tote_far = t.TOTE["center"][0] + t.TOTE["inner"][0] / 2 + t.TOTE["wall"]
    assert t.PRESS_FROM_X - 0.0125 > tote_far  # closed pads behind the plunger clear the tote wall


def test_demo_fits_run_limit_and_ik_reaches_every_waypoint():
    for name in t.PATTERNS:
        wps = t.demo(name)
        assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 2400
        pe, re = t.ik_errors(wps)
        assert pe < 1e-3 and re < math.radians(0.5)


def test_demo_does_the_order():
    """Shortage on the empty bin first, the stocked bin's parts, then its confirm, and nothing else."""
    for name, p in t.PATTERNS.items():
        wps = t.demo(name)
        steps = sorted(((sum(w.duration for w in ws[:i]), w.label.split(":")[0]) for ws in wps.values()
                        for i, w in enumerate(ws) if ":" in w.label))
        b = stocked(p)
        assert [s for _, s in steps] == [f"shortage_{p['empty']}", *t.NAMES[b][: p["lamps"][b]], f"confirm_{b}"]
        assert all(w.pos[0] == t.PRESS_TO_X for ws in wps.values() for w in ws if w.label == "push it in")


def _slowly(a, b, step=0.005):
    """Steps moving every object from state a to state b at most `step` per control step, then at rest at b."""
    n = max(1, int(max(np.linalg.norm(b[k].pos - a[k].pos) for k in b) / step) + 1)
    out = [{k: obj(a[k].pos + (b[k].pos - a[k].pos) * i / n, moving=True) for k in b} for i in range(1, n)]
    return [*out, b]


def test_robot_cannot_erase_a_wrong_press_by_putting_everything_back(pat):
    """Council exploit (press-jam, lockout): press wrong buttons, pull them back and return the parts slowly; the
    mistakes must survive, and redoing the order correctly afterwards stays capped."""
    every = press(picked(pat), *t.BUTTONS)
    states = [start(), picked(pat), every, *_slowly(every, start()), *done(pat)]
    sc, ok = run(*states)
    assert sc <= t.WRONG_CAP and not ok
    assert f"confirm_{dark(pat)}" in t._PRESSED


def test_harness_reset_teleport_clears_presses(pat):
    """A real reset teleports everything back in one step: a new trial, and a clean run then scores 1.0."""
    every = press(picked(pat), *t.BUTTONS)
    assert run(start(), picked(pat), every, start(), *done(pat)) == (1.0, True)


def test_teleport_without_coming_to_rest_is_not_a_reset(pat):
    every = press(picked(pat), *t.BUTTONS)
    moving = {k: obj(o.pos, moving=True) for k, o in start().items()}
    run(start(), picked(pat), every, moving)
    assert f"confirm_{dark(pat)}" in t._PRESSED
