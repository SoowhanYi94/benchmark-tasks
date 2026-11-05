"""One bin is lit at a time. The score must credit only parts let go in the tote as the lamps show, each stocked bin's
confirm pressed after its picks and the empty bin's shortage button, in order, and penalise extra, substitute or early
parts, parts on the floor and every other button press, in every order. Only a press on the lit bin moves the order on."""
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
    return SimpleNamespace(pos=pos, low=pos - np.array(half), high=pos + np.array(half), resting=lambda: not moving)


TCPS = {f"{a}_tcp": obj((t.PARK[0], t.SIDE[a] * t.PARK[1], t.PARK[2]), half=(0, 0, 0)) for a in t.SIDE}
LOOSE = lambda n: n not in t.BUTTONS and n not in TCPS  # noqa: E731  (a part)


def part(x, y, z, moving=False):
    """A 3.5 cm tall part whose underside is at height z."""
    return obj((x, y, z + 0.0175), moving)


def tote(dz=0.0, **kw):
    return part(CX, CY, TOTE_FLOOR + dz, **kw)


def start():
    """The scene as a reset leaves it: parts in their bins (none in the empty bin), every button out."""
    s = {n: obj(p) for n, p in t.START.items() if n in t.BUTTONS or not n.startswith(t.EMPTY + "_")}
    return {**s, **TCPS}  # both arms parked


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
    monkeypatch.setattr(t, "ORDER", p["order"])
    monkeypatch.setattr(t, "LAMPS", {b: dict(p["order"]).get(b, 0) for b in t.BIN_Y})
    monkeypatch.setattr(t, "EMPTY", p["empty"])
    return p


def dark(p):
    """The one bin the order never lights."""
    return next(b for b in t.BIN_Y if b not in dict(p["order"]))


def stocked(p):
    return [b for b, _ in p["order"] if b != p["empty"]]


def total(p):
    return sum(n + 1 for b, n in p["order"] if b != p["empty"]) + 1


def walk(p):
    """Every state of the correct episode: each bin in turn, parts then confirm, or shortage."""
    s, out = start(), [start()]
    for b, n in p["order"]:
        if b == p["empty"]:
            s = press(s, f"shortage_{b}")
            out.append(s)
            continue
        for name in t.NAMES[b][:n]:
            s = dict(s, **{name: tote()})
            out.append(s)
        s = press(s, f"confirm_{b}")
        out.append(s)
    return out


def test_patterns():
    for p in t.PATTERNS.values():
        assert len(p["order"]) == 3 and p["empty"] in dict(p["order"]) and len(dict(p["order"])) == 3
        assert all(n in (1, 2) for _, n in p["order"])
    assert len({p["order"] for p in t.PATTERNS.values()}) == 3


def test_nothing_done(pat):
    assert run(start()) == (0.0, False)


def test_full_score(pat):
    assert run(*walk(pat)) == (1.0, True)


def test_lamps_move_on_only_with_a_press_on_the_lit_bin(pat):
    run(start())
    assert t.active() == pat["order"][0]
    wrong = press(start(), f"confirm_{dark(pat)}")
    run(start(), wrong)
    assert t.active() == pat["order"][0]
    states = walk(pat)
    run(*states)
    assert t.active() is None


def test_each_good_press_lights_the_next_bin(pat):
    t.reset_latch()
    seen = []
    for s in walk(pat):
        t.score(s)
        if not seen or seen[-1] != t.active():
            seen.append(t.active())
    assert seen == [*pat["order"], None]


def test_order_cannot_be_done_up_front(pat):
    """Pressing the last bin's button first is wrong and doesn't move the order on."""
    last = pat["order"][-1][0]
    k = "shortage" if last == pat["empty"] else "confirm"
    assert run(start(), press(start(), f"{k}_{last}"))[0] <= t.WRONG_CAP
    assert t.active() == pat["order"][0]


def test_part_from_a_bin_not_lit_yet_is_wrong_even_if_it_lights_later(pat):
    later = next(b for b, _ in pat["order"][1:] if b != pat["empty"])
    early = dict(start(), **{t.NAMES[later][0]: tote()})
    states = [start(), early, *[{**s, **{t.NAMES[later][0]: tote()}} for s in walk(pat)]]
    sc, ok = run(*states)
    assert sc <= t.WRONG_CAP and not ok and later in t._EARLY


def test_partial_order_scores_its_share(pat):
    w = walk(pat)
    assert run(*w[:2])[0] == pytest.approx((1 if pat["order"][0][0] == pat["empty"] else 1) / total(pat))


def test_confirm_before_picking_is_wrong(pat):
    b, n = pat["order"][0]
    if b == pat["empty"]:
        pytest.skip("first bin is the empty one")
    assert run(start(), press(start(), f"confirm_{b}"))[0] <= t.WRONG_CAP


def test_confirm_short_of_the_count_is_wrong(pat):
    b, n = next((b, n) for b, n in pat["order"] if b != pat["empty"] and n == 2) if any(
        b != pat["empty"] and n == 2 for b, n in pat["order"]) else (None, 0)
    if b is None:
        pytest.skip("no two-part bin")
    i = [x for x, _ in pat["order"]].index(b)
    w = walk(pat)
    upto = next(k for k, s in enumerate(w) if t.active is not None and t.NAMES[b][0] in s
                and np.allclose(s[t.NAMES[b][0]].pos, tote().pos))
    s = press(w[upto], f"confirm_{b}")
    assert run(*w[: upto + 1], s)[0] <= t.WRONG_CAP


def test_confirm_on_the_empty_bin_is_wrong(pat):
    w = walk(pat)
    i = [b for b, _ in pat["order"]].index(pat["empty"])
    upto = next(k for k, s in enumerate(w) if f"shortage_{pat['empty']}" in s and not np.allclose(
        s[f"shortage_{pat['empty']}"].pos, t.START[f"shortage_{pat['empty']}"]))
    s = press(w[upto - 1], f"confirm_{pat['empty']}")
    assert run(*w[:upto], s)[0] <= t.WRONG_CAP


def test_shortage_on_a_stocked_bin_is_wrong(pat):
    b = stocked(pat)[0]
    assert run(*walk(pat), press(walk(pat)[-1], f"shortage_{b}"))[0] <= t.WRONG_CAP


def test_substitute_for_the_empty_bin(pat):
    sub = t.NAMES[dark(pat)][0]
    last = dict(walk(pat)[-1], **{sub: tote()})
    assert run(*walk(pat), last)[0] == t.WRONG_CAP


@pytest.mark.parametrize("k", ["confirm", "shortage"])
def test_press_on_the_dark_bin_is_wrong(pat, k):
    assert run(*walk(pat), press(walk(pat)[-1], f"{k}_{dark(pat)}")) == (t.WRONG_CAP, False)


def test_mashing_every_button(pat):
    assert run(start(), press(start(), *t.BUTTONS))[0] == 0.0
    assert run(*walk(pat), press(walk(pat)[-1], *t.BUTTONS))[0] <= t.WRONG_CAP


def test_wrong_never_beats_a_clean_partial(pat):
    assert run(*walk(pat), press(walk(pat)[-1], f"confirm_{dark(pat)}"))[0] < 2 / total(pat) + t.WRONG_CAP


def test_unpressing_does_not_clear_a_wrong_press(pat):
    w = walk(pat)
    wrong = press(w[-1], f"confirm_{dark(pat)}")
    assert run(*w, wrong, w[-1]) == (t.WRONG_CAP, False)


def test_button_forced_out_of_its_rails_counts_as_pressed(pat):
    last = dict(walk(pat)[-1])
    b = f"shortage_{dark(pat)}"
    last[b] = obj(t.START[b] + (0, 0.02, 0.03))
    assert run(*walk(pat), last)[0] <= t.WRONG_CAP


def test_small_nudge_is_not_a_press(pat):
    assert run(*walk(pat), press(walk(pat)[-1], f"confirm_{dark(pat)}", dx=0.002)) == (1.0, True)


def test_extra_from_a_stocked_bin(pat):
    b = stocked(pat)[0]
    extra = dict(walk(pat)[-1], **{t.NAMES[b][2]: tote()})
    assert run(*walk(pat), extra)[0] == t.WRONG_CAP


def test_held_part_not_credited(pat):
    """A part still in the gripper over the tote floor, or still moving, isn't placed."""
    b = stocked(pat)[0]
    for held in (tote(dz=0.02), tote(moving=True)):
        s = dict(walk(pat)[-1], **{t.NAMES[b][0]: held})
        assert run(*walk(pat), s) == (pytest.approx((total(pat) - 1) / total(pat)), False)


def test_part_on_the_floor_costs(pat):
    s = dict(walk(pat)[-1], **{t.NAMES[dark(pat)][2]: part(0.5, 0.5, t.TOP - 0.7475)})
    assert run(*walk(pat), s) == (t.WRONG_CAP, False)


def test_part_knocked_onto_table_is_free(pat):
    s = dict(walk(pat)[-1], **{t.NAMES[dark(pat)][2]: part(0.30, 0.25, t.TOP)})
    assert run(*walk(pat), s) == (1.0, True)


def test_part_just_outside_tote_not_counted(pat):
    b = pat["order"][0][0]
    if b == pat["empty"]:
        pytest.skip("first bin is the empty one")
    s = dict(start(), **{t.NAMES[b][0]: part(CX + t.TOTE["inner"][0] / 2 + 0.005, CY, t.TOP)})
    assert run(start(), s)[0] == 0.0


def test_emptying_the_rack(pat):
    s = {n: (tote() if LOOSE(n) else o) for n, o in start().items()}
    assert run(start(), s)[0] == 0.0


def test_scene_matches_pattern(monkeypatch):
    for name, p in t.PATTERNS.items():
        monkeypatch.setenv("PICK_TO_LIGHT_PATTERN", name.lower())
        m = importlib.reload(t)
        lit = {o.name for o in m.OBJECTS if o.name.startswith("lamp_") and tuple(o.color) == m.LIT}
        b, n = p["order"][0]
        assert lit == {f"lamp_{b}_{k}" for k in range(n)}  # only the first bin starts lit
        names = {o.name for o in m.OBJECTS}
        assert not any(x in names for x in m.NAMES[p["empty"]])
        assert all(x in names for y in m.BIN_Y if y != p["empty"] for x in m.NAMES[y])
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
    """Each bin in turn: its parts then confirm, or its shortage, and nothing else."""
    for name, p in t.PATTERNS.items():
        wps = t.demo(name)
        steps = sorted(((sum(w.duration for w in ws[:i]), w.label.split(":")[0]) for ws in wps.values()
                        for i, w in enumerate(ws) if ":" in w.label))
        want = []
        for b, n in p["order"]:
            want += [f"shortage_{b}"] if b == p["empty"] else [*t.NAMES[b][:n], f"confirm_{b}"]
        assert [s for _, s in steps] == want
        assert all(w.pos[0] == t.PRESS_TO_X for ws in wps.values() for w in ws if w.label == "push it in")


def _slowly(a, b, step=0.005):
    """Steps moving every object from state a to state b at most `step` per control step, then at rest at b."""
    n = max(1, int(max(np.linalg.norm(b[k].pos - a[k].pos) for k in b) / step) + 1)
    out = [{k: obj(a[k].pos + (b[k].pos - a[k].pos) * i / n, moving=True) for k in b} for i in range(1, n)]
    return [*out, b]


def test_robot_cannot_erase_a_wrong_press_by_putting_everything_back(pat):
    """Press wrong buttons, pull them back and return the parts slowly; the mistakes must survive."""
    every = press(start(), *t.BUTTONS)
    sc, ok = run(start(), every, *_slowly(every, start()), *walk(pat))
    assert sc <= t.WRONG_CAP and not ok and f"confirm_{dark(pat)}" in t._PRESSED


def test_part_still_gripped_on_the_tote_floor_is_not_placed(pat):
    """A part held still on the tote floor isn't placed; once the fingertips are 2 cm clear it is."""
    b = stocked(pat)[0]
    arm = f"{t._arm(b)}_tcp"
    held = part(CX - 0.03, CY + 0.04, TOTE_FLOOR)
    last = dict(walk(pat)[-1], **{t.NAMES[b][0]: held})
    gripping = dict(last, **{arm: obj(held.pos + (0, 0, 0.0075), half=(0, 0, 0))})
    assert run(*walk(pat), gripping) == (pytest.approx((total(pat) - 1) / total(pat)), False)
    clear = dict(last, **{arm: obj((*held.pos[:2], held.high[2] + 0.021), half=(0, 0, 0))})
    assert run(*walk(pat), clear) == (1.0, True)


def test_scorer_needs_the_fingertips(pat):
    """Run with plain yam_arena (no fingertip states), the scorer fails loudly rather than credit held parts."""
    s = {n: o for n, o in dict(start(), **{t.NAMES[stocked(pat)[0]][0]: tote()}).items() if n not in TCPS}
    with pytest.raises(KeyError):
        run(s)


def test_demo_lets_go_well_clear_of_each_part():
    """After each release the TCP rises to 10 cm: over 2 cm above a 3.5 cm part standing on the tote floor."""
    part_top = t.TOP + t.TOTE["floor"] + 0.035
    for name in t.PATTERNS:
        for ws in t.demo(name).values():
            for i, w in enumerate(ws):
                if w.label == "release":
                    assert w.pos[2] - part_top > 0.03 and ws[i + 1].pos[2] == t.Z_UP


def _first_bin_picked(pat):
    b, lit = t.ORDER[0]
    return b, dict(start(), **{t.NAMES[b][i]: tote() for i in range(lit)})


def test_only_pushing_in_is_a_right_press(pat):
    b, s = _first_bin_picked(pat)
    n = f"confirm_{b}"
    run(start(), press(s, n))
    assert t._PRESSED[n] is True
    run(start(), press(s, n, dx=-0.006))  # pulled out toward the arms
    assert t._PRESSED[n] is False


def test_dumping_the_first_lit_bin_scores_nothing(pat):
    b = t.ORDER[0][0]
    assert run(start(), dict(start(), **{n: tote() for n in t.NAMES[b]}))[0] == 0
