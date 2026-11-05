"""The score must credit each block only in the tray its two gauge limits send it to, once it is let go and still;
cancel three correct blocks for each block in a wrong tray or off the table; and give blind or one-sided play nothing
(or next to nothing)."""
import importlib
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t

PARK = {f"{a}_tcp": p for a, p in t.PARK.items()}


def point(p):
    p = np.array(p, float)
    return SimpleNamespace(pos=p, low=p, high=p, resting=lambda: True)


def block(name, x, y, z, moving=False, on_side=False):
    """A block whose underside is at height z, lying flat (or on its side)."""
    L, W, H = t.PARTS[name]["size"]
    size = np.array((L, H, W) if on_side else (L, W, H))
    c = np.array((x, y, z + size[2] / 2))
    return SimpleNamespace(pos=c, low=c - size / 2, high=c + size / 2, resting=lambda: not moving)


def at(name, tray, k=0, **kw):
    (cx, cy) = t.TRAYS[tray]["center"]
    return block(name, cx + (-0.04, 0.04)[k], cy, t.TOP + t.TRAY_FLOOR, **kw)


def state(**blocks):
    s = {n: block(n, *p["start"], t.TOP) for n, p in t.PARTS.items()}
    s.update({k: point(v) for k, v in PARK.items()})
    s["lot_tag"] = tag(*t.TAG["start"])
    s.update(blocks)
    return s


def tag(x, y, moving=False, dz=0.0):
    h = t.TAG["size"] / 2
    c = np.array((x, y, t.TOP + t.PAD_H + h + dz))
    return SimpleNamespace(pos=c, low=c - h, high=c + h, resting=lambda: not moving)


def on_pad(pad=None, **kw):
    return tag(*t.LOT_PADS[pad or t.lot_decision()]["center"], **kw)


def done():
    used = {k: 0 for k in t.TRAYS}
    out = {}
    for n in t.PARTS:
        out[n] = at(n, t.kind(n), used[t.kind(n)])
        used[t.kind(n)] += 1
    return out


def sort_score(s):
    """The score on the sort's own 0 to 1 scale (its share of the task's score), with the tag left at its start."""
    return t.score(s) / t.SORT_SHARE


def approx(a, b):
    return abs(a - b) < 1e-9


# -- the gauge itself ----------------------------------------------------------------------------------------------
def test_two_of_each_kind():
    kinds = [t.kind(n) for n in t.PARTS]
    assert sorted(kinds) == ["accept"] * 2 + ["over"] * 2 + ["under"] * 2


def test_limits_and_clearances():
    """Passing blocks clear the bar by 2 mm; blocked ones are 1.5 mm too tall for it."""
    for n, p in t.PARTS.items():
        h = p["size"][2]
        if t.kind(n) == "over":
            assert h - t.GO_H >= 0.0015 - 1e-9
        else:
            assert t.GO_H - h >= 0.002 - 1e-9
        if t.kind(n) == "under":
            assert t.NOGO_H - h >= 0.002 - 1e-9
        else:
            assert h - t.NOGO_H >= 0.0015 - 1e-9


def test_height_is_the_smallest_side_so_no_other_orientation_passes():
    """Gauged on its side or end, every block is wider than the GO gap: lying flat is the only real test."""
    for p in t.PARTS.values():
        L, W, H = p["size"]
        assert H < W < L and W >= t.GO_H + 0.0015 - 1e-9


def test_masses_overlap_between_kinds():
    m = {n: t.ALU_DENSITY * float(np.prod(p["size"])) for n, p in t.PARTS.items()}
    rng = {k: [m[n] for n in t.PARTS if t.kind(n) == k] for k in ("accept", "over", "under")}
    for a, b in itertools.combinations(rng.values(), 2):
        assert min(a) < max(b) and min(b) < max(a)
    assert t.kind(max(m, key=m.get)) != "over" and max(m.values()) < 0.27


def test_heights_differ_only_by_millimetres_and_less_than_footprints():
    hs = sorted({p["size"][2] for p in t.PARTS.values()})
    steps = np.diff(hs)
    ls = [p["size"][0] for p in t.PARTS.values()]
    assert max(steps) <= 0.0035 + 1e-9 and max(ls) - min(ls) > max(hs) - min(hs)


def test_start_layout_has_no_overlaps():
    names = list(t.PARTS)
    for a, b in itertools.combinations(names, 2):
        A, B = block(a, *t.PARTS[a]["start"], t.TOP), block(b, *t.PARTS[b]["start"], t.TOP)
        assert np.any(A.high[:2] + 0.01 < B.low[:2]) or np.any(B.high[:2] + 0.01 < A.low[:2])


# -- outcomes ------------------------------------------------------------------------------------------------------
def test_nothing_done():
    assert sort_score(state()) == 0.0 and not t.success(state())


def test_full_score():
    s = state(**done(), lot_tag=on_pad())
    assert t.score(s) == pytest.approx(1.0) and t.success(s)


def test_sort_alone_is_short_of_full_marks():
    s = state(**done())
    assert t.score(s) == pytest.approx(0.7) and not t.success(s)


def test_tag_on_the_right_pad_without_the_sort_earns_nothing():
    assert t.score(state(lot_tag=on_pad())) == 0.0


def test_tag_on_the_wrong_pad_is_capped():
    wrong = "hold" if t.lot_decision() == "release" else "release"
    s = state(**done(), lot_tag=on_pad(wrong))
    assert t.score(s) == pytest.approx(t.WRONG_LOT_CAP) and not t.success(s)


def test_tag_still_held_or_moving_does_not_count():
    arm = {"left_tcp": point(on_pad().pos)}
    assert t.tag_on(state(lot_tag=on_pad(), **arm)) is None
    assert t.tag_on(state(lot_tag=on_pad(moving=True))) is None


def test_tag_off_its_pad_or_stacked_up_does_not_count():
    (cx, cy) = t.LOT_PADS[t.lot_decision()]["center"]
    assert t.tag_on(state(lot_tag=tag(cx + t.PAD_SIZE, cy))) is None
    assert t.tag_on(state(lot_tag=on_pad(dz=0.02))) is None


def test_lots_call_for_different_decisions_by_counting(layouts):
    seen = {}
    for name, m in layouts():
        rejects = sum(m.kind(n) != "accept" for n in m.PARTS)
        assert sum(m.kind(n) == "accept" for n in m.PARTS) <= 4
        assert all(sum(m.kind(n) == k for n in m.PARTS) <= 2 for k in ("over", "under"))
        seen[name] = (rejects, m.lot_decision())
    assert seen == {"A": (4, "hold"), "B": (2, "release"), "C": (3, "hold")}


def test_demo_places_every_block_and_the_tag_by_the_lot(layouts):
    for _, m in layouts():
        ws = m.demo()["left"]
        assert any(w.label == "over the lot's pad" and w.pos[:2] == m.LOT_PADS[m.lot_decision()]["center"] for w in ws)
        assert max(sum(w.duration for w in v) for v in m.demo().values()) * 20 < 3600


@pytest.mark.parametrize("k", range(7))
def test_partial_correct(k):
    d = done()
    s = state(**{n: d[n] for n in list(t.PARTS)[:k]})
    assert approx(sort_score(s), k / 6) and not t.success(s)


def test_one_wrong_cancels_three():
    d = done()
    wrong = "block_1"  # undersize, into ACCEPT
    s = state(**dict(d, block_1=at(wrong, "accept", 1)))
    s["block_2"] = at("block_2", "accept", 0)
    assert approx(sort_score(s), (5 - 3) / 6) and not t.success(s)


def test_two_wrong_score_zero():
    d = done()
    s = state(**dict(d, block_1=at("block_1", "over", 1), block_3=at("block_3", "under", 1)))
    assert sort_score(s) == 0.0


def test_wrong_run_never_beats_correct_partial():
    """Two right and one wrong (0) is below one right alone (1/6)."""
    d = done()
    assert sort_score(state(block_2=d["block_2"], block_6=d["block_6"], block_1=at("block_1", "accept"))) == 0.0
    assert approx(sort_score(state(block_2=d["block_2"])), 1 / 6)


@pytest.mark.parametrize("tray", list(t.TRAYS))
def test_everything_in_one_tray_scores_zero(tray):
    s = state(**{n: at(n, tray, i % 2) for i, n in enumerate(t.PARTS)})
    assert sort_score(s) == 0.0


def test_go_side_only_scores_zero():
    """Gauging with the GO bar alone (as a single-limit check would): undersize blocks land in ACCEPT."""
    d = done()
    s = state(**dict(d, block_1=at("block_1", "accept", 0), block_5=at("block_5", "accept", 1)))
    assert sort_score(s) == 0.0


def test_nogo_side_only_scores_zero():
    """Gauging with the NO-GO bar alone: oversize blocks land in ACCEPT."""
    d = done()
    s = state(**dict(d, block_3=at("block_3", "accept", 0), block_4=at("block_4", "accept", 1)))
    assert sort_score(s) == 0.0


def test_over_and_under_swapped_scores_zero():
    d = done()
    s = state(**dict(d, **{n: at(n, "under" if t.kind(n) == "over" else "over", i % 2)
                           for i, n in enumerate(t.PARTS) if t.kind(n) != "accept"}))
    assert sort_score(s) == 0.0


def test_blind_play_near_zero():
    """A policy that can't tell the blocks apart: every way of sending blocks to {bench, three trays}, averaged over
    every arrangement of the kinds. The best blind strategy (one block into one tray) earns 1/18; the average is
    about 0.01."""
    kinds = [t.kind(n) for n in t.PARTS]
    arrangements = set(itertools.permutations(kinds))

    def expected(dest):
        total = 0.0
        for arr in arrangements:
            c = sum(d == k for d, k in zip(dest, arr) if d)
            w = sum(d != k for d, k in zip(dest, arr) if d)
            total += max(0.0, c - t.WRONG_PENALTY * w) / len(kinds)
        return total / len(arrangements)

    evs = [expected(d) for d in itertools.product((None, "accept", "over", "under"), repeat=len(kinds))]
    assert max(evs) <= 1 / 18 + 1e-9 and sum(evs) / len(evs) < 0.015


# -- exploits ------------------------------------------------------------------------------------------------------
def test_held_block_does_not_count():
    d = done()
    b = d["block_2"]
    s = state(**d)
    s["left_tcp"] = point((b.pos[0] - 0.005, b.pos[1], b.low[2] + t.GRIP_Z))
    assert approx(sort_score(s), 5 / 6) and not t.success(s)


def test_fingers_just_above_do_not_count_until_2cm_clear():
    d = done()
    b = d["block_6"]
    s = state(**d)
    s["left_tcp"] = point((b.pos[0], b.pos[1], b.high[2] + 0.019))
    assert not t.in_tray("block_6", "accept", s)
    s["left_tcp"] = point((b.pos[0], b.pos[1], b.high[2] + 0.021))
    assert t.in_tray("block_6", "accept", s)


def test_held_wrong_block_still_counts_as_wrong_once_let_go_only():
    """A wrong block held in a tray counts neither way, but it can't be held there forever: the final state is
    scored, and released it counts as wrong."""
    d = done()
    (cx, cy) = t.TRAYS["accept"]["center"]
    wrong = block("block_1", cx, cy + 0.06, t.TOP + t.TRAY_FLOOR)
    s = state(**dict(d, block_1=wrong))
    s["right_tcp"] = point(wrong.pos)
    assert approx(sort_score(s), 5 / 6)
    s["right_tcp"] = point(PARK["right_tcp"])
    assert approx(sort_score(s), (5 - 3) / 6)


def test_moving_block_does_not_count():
    d = done()
    s = state(**dict(d, block_4=at("block_4", "over", 0, moving=True)))
    assert approx(sort_score(s), 5 / 6)


def test_block_dropped_off_table_counts_as_wrong():
    """Knocking a block off the table doesn't dodge the penalty."""
    d = done()
    s = state(**dict(d, block_1=block("block_1", 0.2, 0.7, t.TOP - 0.75)))
    assert approx(sort_score(s), (5 - 3) / 6)
    assert sort_score(state(block_1=block("block_1", 0.2, 0.7, t.TOP - 0.75))) == 0.0


def test_stacked_block_does_not_count():
    d = done()
    under = d["block_5"]
    s = state(**dict(d, block_1=block("block_1", under.pos[0], under.pos[1], under.high[2])))
    assert approx(sort_score(s), 5 / 6)


def test_block_on_divider_or_wall_does_not_count():
    s = state(block_4=block("block_4", 0.30, -0.265, t.TOP + t.TRAY_H))
    assert sort_score(s) == 0.0


def test_block_outside_tray_walls_does_not_count():
    (cx, cy), (ix, _) = t.TRAYS["accept"]["center"], t.TRAYS["accept"]["inner"]
    s = state(block_2=block("block_2", cx + ix / 2 + 0.03, cy, t.TOP))
    assert sort_score(s) == 0.0


def test_block_on_its_side_in_right_tray_counts():
    """Orientation in the tray doesn't matter: only the sort is scored."""
    d = done()
    s = state(**dict(d, block_2=at("block_2", "accept", 0, on_side=True)))
    assert sort_score(s) == pytest.approx(1.0)


def test_block_left_in_gauge_counts_neither_way():
    s = state(block_5=block("block_5", t.BAR["x0"], t.LANE["nogo"], t.TOP))
    assert sort_score(s) == 0.0


def test_score_bounded():
    for s in (state(), state(**done())):
        assert 0.0 <= sort_score(s) <= 1.0


# -- the demo ------------------------------------------------------------------------------------------------------
def test_demo_fits_run_limit_and_uses_three_trays():
    wps = t.demo()
    durations = [sum(w.duration for w in v) for v in wps.values()]
    assert max(durations) * 20 < 2400 and math.isclose(durations[0], durations[1])
    labels = [w.label for w in wps["left"] + wps["right"]]
    assert sum("NO-GO bar" in lab for lab in labels) == 4 and sum("GO bar" in lab for lab in labels) == 10


# -- the layouts -----------------------------------------------------------------------------------------------------



@pytest.fixture
def layouts(monkeypatch):
    def each():
        for name in t.LAYOUTS:
            monkeypatch.setenv("GAUGE_LAYOUT", name.lower())
            yield name, importlib.reload(t)
        monkeypatch.delenv("GAUGE_LAYOUT")
        importlib.reload(t)
    return each


def test_scene_matches_layout(layouts):
    for name, m in layouts():
        for k, xy in zip(m.LAYOUTS[name], m.SLOTS):
            o = next(o for o in m.OBJECTS if o.name == f"block_{k}")
            assert tuple(o.pos[:2]) == xy and tuple(o.size) == m.SIZES[f"block_{k}"]


def test_every_layout_has_no_overlaps(layouts):
    for _, m in layouts():
        boxes = [(np.array(p["start"]) - np.array(p["size"][:2]) / 2, np.array(p["start"]) + np.array(p["size"][:2]) / 2)
                 for p in m.PARTS.values()]
        for (alo, ahi), (blo, bhi) in itertools.combinations(boxes, 2):
            assert np.any(ahi + 0.01 < blo) or np.any(bhi + 0.01 < alo)


def test_every_layout_demo_reaches_and_gauges(layouts):
    for name, m in layouts():
        wps = m.demo()
        assert max(sum(w.duration for w in ws) for ws in wps.values()) * 20 < 2400
        pe, re = m.ik_errors(wps)
        assert pe < 1e-3 and re < math.radians(0.5), name
        assert [m.PARTS[n]["start"][0] for n in m.ORDER] == [m.BACK_X] * 3 + [m.FRONT_X] * 3


def test_full_and_blind_scores_in_every_layout(layouts):
    for _, m in layouts():
        used, d = {k: 0 for k in m.TRAYS}, {}
        for n in m.PARTS:
            (cx, cy) = m.TRAYS[m.kind(n)]["center"]
            L, W, H = m.PARTS[n]["size"]
            c = np.array((cx + (-0.04, 0.04)[used[m.kind(n)] % 2], cy + 0.05 * (used[m.kind(n)] // 2), m.TOP + m.TRAY_FLOOR + H / 2))
            d[n] = SimpleNamespace(pos=c, low=c - np.array((L, W, H)) / 2, high=c + np.array((L, W, H)) / 2,
                                   resting=lambda: True)
            used[m.kind(n)] += 1
        tcps = {f"{a}_tcp": point(p) for a, p in m.PARK.items()}
        h = m.TAG["size"] / 2
        c = np.array((*m.LOT_PADS[m.lot_decision()]["center"], m.TOP + m.PAD_H + h))
        d["lot_tag"] = SimpleNamespace(pos=c, low=c - h, high=c + h, resting=lambda: True)
        assert m.score({**d, **tcps}) == pytest.approx(1.0)


