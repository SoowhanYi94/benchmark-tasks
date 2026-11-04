"""The score must credit only refills of bins at or below their min, to exactly their max, and give 0 for any change
to a bin above its min, any overfill and any part taken out; blind strategies must average near 0."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import task as t
from yam_arena import ik
from yam_arena.constants import READY
from yam_arena.demo import grip_rotation
from yam_arena.station import BASES

BULK = [n for n in t.PARTS if n.startswith("bulk")]
BINS = list(t.BINS)


def part(x, y, z=t.TOP, moving=False, lying=False):
    """A socket whose underside is at height z, standing (or lying along x)."""
    h, w = (t.PART[0], t.PART[2]) if lying else (t.PART[2], t.PART[0])
    return SimpleNamespace(pos=np.array((x, y, z + h / 2)), low=np.array((x - w / 2, y - t.PART[0] / 2, z)),
                           resting=lambda: not moving)


def start(**kw):
    return dict({n: part(x, y, z) for n, (x, y, z) in t.PARTS.items()}, **kw)


def into(b, k=0):
    """A part at slot k of bin b."""
    return part(t.BINS[b][0] + t.SLOTS[k][0], t.BINS[b][1] + t.SLOTS[k][1])


def place(dest):
    """Start, with bulk part i moved into bin dest[i] (None: left in the tote)."""
    s, fill = start(), dict(t.START_COUNT)
    for n, b in zip(BULK, dest):
        if b is not None:
            s[n] = into(b, min(fill[b], 2))
            fill[b] += 1
    return s


def close(a, b):
    return abs(a - b) < 1e-9


def test_rule():
    assert t.NEED == {"bin_a": 3, "bin_b": 0, "bin_c": 1, "bin_d": 0}
    kinds = {b: ("below" if n < t.LEVELS[b][0] else "at min" if n == t.LEVELS[b][0] else
                 "at max" if n == t.LEVELS[b][1] else "between") for b, n in t.START_COUNT.items()}
    assert set(kinds.values()) == {"below", "at min", "between", "at max"}
    assert t.START_COUNT["bin_b"] == t.START_COUNT["bin_c"]  # same count, different label, different action


def test_nothing_done():
    assert t.score(start()) == 0.0 and not t.success(start())


def test_full_score():
    s = place(["bin_a", "bin_a", "bin_a", "bin_c", None, None])
    assert t.score(s) == 1.0 and t.success(s)


@pytest.mark.parametrize("dest,want", [
    (["bin_a"], 0.25), (["bin_a", "bin_a"], 0.5), (["bin_c"], 0.25), (["bin_a", "bin_a", "bin_a"], 0.75),
    (["bin_a", "bin_a", "bin_c"], 0.75)])
def test_partial(dest, want):
    s = place(dest)
    assert close(t.score(s), want) and not t.success(s)


@pytest.mark.parametrize("dest", [
    ["bin_a", "bin_a", "bin_a", "bin_c", "bin_b"],             # everything to its own max (council exploit)
    ["bin_a", "bin_a", "bin_a", "bin_c", "bin_b", "bin_d"],    # everything to 3
    ["bin_a", "bin_b", "bin_c", "bin_d"],                      # one into each bin
    ["bin_a", "bin_a", "bin_a", "bin_a", "bin_c"],             # overfill A
    ["bin_a", "bin_a", "bin_a", "bin_c", "bin_c"],             # overfill C
    ["bin_b"], ["bin_d"]])                                     # topping up a bin above its min
def test_mistakes_score_zero(dest):
    assert t.score(place(dest)) == 0.0 and not t.success(place(dest))


def test_part_taken_out():
    s = place(["bin_a", "bin_a", "bin_a", "bin_c"])
    s["bin_b_part_1"] = part(0.5, 0.0)  # off the bin, onto the table
    assert t.score(s) == 0.0


def test_robbing_a_bin():
    s = start(bin_b_part_1=into("bin_a", 0))  # B's part moved into A
    assert t.score(s) == 0.0
    s = place(["bin_a", "bin_a", "bin_c"])
    s["bin_d_part_1"] = into("bin_a", 2)
    assert t.score(s) == 0.0


def test_swap_within_counts_is_fine():
    s = place(["bin_a", "bin_a", "bin_a", "bin_c", "bin_b"])
    s["bin_b_part_1"] = part(t.BULK["center"][0], 0.0)  # B's part swapped for a tote part: B still holds 2
    assert t.score(s) == 1.0


def test_held_part_does_not_count():
    s = place(["bin_a", "bin_a", "bin_c"])
    x, y = t.BINS["bin_a"][0] + t.SLOTS[2][0], t.BINS["bin_a"][1]
    s[BULK[3]] = part(x, y, t.TOP + 0.039)  # still in the fingers, 3.9 cm up
    assert close(t.score(s), 0.75) and not t.success(s)
    s[BULK[3]] = part(x, y, moving=True)  # just dropped, still moving
    assert close(t.score(s), 0.75)
    s[BULK[3]] = part(x, y)
    assert t.score(s) == 1.0 and t.success(s)


def test_perched_on_wall_or_stacked():
    s = place(["bin_a", "bin_a", "bin_c"])
    cx, cy = t.BINS["bin_a"]
    s[BULK[3]] = part(cx + 0.044, cy, t.TOP + t.BIN_H)  # on the far wall's rim
    assert close(t.score(s), 0.75)
    s[BULK[3]] = part(cx + 0.034, cy)  # centre 3.4 cm out: over the wall
    assert close(t.score(s), 0.75)
    s[BULK[3]] = part(cx - 0.022, cy - 0.022, t.TOP + t.PART[2])  # on top of another socket
    assert close(t.score(s), 0.75)


def test_lying_part_counts():
    s = place(["bin_a", "bin_a", "bin_c"])
    s[BULK[3]] = part(t.BINS["bin_a"][0] + 0.01, t.BINS["bin_a"][1], lying=True)
    assert t.score(s) == 1.0


def test_part_off_table_counts_neither_way():
    s = place(["bin_a", "bin_a", "bin_a", "bin_c"])
    s[BULK[4]] = part(0.2, 0.0, 0.0)  # on the floor
    assert t.score(s) == 1.0


def test_blind_strategies_average_near_zero():
    """Each of the six tote parts left in the tote or put in one of the four bins, all 5^6 ways equally likely."""
    scores = [t.score(place(d)) for d in itertools.product([None, *BINS], repeat=6)]
    mean = sum(scores) / len(scores)
    assert mean < 0.05, mean
    # a fixed number of parts per bin, chosen blind: any count into bins without looking
    into_bins = [t.score(place(d)) for d in itertools.product(BINS, repeat=6)]
    assert sum(into_bins) / len(into_bins) < 0.01


def test_demo_fits_run_limit():
    steps = max(sum(w.duration for w in wps) for wps in t.demo().values()) * 20
    assert steps < 2400, steps


@pytest.mark.parametrize("arm", list(t.SIDE))
def test_demo_waypoints_reachable(arm):
    q = np.array(READY, float)
    for w in t.demo()[arm]:
        q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q,
                             base_pos=BASES[arm])
        assert pe < 0.001 and re < 0.01, (w.label, pe, re)


def test_demo_drops_into_the_right_slots():
    """Where the demo lets go: three empty slots of A and C's free slot; B and D never visited."""
    drops = [w.pos[:2] for wps in t.demo().values() for w in wps if w.label == "lower"]
    assert len(drops) == t.TOTAL
    s = start()
    for k, (x, y) in enumerate(drops):
        s[BULK[k]] = part(x, y)
    assert t.score(s) == 1.0
    for wps in t.demo().values():
        for w in wps:
            for b in ("bin_b", "bin_d"):
                assert not (abs(w.pos[0] - t.BINS[b][0]) < 0.06 and abs(w.pos[1] - t.BINS[b][1]) < 0.06), w.label
