"""The score must pay 0.3 for filling the pockets one kanban each and 0.7 only for a levelled box (the red kanbans in
alternate pockets), so blind filling scores far less than levelling."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import task as t
from yam_arena import ik
from yam_arena.constants import READY
from yam_arena.demo import grip_rotation
from yam_arena.station import BASES

H = t.KANBAN[2]


def cube(x, y, z):
    """A kanban whose underside is at height z."""
    return SimpleNamespace(pos=np.array((x, y, z + H / 2)), low=np.array((x - H / 2, y - H / 2, z)))


START = {n: cube(t.TRAY_X, y, t.TOP + 0.004) for n, (_, y) in t.KANBANS.items()}


def put(order, **extra):
    """The kanbans named in order placed in pockets 1, 2, ... (None leaves a pocket empty)."""
    s = dict(START, **{n: cube(t.BOX_X, t.POCKET_Y[k], t.FLOOR_Z) for k, n in enumerate(order) if n})
    return dict(s, **extra)


LEVELLED = ["kanban_a1", "kanban_b1", "kanban_a2", "kanban_c1", "kanban_a3", "kanban_b2"]  # A B A C A B


def close(a, b):
    return abs(a - b) < 1e-9


def test_start_scores_zero():
    assert t.score(START) == 0.0 and not t.success(START)


def test_levelled_full_score():
    assert close(t.score(put(LEVELLED)), 1.0) and t.success(put(LEVELLED))


def test_reds_in_even_pockets_full_score():
    order = ["kanban_c1", "kanban_a1", "kanban_b1", "kanban_a2", "kanban_b2", "kanban_a3"]  # C A B A B A
    assert close(t.score(put(order)), 1.0)


def test_demo_order_is_levelled():
    order = [n for _, n, _ in sorted(t.MOVES, key=lambda m: m[2])]
    assert close(t.score(put(order)), 1.0)


def test_tray_order_scores_fill_only():
    assert close(t.score(put(list(t.KANBANS))), 0.3)  # A A A B B C


def test_neighbours_differ_but_c_left_to_the_end():
    """Council: A B A B C A has no matching neighbours but bunches the shift's end; it isn't levelled."""
    order = ["kanban_a1", "kanban_b1", "kanban_a2", "kanban_b2", "kanban_c1", "kanban_a3"]
    assert close(t.score(put(order)), 0.3) and not t.success(put(order))


def test_partial_fill():
    assert close(t.score(put(LEVELLED[:3])), 0.15)
    assert close(t.score(put(LEVELLED[:5])), 0.25)


def test_full_with_mistake_beats_correct_unfinished():
    """Accepted and stated in the scoring: levelling is only credited for the full box."""
    wrong = put(["kanban_a1", "kanban_a2", "kanban_b1", "kanban_a3", "kanban_c1", "kanban_b2"])
    assert close(t.score(wrong), 0.3) > 0 and t.score(wrong) > t.score(put(LEVELLED[:5]))


def test_reds_alone_alternate():
    """Council exploit: three reds alone in alternate pockets."""
    assert close(t.score(put(["kanban_a1", None, "kanban_a2", None, "kanban_a3"])), 0.15)


def test_two_in_one_pocket_spoils_it():
    s = put(LEVELLED[:5], kanban_b2=cube(t.BOX_X, t.POCKET_Y[4] + 0.01, t.FLOOR_Z))  # pocket 5 holds two
    assert close(t.score(s), 0.2) and not t.success(s)


def test_stacked_kanban_not_counted():
    """A kanban resting on another (underside 3 cm up) doesn't count for the pocket."""
    s = put(LEVELLED[:5], kanban_b2=cube(t.BOX_X, t.POCKET_Y[5], t.FLOOR_Z + 0.011))
    assert close(t.score(s), 0.25)
    s = put(LEVELLED[:5], kanban_b2=cube(t.BOX_X, t.POCKET_Y[4], t.FLOOR_Z + H))
    assert close(t.score(s), 0.25)


def test_kanban_on_a_wall_counts_for_neither_pocket():
    wall_y = (t.POCKET_Y[4] + t.POCKET_Y[5]) / 2
    s = put(LEVELLED[:5], kanban_b2=cube(t.BOX_X, wall_y, t.TOP + t.WALL_H))
    assert close(t.score(s), 0.25)


def test_kanban_off_the_table_counts_neither_way():
    s = put(LEVELLED[:5], kanban_b2=cube(0.6, 0.6, 0.0))
    assert close(t.score(s), 0.25)


def test_blind_filling_scores_far_below_levelling():
    """Council: random filling averaged 0.59. Over all 720 orders it now averages 0.37 and 1 in 10 reach 1.0."""
    scores = [t.score(put(list(p))) for p in itertools.permutations(t.KANBANS)]
    mean = sum(scores) / len(scores)
    assert close(mean, 0.3 + 0.7 * 0.1) and sum(s == 1.0 for s in scores) == 72


def test_demo_fits_run_limit():
    steps = max(sum(w.duration for w in wps) for wps in t.demo().values()) * 20
    assert steps < 2400, steps


@pytest.mark.parametrize("arm", list(t.SIDE))
def test_demo_waypoints_reachable(arm):
    q = np.array(READY, float)
    for w in t.demo()[arm]:
        q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
        assert pe < 0.005 and re < 0.05, (w.label, pe, re)
