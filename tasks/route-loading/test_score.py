"""The score must credit only a full load in the board's reverse order, with one step of partial credit, give a blind
load (random or by layout) about nothing, and not count a parcel half out of the door, held up or still moving."""
import itertools
import statistics
from types import SimpleNamespace

import numpy as np
import pytest
import task as t
from yam_arena import ik
from yam_arena.constants import READY
from yam_arena.demo import grip_rotation
from yam_arena.station import BASES

ROUTE0 = ["blue", "red", "yellow"]
SLOTS = t.SLOTS[::-1]  # door to nose


def box(x, y, z, moving=False):
    """A parcel standing with its underside at z."""
    lx, ly, lz = t.PARCEL
    return SimpleNamespace(pos=np.array((x, y, z + lz / 2)), low=np.array((x - lx / 2, y - ly / 2, z)),
                           high=np.array((x + lx / 2, y + ly / 2, z + lz)), resting=lambda: not moving)


def start(**kw):
    return dict({n: box(*xy, t.TOP + 0.004) for n, (c, xy) in t.PARCELS.items()}, **kw)


def in_trailer(names, xs=SLOTS):
    """The named parcels down the middle of the trailer, door to nose."""
    return start(**{n: box(x, 0.0, t.FLOOR_Z) for n, x in zip(names, xs)})


def score(objects, route=ROUTE0):
    return t.score(objects, route)


def test_nothing_done():
    assert score(start()) == 0.0 and not t.success(start())


@pytest.mark.parametrize("route", t.VARIANTS)
def test_demo_end_state_scores_one(route):
    done = start(**{n: box(x, 0.0, t.FLOOR_Z) for _, n, x in t.load_plan(list(route))})
    assert score(done, list(route)) == 1.0


def test_success_uses_the_board():
    done = start(**{n: box(x, 0.0, t.FLOOR_Z) for _, n, x in t.load_plan()})
    assert t.success(done)


def test_route_order_scores_zero():  # blue at the nose
    assert score(in_trailer(["parcel_1", "parcel_5", "parcel_3", "parcel_2", "parcel_4"])) == 0.0


def test_one_parcel_out_of_place_scores_a_third():  # door to nose: blue, red, blue, yellow, yellow
    assert abs(score(in_trailer(["parcel_2", "parcel_3", "parcel_4", "parcel_1", "parcel_5"])) - 1 / 3) < 1e-9


def test_two_parcels_counting_score_zero():  # blue, yellow, blue, red, yellow: only the door blue and nose yellow count
    assert score(in_trailer(["parcel_2", "parcel_1", "parcel_4", "parcel_3", "parcel_5"])) == 0.0


def test_any_partial_load_scores_zero():
    """A missing parcel stops every parcel for another stop counting, so no partial load scores."""
    full = ["parcel_2", "parcel_4", "parcel_3", "parcel_1", "parcel_5"]
    for k in range(1, 5):
        for names in itertools.combinations(full, k):
            assert score(in_trailer(list(names), [x for n, x in zip(full, SLOTS) if n in names])) == 0.0


def test_one_stop_only_scores_zero():
    assert score(in_trailer(["parcel_1", "parcel_5"])) == 0.0


def test_old_pair_exploit_scores_zero():  # a yellow at the nose and a blue at the door used to score 0.125
    assert score(start(parcel_1=box(0.37, 0.0, t.FLOOR_Z), parcel_2=box(0.17, 0.0, t.FLOOR_Z))) == 0.0


def test_too_close_does_not_count():
    """Neighbours for different stops less than 2 cm apart (centre to centre) aren't in order: here the red and the
    yellow 1.9 cm behind it don't count, the blues and the other yellow do (3 of 5)."""
    xs = [0.17, 0.22, 0.27, 0.27 + 0.019, 0.33]
    assert abs(score(in_trailer(["parcel_2", "parcel_4", "parcel_3", "parcel_1", "parcel_5"], xs)) - 1 / 3) < 1e-9


def test_random_loading_averages_near_zero():
    """Council blocker: every full load in every order used to average 0.5. Now 0.06, on every board."""
    for route in t.VARIANTS:
        m = statistics.mean(score(in_trailer(list(p)), list(route)) for p in itertools.permutations(t.PARCELS))
        assert m < 0.07, (route, m)


def test_blind_orders_average_low_over_boards():
    """Council: start-x or name order scored 0.5 on the fixed board. Over the six boards no fixed order beats 0.17."""
    best = max(statistics.mean(score(in_trailer(list(p)), list(r)) for r in t.VARIANTS)
               for p in itertools.permutations(t.PARCELS))
    assert best < 0.17 + 1e-9
    by_x = sorted(t.PARCELS, key=lambda n: t.PARCELS[n][1][0])
    for order in (by_x, by_x[::-1], list(t.PARCELS), list(t.PARCELS)[::-1]):
        assert statistics.mean(score(in_trailer(order), list(r)) for r in t.VARIANTS) < 0.17 + 1e-9


def test_memorised_colours_fail_on_other_boards():
    """Council: 'yellow deepest, blue at the door' scored 1.0 without reading the board; now only on board 0."""
    memorised = start(**{n: box(x, 0.0, t.FLOOR_Z) for _, n, x in t.load_plan(ROUTE0)})
    assert [score(memorised, list(r)) for r in t.VARIANTS].count(1.0) == 1


def test_parcel_half_out_of_the_door_not_loaded():
    """Council: a centre-only check counted a parcel half out of the door. The whole footprint must be inside."""
    xs = list(SLOTS)
    xs[0] = t.TRAILER["door_x"] + 0.01  # blue at the door, half out
    assert score(in_trailer(["parcel_2", "parcel_4", "parcel_3", "parcel_1", "parcel_5"], xs)) == 0.0
    xs[0] = t.TRAILER["door_x"] + 0.02  # just inside
    assert score(in_trailer(["parcel_2", "parcel_4", "parcel_3", "parcel_1", "parcel_5"], xs)) == 1.0


def test_held_up_parcel_not_loaded():
    """Council: a parcel held 2 cm over the floor counted. Now its underside must be within 5 mm of the floor."""
    s = in_trailer(["parcel_2", "parcel_4", "parcel_3", "parcel_1", "parcel_5"])
    s["parcel_5"] = box(0.37, 0.0, t.FLOOR_Z + 0.015)
    assert score(s) == 0.0
    s["parcel_5"] = box(0.37, 0.0, t.FLOOR_Z + 0.004)
    assert score(s) == 1.0


def test_moving_parcel_not_loaded():
    s = in_trailer(["parcel_2", "parcel_4", "parcel_3", "parcel_1", "parcel_5"])
    s["parcel_5"] = box(0.37, 0.0, t.FLOOR_Z, moving=True)
    assert score(s) == 0.0 and not t.success(s)


def test_stacked_parcel_not_loaded():
    s = in_trailer(["parcel_2", "parcel_4", "parcel_3", "parcel_1", "parcel_5"])
    s["parcel_5"] = box(0.32, 0.0, t.FLOOR_Z + t.PARCEL[2])
    assert score(s) == 0.0


def test_variants_are_the_six_orders():
    assert len(set(t.VARIANTS)) == 6 and t.VARIANTS[0] == tuple(ROUTE0)


@pytest.mark.parametrize("route", t.VARIANTS)
def test_demo_fits_run_limit(route):
    steps = max(sum(w.duration for w in wps) for wps in t.demo(list(route)).values()) * 20
    assert steps < 2400, steps


@pytest.mark.parametrize("route", t.VARIANTS)
@pytest.mark.parametrize("arm", ["left", "right"])
def test_demo_waypoints_reachable(route, arm):
    q = np.array(READY, float)
    for w in t.demo(list(route))[arm]:
        q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
        assert pe < 0.005 and re < 0.05, (w.label, pe, re)
