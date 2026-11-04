"""The score must credit the rails set for the carton (0.6) and the carton between them on the outfeed mark (0.4 more),
and give nothing for the exploits found in review: rails off the bed or short of the carton, rails splayed into a
funnel, the carton turned so its narrow side goes across, the carton held just above the bed."""
import math
from types import SimpleNamespace

import numpy as np
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES

import task as t

X = t.BED["center"][0]


def box(x, y, z, size, yaw=0.0, roll=0.0, moving=False):
    """A box whose underside is at height z (before rolling), turned yaw and rolled about x."""
    q = np.array((math.sin(roll / 2) * math.cos(yaw / 2), -math.sin(roll / 2) * math.sin(yaw / 2),
                  math.cos(roll / 2) * math.sin(yaw / 2), math.cos(roll / 2) * math.cos(yaw / 2)))
    o = SimpleNamespace(pos=np.array((x, y, z + size[2] / 2)), quat_xyzw=q, resting=lambda: not moving)
    c = t.corners(o, size)
    o.low, o.high = c.min(axis=0), c.max(axis=0)
    return o


def rail(y, x=X, z=t.BED_Z, **kw):
    return box(x, y, z, t.RAIL, **kw)


def carton(x=t.OUTFEED[0], y=t.OUTFEED[1], z=t.BED_Z, **kw):
    return box(x, y, z, t.CARTON, **kw)


def rails_at_gap(gap, offset=0.0):
    c = gap / 2 + t.RAIL[1] / 2
    return dict(rail_left=rail(c + offset), rail_right=rail(-c + offset))


START = dict(rail_left=rail(t.RAIL_START_Y["rail_left"]), rail_right=rail(t.RAIL_START_Y["rail_right"]),
             carton=carton(*t.CARTON_START, t.TOP + t.STAGING["size"][2]))


def state(**kw):
    return dict(START, **kw)


def done(gap=0.088, **kw):
    return state(**dict(dict(rails_at_gap(gap), carton=carton()), **kw))


def close(a, b):
    return abs(a - b) < 1e-9


# -- outcomes in the scoring rules ------------------------------------------------------------------------------------
def test_nothing_done():
    assert t.score(START) == 0.0 and not t.success(START)


def test_full_score():
    assert close(t.score(done()), 1.0) and t.success(done())


def test_gap_limits():
    assert close(t.score(done(0.0821)), 1.0) and close(t.score(done(0.0919)), 1.0)
    assert t.score(done(0.079)) == 0.0  # closed onto the carton: narrower than it
    assert t.score(done(0.0935)) == 0.0  # too wide


def test_off_centre():
    assert close(t.score(done(**rails_at_gap(0.088, 0.004))), 1.0)
    assert t.score(done(**rails_at_gap(0.088, 0.008))) == 0.0


def test_rails_set_no_carton():
    assert close(t.score(state(**rails_at_gap(0.088))), 0.6)


def test_swapped_rails_count():
    d = done()
    assert close(t.score(dict(d, rail_left=d["rail_right"], rail_right=d["rail_left"])), 1.0)


def test_rail_not_along_conveyor():
    d = done(0.088)
    assert t.score(dict(d, rail_left=rail(d["rail_left"].pos[1], yaw=math.radians(3)))) == 0.0


def test_rail_tilted_or_moving_or_held():
    c = 0.044 + t.RAIL[1] / 2
    assert t.score(done(rail_left=rail(c, roll=math.radians(5)))) == 0.0
    assert t.score(done(rail_left=rail(c, moving=True))) == 0.0
    assert t.score(done(rail_left=rail(c, z=t.BED_Z + 0.003))) == 0.0  # held still just above the bed


def test_carton_outcomes():
    assert close(t.score(done(carton=carton(x=t.OUTFEED[0] - 0.02))), 0.6)  # off the mark
    assert close(t.score(done(carton=carton(roll=math.pi / 2))), 0.6)  # knocked over
    assert close(t.score(done(carton=carton(moving=True))), 0.6)
    assert close(t.score(done(carton=carton(z=t.BED_Z + t.RAIL[2]))), 0.6)  # on top of the rails' height


def test_carton_either_way_round_and_slightly_turned():
    assert close(t.score(done(carton=carton(yaw=math.pi))), 1.0)
    assert close(t.score(done(0.090, carton=carton(yaw=math.radians(8)))), 1.0)


# -- exploits found in review -----------------------------------------------------------------------------------------
def test_rails_on_bare_table():
    c = 0.044 + t.RAIL[1] / 2
    s = state(rail_left=rail(c, x=0.05, z=t.TOP), rail_right=rail(-c, x=0.05, z=t.TOP))
    assert t.score(s) == 0.0


def test_rails_slid_back_short_of_carton():
    c = 0.044 + t.RAIL[1] / 2
    assert t.score(done(rail_left=rail(c, x=0.17), rail_right=rail(-c, x=0.17))) == 0.0
    assert t.score(done(rail_left=rail(c, x=0.25), rail_right=rail(-c, x=0.25))) == 0.0  # on the bed, short of the mark


def test_rails_splayed_into_funnel():
    """8.6 cm at the near end, about 10.2 cm at the far end, each rail turned under 2 degrees."""
    a = math.radians(1.9)
    off = 0.12 * math.sin(a)
    c = 0.043 + off + t.RAIL[1] / 2
    s = done(rail_left=rail(c, yaw=a), rail_right=rail(-c, yaw=-a))
    gaps, _ = t.end_gaps(s)
    assert abs(gaps[0] - 0.086) < 0.001 and gaps[1] > 0.1
    assert t.score(s) == 0.0


def test_carton_turned_90():
    """Its 6 cm side across fits any gap easily: the test placement doesn't count."""
    assert close(t.score(done(carton=carton(yaw=math.pi / 2))), 0.6)
    assert close(t.score(done(carton=carton(yaw=math.radians(15)))), 0.6)


def test_carton_held_above_bed():
    assert close(t.score(done(carton=carton(z=t.BED_Z + 0.003))), 0.6)


# -- the scripted demo --------------------------------------------------------------------------------------------------
def test_demo_end_state_scores_full():
    """Where the demo leaves things: rails set down at RAIL_SET_Y (9.1 cm gap) and settled about 1.7 mm in each."""
    for c in (t.RAIL_SET_Y, t.RAIL_SET_Y - 0.0017):
        s = done(rail_left=rail(c), rail_right=rail(-c))
        assert close(t.score(s), 1.0), c


def test_demo_fits_run_limit():
    steps = max(sum(w.duration for w in wps) for wps in t.demo().values()) * 20
    assert steps < 2400, steps


def test_demo_ik_reaches_every_step():
    """Played at 20 Hz, as the scripted policy does, the IK reaches every target (all waypoints included)."""
    for arm, wps in t.demo().items():
        plan = ArmPlan(wps, BASES[arm])
        for k in range(int(round(plan.duration * 20)) + 1):
            plan.action7(k / 20)
        assert plan.worst_ik_mm < 1.0 and plan.worst_ik_deg < 1.0, (arm, plan.worst_ik_mm, plan.worst_ik_deg)
