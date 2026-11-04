"""The score must credit the jam cleared, the blank seated and the selector back at AUTO, and cap the trial at 0.1 once
the jam has been away from where it sat while the press was not OFF, whatever happens afterwards."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import task as t


def box(x, y, z, size=t.BLANK, yaw=0.0, roll=0.0, moving=False):
    """A box whose underside is at height z (before rolling), turned yaw and rolled about x."""
    q = np.array((math.sin(roll / 2) * math.cos(yaw / 2), -math.sin(roll / 2) * math.sin(yaw / 2),
                  math.cos(roll / 2) * math.sin(yaw / 2), math.cos(roll / 2) * math.cos(yaw / 2)))
    o = SimpleNamespace(pos=np.array((x, y, z + size[2] / 2)), quat_xyzw=q,
                        resting=lambda: not moving)
    c = t.corners(o, size)
    o.low, o.high = c.min(axis=0), c.max(axis=0)
    return o


DC = t.DIE["center"]
JAM_Z = t.DIE_Z + t.NEST_H
BIN = (*t.REJECT["center"], t.TOP + t.REJECT["floor"])


def handle(x):
    return box(x, t.TRACK_Y, t.TOP, t.HANDLE)


def start(**kw):
    s = {"selector_handle": handle(t.AUTO_X), "jammed_blank": box(*DC, JAM_Z, yaw=math.pi / 4),
         "next_blank": box(*t.NEXT_START, t.TOP + t.INFEED["size"][2])}
    return dict(s, **kw)


def done(**kw):
    return start(**dict(dict(jammed_blank=box(*BIN), next_blank=box(*DC, t.DIE_Z)), **kw))


def run(*states):
    """Score a trial step by step; return the last score."""
    return [t.score(s) for s in states][-1]


@pytest.fixture(autouse=True)
def fresh_process():
    t._trial.update(violated=False, start=None, last=None)


def proper(*before):
    """The task done in the right order after the given steps."""
    off = dict(selector_handle=handle(t.OFF_X))
    return [start(**off), start(jammed_blank=box(*DC, JAM_Z + 0.06, yaw=math.pi / 4), **off),
            start(jammed_blank=box(*BIN), **off), done(**off), done()]


def test_nothing_done():
    assert run(start()) == 0.0 and not t.success(start())


def test_full_score():
    assert abs(run(start(), *proper()) - 1.0) < 1e-9 and t.success(done())


def test_jam_cleared_only():
    off = dict(selector_handle=handle(t.OFF_X))
    assert abs(run(start(), start(**off), start(jammed_blank=box(*BIN), **off)) - 0.3) < 1e-9


def test_left_at_off():
    off = dict(selector_handle=handle(t.OFF_X))
    assert abs(run(start(), *proper()[:-1], done(**off)) - 0.8) < 1e-9


def test_seated_crooked():
    states = proper()
    states[-1] = done(next_blank=box(*DC, t.DIE_Z, yaw=math.radians(30)))
    assert abs(run(start(), *states) - 0.3) < 1e-9


def test_seated_upside_down_counts():
    states = proper()
    states[-1] = done(next_blank=box(*DC, t.DIE_Z, roll=math.pi))
    assert abs(run(start(), *states) - 1.0) < 1e-9


def test_jam_lifted_in_auto_capped():
    lifted = start(jammed_blank=box(*DC, JAM_Z + 0.06, yaw=math.pi / 4))
    assert abs(run(start(), lifted, start(jammed_blank=box(*BIN)), done()) - 0.1) < 1e-9


def test_jam_set_back_in_auto_stays_capped():
    nudged = start(jammed_blank=box(DC[0] + 0.01, DC[1], JAM_Z + 0.01, yaw=math.pi / 4))
    back = start(jammed_blank=box(DC[0] + 0.001, DC[1], JAM_Z, yaw=math.pi / 4))
    assert abs(run(start(), nudged, back, *proper()) - 0.1) < 1e-9


def test_jam_twisted_in_auto_capped():
    """Council exploit: turning the jam 15 degrees in place moved no centre but its corners 9 mm; then turned back."""
    turns = [start(jammed_blank=box(*DC, JAM_Z, yaw=math.pi / 4 + math.radians(d))) for d in (5, 10, 15, 10, 5, 0)]
    assert abs(run(start(), *turns, *proper()) - 0.1) < 1e-9


def test_fake_reset_stays_capped():
    """Council exploit: put everything back within 3 mm, then drop the next blank 4 cm, to look like a reset."""
    lifted = start(jammed_blank=box(*DC, JAM_Z + 0.06, yaw=math.pi / 4))
    back = start(jammed_blank=box(DC[0] + 0.002, DC[1], JAM_Z, yaw=math.pi / 4))
    dropped = dict(back, next_blank=box(*t.NEXT_START, t.TOP + t.INFEED["size"][2] + 0.04))
    assert abs(run(start(), lifted, back, dropped, back, *proper()) - 0.1) < 1e-9


def test_real_reset_clears_cap():
    """A reset puts every free object back on its start pose in one step."""
    lifted = start(jammed_blank=box(*DC, JAM_Z + 0.06, yaw=math.pi / 4))
    assert abs(run(start(), lifted, start(jammed_blank=box(*BIN)), done()) - 0.1) < 1e-9
    assert abs(run(start(), *proper()) - 1.0) < 1e-9


def test_jam_parked_off_nest_then_auto_capped():
    """The jam away from where it sat, not in the bin, with the selector back at AUTO counts as touched outside OFF."""
    off = dict(selector_handle=handle(t.OFF_X))
    parked = box(0.20, 0.10, t.TOP)
    assert abs(run(start(), start(**off), start(jammed_blank=parked, **off),
                   start(jammed_blank=parked, next_blank=box(*DC, t.DIE_Z), **off),
                   start(jammed_blank=parked, next_blank=box(*DC, t.DIE_Z))) - 0.1) < 1e-9


def test_start_is_where_the_jam_settled():
    """The jam settling 3.5 mm from its designed pose before the first step doesn't cap the trial."""
    settled = box(DC[0] + 0.0035, DC[1], JAM_Z, yaw=math.pi / 4)
    off = dict(selector_handle=handle(t.OFF_X))
    assert run(start(jammed_blank=settled), start(jammed_blank=settled, **off)) == 0.0
    assert not t._trial["violated"]


def test_handle_upright_upside_down():
    assert t.selector_at(t.AUTO_X, start(selector_handle=box(t.AUTO_X, t.TRACK_Y, t.TOP, t.HANDLE, roll=math.pi)))


def test_demo_fits_run_limit():
    steps = max(sum(w.duration for w in wps) for wps in t.demo().values()) * 20
    assert steps < 2400, steps
