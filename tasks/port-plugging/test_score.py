"""Scoring must credit only the two open ports, penalise the test port and the fitted plug, and ignore near misses."""
from types import SimpleNamespace

import numpy as np
import task as t


def plug(x, y, z):
    """A standing plug whose underside is at height z."""
    return SimpleNamespace(pos=np.array((x, y, z + t.PLUG_H / 2)), low=np.array((x - t.PLUG_R, y - t.PLUG_R, z)))


def in_port(port, dz=0.0, dx=0.0):
    return plug(t.MANIFOLD["x"] + dx, t.PORTS[port], t.FLOOR_Z + dz)


def start():
    return {n: plug(*xy, t.FLOOR_Z if n == "plug_fitted" else t.TOP + 0.004) for n, xy in t.PLUGS.items()}


def done():
    return dict(start(), plug_1=in_port("port_1"), plug_3=in_port("port_3"))


def test_nothing_done(): assert t.score(start()) == 0.0 and not t.success(start())

def test_full_score(): assert t.score(done()) == 1.0 and t.success(done())

def test_one_port(): assert t.score(dict(start(), plug_1=in_port("port_1"))) == 0.5

def test_test_port_plugged_too(): assert t.score(dict(done(), plug_2=in_port("port_2"))) == 0.25

def test_only_test_port_plugged(): assert t.score(dict(start(), plug_2=in_port("port_2"))) == 0.0

def test_fitted_plug_pulled_out():
    assert t.score(dict(done(), plug_fitted=plug(0.18, 0.0, t.TOP))) == 0.25


def test_fitted_plug_swapped_into_open_port():
    d = dict(done(), plug_fitted=in_port("port_1"), plug_1=in_port("port_4"))
    assert abs(t.score(d) - 0.25) < 1e-9


def test_two_plugs_in_one_port_count_once():
    assert t.score(dict(start(), plug_1=in_port("port_1"), plug_3=in_port("port_1", dz=0.03))) == 0.5


def test_plug_resting_on_face_not_plugged():
    assert t.score(dict(start(), plug_1=in_port("port_1", dz=t.MANIFOLD["depth"]))) == 0.0


def test_plug_less_than_1cm_in_not_plugged():
    assert t.score(dict(start(), plug_1=in_port("port_1", dz=0.011))) == 0.0


def test_plug_beside_port_not_plugged():
    assert t.score(dict(start(), plug_1=in_port("port_1", dx=t.HOLE / 2 + 0.002))) == 0.0


def test_demo_fits_run_limit():
    steps = max(sum(w.duration for w in wps) for wps in t.demo().values()) * 20
    assert steps < 1200, steps
