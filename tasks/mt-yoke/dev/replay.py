"""Dev only, no simulator: play the demo's TCP plan with perfect tracking (the tools rigid in the hands while gripped, the
yoke's poles stopped by the plate, the bulb's squeeze from the finger width) through note_step and score.
python dev/replay.py   (MT_DEMO=<variant> for the shortcut variants)"""
import math
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import task as t  # noqa: E402
from mt_yoke import mech as M  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.station import BASES  # noqa: E402


def quat_z(a):
    return np.array((0.0, 0.0, math.sin(a / 2), math.cos(a / 2)))


def st(p, a=0.0, v=0.0, low=None, high=None):
    p = np.asarray(p, float)
    return NS(pos=p, quat_xyzw=quat_z(a), lin_vel=np.array((v, 0, 0)), ang_vel=np.zeros(3), low=p if low is None else low, high=p if high is None else high)


def main():
    wps = t.demo()
    plans = {a: ArmPlan(wps[a], BASES[a]) for a in wps}
    T = max(p.duration for p in plans.values())
    t.reset_latch()
    yoke = [np.array((M.YOKE_REST["x"], M.YOKE_REST["y"], t.TOP + M.YOKE_REST["z"])), M.YOKE_REST["heading"]]
    bulb = [np.array((M.BULB_REST["x"], M.BULB_REST["y"], t.TOP + M.BULB_REST_Z)), M.BULB_REST["heading"]]
    pegs = {n: [np.array((x, y, t.TOP + M.TRAY["t"])), 0.0] for n, (x, y) in zip(t.PEGS, M.PEG_HOME)}
    held = {"left": None, "right": None}
    for k in range(int(round(T / 0.05)) + 1):
        tt = k * 0.05
        objs, q = {}, 0.0
        for a, pl in plans.items():
            pos, R, finger, _ = pl.target(tt)
            Rz = R @ np.linalg.inv(t.grip_rotation(0.0))
            yaw = math.atan2(Rz[1, 0], Rz[0, 0])
            width = -finger / t.FINGER_PER_M
            if held[a] is None and width < 0.05:
                for name, part, grip_c, w in (("yoke", yoke, lambda: t._world(st(*yoke), M.HANDLE_C), M.HANDLE["x"]), ("bulb", bulb, lambda: bulb[0], M.BULB_GRIP_W),
                                              *((n, pegs[n], (lambda n=n: pegs[n][0] + M.PEG_HEAD_C), M.PEG_HEAD["w"]) for n in t.PEGS)):
                    if np.linalg.norm(pos - (grip_c() + (0, 0, t.GRIP_DZ))) < 0.01 and width <= w + 0.001:
                        held[a] = (name, yaw)
            if held[a] is not None and width > {"yoke": M.HANDLE["x"], "bulb": M.BULB_GRIP_W}.get(held[a][0], M.PEG_HEAD["w"]) + 0.005:
                held[a] = None
            if held[a] is not None:
                name, y0 = held[a]
                g = pos - (0, 0, t.GRIP_DZ)
                if name == "yoke":
                    h = yaw - 0.5 * math.pi
                    o = g - t._rz(h) @ M.HANDLE_C
                    o[2] = max(o[2], t.PLATE_Z if M.PLATE["x"][0] < o[0] < M.PLATE["x"][1] else o[2])
                    yoke[:] = [o, h]
                elif name == "bulb":
                    bulb[:] = [g, yaw + 0.5 * math.pi]
                    q = float(np.clip(M.BULB_GRIP_W - width, 0.0, M.SQUEEZE_TRAVEL))
                else:
                    pegs[name][0] = g - M.PEG_HEAD_C
                    pegs[name][0][2] = max(pegs[name][0][2], t.PLATE_Z if M.PLATE["y"][0] < pegs[name][0][1] < M.PLATE["y"][1] and M.PLATE["x"][0] < pegs[name][0][0] < M.PLATE["x"][1] else t.TOP + M.TRAY["t"])
            gap = min(width, 0.094)
            if held[a] is not None:
                gap = max(gap, {"yoke": M.HANDLE["x"], "bulb": M.BULB_GRIP_W - q}.get(held[a][0], M.PEG_HEAD["w"]))
            objs[f"{a}_tcp"] = st(pos, low=pos - (gap / 2, 0, 0), high=pos + (gap / 2, 0, 0))
        objs["yoke"], objs["bulb"] = st(*yoke), st(*bulb)
        objs.update({n: st(*v) for n, v in pegs.items()})
        objs["squeeze"] = st((q, 0.0, 0.0))
        t.note_step(objs, 0.05)
        if k % 40 == 0 or os.environ.get("ALL"):
            f = t.feet(objs)
            print(f"t={tt:5.1f} field {int(t._trial['field'])} feet z {np.round(1000 * (f[:, 2] - t.PLATE_Z), 1).tolist()} nozzle {np.round(1000 * t.nozzle(objs), 0).tolist()} "
                  f"q {1000 * q:.1f} held {held} | {t.report(objs)}")
    print("A:", t._trial["A"].astype(int).tolist(), "B:", t._trial["B"].astype(int).tolist())
    print(t.report(objs), "score", t.score(objs))


main()
