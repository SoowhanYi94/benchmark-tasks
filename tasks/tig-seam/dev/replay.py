"""Dev only, no simulator: play the demo's TCP plan with perfect tracking, the tools rigid in the hands while gripped, through
note_step and score. python dev/replay.py"""
import math
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import task as t  # noqa: E402
from tig_seam import mech as M  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.station import BASES  # noqa: E402


def quat_z(a):
    return np.array((0.0, 0.0, math.sin(a / 2), math.cos(a / 2)))


def st(p, a, v=0.0):
    p = np.asarray(p, float)
    return NS(pos=p, quat_xyzw=quat_z(a), lin_vel=np.array((v, 0, 0)), ang_vel=np.zeros(3), low=p, high=p)


def main():
    if os.environ.get("NOCAL"):  # perfect tracking: drop the arm-give offsets the plan aims against
        t._cal = lambda s: (0.0, 0.0)
        t.AHEAD_A, t.AHEAD_B, t.ROD_SIDE_A, t.ROD_SIDE_B, t.ROD_LOW_B, t.ROD_AIM = t.AHEAD, t.AHEAD, 0.0, 0.0, 0.0, t.ROD_AIM - 0.003
    wps = t.demo()
    plans = {a: ArmPlan(wps[a], BASES[a]) for a in wps}
    T = max(p.duration for p in plans.values())
    t.reset_latch()
    torch = (np.array((M.TORCH_REST["x"], M.TORCH_REST["y"], t.TOP + M.TORCH_REST_Z)), M.TORCH_REST["lean"])
    rod = (np.array((M.ROD_REST["x"], M.ROD_REST["y"], t.TOP + M.ROD_REST_Z)), M.ROD_REST["heading"])
    held = {"left": False, "right": False}
    prev = {}
    n = int(round(T / 0.05)) + 1
    for k in range(n):
        tt = k * 0.05
        objs = {}
        for a, pl in plans.items():
            pos, R, finger, label = pl.target(tt)
            yaw = math.atan2(R[1, 1], R[0, 1]) - math.pi / 2  # the down grip's +y column turned by yaw
            Rz = R @ np.linalg.inv(t.grip_rotation(0.0))
            yaw = math.atan2(Rz[1, 0], Rz[0, 0])
            closed = finger > t.TORCH_OPEN / 2 if a == "left" else finger > t.ROD_OPEN / 2
            objs[f"{a}_tcp"] = st(pos, 0.0)
            if a == "left":
                if closed and np.linalg.norm(pos - (torch[0] + (0, 0, t.GRIP_DZ))) < 0.01:
                    held[a] = True
                if not closed:
                    held[a] = False
                if held[a]:
                    torch = (pos - (0, 0, t.GRIP_DZ), yaw + 1.5 * math.pi)
            else:
                if closed and np.linalg.norm(pos - (rod[0] + (0, 0, t.SLEEVE_DZ))) < 0.01:
                    held[a] = True
                if not closed:
                    held[a] = False
                if held[a]:
                    rod = (pos - (0, 0, t.SLEEVE_DZ), yaw + 0.5 * math.pi)
        objs["torch"] = st(*torch)
        objs["rod"] = st(*rod)
        if "laid" == t.VARIANT and not held["right"] and tt > 5:
            pass
        t.note_step(objs, 0.05)
        w = t.weld_state(objs)
        if k % 20 == 0 or (os.environ.get("ALL") and 300 < k < 400):
            print(f"t={tt:5.1f} h={1000 * w['h']:5.1f} lat={1000 * w['lat']:4.1f} arc={t._trial['arc']} ahead={1000 * w['ahead']:5.1f} rh={1000 * w['rh']:5.1f} "
                  f"rlat={1000 * w['rlat']:5.1f} gap={1000 * w['gap']:5.1f} hands={w['torch_hand']},{w['rod_hand']} dabs={t._trial['dabs']} cold={t._trial['dabs_cold']} "
                  f"good={int(t.good_segments().sum())} dirty={t._trial['dirty']}")
    print("faults:", t.faults())
    print("heat:", np.round(t._trial["heat"], 1).tolist())
    print("filler:", t._trial["filler"].tolist())
    print("rested", t.tools_rested(objs), "clear", t.hands_clear(objs), "still", round(t._trial["still"], 2), "score", t.score(objs))


main()
