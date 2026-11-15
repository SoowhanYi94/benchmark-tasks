"""IK check of every demo waypoint (and the motion between them) for each demo variant. CPU only.

    python dev/ik_check.py          (from the task folder)
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from yam_arena import station  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402

worst = 0
for variant in ("correct", "loose", "shut_first"):
    os.environ["JD_DEMO"] = variant
    import importlib

    import task
    importlib.reload(task)
    plan = task.demo_plan()
    for arm, wps in plan.items():
        p = ArmPlan(list(wps), station.BASES[arm])
        t, dt = 0.0, 0.05
        while t < p.duration:
            p.action7(t)
            t += dt
        print(f"{variant:10s} {arm:5s} {p.duration:5.1f}s worst IK {p.worst_ik_mm:6.2f} mm {p.worst_ik_deg:5.2f} deg")
        if p.worst_ik_mm > 1.0 or p.worst_ik_deg > 1.0:
            worst += 1
            # find which waypoints fail
            q0 = None
            for w in wps:
                from yam_arena import ik
                from yam_arena.demo import grip_rotation
                q, pe, re = ik.solve(w.pos, ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), None,
                                     base_pos=station.BASES[arm], iters=400)
                if pe > 1e-3 or np.degrees(re) > 1:
                    print(f"   FAIL {w.label}: pos {np.round(w.pos, 3)} yaw {w.yaw:.2f} err {pe*1000:.1f} mm {np.degrees(re):.1f} deg")
print("FAILURES" if worst else "all waypoints within 1 mm / 1 deg")
