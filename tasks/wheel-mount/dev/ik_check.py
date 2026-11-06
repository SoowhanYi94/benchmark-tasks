"""Every demo waypoint must pass IK (error under 1 mm, rotation under 1 degree). NumPy only.

    python dev/ik_check.py      # from tasks/wheel-mount
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import task  # noqa: E402
from yam_arena import ik  # noqa: E402
from yam_arena.constants import READY  # noqa: E402
from yam_arena.demo import grip_rotation  # noqa: E402
from yam_arena.station import BASES  # noqa: E402

bad, t_end = 0, {}
for arm, wps in task.demo().items():
    q = np.array(READY, float)
    t = 0.0
    for w in wps:
        t += w.duration
        q, e, r = ik.solve(w.pos, ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q_init=q, base_pos=BASES[arm])
        if e > 1e-3 or r > math.radians(1):
            bad += 1
            print(f"{arm} t={t:6.2f} {w.label}: pos {np.round(w.pos, 4)} err {1000 * e:.1f} mm rot {math.degrees(r):.1f}")
    t_end[arm] = t
print(f"{bad} bad waypoints; plan length {t_end} s = {max(t_end.values()) * 20:.0f} steps at 20 Hz")
