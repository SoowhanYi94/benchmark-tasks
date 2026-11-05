"""`yam_gauge`: the stock yam_arena embodiment, plus two states for the scorer each step: `left_tcp` and `right_tcp`
(each gripper's fingertip centre, from the arm's joint positions), so it can tell whether a block has been let go.
Set GAUGE_DEBUG=1 to print the blocks' heights, the TCPs and the score four times a second."""

import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("GAUGE_DEBUG") == "1"


class GaugeEmbodiment(YamArenaEmbodiment):
    def _tcp_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        base, out, zero = self._base, {}, np.zeros(3)
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            q = base.scene[robot].data.joint_pos.torch[0, arm_ids].detach().cpu().numpy().astype(float)
            p = np.asarray(ik.tcp_pose(q, np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=np.array((0.0, 0.0, 0.0, 1.0)), lin_vel=zero,
                                            ang_vel=zero, low=p, high=p)
        return out

    def _task_step(self):
        from yam_arena import scene

        original, last = scene.object_states, {}

        def object_states(env, names):
            last.update(original(env, names), **self._tcp_states())
            return dict(last)

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if DEBUG and last and round(self._t * self.info.control_hz) % round(self.info.control_hz / 4) == 0:
            m = self.task.module
            blocks = " ".join(f"{n[-1]}:{1000 * last[n].low[0]:.0f},{1000 * last[n].low[2]:.0f}" for n in m.PARTS)
            print(f"GAUGE t={self._t:5.1f} low x,z mm {blocks} | L {np.round(1000 * last['left_tcp'].pos).tolist()} "
                  f"R {np.round(1000 * last['right_tcp'].pos).tolist()} | sorted {m._sorted(last)} score "
                  f"{m.score(last):.2f}", flush=True)
        return result


def make_embodiment(**kwargs):
    return GaugeEmbodiment(**kwargs)
