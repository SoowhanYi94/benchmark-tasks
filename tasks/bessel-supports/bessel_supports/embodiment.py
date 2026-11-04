"""`yam_bessel`: the stock yam_arena embodiment, the harness itself unchanged, plus each gripper's TCP (fingertip
centre) in the states the task's scorer gets, as `left_tcp` / `right_tcp`, so it can tell whether the hands have let go
of the straightedge."""

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment


class BesselEmbodiment(YamArenaEmbodiment):
    def _tcp_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        zero, out = np.zeros(3), {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            q = self._base.scene[robot].data.joint_pos.torch[0, arm_ids].detach().cpu().numpy().astype(float)
            p = np.asarray(ik.tcp_pose(q, np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=np.array((0.0, 0.0, 0.0, 1.0)), lin_vel=zero,
                                            ang_vel=zero, low=p, high=p)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        scene.object_states = lambda env, names: {**original(env, names), **self._tcp_states()}
        try:
            return super()._task_step()
        finally:
            scene.object_states = original


def make_embodiment(**kwargs):
    return BesselEmbodiment(**kwargs)
