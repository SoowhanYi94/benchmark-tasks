"""`yam_ptl`: the stock yam_arena embodiment, plus two states for the scorer each step: `left_tcp` and `right_tcp`
(each gripper's fingertip centre, from the arm's joint positions), so it can tell whether a part has been let go."""

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment


class PickToLightEmbodiment(YamArenaEmbodiment):
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

        original = scene.object_states

        def object_states(env, names):
            return {**original(env, names), **self._tcp_states()}

        scene.object_states = object_states
        try:
            return super()._task_step()
        finally:
            scene.object_states = original


def make_embodiment(**kwargs):
    return PickToLightEmbodiment(**kwargs)
