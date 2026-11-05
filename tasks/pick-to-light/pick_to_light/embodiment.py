"""`yam_ptl`: the stock yam_arena embodiment, plus two states for the scorer each step: `left_tcp` and `right_tcp`
(each gripper's fingertip centre, from the arm's joint positions), so it can tell whether a part has been let go. It
also switches the lamps: after each step it lights what the task's `active()` says and dims the rest."""

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment


class PickToLightEmbodiment(YamArenaEmbodiment):
    _shown = None
    _shaders = None

    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        obs = super().reset(scene, seed=seed)
        self._show(self.task.module.active())
        return obs

    def _show(self, lit):
        """Set every lamp's colour to match ``lit`` (bin, lamps), or dark if None. The Arena spawner gives each object
        its own preview-surface shader; finding none is an error, or the task would silently show the wrong order."""
        import omni.usd
        from pxr import Gf, Sdf

        m = self.task.module
        stage = omni.usd.get_context().get_stage()
        if self._shaders is None:
            self._shaders = {}
            for prim in stage.Traverse():
                name = next((p for p in str(prim.GetPath()).split("/") if p.startswith("lamp_") and p != "lamp_strip"), None)
                if name and prim.HasAttribute("inputs:diffuseColor"):
                    self._shaders.setdefault(name, []).append(prim)
            want = {f"lamp_{b}_{k}" for b in m.BIN_Y for k in (0, 1)}
            if want - set(self._shaders):
                raise RuntimeError(f"no shader found for {sorted(want - set(self._shaders))}")
        for name, prims in self._shaders.items():
            _, b, k = name.split("_")
            on = lit is not None and b == lit[0] and int(k) < lit[1]
            for sh in prims:
                sh.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*(m.LIT if on else m.DARK)))
                sh.CreateAttribute("inputs:emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*(m.LIT if on else (0.0, 0.0, 0.0))))
        self._shown = lit

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

        last = {}

        def object_states(env, names):
            last["s"] = {**original(env, names), **self._tcp_states()}
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if "s" in last:
            m = self.task.module
            m._latch(last["s"])
            if m.active() != self._shown:
                self._show(m.active())
        return result


def make_embodiment(**kwargs):
    return PickToLightEmbodiment(**kwargs)
