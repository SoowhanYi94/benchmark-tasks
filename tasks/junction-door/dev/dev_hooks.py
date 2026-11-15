"""No-robot mechanism checks for task.py (JD_DEV=<mode>; never set in real trials).

scene: print the cord's joint names once; at 3 s a dev "hand" (the hinge drive) swings the door shut; once it is
       latched the tracker is put straight into the door test (unlatch, open to the stop, shut again), so the catch,
       the friction hinge and the test sequence are checked without a robot or a plugged cord.
"""
import os

import bodykit

MODE = os.environ.get("JD_DEV", "")
_S = {"printed": False, "closing": False, "forced": False}


def step(env, st, m, tr, gripper_points):
    """Never import task.py here: a second copy would register its own hooks."""
    t = m["t"]
    if not _S["printed"]:
        _S["printed"] = True
        art = env.unwrapped.scene["cord"]
        print("[dev] cord joints:", list(art.joint_names), "bodies:", list(art.body_names), flush=True)
    if MODE == "print" and int(round(t * 20)) % 10 == 0:
        import numpy as np

        c, g, top = st["cord"], gripper_points(env), [0, 0, 0.7475]
        print(f"[dev] t={t:.1f} plug={np.round((c.end_plug - top) * 1000).astype(int).tolist()} "
              f"dir={np.round(c.end_dir, 2).tolist()} rtcp={np.round((g[7] - top) * 1000).astype(int).tolist()} "
              f"rgap={np.linalg.norm(g[5] - g[6]) * 1000:.0f} ltcp={np.round((g[3] - top) * 1000).astype(int).tolist()}",
              flush=True)
    if int(round(t * 20)) % 40 == 0:
        import numpy as np
        c = st["cord"]
        print(f"[dev] t={t:.1f} cord pts (mm, rel. table):", np.round((c.points - [0, 0, 0.7475]) * 1000).astype(int).tolist(),
              "plug", None if c.end_plug is None else np.round((c.end_plug - [0, 0, 0.7475]) * 1000).astype(int).tolist(),
              flush=True)
    if MODE in ("scene", "catch"):
        if t >= float(os.environ.get("JD_CLOSE_AT", 3.0)) and not _S["closing"]:
            _S["closing"] = True
            bodykit.set_hinge_drive("door_hinge", target_deg=30.0, spring=0.02, damping=0.004, max_torque=0.8)
            tr.dev_hand = True
            print("[dev] closing the door", flush=True)
        if _S["closing"] and not _S["forced"] and m["door"] < 0.6 and m["bolt"] > -0.0008:
            _S["forced"] = True
            print(f"[dev] latched at t={t:.2f}; starting the door test", flush=True)
            tr.phase, tr.t_test, tr.dev_hand = "test", t, False
    if MODE == "catch" and tr.phase != "robot" and int(round(t * 20)) % 4 == 0:
        print(f"[dev] test t={t:.2f} door={m['door']:.2f} bolt={m['bolt'] * 1000:.2f}", flush=True)


def tie(env, st, tr):
    """JD_DEV=tie: when the door test starts, tie the cord to the door JD_TIE_SEG links (15 mm each) along from the
    lamp, as a cable tie on the door would: a ball joint pinning that link where it is, relative to the door. The
    free hinge loop is then that much shorter, with nothing else changed (a mechanism check of the slack window)."""
    if MODE != "tie" or _S.get("tied") or tr.phase != "test":
        return
    _S["tied"] = True
    import numpy as np
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    k = int(os.environ.get("JD_TIE_SEG", "5"))
    door, cord = st["door"], st["cord"]
    origin = bodykit.as_numpy(env.unwrapped.scene.env_origins[0])
    p_seg = cord.points[k]
    local = door.rotation().T @ (p_seg - door.pos)
    stage = get_current_stage()
    root = next(str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).endswith("/door/body")).rsplit("/", 2)[0]
    j = UsdPhysics.Joint.Define(stage, f"{root}/door/dev_tie")
    j.CreateBody0Rel().SetTargets([f"{root}/door/body"])
    j.CreateBody1Rel().SetTargets([f"{root}/cord/seg_{k}"])
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in local]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateExcludeFromArticulationAttr(True)
    for dof in ("transX", "transY", "transZ"):
        lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), dof)
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)
    print(f"[dev] tied cord link {k} ({(k + 0.5) * 0.015:.3f} m from the lamp) to the door at {np.round(p_seg, 3)}"
          f" (env origin {origin})", flush=True)
