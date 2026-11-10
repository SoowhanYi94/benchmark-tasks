# Break a seized pulley loose with two pry bars and take it off

A cast-iron V-belt pulley is seized on a gearmotor's output shaft, its back face 19 mm from the bearing housing. Two pry bars lie on wooden rests in front of the machine. The robot has to lever the pulley loose with the bars without touching it by hand, then take it off the shaft and stand it in the V-block. Instruction: "Use the pry bars to break the pulley loose on the motor shaft without touching it by hand, then take it off the shaft and stand it in the V-block."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_pulley -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Use the pry bars to break the pulley loose on the motor shaft without touching it by hand, then take it off the shaft and stand it in the V-block." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** 5-8 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`. There are no lots.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`. `python task.py` checks the demo's reach and length.
- **The scripted demo** is open loop. It uses only the fixed starting layout (where the bars, housing face and pulley are), which a policy can see; it is told nothing about the fit. Each arm takes its bar from the rests, lays it on the slide rail, slides its tip into the gap behind the pulley up to the bearing cap, and both arms swing their handles toward the robot together, levering on the housing's bevelled corners. The arms take the bars back to the rests, and the right arm pinches the loose pulley at the top across its back face and rim, slides it off the shaft and stands it in the V-block.
- **What the physics does:** the seized fit is a capped axial drag (60 N) until the pulley has come 8 mm; the bore's fit on the shaft is real contact (0.3 mm a side, friction 0.5). Measured with forces applied at the handles and no arm: the two bars together move it at about 1-2.5 N per handle; one bar alone cocks it 1.4 degrees and it jams, with no travel to 25 N at the handle; a straight pull on the hub needs about 64 N, and a pinched hub pulled by one arm slips after 0.4 mm. `PULLEY_PUSH`, `PULLEY_START=seated` and `PULLEY_DEMO` are debugging switches only (see the code).
- **Third-party assets:** none. Every part is built from boxes and lathe-turned meshes in the task's own code.
