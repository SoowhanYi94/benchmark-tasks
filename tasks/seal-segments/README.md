# Run seal segments round a turbine ring groove, passing them between arms

A turbine fitter's job: four identical seal segments go in through a casing ring's gate and run clockwise round its covered groove, the wrist turning with the arc, and the gate is closed with a bar. A step on the groove's floor stops a segment being pushed round by the one behind it, so each of the first two has to be lifted across by its knob, beyond where the arm that fed it can reach. With a straight-down grip the left arm stops reaching at about -30 degrees and the right arm at about +30 degrees (`python task.py` prints this), so the left arm carries a segment round its side and the right arm takes it on.

Instruction: "Run all four seal segments in through the gate and clockwise round the ring's groove until all four lie under the cover, lifting a segment over the floor step at the yellow mark where it has to cross it, then lay the orange bar in the gate and move both hands away."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_seal -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Run all four seal segments in through the gate and clockwise round the ring's groove until all four lie under the cover, lifting a segment over the floor step at the yellow mark where it has to cross it, then lay the orange bar in the gate and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** 5 to 10 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`. There are no lots or variants.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is open loop, about 2,630 steps. It works from the fixed layout in the task's text and the ring's geometry, all of which a policy can see in the screenshot or read in the setup, and is told nothing else. Debug switches: `SEAL_WRONG=chain` (one arm pushes the segments along), `SEAL_WRONG=nolift` (slid into the step), `SEAL_WRONG=early` (bar on with a segment out).
- **Third-party assets:** none. Every part is built from boxes in the task's own code.
