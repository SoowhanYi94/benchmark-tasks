# Retain a housing in a rollover fixture, turn it over and lock it, and fit a pad on its far face

A housing goes in a rollover fixture recess down, is retained by a sliding bar, is turned over on the fixture's axle and locked by a bolt, and a wear pad is fitted in the recess, now facing up. Instruction: "Lay the housing in the rollover fixture with its orange recess down, slide the blue bar across it, turn the fixture over, push the red bolt home, fit the white wear pad in the recess, and move both hands away."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_trunnion -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Lay the housing in the rollover fixture with its orange recess down, slide the blue bar across it, turn the fixture over, push the red bolt home, fit the white wear pad in the recess, and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 5 minutes on one GPU for a full 3,000-step run; the scripted demo plays about 69 s of simulated time and then stops.
- **Seeds:** none. The start is fixed and `object_jitter=0`. There are no lots or variants.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is open loop. Everything it uses is the fixed layout in plain sight; it is told nothing a policy couldn't see. `TRUNNION_DEMO` (nobar, nobolt, empty, padfirst, nopad, arc) selects wrong-way runs for debugging, and `TRUNNION_DEBUG=1` logs state.
- **Third-party assets:** none. Every part is built from boxes in the task's own code.
