# Run a dead-weight gauge test: load, float and spin the weights, then trim the gauge

An instrument technician's job on a dead-weight tester: load the slotted weights, pump until the piston floats, spin the weights to take out piston friction, then trim the test gauge to the reference while they spin. Instruction: "Load both weights onto the piston, pump until the orange flag is in the green band, spin the weights, turn the blue knob while they spin until the gauge needle is on the red mark, and move both hands away."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_deadweight -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Load both weights onto the piston, pump until the orange flag is in the green band, spin the weights, turn the blue knob while they spin until the gauge needle is on the red mark, and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 4 minutes on one GPU.
- **Seeds and lots:** none. The start is fixed and `object_jitter=0`.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is open loop, from the fixed start. It pumps a planned five strokes and turns the knob a planned 66.7 degrees; both can be judged from the flag on its band and the needle on its dial, which face up. It reads no internal state.
- **Wrong-way switches, for debugging only:** `DW_WRONG=null_first`, `no_spin`, `null_late`, `overpump`; `DW_SWAP=1` loads the other weight first.
- **Third-party assets:** none. Every part is built in the task's own code.
