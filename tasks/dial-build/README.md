# Build two parts on a turning assembly dial, then stop it

An assembly dial keeps turning while two parts are built on it: a block in each pocket and a peg in each block. Either block and peg will do. Instruction: "Fit a block in each blue pocket on the turning dial and a peg in each block, press the red pad to stop the dial, and move both hands away."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_dial -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Fit a block in each blue pocket on the turning dial and a peg in each block, press the red pad to stop the dial, and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 4 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`. There are no lots: the dial starts the same each time.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is open loop. Its timing comes from the dial's known start and rate, and it's told nothing a policy couldn't see from the pockets' positions.
- **Third-party assets:** none. Every part is built from boxes in the task's own code.
