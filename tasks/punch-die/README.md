# Change a press to the die that clears its hidden punch

A guard over a small press hides its punch. Each die has its pocket in a different place, and the robot must find which one suits the press. Instruction: "Rack the old die, change the press to the die that clears its punch, cycle it with its guard on, and move both hands away."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    PUNCH_LOT=A inspect-robots run --embodiment yam_punch -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Rack the old die, change the press to the die that clears its punch, cycle it with its guard on, and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 3 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`.
- **Lots:** `PUNCH_LOT` picks A, B or C, the punch's place across the press. The default is A. Score every lot and average.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is told the lot, to show the task can be done. It lifts the guard and pauses with the punch in view on every lot before choosing a die.
- **Third-party assets:** none. Every part is built from boxes in the task's own code.
