# Hot-tap a live pipe through its valve

A tapping machine bores a coupon out of a live pipe through an open-top valve. The gate and the boring bar physically block each other, and the machine itself has to be set on and lifted off. Instruction: "Hot-tap the live pipe through the valve to cut out its coupon, return the tapping machine to its yellow stand, leave the valve shut with the blank on its outlet, and move both hands away."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_hot_tap -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Hot-tap the live pipe through the valve to cut out its coupon, return the tapping machine to its yellow stand, leave the valve shut with the blank on its outlet, and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 4 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`. There are no lots.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is open loop from the known fixed start; it uses nothing a policy couldn't see. Debug switches: `HT_DEMO_WRONG=lift_open` lifts the machine off with the gate open after the cut; `HT_DEMO_WRONG=shut_on_bar` tries to shut the gate on the fed bar.
- **Third-party assets:** none. Every part is built from boxes and discs in the task's own code.
