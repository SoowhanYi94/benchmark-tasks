# Press a warped flange at its high spot

A warped flange looks plain. Turning it by its crank moves a dial pointer, which peaks when the high spot is under the ram. The robot must turn it to the peak, press, and move both hands away. A press anywhere else scores 0 for good.

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    FLANGE_LOT=A inspect-robots run --embodiment yam_flange -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Press the warped flange at its high spot, then move both hands away." \
      --scorer task_score --max-steps 2400 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 4 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`.
- **Lots:** `FLANGE_LOT` picks A, B, C or D, the turn that brings the high spot under the ram. The default is A.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is told the lot, to show the task can be done. It first turns the flange through the same full search on every lot, so the pointer's peak shows in its video.
