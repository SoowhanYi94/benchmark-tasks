# Set parallels clear of a plate's through-holes

A plate has two through-holes. The robot must stand two parallels on the mat where they won't block them, at least 10 cm apart, rest the plate on them, then drop a pin through each hole. A pin that meets a parallel stops 3 cm proud, as a drill would.

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    PARALLELS_LOT=A inspect-robots run --embodiment yam_parallels -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Stand the two parallels on the mat at least 10 cm apart and clear of the plate's holes, rest the plate on them, drop a pin fully through each hole, and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 8 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`.
- **Lots:** `PARALLELS_LOT` picks A, B, C or D, the plate's hole layout. The default is A.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is told the lot and plans ahead, to show the task can be done.
