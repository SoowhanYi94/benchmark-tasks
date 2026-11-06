# Splice a parted joint in a rod line

A rod line has four joints that look alike. One has parted. Working the handle opens a red gap there, showing where the motion stops. The robot must find it, seat the splice plate across it so the handle opens the gate again, and move both hands away.

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    ROD_LOT=A inspect-robots run --embodiment yam_rod -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Restore the drive, work the handle to confirm the gate opens, then move both hands away." \
      --scorer task_score --max-steps 2400 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 5 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`.
- **Lots:** `ROD_LOT` picks A, B, C or D, the parted joint from the handle end. The default is A.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is told the lot and plans ahead, to show the task can be done. It works the handle first, so the red gap a policy would need to see shows in its video.
