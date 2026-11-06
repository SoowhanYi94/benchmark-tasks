# Find and turn round a backwards anti-runback dog

A carrier has to reach the end of its rail, past three capped dogs. One was fitted backwards and stops it dead, and the caps hide which from above. The robot finds it by pushing the carrier, lifts that dog off, turns it round and seats it again, then pushes the carrier to its end stop.

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    DOG_LOT=A inspect-robots run --embodiment yam_dog -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Get the carrier to its end stop with all three dogs seated on their studs facing forward, then move both hands away." \
      --scorer task_score --max-steps 2400 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 6 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`.
- **Lots:** `DOG_LOT` picks A, B or C, the backwards dog from the handle end. The default is A.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is told the lot and plans ahead, to show the task can be done. It pushes the carrier first, so where it stops shows in the video.
