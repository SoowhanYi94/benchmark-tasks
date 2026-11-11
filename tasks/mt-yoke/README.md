# Magnetic-particle test a weld in two directions and mark the cracks

A butt-welded steel test plate lies on the table with 160 mm of weld between two white marks. An AC magnetic yoke stands on its rest, a dry-powder squeeze bulb on its own, and three blue marker pegs in a tray. The robot has to examine the weld by the continuous method: one hand sets the yoke's poles on the plate with its trigger held, the other puffs powder between the poles while the field is on, and the yoke is stepped along the whole weld twice, the second time turned about 90 degrees. A crack only shows (a red line) where the field crosses it. Each crack found gets a peg, and both tools go back on their rests. Instruction: "Examine the weld between the white marks for cracks with the magnetic yoke and the powder bulb, mark each crack you find with a blue peg, then put both tools back on their rests and move both hands away."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_mt -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Examine the weld between the white marks for cracks with the magnetic yoke and the powder bulb, mark each crack you find with a blue peg, then put both tools back on their rests and move both hands away." \
      --scorer task_score --max-steps 3600 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** 8-9 minutes on one GPU (an L40S: about 3 minutes to start, 5.5 to run the 1,730-step demo).
- **Seeds:** none. The start is fixed and `object_jitter=0`; `--seed 0` was used for the demo.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`. `python task.py` checks the demo's reach, length and the hands' spacing. `python dev/replay.py` plays the demo's plan with perfect tracking through the scorer.
- **The scripted demo** is open loop and uses only the fixed starting layout. The left arm takes the yoke and the right arm the bulb. In direction A the yoke is set across the weld at four places 40 mm apart. At each one the nozzle, staged back along its own line, slides in between the poles from the front-right, and the bulb is squeezed for 0.6 s while the field is on. The yoke is then turned 90 degrees and set along the weld at three places 50 mm apart, the nozzle coming in from the front. Between placements the yoke is lifted clear of the plate. The yoke's hand bears down 12 mm past contact so both poles sit flat. Its aim allows for the arm's measured give, taken from logged runs: it lands about 1 cm low in the air and a few mm off in plan. Both tools go back on their rests, and the right arm sets a peg on each red line.
- **What the physics does:** the yoke, bulb, pegs, plate and hands are rigid bodies with real contact, and the bulb's plunger is a sprung prismatic joint that the fingers squeeze. The field, powder and indications are geometric events (see `task.json`'s fidelity and the top of `task.py`). The lamp, puff, powder and red lines are shown from the score's record.
- **Shortcut variants** (`MT_DEMO`) score 0: `dust_first` (one arm at a time: the weld dusted with the yoke on its rest, then the yoke stepped through both passes alone), `one_dir` (direction A only, its crack marked), `wrong_peg` (the second peg 40 mm off its crack, on a clean stretch of weld) and `no_peg`. `MT_DEBUG=1` prints the state every 5 steps; `MT_PRESS`, `MT_NOZZLE`, `MT_YSAG` and `MT_BSAG` are tuning switches only.
- **Third-party assets:** none. Every part is built from boxes and cylinders in the task's own code.
