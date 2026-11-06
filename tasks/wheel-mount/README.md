# Mount a grinding wheel safely on a bench grinder

A bench grinder stands on the table with its spindle bare. A parts tray in front of it holds the mounting parts: a
black inner flange keyed to the spindle's flat, a steel outer flange, two paper blotters, a good grinding wheel and a
damaged one (a chip out of its rim and a crack). The spindle nut, with a four-spoke handle, stands on a post in line
with the spindle, and the spindle lock's red button is on the grinder's left.

The robot must mount the good wheel the standard way: from the collar out, inner flange, blotter, wheel, blotter,
outer flange; then run the nut on with the spindle lock held down (unpressed, the spindle turns with the nut), tighten
it enough to drive the wheel but not so far that it cracks, and take both hands away. The nut's free run before it
seats is 1.81 mm. Once the stack is complete, the nut seated and the
hands clear, the judge spins the spindle with the wheel braked (only the clamp's friction drives the wheel, so a
loose nut lets it slip) and pulls the wheel 18 N toward the nose (the clamp spring gives unless the nut has preloaded
it). The damaged wheel on the spindle, or turning the nut 240 degrees or more past where it seated (which cracks the wheel), scores 0.
`task.json` has the exact scoring.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). The grinder is our own model in
[`wheel_mount/`](wheel_mount): `fixture.py` builds it (the grinder body and tray as one kinematic body; the spindle,
the nut's joint chain and the lock button as one articulation; each mounting part a free rigid ring of convex
segments, so its bore really fits round the spindle). `embodiment.py` is the `yam_wheel` embodiment: it spawns these
into the station scene, threads the nut on when it is pushed to the thread's start, locks the spindle while the button
is pressed, raises the nut's turning torque with the clamp force, latches the crack, runs the judge and reports the
parts, nut, judge and both fingertip centres to the scorer.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`, then install the embodiment once from this folder:

```bash
uv pip install -e . --no-deps
```

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_wheel --policy dream \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount the good grinding wheel on the grinder's spindle the safe way, tighten its nut, and move both hands away." \
  --scorer task_score --max-steps 3000 --seed 0 --no-prompt --store-frames --log-dir logs/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_wheel --policy yam_scripted \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount the good grinding wheel on the grinder's spindle the safe way, tighten its nut, and move both hands away." \
  --scorer task_score --max-steps 3000 --seed 0 --no-prompt --store-frames --log-dir logs/demo
```

The run prints `task_score` at the end. Run one trial per `inspect-robots run` (no `--epochs`). Videos:
`inspect-robots video <log>.json`. Set `WHEEL_DEBUG=1` to print the parts, nut, spindle, judge and hands four times a
second.

## Runtime and seeds

The scripted demo takes 2,407 steps (120 s simulated), succeeding once the judge's spin and tug pass, about 15 minutes of wall clock on an RTX 4070 once Isaac Sim has started. A Dream-1 trial runs up to 3,000 steps.

Seeds: none needed; the layout and the nut's 1.81 mm free run are fixed.

## Checks

- `python -m pytest -q` (NumPy only): every scoring outcome and exploit on synthetic states, the inner flange's D key
  (it slides on only with its flat lined up with the spindle's, and is scored only that way up), and the nut's rising
  torque.
- `python dev/ik_check.py`: every demo waypoint passes IK.
- `python dev/mech_check.py` (Isaac Sim, no robots): the nut drags an unlocked spindle; locked, it runs on at 2.5 mm
  per turn; then the judge's spin and tug. The demo seats the nut at 277 degrees of turn and stops at 391, 114 past the seat, with 0.8 mm of squeeze; the wheel slips 1 degree in the spin and moves 0.025 mm in the tug.

The model: the nut's thread is a NewtonMimicAPI mimic joint (which the PhysX GPU pipeline honours) coupling its turn
to its advance, 2.5 mm per turn; its turning torque is a capped damping drive, 0.12 N m of free-run drag plus 0.008 N m
per N of clamp force; the clamp is a 30 N/mm spring between the nut and its washer face; the spindle lock closes the
spindle joint's limits while the button is down 2.5 mm or more.
