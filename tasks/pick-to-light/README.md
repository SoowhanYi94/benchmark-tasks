# Pick, confirm and report shortages on a pick-to-light rack

A pick-to-light rack of four parts bins has two indicator lamps on a strip in front of every bin; each lit (green) lamp
means one part to pick from that bin. In front of the strip each bin has a white confirm button and a red shortage
button, plungers pushed in toward the strip. One bin is lit at a time: pressing the lit bin's button dims it and lights the
next, so the robot can't see the order up front and must read the rack again after every press. Each trial is one of three
orders of three bins, one of them empty (A: yellow 2, orange 1, blue 1, orange empty; B: blue 2, grey 1, yellow 1, grey
empty; C: orange 1, grey 2, blue 2, grey empty). Every extra or substitute part, part taken before its bin lit, part on
the floor and wrong press costs 1.5 correct ones and caps the trial at 0.2, so pressing every button scores 0. The task's
score is the mean over one trial of each order.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). Setup, scoring and fidelity are in
[`task.json`](task.json). Everything in the scene is a simulator primitive (boxes and cylinders). The embodiment
`yam_ptl` ([`pick_to_light/embodiment.py`](pick_to_light/embodiment.py)) is the stock `yam_arena` one that also gives
the scorer each gripper's fingertip centre, so a part still held in the tote isn't counted as placed, and switches
the lamps as the order moves on.

## Terms

- **Pick-to-light:** a picking station where a light at each bin tells the picker what to take; here one green lamp
  per part to take.
- **Lit / dark bin:** a bin with at least one / no green lamp in front of it.
- **Confirm button (white):** pressed once a bin's parts are in the tote, to tell the order system the pick is done.
- **Shortage button (red):** pressed instead of picking when a lit bin is empty (a stock-out).
- **Substitute:** a part from any other bin put in the tote in place of the missing one; it is wrong.
- **Placed:** in the tote, at rest on its floor, and let go (both fingertip centres 2 cm or more away).
- **Pressed:** a plunger moved 4 mm or more from its start; it stays counted for the trial.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`, then install this folder's embodiment once: `uv pip install -e . --no-deps`.

## Run a trial

From this folder, once per order (`PICK_TO_LIGHT_PATTERN` = `A`, `B` or `C`; `A` if unset), one trial per run:

```bash
# Dream-1
PICK_TO_LIGHT_PATTERN=A inspect-robots run --embodiment yam_ptl --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pick the order shown by the lamps into the order tote, one bin at a time: take one part for every lit green lamp from the lit bin, then press its white confirm button to light the next bin; if the lit bin is empty, press its red shortage button instead and take nothing in its place; stop when every lamp is dark." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pick-to-light/dream-A

# The scripted demo (reaches 1.0)
PICK_TO_LIGHT_PATTERN=A inspect-robots run --embodiment yam_ptl --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pick the order shown by the lamps into the order tote, one bin at a time: take one part for every lit green lamp from the lit bin, then press its white confirm button to light the next bin; if the lit bin is empty, press its red shortage button instead and take nothing in its place; stop when every lamp is dark." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pick-to-light/demo-A
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once the order,
its confirm and the shortage have all held for 1 s. Button presses are latched in the task file for the episode (with
the tote's count at the moment of each press), and it clears only on a harness reset: a step on which a loose object jumped more
than 1 cm and everything is then at rest within 1 mm of its start (a robot putting things back can't do that). Videos: `inspect-robots video <log>.json`.

`python -m pytest -q test_score.py` tests the scorer against every outcome in the scoring rules in all three orders:
the full order, picks without buttons, a part still gripped on the tote floor, each button alone, a confirm before or after a short pick, substitutes for the
empty bin, each kind of wrong press, mashing every button (before or after the picks), pulling a wrongly pressed
plunger back out, forcing one out of its rails, extra parts stacked or held over the tote, held parts, parts on the
floor; it also checks that IK reaches every demo waypoint. `python task.py` runs the demo-plan and IK check alone.
Neither needs the simulator.

- **Seeds:** none needed. `-E object_jitter=0` is required: it keeps the fixed layout, which the scripted demo (open
  loop) needs, and jitter would also shift the buttons in their rails. The default seed is 0. The order is set
  by `PICK_TO_LIGHT_PATTERN`, not by the seed.
- **Runtime per trial:** the scripted demo's plan is 866 steps (43 s simulated) for each order, about 1.5 minutes of
  wall clock on an RTX 4070 once Isaac Sim has started. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

Both arms park. Then, one bin at a time in the order the lamps call them, the arm on that bin's side (left for the yellow
and grey bins, right for orange and blue) either picks the lit number of parts, the one nearest the arms first, gripping
each across its width with the fingers closing along y, drops them into the tote and presses the bin's white confirm
button with closed fingers (the plunger stops against the strip after 7.5 mm); or, at the empty bin, presses its red
shortage button. The other arm waits parked. The demo is open loop: it reads the order from the task file, and doesn't
look at the lamps.
