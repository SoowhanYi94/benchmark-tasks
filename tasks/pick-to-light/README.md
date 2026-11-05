# Pick, confirm and report shortages on a pick-to-light rack

A pick-to-light rack of four parts bins has two indicator lamps on a strip in front of every bin; each lit (green) lamp
means one part to pick from that bin. In front of the strip each bin has a white confirm button and a red shortage
button, plungers pushed in toward the strip. Each trial lights three lamps over two bins in one of three patterns, and
one of the two lit bins is empty (A: both yellow-bin lamps, and the orange bin's left one with the orange bin empty;
B: both blue-bin lamps, and the grey bin's left one with the grey bin empty; C: the orange bin's left lamp, and both
grey-bin lamps with the grey bin empty). The robot must pick the stocked bin's parts into the order tote, then press
that bin's confirm button, and press the empty bin's shortage button instead of picking a substitute. Every extra or
substitute part, part on the floor and wrong press (a confirm before the picks are done, on a dark or the empty bin; a
shortage on any other bin) costs 1.5 correct ones and caps the trial at 0.2, so pressing every button scores 0. The
task's score is the mean over one trial of each pattern.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). Setup, scoring and fidelity are in
[`task.json`](task.json). Everything in the scene is a simulator primitive (boxes and cylinders).

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`.

## Run a trial

From this folder, once per pattern (`PICK_TO_LIGHT_PATTERN` = `A`, `B` or `C`; `A` if unset), one trial per run:

```bash
# Dream-1
PICK_TO_LIGHT_PATTERN=A inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pick the order shown by the lamps into the order tote, taking one part for every lit green lamp in front of a bin and nothing from bins whose lamps are dark, and after picking from a bin press its white confirm button; if a lit bin is empty, press its red shortage button instead and take nothing in its place." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pick-to-light/dream-A

# The scripted demo (reaches 1.0)
PICK_TO_LIGHT_PATTERN=A inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pick the order shown by the lamps into the order tote, taking one part for every lit green lamp in front of a bin and nothing from bins whose lamps are dark, and after picking from a bin press its white confirm button; if a lit bin is empty, press its red shortage button instead and take nothing in its place." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pick-to-light/demo-A
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once the order,
its confirm and the shortage have all held for 1 s. Button presses are latched in the task file for the episode (with
the tote's count at the moment of each press), and it clears only on a harness reset: a step on which a loose object jumped more
than 1 cm and everything is then at rest within 1 mm of its start (a robot putting things back can't do that). Videos: `inspect-robots video <log>.json`.

`python -m pytest -q test_score.py` tests the scorer against every outcome in the scoring rules in all three patterns:
the full order, picks without buttons, each button alone, a confirm before or after a short pick, substitutes for the
empty bin, each kind of wrong press, mashing every button (before or after the picks), pulling a wrongly pressed
plunger back out, forcing one out of its rails, extra parts stacked or held over the tote, held parts, parts on the
floor; it also checks that IK reaches every demo waypoint. `python task.py` runs the demo-plan and IK check alone.
Neither needs the simulator.

- **Seeds:** none needed. `-E object_jitter=0` is required: it keeps the fixed layout, which the scripted demo (open
  loop) needs, and jitter would also shift the buttons in their rails. The default seed is 0. The lamp pattern is set
  by `PICK_TO_LIGHT_PATTERN`, not by the seed.
- **Runtime per trial:** the scripted demo's plan is 584 steps (29.2 s simulated) for patterns A and B and 416 steps
  (20.8 s) for C, about 1.5 minutes of wall clock on an RTX 4070 once Isaac Sim has started. A Dream-1 trial runs up to
  2,400 steps.

## How the demo does it

Both arms first move to a park pose beside the tote. The arm on the empty bin's side presses its shortage button:
fingers closed, it comes down 3 cm in front of the plunger, pad tips 4 mm above the table, pushes it 2.5 cm forward
(the plunger stops against the strip after 7.5 mm) and backs off. Then, one part at a time, the arm on the stocked
bin's side (left for the yellow and grey bins, right for orange and blue) picks its parts, the one nearest the arms
first, gripping each across its width with the fingers closing along y, and drops them into the tote; then it presses
that bin's confirm button the same way. The other arm waits parked. The demo reads the pattern from the task file.
