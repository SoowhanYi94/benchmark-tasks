# Sort machined blocks with a two-limit go/no-go gauge

Six machined aluminium blocks lie in two rows on the bench, in one of three layouts (A, B, C). They look alike, but they come in three heights, 3.5 mm
apart: two are in tolerance (3.55 cm), two oversize (3.90 cm) and two undersize (3.20 cm); their lengths, widths and
masses vary so that nothing else gives them away. A fixed two-limit bridge gauge stands beyond them: a green GO bar
whose underside is 3.75 cm above the table and, beside it, a red NO-GO bar at 3.40 cm. A good block slides under the
GO bar but stops at the NO-GO bar; an oversize one stops at the GO bar; an undersize one slides under both. The robot
must gauge every block against both limits and put it in the green ACCEPT, red OVERSIZE or blue UNDERSIZE tray. A
wrongly sorted (or dropped) block cancels three correct ones, so guessing, gauging with one bar only or putting
everything in one tray scores 0 (the best blind strategy averages 0.056). Across the three layouts every place holds an
undersize, a good and an oversize block once, so where a block lies says nothing about it; the task's score is the
mean over one trial of each layout.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). Setup, scoring and fidelity are in
[`task.json`](task.json). Everything in the scene is a simulator primitive (boxes). The embodiment `yam_gauge`
([`gauge_sorting/embodiment.py`](gauge_sorting/embodiment.py)) is the stock `yam_arena` one that also gives the
scorer each gripper's fingertip centre, so a block still held doesn't count.

## Terms

- **Go/no-go gauge:** a fixed gauge that tells good parts from bad without a reading: a good part goes on the GO side
  and does not go on the NO-GO side.
- **Two-limit:** the GO bar checks the upper limit (oversize blocks stop at it); the lower NO-GO bar checks the lower
  limit (undersize blocks also slide under it).
- **ACCEPT / OVERSIZE / UNDERSIZE:** the green, red and blue trays for good, too-tall and too-short blocks.
- **In a tray:** centre inside its walls, on its floor (not stacked or on a wall), still, and let go (fingertips 2 cm
  or more away).
- **Wrong:** a block in any other tray, or off the table.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`, then install this folder's embodiment once: `uv pip install -e . --no-deps`.

## Run a trial

From this folder, once per layout (`GAUGE_LAYOUT` = `A`, `B` or `C`; `A` if unset), one trial per run:

```bash
# Dream-1
GAUGE_LAYOUT=A inspect-robots run --embodiment yam_gauge --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Check each block, lying flat, with the two-bar gauge: put blocks that slide under the green GO bar but not under the red NO-GO bar in the green ACCEPT tray, blocks that don't fit under the GO bar in the red OVERSIZE tray, and blocks that also slide under the NO-GO bar in the blue UNDERSIZE tray." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/gauge-sorting/dream-A

# The scripted demo (reaches 1.0)
GAUGE_LAYOUT=A inspect-robots run --embodiment yam_gauge --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Check each block, lying flat, with the two-bar gauge: put blocks that slide under the green GO bar but not under the red NO-GO bar in the green ACCEPT tray, blocks that don't fit under the GO bar in the red OVERSIZE tray, and blocks that also slide under the NO-GO bar in the blue UNDERSIZE tray." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/gauge-sorting/demo-A
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once every block
has been in its tray for 1 s. Videos: `inspect-robots video <log>.json`. `GAUGE_DEBUG=1` prints the blocks, the
fingertips and the score four times a second. `python -m pytest -q test_score.py` tests the scorer against every
outcome in the scoring rules and the exploits we found (held, moving, stacked or dropped blocks, one-bar gauging,
blind play, memorising one layout's answer by place), checks every layout's scene, and checks that each layout's
demo fits the run limit and that IK reaches every waypoint. `python task.py` runs the demo-plan and IK check alone
(for `GAUGE_LAYOUT`'s layout). Neither needs the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0. The layout is set by `GAUGE_LAYOUT`, not by the seed.
- **Runtime per trial:** the scripted demo's plan is 2,056 steps (103 s simulated) in every layout, and it succeeds 1 s after the
  last block is let go. A Dream-1 trial runs up to 3,600 steps.

## How the demo does it

Blocks go one at a time, back row first, each with the arm on its tray's side (left for ACCEPT, right for OVERSIZE
and UNDERSIZE); the other arm waits out of the way. The demo reads the layout from the task file. For each block the arm:

1. grips it across its width 2.5 cm in from its rear end, so the fingers never reach a bar;
2. sets it on the table in the GO lane, 2 cm before the bar, slides it toward the bar and back out: a good or
   undersize block's front goes 1.5 cm under the bar, an oversize one stops at its face;
3. if it went under, eases it 1 cm up, moves it across to the NO-GO lane, and slides it toward that bar and back out:
   an undersize block goes under, a good one stops at its face;
4. lifts it and lays it in its tray, opening only 2 cm wider than the block.

The demo is open loop: it knows each block's kind, and the gauge shows the result rather than deciding it. Each slide
aims 1.5 cm past its target because the arms reach about that much short at the gauge (measured in an earlier
one-bar version of this task). Every step of the plan passes IK within 0.1 mm and 0.02 degrees.
