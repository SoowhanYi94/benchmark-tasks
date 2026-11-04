# Support a straightedge at its Bessel points

A 30 cm precision straightedge, deliberately unmarked, lies on a cloth in front of a granite surface plate, with a
support block on each side. A gauge bar or straightedge supported at its Bessel points, 0.2203 of its length in from
each end, keeps the length of its neutral axis most nearly unchanged as it sags (the Airy points, 0.2113, keep its end
faces parallel; overall sag is least at about 0.2232). The robot must stand the two blocks on the plate, lay the
straightedge across them, level and along y, with each block within 1 cm of a Bessel point (6.61 cm in from an end,
16.78 cm apart), and let go of it. Supported at any spacing scores 0.5, plus 0.25 per block at a Bessel point.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). Setup, scoring and fidelity are in
[`task.json`](task.json). Everything in the scene is a simulator primitive (boxes). The scorer checks that both
hands have let go, so [`bessel_supports/embodiment.py`](bessel_supports/embodiment.py) is the `yam_bessel`
embodiment: the stock `yam_arena` one plus each gripper's fingertip centre (`left_tcp`, `right_tcp`) in the states the
scorer gets.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`, then install the embodiment once from this folder:

```bash
pip install -e .
```

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_bessel --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Support the 30 cm straightedge at its Bessel points: stand the two support blocks on the surface plate, lay the straightedge across them, level, with each support 22 percent of its length in from its end, and let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/bessel-supports/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_bessel --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Support the 30 cm straightedge at its Bessel points: stand the two support blocks on the surface plate, lay the straightedge across them, level, with each support 22 percent of its length in from its end, and let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/bessel-supports/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once everything
has been in place for 1 s. Videos: `inspect-robots video <log>.json`.
`python -m pytest -q test_score.py` tests the scorer against every outcome in the scoring rules, including the
exploits found in review (the bar held still in the grippers, laid yawed or rolled, both blocks on one side of its
middle), and checks the demo plan; `python task.py` checks that the demo fits the run limit and that IK reaches every
waypoint. Neither needs the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo's plan is 566 steps (28 s simulated) and it succeeds about then, once both
  hands have moved clear and the straightedge has stayed in place for 1 s; about 1.2 minutes on an RTX 4070 once
  Isaac Sim has started. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

1. Each arm stands its support block on the plate, 8.4 cm either side of the plate's centre line.
2. Both arms grip the straightedge across its width 12 cm either side of its middle (outside the supports; closer
   together, the two hands collide), lift it, carry it over the supports and lower it onto them, centred.
