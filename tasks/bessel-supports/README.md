# Support a straightedge at its Bessel points

A 30 cm precision straightedge lies on a cloth in front of a granite surface plate, with a support block on each
side. Precision straightedges are supported at their Bessel points, 22 % of their length in from each end, where they
sag least. The robot must stand the two supports on the plate 16.8 cm apart and lay the straightedge across them,
level, with each support under a Bessel point (6.6 cm in from an end). Laying it on supports at the wrong spacing earns
half credit.

The task file is [`bessel_supports.py`](bessel_supports.py) (scene, scoring and scripted demo). Setup, scoring and
fidelity are in [`task.json`](task.json). Everything in the scene is a simulator primitive (boxes).

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`.

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=bessel_supports.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "support the 30 cm straightedge at its Bessel points: stand the two support blocks on the surface plate and lay the straightedge across them, level, with each support 22 percent of its length in from its end" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/bessel-supports/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=bessel_supports.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "support the 30 cm straightedge at its Bessel points: stand the two support blocks on the surface plate and lay the straightedge across them, level, with each support 22 percent of its length in from its end" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/bessel-supports/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once everything
has been in place for 1 s. Videos: `inspect-robots video <log>.json`. `python bessel_supports.py` runs a quick
self-check of the scorer without the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo succeeds after 530 steps (27 s simulated), about 1.1 minutes on an RTX 4070
  once Isaac Sim has started. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

1. Each arm stands its support block on the plate, 8.4 cm either side of the plate's centre line.
2. Both arms grip the straightedge across its width 12 cm either side of its middle (outside the supports; closer
   together, the two hands collide), lift it, carry it over the supports and lower it onto them, centred.
