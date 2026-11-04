# Plug a manifold's open ports for a pressure test

A hydraulic manifold has four ports on its top face: one already plugged, one test port marked blue (which takes the test
gauge), and two open. Before the pressure test the robot must plug the two open ports and leave the test port open.
Plugging the test port, or pulling the fitted plug, costs 1.5 ports.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). Setup, scoring and fidelity
are in [`task.json`](task.json). Everything in the scene is a simulator primitive (boxes and cylinders).

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`.

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "get the manifold ready for its pressure test: plug every open port, but leave the blue test port open for the gauge" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/port-plugging/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "get the manifold ready for its pressure test: plug every open port, but leave the blue test port open for the gauge" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/port-plugging/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once both ports
have been plugged for 1 s. Videos: `inspect-robots video <log>.json`. `python -m pytest -q test_score.py` tests the
scorer against every outcome in the scoring rules, and `python task.py` runs a quick self-check; neither needs
the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo succeeds after 351 steps (18 s simulated), under a minute on an RTX 4070 once
  Isaac Sim has started. A Dream-1 trial runs up to 1,200 steps.

## How the demo does it

The left arm takes the left plug to the port at y = 10.5 and the right arm the right plug to the port at y = -3.5. Each
plug is held near its top and let go 4 mm above the manifold face, centred on its port, so it drops straight in. The
middle plug is a spare.
