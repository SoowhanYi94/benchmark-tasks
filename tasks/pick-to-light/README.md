# Pick an order from a pick-to-light rack

A pick-to-light rack of four parts bins, three identical parts in each, has two indicator lamps in front of every
bin; each lit (green) lamp means one part to pick from that bin. Now the yellow bin's two lamps are lit and the orange
bin's left one, and the rest are dark. The robot must pick exactly the order the lamps show into the order tote. Each
extra part, or any part from a dark bin, costs 1.5 correct ones.

The task file is [`pick_to_light.py`](pick_to_light.py) (scene, scoring and scripted demo). Setup, scoring and
fidelity are in [`task.json`](task.json). Everything in the scene is a simulator primitive (boxes and cylinders).

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`.

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=pick_to_light.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "pick the order shown by the lamps into the order tote: from each bin, one part for every lit green lamp in front of it, and nothing from bins whose lamps are dark" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pick-to-light/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=pick_to_light.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "pick the order shown by the lamps into the order tote: from each bin, one part for every lit green lamp in front of it, and nothing from bins whose lamps are dark" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pick-to-light/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once the order has
been complete for 1 s. Videos: `inspect-robots video <log>.json`. `python pick_to_light.py` runs a quick self-check
of the scorer without the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo succeeds after 484 steps (24 s simulated), about 1.5 minutes on an RTX 4070
  once Isaac Sim has started. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

The left arm picks two yellow cylinders and the right arm one orange cylinder, one at a time, each gripped across its
width with the fingers along the row (so the open hand clears the neighbouring parts and the bin walls) and dropped
into the tote.
