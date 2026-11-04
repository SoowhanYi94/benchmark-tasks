# Replenish line-side bins by a min/max rule

Three line-side bins run on a min/max rule (minimum 2, maximum 3): a bin below 2 is refilled to 3, others are left
alone. Now they hold 1, 3 and 0 parts. The robot must add 2 parts to the left bin and 3 to the right bin from the bulk
tote, and leave the middle bin alone. Each part beyond a bin's maximum, or taken from the full bin, costs 1.5 correct
ones.

The task file is [`min_max_replenish.py`](min_max_replenish.py) (scene, scoring and scripted demo). Setup, scoring
and fidelity are in [`task.json`](task.json). The sockets are an NVIDIA SimReady model; the bins and tote are simulator boxes.

**Assets:** Socket_3_4in_A01, from NVIDIA SimReady (Isaac Sim 6.1 asset library), CC BY 4.0. The task file registers it at load time and Isaac Sim fetches it from NVIDIA's asset server, so the first run needs internet access.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`.

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=min_max_replenish.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "replenish the line-side bins by their min/max rule: refill any bin holding fewer than 2 parts up to 3 parts from the bulk tote, and leave bins holding 2 or more alone" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/min-max-replenish/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=min_max_replenish.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "replenish the line-side bins by their min/max rule: refill any bin holding fewer than 2 parts up to 3 parts from the bulk tote, and leave bins holding 2 or more alone" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/min-max-replenish/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once the bins
have been replenished for 1 s. Videos: `inspect-robots video <log>.json`. `python min_max_replenish.py` runs a quick
self-check of the scorer without the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo succeeds after 830 steps (42 s simulated), about 2 minutes on an RTX 4070
  once Isaac Sim has started. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

The left arm moves two sockets from the bulk tote into the left bin and the right arm three into the right bin, one at a
time, each let go 2.2 cm above the bin floor so the finger pads clear the sockets already there. The middle bin is not
touched.
