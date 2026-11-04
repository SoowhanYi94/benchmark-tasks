# Replenish line-side bins by their min/max labels

Four line-side bins each carry a min/max label: a row of squares, one per part of the bin's maximum, red up to its
minimum (the reorder point) and green beyond. Only a bin holding no more than its minimum is refilled, to exactly its
maximum; any other bin is left alone. The labels read A 1/3, B 1/3, C 2/3 and D 1/2, and the bins hold 0, 2, 2 and 2.
So A needs 3 sockets and C needs 1 from the bulk tote, while B (above its min) and D (at its max) must not be
touched. B and C hold the same count but need different actions. Each socket correctly added scores 0.25; any
mistake (topping up B or D, overfilling A or C, taking a socket out of any bin) scores 0.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). Setup, scoring and fidelity are in
[`task.json`](task.json). The sockets are an NVIDIA SimReady model; the bins, tote and labels are simulator boxes.

**Assets:** Socket_3_4in_A01, from NVIDIA SimReady (Isaac Sim 6.1 asset library), CC BY 4.0. The task file registers
it at load time and Isaac Sim fetches it from NVIDIA's asset server, so the first run needs internet access.

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
  --instruction "Replenish the line-side bins by their min/max labels: refill only the bins holding no more parts than their minimum (the red squares on the label) up to exactly their maximum (all the squares) with sockets from the bulk tote, and leave every other bin as it is." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/min-max-replenish/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Replenish the line-side bins by their min/max labels: refill only the bins holding no more parts than their minimum (the red squares on the label) up to exactly their maximum (all the squares) with sockets from the bulk tote, and leave every other bin as it is." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/min-max-replenish/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once the bins
have been right for 1 s. Videos: `inspect-robots video <log>.json`. `python -m pytest -q test_score.py` tests the
scorer against every outcome in the scoring rules and the exploits found in review (filling every bin to its max, held
or perched sockets, robbing one bin for another, blind placement over all 5^6 ways: mean 0.009), plus the demo's reach
and length; it doesn't need the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo plan is 516 steps (26 s simulated); success follows 1 s after the last
  socket settles. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

The left arm moves three sockets from the tote into bin A (places 1, 2, 3) while the right arm moves one into bin C's
free place, each let go 2.2 cm above the bin floor so the finger pads clear the sockets already there. B and D are not
touched.
