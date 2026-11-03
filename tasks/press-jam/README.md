# Clear a press jam safely and reseat the next blank

The last blank has jammed in a press: it sits crooked across the locating nest, turned 45 degrees and resting on the
nest's edges. The press's mode selector is at AUTO. The robot must slide the selector to SETUP first, clear the jammed
blank into the reject bin, seat the next blank squarely in the nest and slide the selector back to AUTO. Touching the
jam while the press is not in SETUP caps the whole trial at 0.1.

The task file is [`press_jam.py`](press_jam.py) (scene, scoring and scripted demo). Setup, scoring and fidelity are
in [`task.json`](task.json). Everything in the scene is a simulator primitive (boxes).

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`.

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=press_jam.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "clear the jam safely: slide the mode selector to SETUP first, put the jammed blank in the reject bin, seat the next blank squarely in the nest, then slide the selector back to AUTO" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/press-jam/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=press_jam.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "clear the jam safely: slide the mode selector to SETUP first, put the jammed blank in the reject bin, seat the next blank squarely in the nest, then slide the selector back to AUTO" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/press-jam/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once everything
has been in place for 1 s. The score remembers, within a trial, whether the jam was touched outside SETUP (it resets
when the scene is back in its starting state). Videos: `inspect-robots video <log>.json`. `python press_jam.py` runs a
quick self-check of the scorer (including the SETUP rule) without the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo succeeds after 678 steps (34 s simulated), about 1.4 minutes on an RTX 4070
  once Isaac Sim has started. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

1. The right arm grips the selector handle along the track (clear of the rails), lifts it 2 mm and slides it to SETUP.
2. The left arm grips the jammed blank square to its turned faces, lifts it off the nest, turns it on the way and drops
   it in the reject bin.
3. The left arm lowers the next blank into the nest, square to it.
4. The right arm slides the selector back to AUTO.
