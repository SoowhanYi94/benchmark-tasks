# Change over a conveyor's guide rails for a new carton

The end of a packaging line's conveyor has two loose guide rails set 14 cm apart for the last product. The next
product is a carton 8 cm wide. The robot must move both rails in so the carton just fits between them (2 to 12 mm of
clearance in all), centred on the conveyor's centre line and parallel to it, then set the carton between the rails on
the outfeed mark as a test.

The task file is [`guide_rail_changeover.py`](guide_rail_changeover.py) (scene, scoring and scripted demo). Setup,
scoring and fidelity are in [`task.json`](task.json). Everything in the scene is a simulator primitive (boxes).

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`.

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=guide_rail_changeover.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "change the conveyor over for the new carton: move both guide rails in so the carton just fits between them, centred on the white line and parallel to it, then put the carton between the rails on the green outfeed mark" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/guide-rail-changeover/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=guide_rail_changeover.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "change the conveyor over for the new carton: move both guide rails in so the carton just fits between them, centred on the white line and parallel to it, then put the carton between the rails on the green outfeed mark" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/guide-rail-changeover/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once everything
has been in place for 1 s. Videos: `inspect-robots video <log>.json`. `python guide_rail_changeover.py` runs a quick
self-check of the scorer without the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo succeeds after 518 steps (26 s simulated), about 1.1 minutes on an RTX 4070
  once Isaac Sim has started. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

1. Each arm in turn lifts its rail and sets it down 5.55 cm from the centre line (aimed 5.7 mm further in, the
   arms' measured shortfall), giving an 8.8 cm gap.
2. The left arm stands the carton between the rails on the outfeed mark.
