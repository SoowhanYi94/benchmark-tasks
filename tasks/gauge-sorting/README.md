# Sort machined blocks with a go/no-go gauge

Five machined aluminium blocks lie in a row on the bench. They look alike, but two are 4 mm too tall, and their
lengths and widths vary so that nothing else gives them away. A fixed bridge gauge stands beyond them: a steel bar
whose underside is 3.7 cm above the table. A good block (3.5 cm tall) slides under the bar; an oversize one (3.9 cm)
stops against its face. The robot must check every block with the gauge and put the good ones in the green ACCEPT tray
and the oversize ones in the red REJECT tray. A wrongly sorted block costs 1.5 correct ones, so sorting by guesswork
or putting every block in one tray scores 0.

The task file is [`gauge_sorting.py`](gauge_sorting.py) (scene, scoring and scripted demo). Setup, scoring and
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
  -E task=gauge_sorting.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "check each block with the bridge gauge: put the blocks that slide under the bar into the green ACCEPT tray and the ones that don't into the red REJECT tray" \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/gauge-sorting/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=gauge_sorting.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "check each block with the bridge gauge: put the blocks that slide under the bar into the green ACCEPT tray and the ones that don't into the red REJECT tray" \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/gauge-sorting/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once every block
has been in its tray for 1 s. Videos: `inspect-robots video <log>.json`. `python gauge_sorting.py` runs a quick
self-check of the scorer and the demo's length without the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo succeeds after 1,355 steps (68 s simulated), about 2.7 minutes on an
  RTX 4070 once Isaac Sim has started (the first run on a machine adds a few minutes for shader compilation).
  A Dream-1 trial runs up to 3,600 steps.

## How the demo does it

Each block goes with the arm on its tray's side, one arm at the gauge at a time. For each block the arm:

1. grips it across its width 2.5 cm in from its rear end, so the fingers never reach the bar;
2. sets it on the table in front of the gauge and slides it toward the bar: in the logged run the good blocks' fronts
   went 0.8-1.8 cm under the bar and the oversize blocks' fronts stopped at its face (within 1 mm);
3. slides it back out, lifts it and lays it in its tray.

The demo is open loop: it knows which blocks are good, and the gauge test shows the result rather than deciding it.
The slide aims 1.5 cm past its target because the arms reach about that much short at the gauge (measured).
