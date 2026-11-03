# Load a trailer in reverse delivery order

Five parcels for one delivery route, coloured by stop, must be loaded into a trailer that is unloaded from its door.
The route board shows the stops in delivery order (blue, red, yellow), so the robot must load in reverse: the yellow
parcels deepest at the nose, then red, then blue nearest the door. A parcel scores when it is loaded and every parcel
for another stop lies on the correct side of it.

The task file is [`route_loading.py`](route_loading.py) (scene, scoring and scripted demo). Setup, scoring and fidelity
are in [`task.json`](task.json). Everything in the scene is a simulator primitive (boxes).

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`.

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=route_loading.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "load the parcels into the trailer for the route: the route board shows the stops in delivery order from left to right; the trailer is unloaded from the door, so the first stop's parcels must be nearest the door and the last stop's at the nose" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/route-loading/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=route_loading.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "load the parcels into the trailer for the route: the route board shows the stops in delivery order from left to right; the trailer is unloaded from the door, so the first stop's parcels must be nearest the door and the last stop's at the nose" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/route-loading/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once every parcel
has been in sequence for 1 s. Videos: `inspect-robots video <log>.json`. `python route_loading.py` runs a quick
self-check of the scorer without the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo succeeds after 833 steps (42 s simulated), about 1.7 minutes on an RTX 4070
  once Isaac Sim has started. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

The arms take turns loading single file down the middle of the trailer, nose first: yellow at x = 38 and 33, red at
28, blue at 23 and 18. Each parcel is gripped across its 6 cm width, the fingers going down into the 2 cm beside the
side walls.
