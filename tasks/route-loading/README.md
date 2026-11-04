# Load a trailer in reverse delivery order

Five parcels for one delivery route, coloured by stop, must be loaded into a trailer that is unloaded from its door.
The route board shows the stops in delivery order, so the robot must load in reverse: the last stop's parcels deepest
at the nose, the first stop's nearest the door. The board's order is one of six variants (`ROUTE_VARIANT`), so the
colours can't be memorised. A parcel counts when it is loaded and every parcel for another stop is on the correct side
of it; the score is (parcels that count − 2) / 3, not below 0, so a blind load (about 2 right by chance) scores about 0.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). Setup, scoring and fidelity are in
[`task.json`](task.json). Everything in the scene is a simulator primitive (boxes).

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`.

## Run a trial

From this folder, with `ROUTE_VARIANT` set to the board variant (0 to 5, default 0; the variants are listed in the
setup and in `VARIANTS` in `task.py`):

```bash
# Dream-1
ROUTE_VARIANT=0 inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the parcels into the trailer so they can be unloaded from its door in the delivery order shown on the route board." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/route-loading/dream

# The scripted demo (reaches 1.0 on every variant)
ROUTE_VARIANT=0 inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the parcels into the trailer so they can be unloaded from its door in the delivery order shown on the route board." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/route-loading/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once every parcel
has counted for 1 s. Videos: `inspect-robots video <log>.json`. `python -m pytest -q test_score.py` tests the scorer
against every outcome in the scoring rules and the exploits found in review (blind or memorised orders, a parcel half
out of the door, held up or still moving), and checks the demo's waypoints for IK reach on every variant; it doesn't
need the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0. The board's order is set by `ROUTE_VARIANT`, one trial per variant.
- **Runtime per trial:** the scripted demo's plan is 860 steps (43 s simulated); it reaches success shortly before
  the plan ends, after the last parcel is released, about 1.7 minutes on an RTX 4070 once Isaac Sim has started. A
  Dream-1 trial runs up to 2,400 steps.

## How the demo does it

The arms take turns loading single file down the middle of the trailer, the last stop's parcels first: slots at x =
37, 32, 27, 22 and 17 cm, nose to door, each arm taking the parcels on its side. Each parcel is gripped across its 6 cm
width, the fingers going down into the 2 cm beside the side walls and opening to 6.8 cm to release it.
