# Level a production mix in a heijunka box

Six production kanbans (three red for product A, two blue for B, one yellow for C) wait on a tray grouped by product.
The robot must load them into the six time-slot pockets of a heijunka box, one per pocket, so that no two neighbouring
pockets hold the same product: the mix is levelled across the shift. Any levelled sequence scores 1.0; with three A
kanbans in six slots, the robot has to plan where the A's go.

The task file is [`heijunka_box.py`](heijunka_box.py) (scene, scoring and scripted demo). Setup, scoring and fidelity
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
  -E task=heijunka_box.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "load the heijunka box to level the production mix: put one kanban in each of the six pockets so that no two neighbouring pockets hold the same product colour" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/heijunka-box/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=heijunka_box.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "load the heijunka box to level the production mix: put one kanban in each of the six pockets so that no two neighbouring pockets hold the same product colour" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/heijunka-box/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once the box
has been levelled for 1 s. Videos: `inspect-robots video <log>.json`. `python heijunka_box.py` runs a quick self-check
of the scorer (including a brute-force check that exactly the levelled sequences score 1.0) without the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo succeeds after 1,003 steps (50 s simulated), about 2 minutes on an RTX 4070
  once Isaac Sim has started. A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

It loads red, blue, red, yellow, red, blue from left to right. The left arm fills the three left pockets and the right
arm the three right ones, taking turns. Each kanban is let go 4 mm above the pocket walls and drops in.
