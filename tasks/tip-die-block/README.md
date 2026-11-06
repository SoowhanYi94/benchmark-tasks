# Tip a block over its edges until its red face is up

A 12 cm hollow steel block, too wide for a gripper to pick up, stands in a rubber-matted lane between low curbs. One face,
the next to machine, is painted red. The robot must tip the block over its edges along the lane until the red face is up,
and move both hands away. A push high on a face tips it over its edge; a push low only slides it. Each face next to the
red one has a red band along their shared edge, so the top shows which way the red face lies, and no band means it is
underneath. Lots A, B and C start with the red face to the left, to the right and underneath, needing one tip one way, one
tip the other way, or two tips. The task's score is the mean over one trial of each lot.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). Setup, scoring and fidelity are in
[`task.json`](task.json). The embodiment `yam_tip` ([`tip_die_block/embodiment.py`](tip_die_block/embodiment.py)) spawns
the lane and the marked block, and gives the scorer each gripper's fingertip midpoint.

## Terms

- **Tip:** turning the block 90 degrees over one bottom edge.
- **Hands clear:** each gripper's fingertip midpoint 3 cm or more from the block's surface.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`, then install this folder's embodiment once: `uv pip install -e . --no-deps`.

## Run a trial

From this folder, once per lot (`TIP_LOT` = `A`, `B` or `C`; `A` if unset), one trial per run:

```bash
# Dream-1
TIP_LOT=A inspect-robots run --embodiment yam_tip --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Tip the block over along the lane, without lifting it, until its red face is on top, then move both hands away." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/tip-die-block/dream-A

# The scripted demo
TIP_LOT=A inspect-robots run --embodiment yam_tip --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Tip the block over along the lane, without lifting it, until its red face is on top, then move both hands away." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/tip-die-block/demo-A
```

The run prints `task_score` at the end. There is no early end: the score is read at the end of the run, so a plan can't
stop as soon as it happens to be right. Stillness and the lift limit are tracked each step, and only the embodiment's reset clears them. `TIP_DEBUG=1` prints the block's pose and the score four times a second.
`python -m pytest -q test_score.py` tests the scorer against every outcome in the scoring rules in all three lots, the
tip and slide arithmetic, and fixed plans; `python task.py` runs the demo-plan and IK check alone. Neither needs the simulator.

- **Seeds:** none needed. `-E object_jitter=0` is required: it keeps the block square in its lane. The lot is set by
  `TIP_LOT`, not by the seed.
- **Runtime per trial:** the scripted demo runs 220 steps in lots A and B and 380 in lot C (11 to 19 s simulated), about 1.5 minutes of wall clock on an RTX 4070 once Isaac Sim has started. A Dream-1 trial runs up to 1,800 steps.

## How the demo does it

With fingers closed, the arm comes down beside the block, pushes high on the face and rises with it until the block goes
past its balance point and falls over its edge, and lifts away. In lot C the left arm makes the first tip and the right arm the second. The
demo is open loop: it reads the lot from the task file.
