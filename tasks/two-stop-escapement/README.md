# Meter parts out of a feed chute with a two-stop escapement, keeping the red part back

A gravity feed chute slopes down across the bench, its low end over a catch bin, with six parts queued nose to tail, one
of them red. A slotted cover keeps the parts in, so none can be lifted out. Two stops meter the queue. The orange lower
stop is sprung shut, and its pad opens it when pressed down. The blue upper stop is sprung open, and its pad drops a pin
between the first and second parts when pressed down. To release one part: hold the upper stop down, press the lower one,
let the lower one go, then let the upper one go, and the queue moves up one. A part going by the lower stop without the pin down scores 0. Open both together and the queue runs out.
The robot must release exactly the parts ahead of the red one and keep the red one in the chute. Lots A, B, C and D put
the red part second to fifth, so the count has to come from looking at the queue. The task's score is the mean over one
trial of each lot.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). Setup, scoring and fidelity are in
[`task.json`](task.json). The embodiment `yam_escapement`
([`two_stop_escapement/embodiment.py`](two_stop_escapement/embodiment.py)) spawns the chute, stops and parts
([`two_stop_escapement/mech.py`](two_stop_escapement/mech.py)) and gives the scorer each gripper's fingertip midpoint.

## Terms

- **Escapement:** a mechanism that lets parts through one at a time.
- **Stop:** a plate or pin that blocks the chute; the lower stop is sprung shut and the upper stop sprung open.
- **Lot:** which place in the queue the red part has.
- **Red held:** the red block still in the chute and uphill of the lower stop's face.
- **Hands clear:** each gripper's fingertip midpoint 6 cm or more from the top centre of both pads and from every part.
- **Pin down:** the upper stop pressed 1 cm or more.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`, then install this folder's embodiment once: `uv pip install -e . --no-deps`.

## Run a trial

From this folder, once per lot (`ESCAPEMENT_LOT` = `A`, `B`, `C` or `D`; `A` if unset), one trial per run:

```bash
# Dream-1
ESCAPEMENT_LOT=A inspect-robots run --embodiment yam_escapement --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Use the two stops on the feed chute to release, one at a time, every part ahead of the red one into the bin, then leave the red part in the chute with both hands off the stops." \
  --scorer task_score --max-steps 2400 --no-prompt --store-frames --log-dir ~/yam-eval/two-stop-escapement/dream-A

# The scripted demo
ESCAPEMENT_LOT=A inspect-robots run --embodiment yam_escapement --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Use the two stops on the feed chute to release, one at a time, every part ahead of the red one into the bin, then leave the red part in the chute with both hands off the stops." \
  --scorer task_score --max-steps 2400 --no-prompt --store-frames --log-dir ~/yam-eval/two-stop-escapement/demo-A
```

The run prints `task_score` at the end. There is no early end: the score is read at the end of the run, so a plan can't
release parts one at a time blind and be stopped at the right count. The stillness clock is tracked each step, and only the embodiment's reset clears it. `ESCAPEMENT_DEBUG=1` prints the joints, the parts' places along the chute and the score four times a second.
`python -m pytest -q test_score.py` tests the scorer against every outcome in the scoring rules in all four lots, the
geometry the mechanism relies on, and fixed-count plans; `python task.py` runs the demo-plan and IK check alone. Neither
needs the simulator.

- **Seeds:** none needed. `-E object_jitter=0` is required: it keeps the parts in their queue. The lot is set by
  `ESCAPEMENT_LOT`, not by the seed.
- **Runtime per trial:** the scripted demo's plan is 224, 336, 448 and 560 steps in lots A to D (11 to 28 s simulated).
  A Dream-1 trial runs up to 2,400 steps.

## How the demo does it

Each arm closes its fingers and hovers over its pad: left over the upper stop, right over the lower. For each part ahead of
the red one, the left presses the upper stop and holds; the right presses the lower stop, waits for the part to run out and
lets it shut; the left lets go, and the queue moves up one. Then both arms move away. The demo is open loop: it reads the
lot from the task file, which stands in for what a policy reads from the camera, where the red part shows through the slot.
