# Chuck a bar in a lathe and take the key out

A small lathe headstock with a 3-jaw chuck stands on the table, its jaws open. The robot must slide a round aluminium bar
into the chuck so 50 mm sticks out of the jaws, tighten the jaws on it with the chuck key (a T-handle seated in a
socket on top of the chuck), then take the key out of the chuck and stand it back in its holder. The bar counts as
tight when the key has turned at least 60 degrees past first contact of all three jaws and, with the key out and every
hand clear, the bar holds a 30 N pull out of the chuck, slipping less than 1 mm. A key left in the
chuck, or resting on it, caps the score at 0.1.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). The lathe is our own model in
[`chuck_key/`](chuck_key): `lathe.py` builds it (one articulation: the headstock, chuck body, key holder and V-block
as its fixed root; the pinion behind the key socket on a revolute joint; three jaws on radial slides that follow the
pinion through mimic joints, each with a 2 mm sprung give), plus the free bar and key. `embodiment.py` is the
`yam_chuck` embodiment: it spawns these into the station scene, reports the jaws, pinion, bar, key and both
fingertip centres to the scorer, and applies the judge's pull. The key turns the pinion only through contact, its
square drive in the socket's square hole.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`, then install the embodiment once from this folder:

```bash
uv pip install -e . --no-deps
```

## Run a trial

From this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_chuck --policy dream \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the bar into the lathe chuck with 50 mm sticking out of the jaws, tighten the jaws on it with the chuck key, then take the key out of the chuck and stand it back in its holder." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir logs/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_chuck --policy yam_scripted \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the bar into the lathe chuck with 50 mm sticking out of the jaws, tighten the jaws on it with the chuck key, then take the key out of the chuck and stand it back in its holder." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir logs/demo
```

The run prints `task_score` at the end. The episode ends early, with `termination_reason="success"`, once the score
has been 1.0 for 1 s. The judge's pull runs while all three jaws are on the bar, the key is out of the chuck and every fingertip centre is
at least 6 cm from the bar's surface; if the bar slips 1 mm the pull stops and counts as failed until the key turns another
2 degrees or a hand goes back to the bar, so the robot can tighten further and be tested again. Run one trial per
`inspect-robots run` (no `--epochs`). Videos: `inspect-robots video <log>.json`. Set `CHUCK_DEBUG=1` to print the
pinion, bar, jaws, key, hands and pull four times a second.

## Runtime and seeds

The demo ends with success after about 1,060 steps (53 s of sim), which took about 4 minutes on our RTX GPU; a full 2,400-step trial takes about 10 minutes. Seed 0 and `object_jitter=0`: the layout is fixed.

`python dev/chuck_check.py` checks the chuck alone, without robots: a torque on the key turns the pinion through the
socket, the jaws close 3 mm per turn, centre the bar and hold it with the key let go. (Its bare scene uses default
PhysX contact settings; the 30 N pull is checked in the full station scene, where the demo's bar slips 0 mm.)

The chuck model: a 3 mm-per-turn coupling (NewtonMimicAPI mimic joints, which the PhysX GPU pipeline honours; PhysX's
own PhysxMimicJointAPI was ignored here), a 0.25 N m friction threshold on the pinion (a capped damping drive) and a
2 mm sprung give in each jaw (50 N/mm). The demo turns the key about 690 degrees, about 80 past first contact. The socket has a 3 mm lead-in and the holder a 6 mm one.
