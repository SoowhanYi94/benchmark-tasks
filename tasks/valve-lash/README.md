# Set a rocker's valve lash with a feeler gauge and lock it

An engine-valve fixture stands on the table: a cylinder-head block with a valve stem, a fixed rocker arm over it, and
in the rocker's end a vertical adjusting screw with a T-bar and a loosened locknut with spoked pins. The gap between
the screw's tip and the stem (the valve lash) starts at 8 mm. The robot must slide a 2 mm feeler blade into the gap,
turn the screw down until it rests on the blade, run the locknut down onto the rocker, then pull the blade out and
move both hands away. The lash must end within 2.0 +- 0.3 mm.

The task file is [`task.py`](task.py) (scene, scoring and scripted demo). The fixture is our own parametric model in
[`valve_lash/`](valve_lash): `fixture.py` builds the fixed frame, the screw-and-locknut articulation (each a
revolute joint driving a prismatic one through a mimic joint, 2 mm per turn, with a torque-capped clutch so it holds
where it is left) and the gauge; `embodiment.py` is the `yam_valve` embodiment, which spawns them into the
station scene, reports the screw, nut, gauge and fingertips to the scorer, locks the screw while the nut is seated, and
latches whether the screw has been set down on the blade (no credit without it).

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
inspect-robots run --embodiment yam_valve --policy dream \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the valve lash with the feeler gauge: slide its blade under the adjusting screw, turn the screw down onto it, tighten the locknut onto the rocker, then pull the blade out and move both hands away." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir logs/dream

# The scripted demo (reaches 1.0)
inspect-robots run --embodiment yam_valve --policy yam_scripted \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the valve lash with the feeler gauge: slide its blade under the adjusting screw, turn the screw down onto it, tighten the locknut onto the rocker, then pull the blade out and move both hands away." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir logs/demo
```

The run prints `task_score` at the end. The episode ends early, with `termination_reason="success"`, once the lash
is in tolerance, the nut is seated, the blade is out and both hands are clear, held for 1 s. Run one trial per
`inspect-robots run` (no `--epochs`). Videos: `inspect-robots video <log>.json`. Set `VALVE_DEBUG=1` to print the
gap, the nut, the gauge and the hands four times a second.

## Runtime and seeds

The fixture's joints are cheap: the demo's 1,895 steps (it ends early, on success) take about 5 minutes on an RTX 4070, and a full 2,400-step trial about 6 minutes. Seed 0 and `object_jitter=0`: the layout is fixed.

`python dev/mech_check.py` checks the mechanism alone, without robots: a torque on the screw takes it down 2 mm per
turn until the blade stops it at a 2.0 mm gap, a torque on the nut runs it down onto the rocker, and the seated nut
locks the screw (its joint limits closed on where it is), holding it under 2 N*m for 30 s.

`pytest test_score.py` checks the scoring on synthetic states.
