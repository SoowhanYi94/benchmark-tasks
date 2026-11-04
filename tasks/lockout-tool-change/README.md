# Run a full lockout cycle to change a machine's cutter

A small machine's tool holder holds a worn cutter; a new cutter waits on a tool tray, and a red scrap bin stands beside
them. The machine's isolator is a handle in a track, now at ON, with a padlock on its mark beside it; its control box
has a green START button and a run lamp. The robot must run the whole lockout cycle in order:

1. **Isolate and lock:** slide the handle to OFF, then drop the padlock into the track behind it (it stops the handle
   going back to ON).
2. **Verify zero energy:** press START (at least 4 mm in); the run lamp stays off.
3. **Service:** the worn cutter into the scrap bin, the new one into the holder.
4. **Restore:** the padlock out of the track and back on its mark, the handle back to ON.

Disturbing a cutter on the machine (more than 3 mm, any corner) at any time it isn't both locked out and verified
scores 0 for the whole trial. Setup, scoring and fidelity are in [`task.json`](task.json).

Files: [`task.py`](task.py) (scene, scoring, scripted demo), [`lockout_tool_change/`](lockout_tool_change) (the
`yam_lockout` embodiment: the stock `yam_arena` one plus the control box, built in USD as one articulation, with the
START cap on a sprung prismatic joint and a lamp that lights when START is pressed while the isolator is at ON),
[`test_score.py`](test_score.py) (the scorer on synthetic states: every outcome and the exploits we could think of).
Everything else in the scene is a simulator primitive.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) at commit `0059d02` (Isaac Lab-Arena 0.3.1,
Isaac Sim 6.1), installed with its `setup.sh`, on an NVIDIA RTX GPU. Start each shell with
`source ~/IsaacLab-Arena/yam.env`, then install this folder's embodiment once: `uv pip install -e . --no-deps`.

## Run a trial

From this folder:

```bash
I="Lock out the machine, verify it is dead, change the cutter and restore it: slide the isolator handle to OFF and put the padlock in the track behind it, press the green START button and check the run lamp stays off, put the worn cutter in the scrap bin and the new cutter in the tool holder, then put the padlock back on its mark and slide the handle back to ON."

# Dream-1
inspect-robots run --embodiment yam_lockout --policy dream \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/lockout-tool-change/dream

# The scripted demo
inspect-robots run --embodiment yam_lockout --policy yam_scripted \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/lockout-tool-change/demo
```

The run prints `task_score` at the end; the episode ends early, with `termination_reason="success"`, once everything
has been in place for 1 s. The score remembers, for the whole trial, whether the machine was locked out, whether START
was pressed while it was, and whether a cutter was disturbed while it wasn't safe; only a reset (every part back at
its start within 1 mm, after jumping there) clears that, so run one trial per run. `LOCKOUT_DEBUG=1` prints the
button, lamp and score four times a second. Videos: `inspect-robots video <log>.json`.

`python -m pytest test_score.py` checks the scorer, and `python task.py` checks the demo fits the run limit and that
every waypoint solves with the arm IK (under 1 mm and 1 degree), all without the simulator.

- **Seeds:** none needed. `-E object_jitter=0` keeps the fixed layout, which the scripted demo (open loop) needs and
  the setup describes; the default seed is 0.
- **Runtime per trial:** the scripted demo's plan is 1,110 steps (55.5 s simulated); a Dream-1 trial runs up to
  2,400 steps.

## How the demo does it

1. The right arm grips the handle along the track (clear of the rails), lifts it 2 mm and slides it to OFF.
2. The right arm drops the padlock into the track behind the handle.
3. The right arm, fingers closed, pushes START down to its 8 mm stop and holds it; the lamp stays off.
4. The left arm lifts the worn cutter into the scrap bin, then stands the new cutter in the holder.
5. The right arm lifts the padlock back onto its mark and slides the handle back to ON.
