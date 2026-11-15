# Wire a door-mounted lamp across the hinge of a junction box and latch the door

A standard small sheet-steel junction box at its real catalogue size (200 x 150 x 80 mm, not a scaled-down cabinet)
is screwed to the bench, lying on its back for wiring, its hinged door standing open at 90 degrees. A red 22 mm LED
pilot lamp is mounted in the door near the hinge. Its 5 mm cord (30 cm) leaves the back of the lamp, runs across the open box and over its front edge, and ends in
a connector plug lying on the bench. Inside the box, on the back plate, are the panel socket for the plug and a blue
spring cable clip. The robot must plug the lamp in, press its cord into the clip and shut the door until the catch
clicks.

Instruction (identical in `task.json` and `task.py`):

> Wire the door lamp to the socket inside the junction box, fixing its cord in the cable clip, and latch the door shut.

What it tests: bringing a cord that hangs out of the box back inside, seating a friction-fit connector in a socket
that faces the robot (lining it up and pushing it home against the socket's contact spring), pressing the cord into a
snap-in spring clip, and swinging a door shut and pressing it until its catch clicks, without anything left across
the rim. The cord crosses the hinge and is clipped inside the box. Its slack across the hinge is fixed by where the clip sits,
not chosen by the robot: the clip grips the cord and the 2.6 cm of stiff cord between the clip and the seated plug leave
no room to move spare cord to the socket side (a demo variant that drags the loop toward the clip hand over hand still
scores 1.0). So the task does not test judging a loop length. Once the robot is done, the door is tested as an
inspector would: the catch is released, the door is swung open to its 120 degree stop and shut again, and the plug must
stay in, the cord stay clipped and nothing be trapped. That shows the finished wiring survives the swing.

Evidence (sim runs, logs and table in the evidence folder, not part of this PR): the scripted demo scores 1.0; the
same demo with the door shut before plugging (cord still across the rim) scores 0 (the door stops at 2.3 degrees on
the cord, the catch cannot click); with the cord never pressed into the clip, 0.2 (plug seated, door latched, no clip
credit, door test never starts). A dev check that ties the cord to the door 8 cm along it when the test starts (a
scorer-side check that the door test detects a strained cord, not something a robot can do) scores 0.6: the cord is
pulled out of the clip at 52 degrees, the plug at 79 degrees, and the door stops at 91 degrees.

## How it is modelled

One task file, [`task.py`](task.py) (scene, scoring, scripted demo), for the `yam_arena` embodiment, plus two helper
modules copied next to it from accepted tasks in this repository: [`cablekit.py`](cablekit.py) (the cord) and
[`bodykit.py`](bodykit.py) (multi-shape bodies, hinge and sliders). Both are the program's own shared helpers, not
third-party assets; this copy adds two small options, marked `junction-door` in the code: a cable start welded to a
moving body (the door) instead of the world, and a hinge drive torque cap.

- Box: one fixed rigid body (back plate, four walls, socket, clip base, catch housing), colliders 2 mm thick.
- Door: a rigid body on a PhysX revolute joint along the hinge line, 0 to 120 degrees. Friction hinge (0.35 N m, holds
  the door where it is left): a spring drive capped at 0.35 N m whose anchor the task file drags along with the door
  whenever the door is pushed past it (Coulomb friction).
- Catch: a spring bolt (PhysX prismatic joint, 300 N/m) with a 45 degree ramp, in a housing on the front wall; a hook
  hanging from the door's free edge pushes it back and the bolt springs over the hook. Contact only.
- Cord: 20 links of 15 mm (cablekit capsule chain, bending springs, no twist, no stretch), its lamp end welded to the
  door's lamp; a rigid 44 x 15 x 12 mm plug body welded to the other end.
- Socket: a cavity 19 x 15 mm, 16 mm deep, with a 4 mm lead-in, opening toward the robot; a sprung contact pad
  (prismatic joint, 400 N/m) in its roof squeezes the plug about 1 mm: the plug is held by contact friction only,
  no latch.
- Clip: a fixed wall and a sprung jaw (prismatic joint, 600 N/m) 4 mm apart; the 5 mm cord pressed in spreads the jaw
  and is gripped by friction, so it cannot slide through.
- Door test: the task file switches the catch bolt back, leads the door with a capped hinge drive (at most 0.8 N m)
  to its stop in 2 s, holds it 1 s, leads it back shut and lets the bolt snap over the hook again.

## Run and score a trial

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
    -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 2400 \
    --instruction "Wire the door lamp to the socket inside the junction box, fixing its cord in the cable clip, and latch the door shut."
```

Scripted demo: add `--policy yam_scripted -P hold_s=10` (the hold keeps the run going while the 7 s door test finishes; with the default 3 s hold the policy stops mid-test and scores 0.8; it plays `demo()`; a few of its waypoints are nudged closed loop by the
plug's measured position while lining the plug up with the socket). Test variants of the demo: `JD_DEMO=loose`
(cord left unclipped), `JD_DEMO=unlatched` (door swung nearly shut, never pressed home), `JD_DEMO=shut_first` (door shut before plugging). Dev checks (no robot work changed):
`JD_DEV=catch` (a dev hand shuts the door, then the door test runs), `JD_DEV=tie JD_TIE_SEG=5` (cord tied to the door
8 cm along when the door test starts: a strained cord).

One trial per process, or one per embodiment reset: the scorer keeps the trial's history, cleared on every reset.

## Scoring

Judged with a ruler and a photo. Terms: plug seated = in line with the socket, its tip at least 12 mm inside (a mark 12 mm from the tip is hidden); cord clipped = the cord lies under the clip's lip, between its wall and jaw, along the clip's whole length; latched = door shut with the catch bolt over its hook; hands clear = no part of either gripper within 3 cm of the box or of the door's swing (up to 26 cm above the bench). Ladder, each rung needing all the ones below it and hands clear: 0.2 plug seated; 0.4 cord clipped; 0.6 door latched. Door test, once 0.6 has held, all still, for 1 s: an inspector presses the catch, swings the door by hand to its 120 degree stop over 2 s (at most 0.8 N m, about 6 N at the free edge), holds it 1 s, swings it shut and pushes until it clicks (in simulation the task file drives the hinge the same way). 0.8: the door reached at least 115 degrees with the plug seated, the cord clipped and hands clear the whole time. 1.0: it then latched again with the plug seated, the cord clipped, no cord outside the box's walls and hands clear, all still for 1 s. A failed door test leaves the score at 0.6 (or 0.8 if only the re-shutting failed). A plug pulled below 12 mm during the door test is a failed test, not a cap: 0.8 if it stayed seated until the door had reached 115 degrees, else 0.6. Zero, whatever else happened: the cord was ever outside the box's walls (across the rim) while the door was within 8 degrees of shut; or the door came within 8 degrees of shut before the plug was first seated. Cap: at most 0.2 once the plug has been below 12 mm at any moment after it was first seated and before the door test began. Any end state matching no rung scores 0.

Real hardware: a ruler, a marker line on the plug 12 mm from its tip, a photo of the closed seam, and a person who
presses the catch, swings the door by hand to its stop and shuts it again (in simulation the task file leads the
hinge the same way, with at most 0.8 N m). Tests: `test_score.py` covers every rung, both zeroing events, the cap,
the door test outcomes and the unintended strategies (plug held in, door held shut, door shut early, cord unclipped).

## Runtime and seeds

Run limit 2,400 steps (2 minutes at 20 Hz). The scripted demo reaches full score and ends with success after 1,046
steps (52.3 s simulated, of which about 7 s is the door test). One trial takes about 5-7 minutes wall clock on one
L40S including Isaac Sim start-up. The layout is fixed (`-E object_jitter=0`; nothing is jittered), so every seed
starts from the same state; the cord settles over the front wall in the first second.

## Files

- `task.py`: scene, scoring, door test, scripted demo.
- `cablekit.py`, `bodykit.py`: shared helpers from this repository (see above).
- `test_score.py`: scorer tests on synthetic states (`python -m pytest -q test_score.py`).
- `dev/ik_check.py`: every demo waypoint through `yam_arena.ik.solve` (CPU).
- `dev/dev_hooks.py`: dev-only mechanism checks (`JD_DEV=catch`, `JD_DEV=tie`, `JD_DEV=print`); never set in trials.

Third-party assets: none.
