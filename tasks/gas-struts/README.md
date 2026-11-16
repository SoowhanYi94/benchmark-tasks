# Replace a failed gas strut on a hinged lid

A small steel machine enclosure faces the robot with its hinge side. Its aluminium lid (0.63 kg) is shut on a piano
hinge with an open stop at 45 degrees and is carried by two gas struts of the ACE GS-10-30 class (10 N, 30 mm stroke,
92 mm extended, 10 mm body, 3 mm rod), rod end down on 8 mm ball studs. The left strut has failed (no gas force) and
carries a red defect tag; a new strut lies across two wooden battens in front of the enclosure. Each strut end is a
steel ball socket held on its ball by a spring clip with a finger loop.

The robot unclips both sockets of the tagged strut and takes it off, fits the new strut (each socket pressed onto its
stud until it snaps; rod end on the lower stud), checks that the lid holds itself open, and closes the lid by pushing
it down past its balance point. One good strut cannot hold the lid, so one hand holds it open while the other fits
the new strut.

Instruction: "Replace the tagged gas strut with the new one, check that the lid holds itself open, then close the lid."

## Run a trial

On a machine with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s `setup.sh`), from
this folder. The standard `yam_arena` embodiment runs it: importing `task.py` installs the enclosure and the struts.

```bash
I="Replace the tagged gas strut with the new one, check that the lid holds itself open, then close the lid."
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --max-steps 3600 --seed 0 --store-frames --log-dir logs/dream

# The scripted demo (hold_s=10 keeps the run going after the plan so the inspection can finish)
inspect-robots run --embodiment yam_arena --policy yam_scripted -P hold_s=10 \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --max-steps 3600 --seed 0 --store-frames --log-dir logs/demo

# Wrong-way variants of the demo: GS_DEMO=wrongside (the good strut taken off instead), noclose (the lid never
# closed), none (nothing replaced; the lid lifted, held and let go)
GS_DEMO=noclose inspect-robots run ...   # same arguments as the demo

# Tests (no simulator), the demo's IK on the CPU, and the no-robot mechanism check (GPU)
python -m pytest test_score.py
python dev/plan_check2.py
python dev/spike2.py probe probehalf release twopull close failed rodup raise start
```

Results (local GPU, seed 0, `object_jitter=0`): scripted demo 1.0 at 1280x704 (and at 320x176): the tagged strut off both studs and on the table by 15.5 s, the new strut on at both ends by 35 s, the lid holding itself open (held) at 55 s, closed at 69.5 s, inspection passed at 75.5 s (step about 1,510 of 3,600; pulls moved 0.29 and 0.56 mm, the lid lifted to 44.2 degrees stayed at 43.9). Wrong ways: GS_DEMO=wrongside 0 (good strut taken off, tagged one still on), GS_DEMO=noclose 0.6, GS_DEMO=none 0. Dream-1 (3,600 steps): 0.0; its arms reach over the lid and handle and never lift the lid or unclip a socket.

Runtime per trial: 3,600 control steps at 20 Hz (3 minutes of simulated time). Seed 0, `object_jitter=0`: one fixed
layout; no seeds are needed (and a fixed layout says nothing about robustness). `GS_DEBUG=20` prints the lid angle, the
sockets, the events and the inspection phase every second.

## Files

- `task.py`: the layout, the hook that reads the state each step and applies the inspection, the scorer, the
  instruction.
- `gsparts.py`: the enclosure with its lid and studs, the struts with their sockets, clips and loops, the battens.
- `gskit.py`: the socket checks, the latched events (held, closed, inspected), the inspection sequence and the score
  ladder (pure Python).
- `strutkit.py` (written for this task): articulated assemblies with any joint tree (torsional contact patch per
  shape, centre of mass per link, initial joint positions).
- `demo_plan.py`: the scripted demo and its variants. `test_score.py`: the scorer's tests.
- `dev/spike2.py`: the mechanism without robots. `dev/plan_check2.py`: the demo's IK and clearances on the CPU.

Assets and licences: none (every shape is built in code).

## What is physics and what is a task-file rule

Physics (PhysX, real contact and joints): the lid's hinge, its open stop and its weight; each strut as a sprung, damped
slider (good: 10 N fully extended rising to 11.5 N at full stroke, 60 N s/m; failed: no force, 5 N s/m) with its 30 mm
stroke; each ball socket on its ball (a bore of flats round the ball and two sprung jaws at its mouth that ride over the
ball as it is pushed in and close behind it; under a pull their 50-degree guides drive them shut); the clip's finger
loops, which a fingertip pushes outboard to drive the jaws open so the socket slides off; the grippers on the handle,
the loops and the strut body; the lid falling shut when let go too early or pushed below its balance point.

Real part: ACE GS-10 catalogue (Industrial Gas Springs, push type, GS-10 page): body 10 mm, piston rod 3 mm,
GS-10-30 stroke 30 mm and extended length 92 mm, "Extension force: 10 N to 100 N (compressed up to 116 N)",
"Progression: Approx. 13 % to 16 %", end fitting "Ball Socket G3.5" (13 mm housing, 8 mm ball), "We recommend
mounting with piston rod downwards". We take 92 mm between ball centres; 66 g per strut is our value (no maker weight).

Clip release: a gas prop replacement sheet (Industrial Hardware Direct, "Gas Prop Replacement Instructions", one
page) describes the common socket with a flat metal clip: "insert the screw driver underneath the metal clip and
slightly pry upwards to dislodge the clip. DO NOT FULLY REMOVE THE CLIP", and to fit: "press lightly until it snaps
into a secure position". Here the clip is modelled as the two sprung jaws with two flat finger loops (16 x 6 x 2.5
mm, drawn larger than a real clip's tab so a gripper can reach them); a fingertip pushing the loops outboard takes the
place of the screwdriver. No-robot checks (`dev/spike2.py`): the socket snaps on at about 1.4 N; it holds a 40 N pull
until released; about 3 N on the loops opens it and it then comes off at about 2.6 N; a socket left on its jaws'
lead-ins comes off at 4 N. No maker publishes these forces: they are this task's own, set so one arm can work them.

Moment balance (our lid and stud geometry): at the 45 degree stop the lid's weight is 0.46 N m about the hinge, one good
strut gives 0.36 N m (the lid sinks shut: 0.4 s in the no-robot check), two give 0.71 N m (it holds at 43.9 degrees).
Below about 28 degrees the weight wins: released at 25 or 20 degrees the lid closes in 0.2 to 0.4 s (the struts' damping
does not make it a slow close, so the task does not claim one). A strut fitted rod end up does not bind: it has the same
length both ways round, so the lid closes the same; the scorer does not count its sockets (each counts only on its own
stud).

Other declared stand-ins: the round strut body has a 4 mm torsional contact patch (PhysX `torsionalPatchRadius`) for
the gripper pads' soft contact; the lid, studs, handle and battens are this task's own sizes.

Task-file rules (`gskit.py`, `task.py`): the readings (a socket is on when its ball's centre is within 1 mm of the
socket's centre and both jaws are within 0.4 mm of shut); the events held, closed and inspected; and the inspection,
applied once the lid is closed and both grippers have been clear (30 mm) and still for 1 s: a 30 N pull outward along
the stud on each socket of the new strut for 1 s (it must move under 1 mm and stay on), then a lift of at most 8 N on the
handle that raises the lid to 44 degrees and lets go (it must stay at 43 degrees or more for 2 s). These forces are this
task's own test values. A gripper coming back within 30 mm stops a check; it starts again when the conditions hold.

The demo takes the tagged strut off with the lid shut (the failed strut carries no load there, and the open lid would be
in the way of the hand at the lower socket); one good strut cannot hold the lid open, so one arm holds it while the
other fits the new strut.

## Real-world trial

Parts: a steel enclosure 400 x 200 x 80 mm with a 2.5 mm aluminium lid (0.63 kg with flanges and handle) on a piano
hinge with a 45 degree open stop; four ball studs with 8 mm balls on 4 mm necks, ball centres 35 mm out from the faces
(body studs on the side walls 288 mm in front of the arm bases' line and 45 mm up; lid studs on the flanges 37.5 mm from
the hinge, 10 mm below the lid's top); three ACE GS-10-30 gas springs with G3.5 ball sockets with spring clips: two
ordered at 10 N, one with its gas let out (the failed one) and a red tag on it; two 20 x 20 x 360 mm battens.
Score with: a protractor on the hinge (45, 43 and 3 degree marks); a 50 N spring scale hooked under each new socket,
pulled along the stud to 30 N for 1 s with a ruler beside it (under 1 mm); a hand lifting the handle to the stop and
letting go; video for the hands' clearance and the order of events. A socket is on when it cannot be lifted off its ball
without releasing its clip.
Safety: a lid let go too early, or pushed below its balance point, falls shut; keep fingers out from under it.
Reset: unclip the new strut, refit the tagged one rod end down, lay the new strut back on the battens.
