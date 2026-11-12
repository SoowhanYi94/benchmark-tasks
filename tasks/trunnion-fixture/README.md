# Retain a housing in a rollover fixture, turn it over and lock it, and fit a pad on its far face

A housing goes in a rollover fixture recess down, is retained by a sliding bar, is turned over on the fixture's axle and locked by a bolt, and a wear pad is fitted in the recess, now facing up. Instruction: "Lay the housing in the rollover fixture with its orange recess down, slide the blue bar across it, turn the fixture over, push the red bolt home, fit the white wear pad in the recess, and move both hands away."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_trunnion -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Lay the housing in the rollover fixture with its orange recess down, slide the blue bar across it, turn the fixture over, push the red bolt home, fit the white wear pad in the recess, and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 5 minutes on one GPU for a full 3,000-step run; the scripted demo plays about 69 s of simulated time and then stops.
- **Seeds:** none. The start is fixed and `object_jitter=0`. There are no lots or variants.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is open loop. Everything it uses is the fixed layout in plain sight; it is told nothing a policy couldn't see. `TRUNNION_DEMO` (nobar, nobolt, empty, padfirst, nopad, arc) selects wrong-way runs for debugging, and `TRUNNION_DEBUG=1` logs state.
- **Third-party assets:** none. Every part is built from boxes in the task's own code.

## Fixture dimensions

Drawing: `fixture_dimensions.svg` (plan and side view, mm), generated from the scene constants by `python dev/fixture_diagram.py`.

The fixture is built from boxes, all with a surface friction coefficient of 0.3. Positions are ranges in mm: x from the arm bases' midpoint (the axle line is x 290), y from the table's centre line (+y toward the left arm), z above the table top. The cradle is listed as it stands at the start (level); it turns about the axle (x 290, z 75). The materials are the stated ones; the simulation sets only colour and friction.

| Part | Material | Size x * y * z (mm) | Position x / y / z (mm) | Moves or fixed |
|---|---|---|---|---|
| Base | steel | 140 * 235 * 6 | 220 to 360 / -135 to 100 / 0 to 6 | Fixed to the table (fixed joint to the world) |
| Near pillar | steel | 60 * 12 * 81 | 260 to 320 / -120 to -108 / 6 to 87 | Fixed to the base |
| Far pillar, lower | steel | 60 * 12 * 88 | 260 to 320 / 72 to 84 / 6 to 94 | Fixed to the base |
| Far pillar, upper | steel | 60 * 12 * 9 | 260 to 320 / 72 to 84 / 106 to 115 | Fixed to the base; the 12 mm gap between lower and upper holds the bolt |
| Axle stubs (2) | steel | 16 * 4 * 16 (far); 16 * 2 * 16 (near) | x 282 to 298 / 68 to 72 (far) and -108 to -106 (near) / 67 to 83 | Part of the cradle; square, no round shaft; the cradle turns about this line, 0 to 195 degrees |
| Cradle rails (2) | aluminium | 8 * 156 * 51 each | x 236 to 244 and 336 to 344 / -98 to 58 / 44 to 95 | Move with the cradle; inner faces 92 mm apart |
| Ledges (2) | aluminium | 16 * 82 * 5 each | x 244 to 260 and 320 to 336 / -41 to 41 / 44 to 49 | Move with the cradle; 60 mm opening between them |
| End beams (2) | aluminium | 92 * 6 * 5 each | x 244 to 336 / 41 to 47 and -47 to -41 / 44 to 49 | Move with the cradle; close the opening at each end |
| Corner stubs (4) | dark steel | 24 * 6 * 8 each | x 244 to 268 and 312 to 336 / 41 to 47 and -47 to -41 / 49 to 57 | Move with the cradle; stop the housing along y, 82 mm apart |
| Near end plate | aluminium | 108 * 8 * 51 | x 236 to 344 / -106 to -98 / 44 to 95 | Move with the cradle |
| Far end plate | aluminium | 80 * 10 * 90 | x 250 to 330 / 58 to 68 / 30 to 120 | Move with the cradle; notch 16 mm wide (x 282 to 298), 30 mm tall (z 30 to 60), open at the bottom edge |
| Crank | dark steel | 56 * 6 * 16 | x 242 to 298 / -128 to -122 / 67 to 83 | Part of the cradle; 40 mm from the axle to the knob's pin |
| Crank pin | dark steel | 8 * 6 * 8 | x 246 to 254 / -134 to -128 / 71 to 79 | Part of the cradle |
| Knob sleeve | plastic | 30 * 24 * 40 | x 235 to 265 / -158 to -134 / 55 to 95 | Turns freely on the pin (two turns each way); the pin rides on the crank |
| Retaining bar plate | aluminium | 91 * 30 * 8 | x 244.5 to 335.5 / -90 to -60 at start / 84 to 92 | Slides along y, 75 mm of travel (to y -15 to 15), joint limit only |
| Retaining bar lug | aluminium | 40 * 8 * 41 | x 270 to 310 / -98 to -90 at start / 84 to 125 | Part of the bar |
| Lock bolt rod | aluminium | 12 * 56 * 12 | x 284 to 296 / 74 to 130 at start / 94 to 106 | Slides along -y, 15 mm of travel, joint limit only |
| Lock bolt lug | aluminium | 40 * 8 * 51 | x 270 to 310 / 122 to 130 at start / 94 to 145 | Part of the bolt |

Simulation joints and their real-fixture equivalents:

- Axle bearings: a hinge joint (the stubs are drawn square). Real fixture: the near stub is a round 16 mm shaft through a bore in the near pillar (close running fit) carrying the crank, clamped with a set screw; the far stub turns in a bore in the far pillar. The knob's pin is 30 mm long, passes through a round bore in the knob and is held by a retaining clip (the simulation draws a square sleeve touching the pin's end).
- Roll stops at 0 and 195 degrees: the hinge joint's limits. Real fixture: a 4 mm pin fixed in the near pillar's inner face, 16 mm below the axle, runs in a 5 mm wide, 195 degree arc slot cut in the near end plate about the axle; the slot's ends are the stops.
- Retaining bar: a sliding joint, 75 mm of travel; the bar fits between the rails' inner faces with 0.5 mm clearance each side. Real fixture: two 3 mm pins in each end of the bar, 20 mm apart, run in a 3.2 mm wide slot in each rail's inner face (z 88, y -86.5 to 11.5, so the pin centres travel y -85 to 10), which guides the bar, keeps it in when the cradle is turned over and sets its travel.
- Lock bolt: a sliding joint, 15 mm of travel, in the far pillar's 12 mm gap. Real fixture: two blocks in the gap, 12 mm apart, make it a square guide hole, and a vertical cross pin through the pillar's upper block runs in a 15 mm slot in the top of the bolt and sets the travel.
- Moving masses: cradle 0.6 kg with its centre 4 mm toward the arms (-x), 20 mm toward the crank end (-y) and 8 mm below the axle, so it rests on its near stop; bar 95 g; bolt 66 g; knob 30 g.
