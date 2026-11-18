# Turn a worm gear reducer over in a rollover fixture, lock it, and fit and screw down its output flange

A Motovario NMRV 030 worm gear reducer (1.2 kg) lies in a rollover fixture with its output face J down. The robot secures it with the sliding bar, turns the fixture over by its crank and pushes the lock bolt home, sets the output flange on face J's 55 mm spigot (its four pre-fitted screws drop into the tapped holes only when it is turned to line up), and runs the four screws down with a stubby screwdriver. A 10 N lift applied by the task file then checks that the screws hold the flange.

Instruction: "Secure the gearbox in the rollover fixture, turn it over and lock it, then fit the output flange on its spigot and screw it down."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_trunnion -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted -P hold_s=10 --no-prompt \
      --instruction "Secure the gearbox in the rollover fixture, turn it over and lock it, then fit the output flange on its spigot and screw it down." \
      --scorer task_score --max-steps 9600 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1. `-P hold_s=10` keeps the scripted policy running after its plan so the lift test (after the hands are clear and all is still for 1 s) can finish.

- **Runtime per trial:** the scripted demo plays about 345 s of simulated time (four screws, 9 turns each at the real M6 pitch, 18 wrist strokes a screw) and reaches full score at about 347 s of the 9,600-step (480 s) limit. Wall clock: about 20 minutes on one RTX 4070 (the simulation runs at about a third of real time; estimate from the earlier 3,600-step runs); 90 to 115 minutes on a shared cluster node (L40S, four runs at once).
- **Seeds:** none. The start is fixed and `object_jitter=0`; runs are deterministic, which says nothing about robustness.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`; `python task.py` checks that the demo fits the run limit and that every waypoint is reachable.
- **Flange start:** it sits on its post turned 35 degrees anticlockwise from the angle at which its screws line up; the demo turns it back by the wrist on the way. Set down unturned, its screw tips rest on face J and it stays 1 mm proud (TRUNNION_DEMO=noalign).
- **The scripted demo** is open loop: it is told nothing while it runs. Its aim uses the fixed layout plus four constants taken from earlier dev runs of this deterministic scene (where the bolted cradle comes to rest, 183.2 degrees; where the turned gearbox lies in its play; where the driver's tip sits in the fingers, 1.7 mm off the hand's axis; and the crank aim). It keeps hold of the driver throughout: each stroke turns the wrist 210 degrees clockwise, lifts the bit 4 mm clear, turns back in the air and sets the bit in again; 18 strokes per screw (10.5 turns: 9 to the seat, then on against the snug resistance). Each stroke's aim comes from the same fixed layout and constants (the screw's axis and, from the pitch, how far its head has gone down); nothing is read during the run. The fingers squeeze the handle 10 mm a side (the gripper's full 50 N a finger) so the grip carries the thread's resistance. `TRUNNION_DEMO` (`noturn`, `nobolt`, `noscrews`, `noalign`, `halfscrew`: screw 3 gets 9 strokes, about half run) selects wrong-way runs, and `TRUNNION_DEBUG=1` logs state every 0.5 s.
- **Third-party assets:** none. The screwdriver's drawn shape (`trunnion_fixture/driver_shape.py`, also written to `assets/wera_367_tx30_stubby.obj`) is our own mesh of the Wera 367 TORX Stubby TX 30 x 25 mm (article 05008861001), made from the maker's published sizes (blade 25 mm, handle 54 mm; 81 mm overall, handle 34 mm at its widest and 37 g as the maker's data is listed by resellers) and product photos; Wera publishes no CAD for it. Every other part is built from boxes and convex shapes in the task's own code, sized from the maker's catalogue (Motovario VSF technical catalogue, Standard IEC, rev0 2017: NMRV 030 dimensions p. 50, output flange FA p. 71, mass p. 68) and the screwdriver maker's size table (Wera 367 TORX Stubby TX 30).

## What the simulation decides by rule (declared)

- **Screw drive and thread.** A screw turns only while the driver's bit sits in its socket: tip within 1.5 mm of the screw's axis and of the socket's floor, driver within 8 degrees of square. The driver's turn about the screw's axis, measured against the gearbox, is the screw's turn; clockwise from the head tightens. The screws come pre-fitted in the flange's holes (a kitting step: the catalogue flange comes with loose screws; a fixed joint to the flange holds each), no thread engaged, tips 1 mm proud and heads 9 mm above the flange. Once the flange is seated each tip rests 1 mm into its tapped hole's mouth (on the real part: the hole's countersink and the screw's chamfered point) with no thread engaged. A screw whose tip is in a hole's mouth (0.5 mm or more below face J, within 2 mm of the hole's axis along each side of the drawn 8 mm square hole: anywhere the hole lets the screw's chamfered point, 4 mm across where it meets face J, sit; the real countersunk mouth guides the point to the axis as it turns) catches its thread on its first tightening turn: the fixed joint lets go and a thread joint to the gearbox (on the hole's axis, along it to a target) takes over, running 1 mm per turn (the real M6 x 1 pitch) to 0.5 mm past the seat, so a seated head clamps the flange through real contact. Each screw is run 9 turns to its seat, where it has 10 mm of thread in the 11 mm deep tapped hole, as the real M6 x 16 screw through the 6 mm flange. A tightening turn made while the bolt is not home marks that screw as not credited. (`screw_step` in task.py; switches applied in trunnion_fixture/embodiment.py.)
- **Thread resistance and snug.** While the bit is in a caught screw's socket, the task file puts a torque on the driver about the screw's axis against its turn: 0.15 N m while the screw runs down (thread drag) and 0.5 N m once its head is down (within 0.5 mm), saturating above 0.3 rad/s. Both are this task's own estimates, not maker values. A screw counts as down only once it has been turned on 15 degrees or more against the 0.5 N m. The robot's grip must carry this: our arm measurements put the wrist's twist through a full grip at about 0.8 N m on a 20 mm handle and 1.1 N m on a 40 mm one (this driver's grip zone is 29 mm). Real trial: a 10 N spring scale 50 mm out on a TX 30 L-key (0.5 N m) must not turn a screw further. (`driver_torque` and `screw_step` in task.py; applied in trunnion_fixture/embodiment.py.)
- **Lift test.** Once all four screws are down, the hands are clear and all is at rest for 1 s, the task file pushes the flange up (world +z) at 10 N, ramped over 0.5 s and held 1 s; it passes if the flange's face lifts 0.5 mm or less off face J. Real trial: a 10 N spring scale hooked under the flange's neck, and a 0.1 mm feeler.
- **Real-world torque.** These arms run the screws down snug only (0.5 N m, about a twentieth of the final torque). The final torque for an M6 property class 8.8 socket head screw is about 10.1 N m (standard tables, friction 0.12), less into an aluminium housing by common practice; we found no value from the reducer maker for the flange screws (catalogue and manual searched). The final tightening, with a torque wrench, is a later station.
- **Start check.** Just before the first step the embodiment puts the free parts back at their start poses at rest and logs the largest offset ("TRN start check").

## Drawn shapes and collision shapes

The gearbox and flange are drawn as their real parts (gearbox: rounded die-cast body with corner bosses, ribbed worm housing, square input flange with motor bore and bolt holes, hollow output bore with keyway, tapped holes, breather plug; flange: square plate with rounded corners, neck, spigot, centre bore, screw holes) from the catalogue drawing and photos. They collide as the blocks and convex pieces listed in `mech.py` (`gearbox_boxes`, `flange_pieces`), which are hidden. The screwdriver is drawn from `driver_shape.py` and collides as its blade and a 29 mm cylinder. Trunnion bosses and round axle stubs are drawn only.

## Sizes that are ours (not catalogue values)

The gearbox's shape between its catalogue dimensions (blocks and a round worm housing), its centre of mass (6 mm toward its top, 4 mm toward the input), the corner bosses that carry the tapped holes; the flange's recess (16-sided, 0.5 to 1 mm loose a side, mouth chamfered 2 mm wide and 1 mm deep; the spigot's edge chamfered 1.5 mm wide and 1 mm tall: together a 4 mm lead-in, and below the chamfers their straight walls overlap 0.5 mm with 0.5 mm of play, so the spigot centres the flange), its neck and its fixing to the housing (four M6 x 16 ISO 14579 socket head screws into the housing's KE tapped holes, pre-fitted 1 mm proud); the flange's mass (0.13 kg); the screwdriver's collision shape (its blade and a 29 mm cylinder over the handle, where the fingers hold it; the drawn handle bulges to the real 34 mm); the fixture, holster and kitting post. The screw's hexalobular socket is drawn round, 0.6 mm loose a side on the 5.6 mm bit. The tapped holes are drawn as plain 8 mm holes 11 deep (the thread is the joint) and the flange's clearance holes as 8.4 mm (real M6 clearance 6.6 mm); the worm housing is drawn within 18 mm of the gearbox's mid-plane so those holes stay open.

## Fixture dimensions

Drawing: `fixture_dimensions.svg` (plan and side view, mm), generated with this table from the scene code by `python dev/fixture_diagram.py`.

Every part is a box with surface friction 0.3 (the flange 0.3, screws 0.2, driver 0.5). Positions are ranges in mm: x from the arm bases' midpoint (the axle line is x 240, 78 above the table), y from the table's centre line (+y toward the left arm), z above the table top. The cradle is listed level, as at the start; it turns about the axle. The materials are the stated ones; the simulation sets only colour, friction and mass.

| Part | Material | Size x * y * z (mm) | Position x / y / z (mm) | Moves or fixed |
|---|---|---|---|---|
| near base plate | dark steel | 90 * 24 * 6 | 195 to 285 / -156 to -132 / 0 to 6 | fixed to the table |
| far base plate | dark steel | 90 * 24 * 6 | 195 to 285 / 32 to 56 / 0 to 6 | fixed to the table |
| tie bar | dark steel | 10 * 212 * 6 | 320 to 330 / -156 to 56 / 0 to 6 | fixed to the table |
| tie bar | dark steel | 10 * 212 * 6 | 150 to 160 / -156 to 56 / 0 to 6 | fixed to the table |
| near pillar | steel | 60 * 12 * 84 | 210 to 270 / -150 to -138 / 6 to 90 | fixed to the table |
| far pillar, lower | steel | 60 * 12 * 96 | 210 to 270 / 38 to 50 / 6 to 102 | fixed to the table |
| far pillar, upper | steel | 60 * 12 * 11 | 210 to 270 / 38 to 50 / 114 to 125 | fixed to the table |
| rail | aluminium | 6 * 150 * 77.5 | 290 to 296 / -130 to 20 / 44 to 121.5 | turns with the cradle, 0 to 195 deg |
| rail | aluminium | 6 * 150 * 77.5 | 184 to 190 / -130 to 20 / 44 to 121.5 | turns with the cradle, 0 to 195 deg |
| ledge (foot side) | aluminium | 3.5 * 65 * 5 | 190 to 193.5 / -70 to -5 / 44 to 49 | turns with the cradle, 0 to 195 deg |
| ledge (worm side) | aluminium | 20.5 * 65 * 5 | 269.5 to 290 / -70 to -5 / 44 to 49 | turns with the cradle, 0 to 195 deg |
| near stop plate | aluminium | 100 * 7.5 * 67 | 190 to 290 / -87.5 to -80 / 44 to 111 | turns with the cradle, 0 to 195 deg |
| near end plate | aluminium | 112 * 6 * 77.5 | 184 to 296 / -136 to -130 / 44 to 121.5 | turns with the cradle, 0 to 195 deg |
| far end plate, above notch | aluminium | 100 * 10 * 60 | 190 to 290 / 20 to 30 / 63 to 123 | turns with the cradle, 0 to 195 deg |
| far end plate, beside notch | aluminium | 42 * 10 * 30 | 248 to 290 / 20 to 30 / 33 to 63 | turns with the cradle, 0 to 195 deg |
| far end plate, beside notch | aluminium | 42 * 10 * 30 | 190 to 232 / 20 to 30 / 33 to 63 | turns with the cradle, 0 to 195 deg |
| far axle stub | dark steel | 16 * 8 * 16 | 232 to 248 / 30 to 38 / 70 to 86 | turns with the cradle, 0 to 195 deg |
| near axle stub | dark steel | 16 * 2 * 16 | 232 to 248 / -138 to -136 / 70 to 86 | turns with the cradle, 0 to 195 deg |
| crank | dark steel | 76 * 6 * 16 | 172 to 248 / -158 to -152 / 70 to 86 | turns with the cradle, 0 to 195 deg |
| crank pin | dark steel | 8 * 12 * 8 | 176 to 184 / -170 to -158 / 74 to 82 | turns with the cradle, 0 to 195 deg |
| knob sleeve | plastic | 30 * 24 * 40 | 165 to 195 / -194 to -170 / 58 to 98 | turns freely on the pin |
| retaining bar | aluminium | 99 * 30 * 8 | 190.5 to 289.5 / -120 to -90 / 113.5 to 121.5 | slides 67.5 along +y |
| bar lug | aluminium | 40 * 8 * 39.5 | 220 to 260 / -128 to -120 / 113.5 to 153 | slides 67.5 along +y |
| lock bolt | aluminium | 12 * 56 * 12 | 234 to 246 / 36 to 92 / 102 to 114 | slides 15 along -y |
| bolt lug | aluminium | 40 * 8 * 48 | 220 to 260 / 84 to 92 / 102 to 150 | slides 15 along -y |
| driver holster | steel | 40 * 40 * 2 | 315 to 355 / -220 to -180 / 0 to 2 | fixed to the table |
| driver holster | steel | 14 * 40 * 40 | 315 to 329 / -220 to -180 / 0 to 40 | fixed to the table |
| driver holster | steel | 14 * 40 * 40 | 341 to 355 / -220 to -180 / 0 to 40 | fixed to the table |
| driver holster | steel | 12 * 14 * 40 | 329 to 341 / -220 to -206 / 0 to 40 | fixed to the table |
| driver holster | steel | 12 * 14 * 40 | 329 to 341 / -194 to -180 / 0 to 40 | fixed to the table |
| flange kitting post (round) | steel | 50 * 50 * 60 | 125 to 175 / 125 to 175 / 0 to 60 | fixed to the table |

Simulation joints and their real-fixture equivalents:

- Axle: a hinge joint between the stubs and the pillars. Real fixture: the stubs are 16 mm round shafts in bores in the pillars; the near one carries the crank.
- Roll stops at 0 and 195 degrees: the hinge's limits. Real fixture: a 4 mm pin in the near pillar runs in a 195 degree arc slot in the near end plate.
- Retaining bar: a sliding joint, 67.5 mm of travel, 0.5 mm clear of each rail. The gearbox has 1.5 mm of play across the cradle and 2.5 mm along it. Real fixture: two 3 mm pins in each end of the bar run in a 3.2 mm slot in each rail's inner face, which guides it, keeps it in when turned over and sets its travel.
- Lock bolt: a sliding joint, 15 mm of travel, in the far pillar's 12 mm gap. Real fixture: two blocks make the gap a square guide hole, and a cross pin in a 15 mm slot sets the travel. It enters the far end plate's notch by real contact, only within about 4 degrees of turned over.
- Knob: a free sleeve on the crank's pin (two turns each way). Real fixture: a plastic knob on a pin with a retaining clip.
- Moving masses: cradle 0.7 kg with its centre 4 mm toward the arms and 20 mm toward the crank end, 8 mm below the axle, so it rests on its near stop; bar 0.11 kg; bolt 0.07 kg; knob 0.03 kg. The gearbox's own centre is 2.5 mm toward the arms of the axle when level.
- Build route (ours): every fixture part is stock steel or aluminium flat bar and plate (sizes in the table), cut and bolted; the knob, holster and kitting post can be 3D printed. Materials about USD 100 to 150 (our estimate).
- Securing to the table: fixed to the world in the simulation. Real bench: bolt both base plates down (M8 through the plates outside the pillars), or clamp them with two toe clamps.
