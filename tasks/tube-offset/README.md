# Bend a 45 degree offset in 1/8 in copper tube on a Swagelok hand tube bender clamped in a vise

An offset is two opposed bends a set distance apart, so a tube steps sideways and carries on parallel. Swagelok's hand
tube bender manual (MS-13-43) gives the layout and the steps: the bend marks lie L = O x 1.414 apart for a 45 degree
offset (p.11), here O = 5.0 cm and L = 7.07 cm; the tube is latched in the bender die with its bend mark at the roll
support's degree mark (pp.13-14); the short handle is pulled round until the roll support's 0 passes the degree mark,
allowing for springback, then swung up and away (p.15, p.19); the reverse bend is made with the tube turned so its
reference mark lies on the other side of the tube latch (p.18). Here the tube is not taken out between the bends: the
latch is eased, the tube slid along the groove until mark B is at the 45 degree mark, the latch let back down onto it
("just enough to hold the tube in place", step 4), the tube turned half a turn about its straight part by the bent leg,
the latch pushed firmly (step 8), and bend B made with the same handle swing, which is now the opposite way in the
tube's own frame.

Instruction: "Use the hand bender to bend a 45 degree offset at the tube's two marks, turn the roll support back to 0, and let go."

Install once, from this folder: `uv pip install -e . --no-deps` (or `pip install -e . --no-deps`)

Run a trial:

    inspect-robots run --embodiment yam_tube_offset -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted -P hold_s=10 --no-prompt \
      --instruction "Use the hand bender to bend a 45 degree offset at the tube's two marks, turn the roll support back to 0, and let go." \
      --scorer task_score --max-steps 5000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1. The shared arena is
https://github.com/kairosiann/yam-isaac-lab-arena.

- **Runtime per trial:** about 25 minutes on one RTX 4070 (the demo ends at 3,891 steps, the limit is 5,000).
- **Seeds:** none needed. The start is fixed and `object_jitter=0`, so runs are deterministic; this says nothing about robustness.
- **At rest** is judged from how far each tube link moves between control steps (0.05 s), not from the solver's
  instantaneous velocities, which chatter at about 2 cm/s in the clamped tube while nothing moves.
- **Each bend must sit at its mark:** its vertex point (the point of the tube's centre line nearest where the bend's two
  straight lengths' centre lines meet, the measuring point of the maker's manual, MS-13-43) within 3.5 mm of the mark's
  centre along the tube, or it is not a correct bend (no credit for it at any rung). 3.5 mm = half the 2 mm mark's width
  + 1 mm reading a ruler + the manual's 1.6 mm gain for 1/8 in tube at 45 degrees; it keeps the mark on the 11.2 mm arc.
  The simulation reads the same point from the tube's hinge angles (`bend_place` in `task.py`).
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`. `python task.py` checks the demo's length and IK.
- **The scripted demo** is open loop. It knows the fixed start, the bender's geometry and where an ideal bend leaves the
  tube, all visible on the bench; it is not told the hinges' state or the scorer's. A few of its constants were measured
  in earlier runs of the same fixed start and are marked so in `task.py`: how far the eased tube rolls under the bent
  leg's weight (9.7 deg), how far the tube lags the hands in the slide (re-measured for this revision: with a 4 mm allowance the tube slid 74.1 mm
  and bend B sat 4.8 mm past mark B; with -0.8 mm it slid 68.7 mm and the roll dies rode over bend B; now 1.5 mm:
  bend A 1.4 mm and bend B 2.4 mm from their marks), how far the roll lags the hand.
- **Wrong-way variants** (debugging only): `TUBE_DEMO=none` (bend A only, hands off at once), `TUBE_DEMO=one`
  (bend A, then the support back to 0), `TUBE_DEMO=same` (slid to mark B but not turned: the second bend goes the same
  way, a U), `TUBE_SLIDE_EXTRA=0.004` (the tube slid about 3.4 mm too far: both bends of the right shape and
  the offset right, but bend B about 5 mm past mark B).
- **Third-party assets:** none. Every part is built from boxes, cylinders and capsules in the task's own code (`tube_offset/mech.py`).

## The real trial

- **Bender:** Swagelok MS-HTB-2 hand tube bender (1/8 in OD tube, 9/16 in = 14.3 mm bend radius), clamped by its vise
  clamp block (MS-13-43 p.8) in a 3 in bench or drill-press vise, the bend plane horizontal, die axis at the place the
  setup gives.
- **Tube:** 1/8 in OD x 0.030 in wall seamless soft-annealed copper, ASTM B280, UNS C12200 (for example Streamline/Mueller
  D 02050, a 50 ft coil, or Midland 972180), cut to 30 cm and straightened, 16 g.
- **Tools to score:** a 9/16 in radius gauge, a protractor, a tape, a ruler, two straightedges, a fine marker; everything is read on the tube where it sits latched in the bender, then it is taken out by hand.
- **Not published by Swagelok, measure on the bought unit before the first trial:** the handle lengths and the grip's
  radius from the die axis (the simulation uses 21 cm), the bender's mass, the latch's closing force. The simulated frame
  sizes are estimates from the manual's photographs.

## The force, and why 1/8 in

The tube resists bending with its plastic moment M_p = Z_p x flow stress. For 1/8 in x 0.030 in tube, Z_p = (D^3 - d^3)/6
= 4.6 mm^3 (D 3.175 mm, d 1.651 mm). Wrapped on the 14.3 mm radius, the outer fibre strains (D/2)/R = 0.111; at the
section's average strain of about 0.083, annealed copper flows at about 320 MPa x 0.083^0.54 = 84 MPa (textbook
constants K = 320 MPa, n = 0.54, not measured). So M_p = 0.38 N m, and at the 21 cm grip the handle needs about
0.38 / 0.21 = 1.8 N, plus pivot and roll-die friction. The arms push about 3-5 N (measured: the S0 handle-sweep runs reached 46.6-50.7 degrees at 2.1-8.6 N). The
earlier 3/8 in x 0.035 in steel tube needs about 23 N m, 90-130 N at the grip, which the arms can't give, so the task
now names the bender's 1/8 in size and soft copper.

## What is simulated, and what is simplified

Real contact does the bending. The roll dies are rigid rollers on the short handle; turning the handle carries them
round the die and they push the tube into the die's groove. Each tube hinge is a spring with the tube's own elastic
stiffness (E I over the 2.5 mm hinge spacing: 117 GPa x 4.6e-12 m^4 / 2.5 mm = 216 N m/rad) whose anchor slips once its
torque reaches M_p (a plastic hinge). So the tube holds any shape it is left in, yields only under the plastic moment,
and springs back by its elastic part, under a degree for a 45 degree bend (the maker allows 1 to 3 degrees for its
tubing). Nothing in the code bends the tube or moves the bender.

Stand-ins, each declared in `task.json`'s fidelity or here:
- **Tube:** rigid round links 3.18 mm across (real OD), 16 g in all (real), with hinges only within 2.5 cm of each mark,
  2.5 mm apart, turning only in the bend plane. No wall thinning, ovality or twist. Solver armature 1e-4 kg m^2 on each
  hinge (numerical only).
- **Bender frame:** boxes and cylinders. Bend radius, tube size and the parts list are the maker's; lengths, the clevis
  pin's place, the roll dies' size (10 mm; the real dies are grooved, the simulated ones are plain 10 x 8 mm cylinders, each with a 2 mm disc above reaching 2.4 mm over the tube, 1.2 mm clear, for the groove's upper lip) and the latch's shape are estimates from MS-13-43's photographs.
  The bender's link (which keeps the roll dies on the tube) is a detent at the clevis pin holding them down up to
  0.08 N m; swung up past 12 degrees the short handle stays where it is left (0.05 N m friction). The roll support turns
  against 0.20 N m of friction (estimate), so pulling the handle back swings it up before the support turns. The tube
  latch (shape estimated; the parts list names a latch, its screw and a spring washer): a keeper on a pin with a lip
  over the tube and a post to push. Off its seat it stays where it is put (0.03 N m friction, estimate); resting on the
  tube its spring holds it down with up to 0.05 N m; pressed down onto the tube for 1 s ("push the tube latch firmly")
  it clamps with up to 0.12 N m (4 N at the lip) until lifted. All three values are estimates.
- **Layout:** the bend plane is horizontal and both handles lie 3.5 cm below the tube's plane, so a hand can pinch
  the tube without touching them. The vise is drawn only; the bender is fixed to the table where its clamp block is.
- **The roll support's limits:** -2 to 100 degrees; the clevis 0 to -45 degrees.
