# Bend a 45 degree offset in 3/8 in tube on a hand tube bender

An offset is two opposed bends a set distance apart, so a tube steps sideways and carries on parallel. Swagelok's hand tube bender manual (MS-13-43, p.11) gives the layout: the bend marks lie L = O x 1.414 apart for a 45 degree offset; here O = 9.9 cm, L = 14.0 cm (the two yellow bands). The tube is fed along the bed until a band is at the bending point, bent with the handle allowing for springback (a set 3 degrees; the manual says to allow 1 to 3, p.19), turned round and fed in again for the second bend, then laid on the template. The result to measure is the offset and the parallel ends, not one angle. Instruction: "Bend the tube into a 45 degree offset with parallel ends at its two yellow bands, lay it on the green template, and move both hands away."

Install once, from this folder: `uv pip install -e . --no-deps` (or `pip install -e . --no-deps`)

Run a trial:

    inspect-robots run --embodiment yam_tube_offset -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Bend the tube into a 45 degree offset with parallel ends at its two yellow bands, lay it on the green template, and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 5 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`. There are no lots: every trial starts the same.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is open loop. It knows the fixed start and where a bend leaves the tube, all of which can be seen on the bed. It isn't told the hinge limits or the scorer's state.
- **Sizes:** same as the Swagelok 3/8 in hand tube bender only in bend radius (2.4 cm drawn; 15/16 in = 23.8 mm) and in the offset formula. The tube is drawn and modelled 2 cm square (real 3/8 in = 9.5 mm round); the handle (15 cm), roll support, tube latch and link of the real tool are not modelled. The tube is three rigid parts on two one-way hinges, not a bending tube (see `fidelity` in `task.json`).
- **Third-party assets:** none. Every part is built from boxes and cylinders in the task's own code.
