# Weld an L-shaped seam with a TIG torch and filler rod

Two 6 mm mild-steel plates are clamped on a fixture table with an L-shaped butt joint between them, 200 mm from a white mark to a yellow one. A TIG torch lies on its rest and a filler rod on its own. The robot has to weld the joint with one hand carrying the torch at arc length and the other dabbing filler into the front of the pool, never letting the rod or the plate touch the tungsten, then put both tools back on their rests and move both hands away. Instruction: "Weld the joint between the white and yellow marks with the TIG torch, dabbing the yellow filler rod into the front of the pool to lay one sound bead, then put both tools back on their rests and move both hands away."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_tig -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Weld the joint between the white and yellow marks with the TIG torch, dabbing the yellow filler rod into the front of the pool to lay one sound bead, then put both tools back on their rests and move both hands away." \
      --scorer task_score --max-steps 3000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** 6-9 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`; `--seed 0` was used for the demo.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`. `python task.py` checks the demo's reach and length (1,830 steps). `python dev/replay.py` plays the demo's plan with perfect tracking through the scorer.
- **The scripted demo** is open loop. It uses only the fixed starting layout (where the tools, rests and joint are), which a policy can see. The left arm takes the torch, strikes the arc 2 mm behind the start mark and lets the pool form, then travels at 3 mm/s, turns the torch about its tip at the corner and finishes at the end mark. The right arm takes the rod and dabs it every 1.25 s, coming in and out of the front of the pool on a 45 degree line clear of the cup. Both tools then go back on their rests and both arms move away. The plan aims against the arms' measured give under load (the tungsten sags 5-6 mm and the rod 1-3 mm; both have a few mm of sideways offset), measured from logged runs.
- **What the physics does:** the tools, plates and hands are rigid bodies with real contact; the arc, heat and filler are scored geometrically (see `task.json`'s fidelity and the top of `task.py`). The bead, pool, arc glow and soot on a contaminated cup are shown from the score's record. `TIG_DEMO` (`torch`, `laid`, `fast`, `dip`), `TIG_LIFT`, `TIG_ROD_LIFT` and `TIG_DEBUG=1` are debugging switches only (see the code). The `torch` and `laid` variants (the torch run alone, or with the rod laid in the joint and let go) score 0.
- **Third-party assets:** none. Every part is built from boxes, cylinders and simple meshes in the task's own code.
