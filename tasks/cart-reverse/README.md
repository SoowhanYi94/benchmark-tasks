# Back a tow cart into its bay by the drawbar

A tow cart has to be backed rear-first along an S from one bay into another by its hinged drawbar. Pushing a hinged, fixed-axle cart in reverse is unstable, so it has to be steered all the way. The chock comes off first and goes on again at the end.

Instruction: "First take the chock away from behind the cart's wheel, then back the cart rear-first into the green bay using only the orange drawbar handle, chock a wheel on its open side, and move both hands away. Don't jackknife or tip the cart, skid its wheels, or touch its body while it moves."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_cart -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "First take the chock away from behind the cart's wheel, then back the cart rear-first into the green bay using only the orange drawbar handle, chock a wheel on its open side, and move both hands away. Don't jackknife or tip the cart, skid its wheels, or touch its body while it moves." \
      --scorer task_score --max-steps 2400 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 4 minutes on one GPU.
- **Seeds:** none. The start is fixed and `object_jitter=0`. There are no lots.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py`
- **The scripted demo** is open loop: it plays a hand path worked out in advance from the cart's known start and a bicycle model of the cart, and makes no corrections. It's told nothing a policy couldn't see from the cart's position. Being open loop on an unstable cart it relies on the simulation repeating; a policy should watch the cart and correct.
- **Two hands:** the demo passes the bar from the right hand to the left near y = 0, and the left hand sets the final chock. The scorer doesn't require a handover, and whether one arm could do it all, by swinging the bar within its reach, hasn't been tested.
- **Final chock:** the demo's chock ends about 1 cm short of the wheel, inside the scorer's band (3.6 to 5.5 cm from the wheel's centre; contact is at 4 cm).
- **Third-party assets:** none. Every part is built from boxes and spheres in the task's own code.
