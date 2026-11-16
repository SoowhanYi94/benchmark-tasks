"""The enclosure with its hinged lid, and the two gas struts, as strutkit assemblies (pure Python, no simulator).

Frame: x forward from the line through the arm bases, y to the robot's left, z up from the table top (TOP added).
Real part: an ACE GS-10-30 gas spring (catalogue: body 10 mm, rod 3 mm, stroke 30 mm, extended length 92 mm, extension
force 10 to 100 N, progression about 13 to 16 %, G3.5 ball sockets, 8 mm ball), ordered at 10 N. Lid, studs and
layout are this task's own values, chosen so the moment balance works (fin/cpu/design.py in the build notes)."""

from __future__ import annotations

import math

import numpy as np

from strutkit import Assembly, Joint, Link, Shape, matrix_quat_xyzw

TOP = 0.75
# enclosure
X0, X1, HW, H, WALL = 0.18, 0.38, 0.200, 0.080, 0.0015
HINGE = (X0, 0.0, H)                    # hinge axis along y on the front top edge
OPEN = math.radians(45.0)               # the lid's open stop
LID_MASS = 0.63                         # 2.5 mm Al plate 400 x 200, 18 mm flanges, 12 x 1 mm Al tube handle
LID_T = 0.0025
LID_COM = (X0 + 0.1052, 0.0, H - 0.0004)  # from the parts (fin/cpu/design.py)
BAR_R, BAR_X = 0.006, X0 + 0.18         # handle bar: 12 mm, 180 mm from the hinge, 30 mm proud
BAR_Z = H + 0.030
# studs: 8 mm balls (the G3.5 socket's ball), 4 mm necks (M4), ball centres 35 mm out from the side faces
BALL_R, NECK_R, BALL_Y = 0.004, float(__import__("os").environ.get("GS_NECK", "0.002")), 0.235  # ball centres 35 mm out: the fitting gripper's inner finger fits
LID_A, LID_E = 0.0375, 0.010            # lid stud: 37.5 mm from the hinge along the lid, 10 mm below its top
STRUT_LEN = 0.092                       # GS-10-30 extended length, here between ball centres
_lx = X0 + LID_A * math.cos(OPEN) + LID_E * math.sin(OPEN)
_lz = H + LID_A * math.sin(OPEN) - LID_E * math.cos(OPEN)
BODY_Z = 0.045
BODY_STUD = (_lx + math.sqrt(STRUT_LEN ** 2 - (_lz - BODY_Z) ** 2), BODY_Z)  # (x, z): 92 mm from the lid stud at 45 deg
# struts: GS-10-30, 10 N extended, 30 mm stroke, progression 1.15
STROKE, P1, PROG = 0.030, 10.0, 1.15
K_GAS = P1 * (PROG - 1) / STROKE        # 50 N/m
GAS_TARGET = -P1 / K_GAS                # spring target 0.2 m beyond full extension (q = compression, 0 = extended)
GAS_DAMP = 60.0
TUBE_R, ROD_R = 0.005, 0.0015
HOUSING = 0.013                         # G3.5 socket housing, 13 mm
TORSION = float(__import__("os").environ.get("GS_TORSION", "0.004"))  # pad contact patch on the body (declared)
JAW_W = float(__import__("os").environ.get("GS_JAWW", "0.0095"))  # jaws span the 8 mm ball across the strut axis
M_TUBE, M_ROD, M_JAW = (float(v) for v in __import__("os").environ.get("GS_M", "0.030,0.012,0.006").split(","))
# ball socket: two spring jaws across the stud axis
GAP = 0.0003
JAW_R = NECK_R + float(__import__("os").environ.get("GS_JGAP", "0.0003"))  # jaws' inner faces at rest, outside the neck
JAW_Y = math.sqrt(BALL_R ** 2 - JAW_R ** 2) + 0.0001  # jaws' back faces, just behind the ball when seated
JAW_K = float(__import__("os").environ.get("GS_JAWK", "250"))      # N/m along the guide
JAW_PRE = float(__import__("os").environ.get("GS_JAWPRE", "0.6"))   # N preload against the shut stop
JAW_TRAVEL = 0.004
# v2: each socket's spring clip = its two sprung jaws (as v1: they snap over the ball and lock under a pull), each
# with a flat finger loop running on past the socket's outer end. Pushing both loops outboard (a fingertip behind
# them) drives the jaws open along their guides; the socket then slides off the ball. (A single jaw was tried first:
# the socket tilted on its ball under the 30 N pull and the ball rolled out past the open side, fin2/t1, t2.)
LEAD = math.radians(15.0)
# WHY angled guides: each jaw opens along a line 50 deg back from the mouth (outward and into the socket), so a ball
# pulled outward drives the jaws shut against their stops (a collet), while a ball pushed in opens them. Radial jaws
# let go at 15 to 18 N (spike 2): PhysX friction alone does not lock them. The jaws are slick (0.05): grippy jaws
# locked by friction on the way in and stalled the push 3.4 mm short (spike 5); the lock is geometric (pull: n.d < 0).
JAW_BETA = math.radians(float(__import__("os").environ.get("GS_BETA", "50")))

MATERIALS = {"default": (0.8, 0.8), "ball": (1.0, 1.0), "ring": (0.3, 0.3, "min"), "slick": (0.05, 0.05, "min"),
             "grip": (0.05, 0.05, "min"), "lid": (0.6, 0.6)}

STRUT_START_X, BATTEN_H = 0.095, 0.020  # struts' centres at the start; batten height
WOOD = (0.62, 0.47, 0.30)
GREY, BLUE, BLACK, STEEL, WHITE = (0.55, 0.57, 0.60), (0.15, 0.35, 0.75), (0.06, 0.06, 0.07), (0.78, 0.79, 0.80), (0.9, 0.9, 0.9)


def lid_point(a, e, th):
    """(x, z) of the lid point a along the lid from the hinge, e below its top, with the lid open th (rad)."""
    return (HINGE[0] + a * math.cos(th) + e * math.sin(th), HINGE[2] + a * math.sin(th) - e * math.cos(th))


def stud_points(side: int, th: float = OPEN):
    """World ball centres (body stud, lid stud) on side +1 (left) or -1 (right), lid open th."""
    bx, bz = BODY_STUD
    lx, lz = lid_point(LID_A, LID_E, th)
    return np.array([bx, side * BALL_Y, TOP + bz]), np.array([lx, side * BALL_Y, TOP + lz])


assert abs(float(np.linalg.norm(stud_points(1)[1] - stud_points(1)[0])) - STRUT_LEN) < 1e-9


def enclosure(name="enclosure") -> Assembly:
    floor = Shape("box", ((X0 + X1) / 2, 0, WALL / 2), (X1 - X0, 2 * HW, WALL), GREY)
    walls = [Shape("box", (X0 + WALL / 2, 0, H / 2), (WALL, 2 * HW, H), GREY),
             Shape("box", (X1 - WALL / 2, 0, H / 2), (WALL, 2 * HW, H), GREY)]
    for s in (1, -1):
        walls.append(Shape("box", ((X0 + X1) / 2, s * (HW - WALL / 2), H / 2), (X1 - X0, WALL, H), GREY))
    studs = []
    for s in (1, -1):
        bx, bz = BODY_STUD
        studs += [Shape("cylinder", (bx, s * (HW + BALL_Y) / 2, bz), (NECK_R, BALL_Y - HW), STEEL, axis="Y"),
                  Shape("sphere", (bx, s * BALL_Y, bz), (BALL_R,), STEEL, material="ball", contact_offset=0.0005)]
    # two wooden battens on the table in front of the enclosure; the struts start laid across them by their sockets
    # WHY: the hand picks a strut rolled 28 deg about it (needed to clear the open lid later); from the bare table the
    # lower fingertip would hit the table, and a shallow grip near the fingertips let the strut squirt out (r1, r2)
    battens = [Shape("box", (STRUT_START_X + k * STRUT_LEN / 2, 0, BATTEN_H / 2), (0.020, 0.360, BATTEN_H), WOOD)
               for k in (1, -1)]
    body = Link("body", 6.0, tuple([floor, *walls, *studs, *battens]))
    lid_shapes = [Shape("box", ((X0 + X1) / 2, 0, H + LID_T / 2), (X1 - X0, 2 * HW + 0.0046, LID_T), BLUE, material="lid")]
    fy = HW + 0.0003 + 0.001
    for s in (1, -1):
        lid_shapes.append(Shape("box", ((X0 + X1) / 2, s * fy, H + 0.002 - 0.009), (X1 - X0, 0.002, 0.018), BLUE,
                                material="lid"))
        lx, lz = lid_point(LID_A, LID_E, 0.0)
        lid_shapes += [Shape("cylinder", (lx, s * (fy + 0.001 + BALL_Y) / 2, lz), (NECK_R, BALL_Y - fy - 0.001), STEEL,
                             axis="Y"),
                       Shape("sphere", (lx, s * BALL_Y, lz), (BALL_R,), STEEL, material="ball", contact_offset=0.0005)]
        lid_shapes.append(Shape("box", (BAR_X, s * 0.14, H + 0.002 + 0.012), (0.010, 0.010, 0.024), BLACK))
    lid_shapes.append(Shape("cylinder", (BAR_X, 0, BAR_Z), (BAR_R, 0.300), BLACK, axis="Y"))
    lid = Link("lid", LID_MASS, tuple(lid_shapes), com=LID_COM)
    hinge = Joint("lid_hinge", "revolute", "body", "lid", HINGE, "Y", -OPEN, 0.0, damping=0.05)
    return Assembly(name=name, pos=(0.0, 0.0, TOP), size=(X1 - X0, 2 * HW, H), links=(body, lid), joints=(hinge,),
                    materials=MATERIALS, fixed=True)


TAB_L, TAB_T, TAB_H = 0.016, 0.0025, 0.006   # finger loop: a flat tab on the jaw running on past the socket's outer end
TAB_PAD = HOUSING / 2 + TAB_L / 2              # its middle, from the socket centre along the strut (where it is pinched)
FAILED_DAMP = 5.0                       # N s/m: a strut that has lost its gas keeps a little oil damping (our value)
RED = (0.85, 0.08, 0.06)


def _socket(xc, s_jaw=1, color=BLACK):
    """Fixed shapes of a ball socket centred at (xc, 0, 0) in the strut frame (mouth toward +y): a back plate and a
    bore of eight flats round the ball; the flats away from the jaw run on to the mouth so the ball can only leave
    along the stud's axis, past the jaw."""
    shapes = [Shape("box", (xc, -(BALL_R + GAP + 0.0015), 0), (HOUSING, 0.003, HOUSING), color, material="ring",
                    contact_offset=0.0005)]
    r = BALL_R + GAP + 0.001
    yl = -(BALL_R + GAP)
    for k in range(8):
        a = math.radians(45 * k)
        th = -a
        yh = 0.0003
        shapes.append(Shape("box", (xc + r * math.cos(a), (yl + yh) / 2, r * math.sin(a)), (0.002, yh - yl, 0.004),
                            color, rot=(math.cos(th / 2), 0.0, math.sin(th / 2), 0.0), material="ring",
                            contact_offset=0.0005))
    return shapes


def _jaw(xc, s, tag, e=1):
    ret = Shape("box", (xc, JAW_Y + 0.0009, s * (JAW_R + 0.002)), (JAW_W, 0.0018, 0.004), WHITE, material="grip",
                contact_offset=0.0005)
    y0 = JAW_Y + 0.0018
    length = 0.008 / math.cos(LEAD)
    mid = np.array([y0 + 0.004, s * (JAW_R + 0.004 * math.tan(LEAD))])
    n = np.array([-math.sin(LEAD), s * math.cos(LEAD)])
    c = mid + 0.0015 * n
    phi = s * LEAD
    cham = Shape("box", (xc, c[0], c[1]), (JAW_W, length, 0.003), WHITE,
                 rot=(math.cos(phi / 2), math.sin(phi / 2), 0.0, 0.0), material="slick", contact_offset=0.0005)
    # the finger loop: a 16 mm tab on the jaw, running on along the strut past the socket's outer end (e = +1 at the
    # body end of the strut, -1 at the rod end); pinched across its faces and pulled, it opens the jaw
    loop = Shape("box", (xc + e * TAB_PAD, JAW_Y + 0.0009, s * (JAW_R + 0.002)), (TAB_L, TAB_T, TAB_H), STEEL,
                 material="slick", contact_offset=0.0005)
    return Link(tag, M_JAW, (ret, cham, loop))


def jaw_joint(link, parent, xc) -> Joint:
    """The jaw's slide: along X of a joint frame turned so X points (0, -sin b, s cos b): outward and back into the
    socket. Positive = opening; the spring preloads it shut against its stop at 0."""
    s = 1 if link.name.endswith("_a") else -1
    # rotate X onto d = (0, -sin b, s cos b): first 90 deg about -Y?? do it as a rotation taking x to d
    d = np.array([0.0, -math.sin(JAW_BETA), s * math.cos(JAW_BETA)])
    x = np.array([1.0, 0.0, 0.0])
    axis = np.cross(x, d); ang = math.acos(float(x @ d)); axis /= np.linalg.norm(axis)
    rot = (math.cos(ang / 2), *(axis * math.sin(ang / 2)))
    return Joint(link.name + "_slide", "prismatic", parent, link.name, (xc, 0, 0), "X", 0.0, JAW_TRAVEL, JAW_K, 2.0,
                 -JAW_PRE / JAW_K, rot=tuple(float(v) for v in rot))


def probe(name, at=(0.30, -0.60, 0.80)) -> Assembly:
    """Test fixture (spike only): its own stud (a block, a 6 mm neck and a 10 mm ball pointing -y) and one ball socket
    on a carriage sliding along the stud's axis (joint "push": 0 = seated, negative = outboard), driven by a drive
    limited to 4 N: a rigid hand that can push at most so hard. Frame: origin at the ball centre, X = world x,
    Y = world +y (the mouth faces the block)."""
    base = Link("base", 1.0, (Shape("box", (0, 0.06, 0.03), (0.01, 0.01, 0.01), GREY, visible=False),))
    sock = Link("carriage", 0.05, tuple(_socket(0.0, 1)))
    jaws = [_jaw(0.0, 1, "jaw_a"), _jaw(0.0, -1, "jaw_b")]
    joints = [Joint("push", "prismatic", "base", "carriage", (0, 0, 0), "Y", -0.03, 0.004, 2000.0, 50.0, 0.0,
                    effort=4.0)]
    joints += [jaw_joint(j, "carriage", 0.0) for j in jaws]
    return Assembly(name=name, pos=tuple(at), size=(0.03, 0.05, 0.03), links=(base, sock, *jaws),
                    joints=tuple(joints), materials=MATERIALS, fixed=True)


def probe_stud(name, at=(0.30, -0.60, 0.80)) -> Assembly:
    """The probe's stud, a separate fixed assembly (links joined in one articulation never touch)."""
    link = Link("stud", 1.0, (Shape("box", (0, 0.035, 0), (0.03, 0.010, 0.03), GREY),
                              Shape("cylinder", (0, 0.015, 0), (NECK_R, 0.030), STEEL, axis="Y"),
                              Shape("sphere", (0, 0, 0), (BALL_R,), STEEL, material="ball", contact_offset=0.0005)))
    tag = Link("tag", 0.01, (Shape("box", (0, 0.06, -0.03), (0.005, 0.005, 0.005), GREY, visible=False),))
    lock = Joint("lock", "prismatic", "stud", "tag", (0, 0.06, -0.03), "Z", 0.0, 1e-5)  # an articulation needs a joint
    return Assembly(name=name, pos=tuple(at), size=(0.03, 0.05, 0.03), links=(link, tag), joints=(lock,),
                    materials=MATERIALS, fixed=True)


def strut(name, p_rod, p_tube, mouth, failed=False, tagged=False) -> Assembly:
    """A gas strut, its tube-end socket centred on world point p_tube and its rod-end socket on p_rod (closer than
    92 mm: the rod starts pushed in by the difference), both sockets' mouths toward the world direction ``mouth``.
    Each socket's one jaw (and its finger loop) is on the side that faces up in this pose. ``failed``: no gas force,
    FAILED_DAMP damping. ``tagged``: a red defect tag on the body."""
    X = np.asarray(p_tube, float) - np.asarray(p_rod, float)
    L = float(np.linalg.norm(X)); X /= L
    Y = np.asarray(mouth, float); Y = Y - X * (Y @ X); Y /= np.linalg.norm(Y)
    Z = np.cross(X, Y)
    q = matrix_quat_xyzw(np.column_stack([X, Y, Z]))
    Lx = STRUT_LEN
    q0 = max(0.0, Lx - L)
    s_jaw = 1 if Z[2] >= 0 else -1
    origin = np.asarray(p_tube, float) - Lx * X
    # body 10 mm from 40 to 80 mm along the strut (a 6 mm neck to the tube-end socket), rod 3 mm from the rod-end
    # socket into the body; 42 g in all with the zinc-plated steel sockets (the task's value: no maker weight)
    tube_shapes = [Shape("cylinder", (0.060, 0, 0), (TUBE_R, 0.040), BLACK, axis="X", torsion=TORSION),
                   Shape("cylinder", (0.0835, 0, 0), (0.003, 0.007), BLACK, axis="X")] + _socket(Lx, s_jaw)
    if tagged:   # a red card on the body's upper side (a defect tag, 30 x 20 mm), lying along the strut
        tube_shapes.append(Shape("box", (0.060, 0, s_jaw * (TUBE_R + 0.0008)), (0.030, 0.020, 0.0015), RED))
    tube = Link("tube", M_TUBE, tuple(tube_shapes))
    rod = Link("rod", M_ROD, tuple([Shape("cylinder", (0.0235, 0, 0), (ROD_R, 0.034), STEEL, axis="X")]
                                   + _socket(0.0, s_jaw)))
    jaws = [_jaw(0.0, 1, "jaw_rod_a", -1), _jaw(0.0, -1, "jaw_rod_b", -1),
            _jaw(Lx, 1, "jaw_tube_a", 1), _jaw(Lx, -1, "jaw_tube_b", 1)]
    k, c, tgt = (0.0, FAILED_DAMP, 0.0) if failed else (K_GAS, GAS_DAMP, GAS_TARGET)
    joints = [Joint("gas", "prismatic", "tube", "rod", (0.0, 0, 0), "X", 0.0, STROKE, k, c, tgt, init=q0)]
    for link, parent, xc in zip(jaws, ("rod", "rod", "tube", "tube"), (0.0, 0.0, Lx, Lx)):
        joints.append(jaw_joint(link, parent, xc))
    return Assembly(name=name, pos=tuple(float(v) for v in origin), size=(Lx, HOUSING, HOUSING), mass=M_TUBE + M_ROD + 4 * M_JAW,
                    links=(tube, rod, *jaws), joints=tuple(joints), materials=MATERIALS, fixed=False,
                    quat=tuple(float(v) for v in q))
