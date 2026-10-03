import glob

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation, HybridTurningController, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)
from odour_steer import antenna_positions, concentration

ETA, TOP, G = 0.15, 0.05, 0.9
N_LEARN, N_REVERSE = 10, 20
BASE, GAIN = 0.8, 10.0
DIST_A, DIST_B, ARRIVE, RUN_TIME, WINDOW = 12.8, 8.0, 3.0, 4.0, 500
ANGLES, SEEDS = [-60, -30, 30, 60], [0, 1]


def read(p, **kw):
    return pd.read_csv(glob.glob(f"connectome/data/{p}*")[0], **kw)


W = sp.load_npz("connectome/brain_W.npz").tocsr()
conn = read("conn", usecols=["pre_root_id", "post_root_id", "nt_type"])
conn = conn[conn["nt_type"].astype(str).str.upper().isin(["ACH", "GABA", "GLUT"])]
ids = pd.Index(np.unique(np.concatenate([conn["pre_root_id"].values, conn["post_root_id"].values])))
assert len(ids) == W.shape[0], "neuron order does not match the cached brain model"
N = W.shape[0]
cls, ct = read("class"), read("*cell_type")


def ix(root):
    a = ids.get_indexer(root)
    return a[a >= 0]


KC = ix(cls.loc[cls["class"] == "Kenyon_Cell", "root_id"])
MB = ix(cls.loc[cls["class"] == "MBON", "root_id"])
orn = ct[ct["primary_type"].astype(str).str.startswith("ORN_")]
orn_idx = {t: ix(g["root_id"]) for t, g in orn.groupby("primary_type")}
types = sorted(orn_idx)
M0 = np.abs(W[MB][:, KC].toarray()).astype(np.float32)


def kc_code(odour_types):
    I = np.zeros(N, np.float32)
    for t in odour_types:
        I[orn_idx[t]] = 1.0
    r = np.zeros(N, np.float32)
    for _ in range(60):
        r = np.maximum(G * (W @ r) + I, 0.0)
    a = r[KC]
    k = int(TOP * len(KC))
    idx = np.argpartition(-a, k)[:k]
    return idx[a[idx] > 0]


rng = np.random.default_rng(0)
perm = rng.permutation(len(types))
SA = kc_code([types[i] for i in perm[:8]])
SB = kc_code([types[i] for i in perm[8:16]])
split = rng.permutation(len(MB))
app, av = split[: len(MB) // 2], split[len(MB) // 2:]

states = {"naive": M0.copy()}
w = M0.copy()
for _ in range(N_LEARN):
    w[np.ix_(av, SA)] *= 1 - ETA
    w[np.ix_(app, SB)] *= 1 - ETA
states["trained (A good, B bad)"] = w.copy()
for _ in range(N_REVERSE):
    w += 0.1 * (M0 - w)                  # depressed synapses slowly recover
    w[np.ix_(app, SA)] *= 1 - ETA
    w[np.ix_(av, SB)] *= 1 - ETA
states["reversed (A bad, B good)"] = w.copy()


def valence(m, S):
    return float(m[app][:, S].sum() - m[av][:, S].sum())      # net approach drive for this odour


tr = states["trained (A good, B bad)"]
VREF = max(abs(valence(tr, SA)), abs(valence(tr, SB)), 1e-9)
print(f"KCs active: A {len(SA)}, B {len(SB)}")
for name, m in states.items():
    print(f"{name:<28} valence A {valence(m, SA) / VREF:+.2f}, B {valence(m, SB) / VREF:+.2f}  (1.0 = trained scale)")

fly = make_locomotion_fly(name="learn", add_adhesion=True, colorize=True)
cam = fly.add_tracking_camera(name="body_cam", pos_offset=(-0.5, -7.5, 0.0),
                              rotation=Rotation3D("euler", (1.57, 0.0, 0.0)), fovy=35.0)
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
              bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
              add_ground_contact_sensors=False)
sim = Simulation(world)
sim.set_renderer([cam], camera_res=(240, 320), playback_speed=0.2, output_fps=25)
steps = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")
controller = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps,
                                     output_dof_order=dof_order)
init = LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dof_order),
                        adhesion_onoff=np.ones(6, dtype=bool))
thorax = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
n_steps = int(RUN_TIME / sim.timestep)


def trial(m, angle, seed):
    vA, vB = valence(m, SA), valence(m, SB)
    sim.reset(); controller.reset(seed=seed)
    apply_locomotion_action(sim, fly.name, init)
    sim.warmup()
    start = sim.get_body_positions(fly.name)[thorax][:2].copy()
    a = np.deg2rad(angle)
    tA = DIST_A * np.array([np.cos(a), np.sin(a)])
    tB = DIST_B * np.array([np.cos(-a), np.sin(-a)])
    path = np.zeros((n_steps, 2))
    drive = np.array([1.0, 1.0])
    for i in range(n_steps):
        p = sim.get_body_positions(fly.name)[thorax][:2] - start
        path[i] = p
        if np.linalg.norm(p - tA) < ARRIVE:
            return "A", i * sim.timestep
        if np.linalg.norm(p - tB) < ARRIVE:
            return "B", i * sim.timestep
        if i >= WINDOW:
            dx, dy = p - path[i - WINDOW]
            lp, rp = antenna_positions(p, np.arctan2(dy, dx))
            aL, bL = concentration(lp, tA), concentration(lp, tB)
            aR, bR = concentration(rp, tA), concentration(rp, tB)
            err = ((aL - aR) * vA + (bL - bR) * vB) / ((aL + bL + aR + bR + 1e-9) * VREF)
            err = np.clip(err, -1.5, 1.5)
            drive = np.clip([BASE - GAIN * err, BASE + GAIN * err], 0.4, 1.2)
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(drive, obs))
        sim.step()
    return "none", np.nan


res = {c: [] for c in states}
for s, a, c in tqdm([(s, a, c) for s in SEEDS for a in ANGLES for c in states], desc="trials"):
    res[c].append(trial(states[c], a, s))

print(f"\nA = {DIST_A} mm, B = {DIST_B} mm (closer, opposite side), {len(ANGLES)} angles x {len(SEEDS)} seeds")
print(f"{'condition':<28}{'reached A':>10}{'reached B':>10}{'neither':>9}{'median time':>13}")
for c, r in res.items():
    out = [x[0] for x in r]
    t = [x[1] for x in r if x[0] != "none"]
    med = f"{np.median(t):.2f} s" if t else "-"
    print(f"{c:<28}{out.count('A'):>10}{out.count('B'):>10}{out.count('none'):>9}{med:>13}")
