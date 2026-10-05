import glob
from pathlib import Path

import matplotlib.pyplot as plt
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

ETA, RECOV, TOP, G = 0.15, 0.1, 0.05, 0.9
N_PHASE, BLOCK = 24, 4
BASE, GAIN, WANDER = 0.8, 10.0, 0.3
noise = np.random.default_rng(7)
DIST_A, DIST_B, ARRIVE, REINF, RUN_TIME, WINDOW = 12.8, 8.0, 3.0, 6.0, 4.0, 500


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
AVOID = ["MBON01", "MBON02", "MBON03", "MBON04", "MBON05"]
APPROACH = ["MBON09", "MBON11", "MBON12"]
tmap = ct.set_index("root_id")["primary_type"].astype(str)
pos = lambda names: np.where(np.isin(MB, ix(tmap.index[tmap.isin(names)].values)))[0]
app, av = pos(APPROACH), pos(AVOID)
print(f"faithful valence groups: {len(av)} avoid MBONs, {len(app)} approach MBONs")


def valence(m, S):
    return float(m[app][:, S].sum() - m[av][:, S].sum())


w_ref = M0.copy()
for _ in range(10):
    w_ref[np.ix_(av, SA)] *= 1 - ETA
    w_ref[np.ix_(app, SB)] *= 1 - ETA
VREF = max(abs(valence(w_ref, SA)), abs(valence(w_ref, SB)), 1e-9)   # fixed unit: the offline-trained scale
print(f"KCs active: A {len(SA)}, B {len(SB)}; naive valence A {valence(M0, SA) / VREF:+.2f}, B {valence(M0, SB) / VREF:+.2f}")


def reinforce(w, key, phase):
    if key == "A":
        w[np.ix_(av if phase == 1 else app, SA)] *= 1 - ETA
    else:
        w[np.ix_(app if phase == 1 else av, SB)] *= 1 - ETA


fly = make_locomotion_fly(name="online", add_adhesion=True, colorize=True)
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
angles = np.random.default_rng(123).choice([-60, -30, 30, 60], size=2 * N_PHASE)


def trial(w, phase, angle, seed, learn):
    sim.reset(); controller.reset(seed=seed)
    apply_locomotion_action(sim, fly.name, init)
    sim.warmup()
    start = sim.get_body_positions(fly.name)[thorax][:2].copy()
    a = np.deg2rad(angle)
    tA = DIST_A * np.array([np.cos(a), np.sin(a)])
    tB = DIST_B * np.array([np.cos(-a), np.sin(-a)])
    path = np.zeros((n_steps, 2))
    got = {"A": False, "B": False}
    vA, vB = valence(w, SA), valence(w, SB)
    drive = np.array([1.0, 1.0])
    bias = 0.0
    for i in range(n_steps):
        p = sim.get_body_positions(fly.name)[thorax][:2] - start
        path[i] = p
        dA, dB = np.linalg.norm(p - tA), np.linalg.norm(p - tB)
        for key, d in (("A", dA), ("B", dB)):
            if d < REINF and not got[key]:
                got[key] = True
                if learn:
                    reinforce(w, key, phase)
                    vA, vB = valence(w, SA), valence(w, SB)
        if dA < ARRIVE:
            return "A"
        if dB < ARRIVE:
            return "B"
        if i >= WINDOW:
            dx, dy = p - path[i - WINDOW]
            lp, rp = antenna_positions(p, np.arctan2(dy, dx))
            aL, bL = concentration(lp, tA), concentration(lp, tB)
            aR, bR = concentration(rp, tA), concentration(rp, tB)
            err = ((aL - aR) * vA + (bL - bR) * vB) / ((aL + bL + aR + bR + 1e-9) * VREF)
            err = np.clip(err, -1.5, 1.5)
            bias += -bias * sim.timestep + WANDER * np.sqrt(sim.timestep) * noise.normal()
            drive = np.clip([BASE - GAIN * err - bias, BASE + GAIN * err + bias], 0.4, 1.2)
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(drive, obs))
        sim.step()
    return "none"


def series(learn, label):
    w = M0.copy()
    rec = []
    for k in tqdm(range(2 * N_PHASE), desc=label):
        phase = 1 if k < N_PHASE else 2
        if learn:
            w += RECOV * (M0 - w)
        out = trial(w, phase, int(angles[k]), k, learn)
        rec.append((phase, out, valence(w, SA) / VREF, valence(w, SB) / VREF))
    return rec


R1 = series(True, "with learning")
R0 = []

for name, rec in (("WITH LEARNING", R1),):
    print(f"\n{name}: trials 1-{N_PHASE} A rewarded / B punished, trials {N_PHASE + 1}-{2 * N_PHASE} flipped")
    print(f"{'trials':<10}{'phase':>6}{'A':>4}{'B':>4}{'none':>6}{'valence A':>11}{'valence B':>11}")
    for b in range(0, 2 * N_PHASE, BLOCK):
        blk = rec[b:b + BLOCK]
        o = [r[1] for r in blk]
        print(f"{b + 1:>3}-{b + BLOCK:<6}{blk[0][0]:>6}{o.count('A'):>4}{o.count('B'):>4}{o.count('none'):>6}"
              f"{np.mean([r[2] for r in blk]):>+11.2f}{np.mean([r[3] for r in blk]):>+11.2f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(1, 2 * N_PHASE + 1)
ax.plot(x, [r[2] for r in R1], "-o", color="green", label="valence of A (with learning)")
ax.plot(x, [r[3] for r in R1], "-o", color="red", label="valence of B (with learning)")
col = {"A": "green", "B": "red", "none": "gray"}
ax.scatter(x, np.full(len(x), -1.25), c=[col[r[1]] for r in R1], marker="s", s=40)
pass
ax.text(0.2, -1.28, "learning", ha="right", fontsize=8)
ax.axvline(N_PHASE + 0.5, color="black", linestyle="--")
ax.set_xlabel("trial (squares: where she ended up; green = A, red = B, gray = neither)")
ax.set_ylabel("net approach drive (1 = offline-trained scale)")
ax.set_title("Online learning, then contingency flip")
ax.legend(fontsize=8, loc="upper right")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/online_learning_faithful.png", dpi=150, bbox_inches="tight")
print("saved figures/online_learning_faithful.png")
