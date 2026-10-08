from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import trange

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
TAU_REC = 30.0     # time constant (s) of the time-based recovery added after seed 3
BASE, GAIN, WANDER = 0.8, 10.0, 0.3
HALF, MARGIN, REINF, WINDOW = 25.0, 6.0, 6.0, 800
T_FLIP, T_TOTAL, SEED = 60.0, 160.0, 0
LIM = HALF - MARGIN - 1.0      # sources spawn inside this square, clear of the wall zone
AVOID = ["MBON01", "MBON02", "MBON03", "MBON04", "MBON05"]
APPROACH = ["MBON09", "MBON11", "MBON12"]


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
tmap = ct.set_index("root_id")["primary_type"].astype(str)
pos = lambda names: np.where(np.isin(MB, ix(tmap.index[tmap.isin(names)].values)))[0]
app, av = pos(APPROACH), pos(AVOID)


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


perm = np.random.default_rng(0).permutation(len(types))
SA = kc_code([types[i] for i in perm[:8]])
SB = kc_code([types[i] for i in perm[8:16]])


def valence(m, S):
    return float(m[app][:, S].sum() - m[av][:, S].sum())


w_ref = M0.copy()
for _ in range(10):
    w_ref[np.ix_(av, SA)] *= 1 - ETA
    w_ref[np.ix_(app, SB)] *= 1 - ETA
VREF = max(abs(valence(w_ref, SA)), abs(valence(w_ref, SB)), 1e-9)
print(f"MBONs: {len(av)} avoid, {len(app)} approach; KCs active A {len(SA)}, B {len(SB)}; "
      f"naive valence A {valence(M0, SA) / VREF:+.2f}, B {valence(M0, SB) / VREF:+.2f}")


def reinforce(w, key, phase):
    if key == "A":
        w[np.ix_(av if phase == 1 else app, SA)] *= 1 - ETA
    else:
        w[np.ix_(app if phase == 1 else av, SB)] *= 1 - ETA


def spawn(rng, keep_clear):
    for _ in range(200):
        p = rng.uniform(-LIM, LIM, size=2)
        if all(np.linalg.norm(p - q) >= 10.0 for q in keep_clear):
            return p
    return p


fly = make_locomotion_fly(name="capstone", add_adhesion=True, colorize=True)
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
n = int(T_TOTAL / sim.timestep)
every = int(0.01 / sim.timestep)


def episode(learn, label, SEED=0):
    rng, noise = np.random.default_rng(SEED), np.random.default_rng(SEED + 1000)
    w = M0.copy()
    sim.reset(); controller.reset(seed=SEED)
    apply_locomotion_action(sim, fly.name, init)
    sim.warmup()
    start = sim.get_body_positions(fly.name)[thorax][:2].copy()
    tA = spawn(rng, [np.zeros(2)])
    tB = spawn(rng, [np.zeros(2), tA])
    path = np.zeros((n, 2))
    visits = []
    drive, bias = np.array([1.0, 1.0]), 0.0
    vA, vB = valence(w, SA), valence(w, SB)
    for i in trange(n, desc=label):
        p = sim.get_body_positions(fly.name)[thorax][:2] - start
        path[i] = p
        if i % every == 0:
            t = i * sim.timestep
            phase = 1 if t < T_FLIP else 2
            if learn:
                w += (every * sim.timestep / TAU_REC) * (M0 - w)      # depressed synapses drift back with time
                vA, vB = valence(w, SA), valence(w, SB)
            for key in ("A", "B"):
                tgt = tA if key == "A" else tB
                if np.linalg.norm(p - tgt) < REINF:
                    if learn:
                        w += RECOV * (M0 - w)
                        reinforce(w, key, phase)
                        vA, vB = valence(w, SA), valence(w, SB)
                    visits.append((t, key, vA / VREF, vB / VREF, p[0], p[1]))
                    new = spawn(rng, [p, tB if key == "A" else tA])
                    if key == "A":
                        tA = new
                    else:
                        tB = new
            if i >= WINDOW:
                dx, dy = p - path[i - WINDOW]
                heading = np.arctan2(dy, dx)
                if max(abs(p[0]), abs(p[1])) > HALF - MARGIN:
                    err_w = (np.arctan2(-p[1], -p[0]) - heading + np.pi) % (2 * np.pi) - np.pi
                    turn = np.clip(err_w, -0.8, 0.8)
                else:
                    lp, rp = antenna_positions(p, heading)
                    aL, bL = concentration(lp, tA), concentration(lp, tB)
                    aR, bR = concentration(rp, tA), concentration(rp, tB)
                    err = ((aL - aR) * vA + (bL - bR) * vB) / ((aL + bL + aR + bR + 1e-9) * VREF)
                    bias += -bias * 0.01 + WANDER * np.sqrt(0.01) * noise.normal()
                    turn = GAIN * np.clip(err, -1.5, 1.5) + bias
                drive = np.clip([BASE - turn, BASE + turn], 0.4, 1.2)
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(drive, obs))
        sim.step()
    return path, visits


def summarise(label, path, visits):
    out = np.maximum(abs(path[:, 0]), abs(path[:, 1]))
    print(f"\n{label}: {len(visits)} visits in {T_TOTAL:.0f} s; time outside the box {np.mean(out > HALF):.1%}; "
          f"furthest from centre {out.max():.1f} mm (wall at {HALF:.0f})")
    print(f"{'time (s)':<12}{'visits A':>9}{'visits B':>9}{'at rewarded':>13}")
    for b in range(0, int(T_TOTAL), 10):
        blk = [v[1] for v in visits if b <= v[0] < b + 10]
        good = "A" if b < T_FLIP else "B"
        frac = f"{blk.count(good) / len(blk):.0%}" if blk else "-"
        print(f"{b:>3}-{b + 10:<8}{blk.count('A'):>9}{blk.count('B'):>9}{frac:>13}")


def plot(path, visits):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    tt = np.arange(0, n, 50) * sim.timestep
    sc = ax[0].scatter(path[::50, 0], path[::50, 1], c=tt, s=3, cmap="viridis")
    ax[0].add_patch(plt.Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False, color="red", linewidth=2))
    for v in visits:
        ax[0].scatter(v[4], v[5], marker="*" if v[1] == "A" else "X", s=110,
                      color="green" if v[1] == "A" else "red", zorder=5, edgecolor="black", linewidth=0.5)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("x (mm)"); ax[0].set_ylabel("y (mm)")
    ax[0].set_title("path (colour = time); stars = visits to A, crosses = visits to B")
    fig.colorbar(sc, ax=ax[0], label="time (s)")
    k = np.arange(1, len(visits) + 1)
    ax[1].plot(k, [v[2] for v in visits], "-o", color="green", label="valence of A")
    ax[1].plot(k, [v[3] for v in visits], "-o", color="red", label="valence of B")
    flip = next((j + 1 for j, v in enumerate(visits) if v[0] >= T_FLIP), None)
    if flip:
        ax[1].axvline(flip - 0.5, color="black", linestyle="--")
    ax[1].set_xlabel("visit number"); ax[1].set_ylabel("net approach drive (1 = offline-trained scale)")
    ax[1].set_title("learned valence after each visit"); ax[1].legend(fontsize=8)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/capstone_decay_seed0.png", dpi=150, bbox_inches="tight")


import json
SEEDS = [0, 1, 2, 3, 4]
Path("results").mkdir(exist_ok=True)
for sd in SEEDS:
    for learn in (True,):
        tag = "learn" if learn else "control"
        f = Path(f"results/capstone_decay_seed{sd}_{tag}.json")
        if f.exists():
            print("skipping", f, "(already done)")
            continue
        path, visits = episode(learn, f"seed {sd} {tag}", SEED=sd)
        out = np.maximum(abs(path[:, 0]), abs(path[:, 1]))
        rows = [[float(v[0]), v[1], float(v[2]), float(v[3]), float(v[4]), float(v[5])] for v in visits]
        f.write_text(json.dumps({"visits": rows, "t_flip": T_FLIP, "t_total": T_TOTAL,
                                 "outside": float(np.mean(out > HALF)), "furthest": float(out.max())}))
        if learn and sd == SEEDS[0]:
            plot(path, visits)
        print(f"saved {f}: {len(visits)} visits")
