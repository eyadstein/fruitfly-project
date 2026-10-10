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
BASE, GAIN, WANDER = 0.8, 10.0, 0.3
HALF, MARGIN, REINF, WINDOW = 25.0, 6.0, 6.0, 800
T_FLIP, T_TOTAL, SEED = 40.0, 100.0, 0
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


import json
from pathlib import Path


def replay(visits, t_flip):
    w = M0.copy()
    out = []
    for x in visits:
        phase = 1 if x[0] < t_flip else 2
        w += RECOV * (M0 - w)
        reinforce(w, x[1], phase)
        out.append((valence(w, SA) / VREF, valence(w, SB) / VREF))
    return out


print("replay of each learning run's visit sequence through the learning rule")
print(f"{'seed':<6}{'max error':>10}{'B visits':>10}{'A after last B':>16}{'end B logged':>14}{'end B replay':>14}{'end B, B visits last':>22}")
rows = {}
for sd in range(10):
    f = Path(f"results/capstone_seed{sd}_learn.json")
    if not f.exists():
        continue
    d = json.load(open(f))
    v, tf = d["visits"], d["t_flip"]
    r = replay(v, tf)
    err = max(max(abs(a - x[2]), abs(b - x[3])) for (a, b), x in zip(r, v))
    post = [x for x in v if x[0] >= tf]
    reordered = [x for x in v if x[0] < tf] + [x for x in post if x[1] == "A"] + [x for x in post if x[1] == "B"]
    r2 = replay(reordered, tf)
    nb = sum(x[1] == "B" for x in post)
    last_b = max((i for i, x in enumerate(post) if x[1] == "B"), default=None)
    aft = sum(x[1] == "A" for x in post[last_b + 1:]) if last_b is not None else len(post)
    rows[sd] = (err, r[-1][1], r2[-1][1])
    print(f"{sd:<6}{err:>10.4f}{nb:>10}{aft:>16}{v[-1][3]:>+14.2f}{r[-1][1]:>+14.2f}{r2[-1][1]:>+22.2f}")

print("\nCRITERION (fixed in advance)")
worst = max(x[0] for x in rows.values())
print(f"replay check: largest error {worst:.4f} -> {'OK (under 0.02)' if worst < 0.02 else 'FAILED: do not use the counterfactual'}")
gain = {k: rows[k][2] - rows[k][1] for k in (3, 9) if k in rows}
print("seeds 3 and 9, gain in end B valence when the B visits come last:", {k: round(g, 2) for k, g in gain.items()})
if len(gain) == 2:
    print("verdict:", "WEARING-DOWN MATTERS (both gains >= 0.15)" if all(g >= 0.15 for g in gain.values()) else "WEARING-DOWN IS SMALL (not both >= 0.15)")
