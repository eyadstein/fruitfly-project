from pathlib import Path
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

BASE, GAIN, DIST, ARRIVE, RUN_TIME, WINDOW = 0.8, 10.0, 12.8, 3.0, 3.0, 500
ANGLES, SEEDS = [-45, 45], [0, 1]
CW, CI = Path("connectome/brain_W.npz"), Path("connectome/brain_idx.npz")


def read(p):
    return pd.read_csv(glob.glob(f"connectome/data/{p}*")[0])


def build_brain():
    conn, cls = read("conn"), read("class")
    conn["s"] = conn["nt_type"].astype(str).str.upper().map({"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0})
    conn = conn.dropna(subset=["s"])
    conn["w"] = conn["syn_count"] * conn["s"]
    e = conn.groupby(["pre_root_id", "post_root_id"], as_index=False)["w"].sum()
    ids = pd.Index(np.unique(np.concatenate([e["pre_root_id"].values, e["post_root_id"].values])))
    N = len(ids)
    pre, post = ids.get_indexer(e["pre_root_id"]), ids.get_indexer(e["post_root_id"])
    W = sp.csr_matrix((e["w"].values.astype(np.float32), (post, pre)), shape=(N, N))
    absrow = np.asarray(abs(W).sum(axis=1)).ravel()
    W = (sp.diags(1.0 / np.maximum(absrow, 1e-9)) @ W).tocsr().astype(np.float32)

    def sel(mask, side):
        idx = ids.get_indexer(cls.loc[mask & (cls["side"] == side), "root_id"])
        return idx[idx >= 0]

    olf, dn = cls["class"] == "olfactory", cls["super_class"] == "descending"
    sp.save_npz(CW, W)
    np.savez(CI, OL=sel(olf, "left"), OR=sel(olf, "right"), DL=sel(dn, "left"), DR=sel(dn, "right"))


class Brain:
    def __init__(self, W, OL, OR, DL, DR, g=0.9):
        self.W, self.OL, self.OR, self.DL, self.DR, self.g = W, OL, OR, DL, DR, g
        self.N = W.shape[0]
        self.I = np.zeros(self.N, np.float32)
        self.r = np.zeros(self.N, np.float32)
        self.reset()

    def step(self, left, right, iters=5):
        self.I[:] = 0
        self.I[self.OL], self.I[self.OR] = left, right
        for _ in range(iters):
            self.r = np.maximum(self.g * (self.W @ self.r) + self.I, 0.0)
        dl, dr = self.r[self.DL].sum(), self.r[self.DR].sum()
        return (dl - dr) / (dl + dr + 1e-12)

    def reset(self):
        self.r[:] = 0
        for _ in range(60):
            self.step(1.0, 1.0, 1)
        self.base = self.step(1.0, 1.0, 1)


if not CW.exists():
    print("building and caching the brain model (first run only)...")
    build_brain()
W = sp.load_npz(CW).tocsr()
d = np.load(CI)
real = Brain(W, d["OL"], d["OR"], d["DL"], d["DR"])
perm = np.random.default_rng(0).permutation(W.shape[0])
sizes = [len(d[k]) for k in ("OL", "OR", "DL", "DR")]
cut = np.cumsum([0] + sizes)
rand = Brain(W, *[perm[cut[i]:cut[i + 1]] for i in range(4)])
print(f"baseline bias: real {real.base:.3f}, random control {rand.base:.3f}")

fly = make_locomotion_fly(name="brain", add_adhesion=True, colorize=True)
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
every = int(0.01 / sim.timestep)


def trial(brain, sign, gain, angle, seed):
    sim.reset(); controller.reset(seed=seed)
    apply_locomotion_action(sim, fly.name, init)
    sim.warmup()
    if brain is not None:
        brain.reset()
    start = sim.get_body_positions(fly.name)[thorax][:2].copy()
    a = np.deg2rad(angle)
    target = DIST * np.array([np.cos(a), np.sin(a)])
    path = np.zeros((n_steps, 2))
    dmin, t_arr, drive = 1e9, np.nan, np.array([1.0, 1.0])
    for i in range(n_steps):
        p = sim.get_body_positions(fly.name)[thorax][:2] - start
        path[i] = p
        dist = np.linalg.norm(p - target)
        dmin = min(dmin, dist)
        if dist < ARRIVE:
            t_arr = i * sim.timestep
            break
        if i % every == 0 and i >= WINDOW:
            dx, dy = p - path[i - WINDOW]
            lp, rp = antenna_positions(p, np.arctan2(dy, dx))
            cl, cr = concentration(lp, target), concentration(rp, target)
            tot = cl + cr + 1e-9
            if brain is None:
                err = (cl - cr) / tot
            else:
                err = brain.step(2 * cl / tot, 2 * cr / tot) - brain.base
            drive = np.clip([BASE - sign * gain * err, BASE + sign * gain * err], 0.4, 1.2)
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(drive, obs))
        sim.step()
    return dmin, t_arr


conds = {
    "plain odour (reference, gain 3.8)": (None, 1.0, 3.8),
    "real brain, DN-left turns left": (real, 1.0, GAIN),
    "real brain, DN-left turns right": (real, -1.0, GAIN),
    "random in/out neurons, turns left": (rand, 1.0, GAIN),
    "random in/out neurons, turns right": (rand, -1.0, GAIN),
}
res = {c: [] for c in conds}
jobs = [(s, a, c) for s in SEEDS for a in ANGLES for c in conds]
for s, a, c in tqdm(jobs, desc="trials"):
    b, sg, g = conds[c]
    res[c].append(trial(b, sg, g, a, s))

print(f"\n{len(ANGLES)} angles x {len(SEEDS)} seeds, target {DIST} mm away, arrive = within {ARRIVE} mm")
print(f"{'controller':<38}{'arrived':>9}{'median time':>13}{'mean closest':>14}")
for c, r in res.items():
    dm = np.array([x[0] for x in r]); t = np.array([x[1] for x in r])
    ok = ~np.isnan(t)
    med = f"{np.median(t[ok]):.2f} s" if ok.any() else "-"
    print(f"{c:<38}{ok.sum():>5}/{len(r):<3}{med:>13}{dm.mean():>11.1f} mm")
