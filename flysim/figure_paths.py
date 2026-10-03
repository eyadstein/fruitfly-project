from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation, HybridTurningController, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)
import circuit_steer as cs
from odour_steer import antenna_positions, concentration

SIGN, BASE, GAIN = 1.0, 0.8, 3.0
DIST_A, DIST_B, ARRIVE, RUN_TIME, WINDOW = 12.8, 8.0, 3.0, 4.0, 500
ANGLES, SEED = [-45, 45], 0

REAL_W = cs.W.copy()
OA = cs.ODOUR.copy()
OB = np.random.default_rng(1).random(REAL_W.shape[1])


def shuffled_W(W, rng, swaps=10):
    ks, ps = np.nonzero(W)
    w = W[ks, ps].copy()
    ks, ps = ks.tolist(), ps.tolist()
    edges, E = set(zip(ks, ps)), len(ks)
    for _ in range(swaps * E):
        i, j = rng.integers(E, size=2)
        k1, p1, k2, p2 = ks[i], ps[i], ks[j], ps[j]
        if k1 == k2 or p1 == p2 or (k1, p2) in edges or (k2, p1) in edges:
            continue
        edges.remove((k1, p1)); edges.remove((k2, p2))
        edges.add((k1, p2)); edges.add((k2, p1))
        ps[i], ps[j] = p2, p1
    S = np.zeros_like(W)
    S[ks, ps] = w
    return S


def make_circuit(W):
    theta = np.percentile(W @ (cs.C_REF * (OA + OB) / 2), 50)
    K = int(0.05 * W.shape[0])

    def kc(pn):
        drive = W @ pn - theta
        idx = np.argpartition(drive, -K)[-K:]
        act = np.zeros(W.shape[0])
        act[idx] = np.maximum(drive[idx], 0.0)
        return act

    r = (kc(cs.C_REF * OA) > 0).astype(float) - (kc(cs.C_REF * OB) > 0).astype(float)
    return lambda pn: float(r @ kc(pn))


scorers = {
    "plain concentration": lambda pn: float(pn.sum()),
    "circuit, real wiring": make_circuit(REAL_W),
    "circuit, shuffled wiring": make_circuit(shuffled_W(REAL_W, np.random.default_rng(0))),
}
style = {"plain concentration": ("tab:gray", "-"),
         "circuit, real wiring": ("tab:blue", "-"),
         "circuit, shuffled wiring": ("tab:orange", "--")}

fly = make_locomotion_fly(name="fig", add_adhesion=True, colorize=True)
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


def trial(score_fn, angle):
    sim.reset(); controller.reset(seed=SEED)
    apply_locomotion_action(sim, fly.name, init)
    sim.warmup()
    start = sim.get_body_positions(fly.name)[thorax][:2].copy()
    a = np.deg2rad(angle)
    tA = DIST_A * np.array([np.cos(a), np.sin(a)])
    tB = DIST_B * np.array([np.cos(-a), np.sin(-a)])
    field = lambda p: concentration(p, tA) * OA + concentration(p, tB) * OB
    path = np.zeros((n_steps, 2))
    for i in range(n_steps):
        p = sim.get_body_positions(fly.name)[thorax][:2] - start
        path[i] = p
        if np.linalg.norm(p - tA) < ARRIVE:
            return path[: i + 1], "reached A", tA, tB
        if np.linalg.norm(p - tB) < ARRIVE:
            return path[: i + 1], "reached B", tA, tB
        if i >= WINDOW:
            dx, dy = p - path[i - WINDOW]
            lp, rp = antenna_positions(p, np.arctan2(dy, dx))
            sl, sr = score_fn(field(lp)), score_fn(field(rp))
            err = (sl - sr) / (abs(sl) + abs(sr) + 1e-9)
            drive = np.clip([BASE - SIGN * GAIN * err, BASE + SIGN * GAIN * err], 0.4, 1.2)
        else:
            drive = np.array([1.0, 1.0])
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(drive, obs))
        sim.step()
    return path, "timed out", tA, tB


fig, axes = plt.subplots(1, len(ANGLES), figsize=(11, 5), sharey=True)
for ax, ang in zip(axes, ANGLES):
    for name, fn in tqdm(scorers.items(), desc=f"angle {ang}"):
        path, outcome, tA, tB = trial(fn, ang)
        c, ls = style[name]
        ax.plot(path[:, 0], path[:, 1], color=c, linestyle=ls, linewidth=2, label=f"{name}: {outcome}")
    ax.scatter(*tA, marker="*", s=300, color="green", zorder=5)
    ax.scatter(*tB, marker="X", s=140, color="red", zorder=5)
    for t, col in ((tA, "green"), (tB, "red")):
        ax.add_patch(plt.Circle(t, ARRIVE, fill=False, color=col, alpha=0.5))
    ax.scatter(0, 0, color="black", s=40, zorder=5)
    ax.text(*(tA + [0.4, 0.8]), "A (rewarded)", color="green")
    ax.text(*(tB + [0.4, -1.6]), "B (distractor)", color="red")
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)")
    ax.set_title(f"A at {ang:+d} deg, B mirrored and closer")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
axes[0].set_ylabel("y (mm)")
fig.suptitle("Two-odour steering (one example per angle): KC-code circuits reach the rewarded odour")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/two_odour_paths.png", dpi=150, bbox_inches="tight")
print("saved figures/two_odour_paths.png")
