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

SIGN, BASE = 1.0, 0.8
DIST, ARRIVE, RUN_TIME, WINDOW = 12.8, 3.0, 4.0, 500
ANGLES = [-75, -45, -15, 15, 45, 75]
SEEDS = [0, 1, 2]


def plain_error(pos, heading, source):
    lp, rp = antenna_positions(pos, heading)
    cl, cr = concentration(lp, source), concentration(rp, source)
    return (cl - cr) / (cl + cr + 1e-9)


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


REAL_W = cs.W.copy()
SHUF_W = shuffled_W(REAL_W, np.random.default_rng(0))


def set_wiring(M):
    cs.W = M
    cs.THETA = np.percentile(M @ (cs.C_REF * cs.ODOUR), 50)


fly = make_locomotion_fly(name="cmp", add_adhesion=True, colorize=True)
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


def trial(err_fn, gain, angle, seed):
    sim.reset(); controller.reset(seed=seed)
    apply_locomotion_action(sim, fly.name, init)
    sim.warmup()
    start = sim.get_body_positions(fly.name)[thorax][:2].copy()
    a = np.deg2rad(angle)
    target = DIST * np.array([np.cos(a), np.sin(a)])
    path = np.zeros((n_steps, 2))
    dmin, t_arr = 1e9, np.nan
    for i in range(n_steps):
        p = sim.get_body_positions(fly.name)[thorax][:2] - start
        path[i] = p
        d = np.linalg.norm(p - target)
        dmin = min(dmin, d)
        if d < ARRIVE:
            t_arr = i * sim.timestep
            break
        if i >= WINDOW:
            dx, dy = p - path[i - WINDOW]
            heading = np.arctan2(dy, dx)
            err = err_fn(p, heading, target)
            drive = np.clip([BASE - SIGN * gain * err, BASE + SIGN * gain * err], 0.4, 1.2)
        else:
            drive = np.array([1.0, 1.0])
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(drive, obs))
        sim.step()
    return dmin, t_arr


conds = {
    "plain odour (gain 3.8)": (plain_error, 3.8, REAL_W),
    "circuit real wiring (gain 3.0)": (cs.circuit_error, 3.0, REAL_W),
    "circuit shuffled wiring (gain 3.0)": (cs.circuit_error, 3.0, SHUF_W),
}
res = {c: [] for c in conds}
jobs = [(s, a, c) for s in SEEDS for a in ANGLES for c in conds]
for s, a, c in tqdm(jobs, desc="trials"):
    fn, gain, M = conds[c]
    set_wiring(M)
    res[c].append(trial(fn, gain, a, s))

print(f"\n{len(ANGLES)} angles x {len(SEEDS)} seeds per controller, target {DIST} mm away, arrive = within {ARRIVE} mm")
print(f"{'controller':<38}{'arrived':>9}{'median time':>13}{'mean closest':>14}")
for c, r in res.items():
    d = np.array([x[0] for x in r]); t = np.array([x[1] for x in r])
    ok = ~np.isnan(t)
    med = f"{np.median(t[ok]):.2f} s" if ok.any() else "-"
    print(f"{c:<38}{ok.sum():>5}/{len(r):<3}{med:>13}{d.mean():>11.1f} mm")
