from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation, HybridTurningController, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)

HALF, MARGIN, RUN_TIME, WINDOW = 20.0, 6.0, 15.0, 1000   # box is 2*HALF mm wide
SIGN, BASE, WALL_GAIN, TAU, SIGMA = 1.0, 0.8, 1.0, 1.0, 0.4

fly = make_locomotion_fly(name="cage", add_adhesion=True, colorize=True)
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
sim.reset(); controller.reset(seed=0)
apply_locomotion_action(sim, fly.name, LocomotionAction(
    joint_angles=steps.default_pose_by_dof_order(dof_order),
    adhesion_onoff=np.ones(6, dtype=bool)))
sim.warmup()

thorax = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
start = sim.get_body_positions(fly.name)[thorax][:2].copy()
n = int(RUN_TIME / sim.timestep)
every = int(0.01 / sim.timestep)          # the "brain" decides every 10 ms
rng = np.random.default_rng(0)
path = np.zeros((n, 2))
bias, drive = 0.0, np.array([1.0, 1.0])
for i in trange(n, desc="in the cage"):
    p = sim.get_body_positions(fly.name)[thorax][:2] - start
    path[i] = p
    if i % every == 0:
        if i >= WINDOW:
            dx, dy = p - path[i - WINDOW]
            heading = np.arctan2(dy, dx)
        else:
            heading = 0.0
        if i >= WINDOW and max(abs(p[0]), abs(p[1])) > HALF - MARGIN:
            err = (np.arctan2(-p[1], -p[0]) - heading + np.pi) % (2 * np.pi) - np.pi
            turn = np.clip(WALL_GAIN * err, -0.8, 0.8)          # head back toward the centre
        else:
            bias += -bias * 0.01 / TAU + SIGMA * np.sqrt(0.01) * rng.normal()
            turn = bias                                          # free wandering
        drive = np.clip([BASE - SIGN * turn, BASE + SIGN * turn], 0.4, 1.2)
    obs = HybridControllerObservation.from_sim(sim, fly.name)
    apply_locomotion_action(sim, fly.name, controller.step(drive, obs))
    sim.step()

outside = np.mean(np.maximum(abs(path[:, 0]), abs(path[:, 1])) > HALF)
cells = {(int((x + HALF) // 5), int((y + HALF) // 5)) for x, y in path
         if abs(x) < HALF and abs(y) < HALF}
print(f"time outside the box: {outside:.1%}")
print(f"furthest from centre: {np.max(np.maximum(abs(path[:, 0]), abs(path[:, 1]))):.1f} mm (wall at {HALF} mm)")
print(f"arena coverage: {len(cells)}/64 cells of 5 mm")

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(path[::50, 0], path[::50, 1], c=np.arange(0, n, 50) * sim.timestep, s=4, cmap="viridis")
ax.add_patch(plt.Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False, color="red", linewidth=2))
ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
fig.colorbar(sc, label="time (s)"); ax.set_title("Fly in a 40 mm cage")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/cage.png", dpi=150, bbox_inches="tight")
print("saved figures/cage.png")
