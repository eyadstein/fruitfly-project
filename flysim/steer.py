from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

SIGN = 1.0          # flip to -1.0 if the fly turns away from the target
BASE, GAIN = 0.8, 0.8
RUN_TIME = 4.0
WINDOW = 500        # steps used to estimate heading from movement

out = Path("flysim/output")
out.mkdir(parents=True, exist_ok=True)

fly = make_locomotion_fly(name="steer", add_adhesion=True, colorize=True)
cam = fly.add_tracking_camera(
    name="body_cam",
    pos_offset=(-0.5, -7.5, 0.0),
    rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
    fovy=35.0,
)
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
              bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
              add_ground_contact_sensors=False)
sim = Simulation(world)
renderer = sim.set_renderer([cam], camera_res=(240, 320), playback_speed=0.2, output_fps=25)

steps = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")
controller = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps,
                                     output_dof_order=dof_order)

sim.reset()
controller.reset(seed=0)
apply_locomotion_action(
    sim, fly.name,
    LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dof_order),
                     adhesion_onoff=np.ones(6, dtype=bool)),
)
sim.warmup()

thorax = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
start = sim.get_body_positions(fly.name)[thorax][:2].copy()
target = start + np.array([10.0, 8.0])   # mm: 10 ahead, 8 to the left

n = int(RUN_TIME / sim.timestep)
path = np.zeros((n, 2))
drives = np.zeros((n, 2))
for i in trange(n, desc="steering"):
    path[i] = sim.get_body_positions(fly.name)[thorax][:2]
    if i >= WINDOW:
        dx, dy = path[i] - path[i - WINDOW]
        heading = np.arctan2(dy, dx)
        bearing = np.arctan2(target[1] - path[i, 1], target[0] - path[i, 0])
        err = (bearing - heading + np.pi) % (2 * np.pi) - np.pi   # >0: target is to the left
        left = BASE - SIGN * GAIN * err
        right = BASE + SIGN * GAIN * err
    else:
        left = right = 1.0
    drive = np.clip([left, right], 0.4, 1.2)
    drives[i] = drive
    obs = HybridControllerObservation.from_sim(sim, fly.name)
    apply_locomotion_action(sim, fly.name, controller.step(drive, obs))
    sim.step()
    sim.render_as_needed()

dist = np.linalg.norm(path - target, axis=1)
print(f"start distance to target: {dist[0]:.1f} mm")
print(f"closest approach:         {dist.min():.1f} mm at t={dist.argmin() * sim.timestep:.2f} s")
print(f"final distance:           {dist[-1]:.1f} mm")
sim.renderer.save_video(out / "steer.mp4")

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(path[:, 0] - start[0], path[:, 1] - start[1], label="fly path")
ax.scatter(*(target - start), color="red", s=80, label="target")
ax.scatter(0, 0, color="green", s=40, label="start")
ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.legend()
fig.savefig(out / "steer_path.png", dpi=150, bbox_inches="tight")
print("saved", out / "steer.mp4", "and", out / "steer_path.png")
