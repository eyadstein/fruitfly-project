from pathlib import Path

import numpy as np
from tqdm import trange

from flygym import Simulation
from flygym.anatomy import BodySegment
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    CPGController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
    make_tripod_cpg_network,
)

out = Path("flysim/output")
out.mkdir(parents=True, exist_ok=True)

# fly with adhesion (so the feet can grip the ground) and a camera that follows it
fly = make_locomotion_fly(name="walker", add_adhesion=True, colorize=True)
cam = fly.add_tracking_camera(
    name="body_cam",
    pos_offset=(-0.5, -7.5, 0.0),
    rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
    fovy=30.0,
)

world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
sim = Simulation(world)
renderer = sim.set_renderer([cam])

# central pattern generator: six coupled oscillators in a tripod gait
steps = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")
cpg = make_tripod_cpg_network(
    timestep=sim.timestep,
    intrinsic_amplitude=1.0,
    coupling_strength=10.0,
    convergence_coef=20.0,
    seed=0,
)
controller = CPGController(cpg_network=cpg, preprogrammed_steps=steps, output_dof_order=dof_order)

# default pose, adhesion on, let the body settle
sim.reset()
apply_locomotion_action(
    sim, fly.name,
    LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dof_order),
                     adhesion_onoff=np.ones(6, dtype=bool)),
)
sim.warmup()

n = int(2.0 / sim.timestep)  # 2 seconds of simulated time
thorax = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
x0 = sim.get_body_positions(fly.name)[thorax][0]
for _ in trange(n, desc="walking"):
    apply_locomotion_action(sim, fly.name, controller.step())
    sim.step()
    sim.render_as_needed()
x1 = sim.get_body_positions(fly.name)[thorax][0]

print(f"forward displacement: {x1 - x0:.1f} mm")
sim.renderer.save_video(out / "walk.mp4")
print("saved", out / "walk.mp4")
