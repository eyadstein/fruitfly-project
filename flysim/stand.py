from pathlib import Path

from tqdm import trange

from flygym import Simulation
from flygym.anatomy import ActuatedDOFPreset, AxisOrder, ContactBodiesPreset, JointPreset, Skeleton
from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, NeuroMechFly
from flygym.utils.math import Rotation3D

out = Path("flysim/output")
out.mkdir(parents=True, exist_ok=True)

# 1. build the fly: body, joints, position-controlled leg actuators
fly = NeuroMechFly()
skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
dofs = fly.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
fly.add_actuators(dofs, actuator_type=ActuatorType.POSITION, kp=50.0,
                  neutral_input=KinematicPosePreset.NEUTRAL)
fly.colorize()
cam = fly.add_tracking_camera()

# 2. put it on flat ground
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
              bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)

# 3. simulate and record
sim = Simulation(world)
sim.set_renderer(cam, playback_speed=0.2, output_fps=25)
sim.reset()
sim.warmup()
for _ in trange(3000):
    sim.step()
    sim.render_as_needed()
sim.renderer.save_video(out / "stand.mp4")
print("saved", out / "stand.mp4")
