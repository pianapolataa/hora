import pybullet as p
import pybullet_data
import time
import numpy as np

# --- SETTINGS ---
URDF_PATH = "/Users/sissi/Downloads/hora/assets/robot.urdf"

# The latest angles provided
test_angles = [
            0.27,      # base pitch -0.661974 0.647023
            0.17,      # wrist_yaw
            -0.16,     # index_splay
            0.77,       # index_mcp (0, 1.9)
            0.32,      # index_pip
            0.32,      # index_dip
            0.77,      # mid_mcp -0.2 1.91986
            0.33,      # mid_pip
            0.33,      # mid_dip
            -0.3,      # ring_splay
            0.77,      # ring_mcp 0.0 1.91986
            0.32,       # ring_pip
            0.32,       # ring_dip
            -0.12,     # pinky_splay
            0.77,      # pinky_mcp 0.0 1.91986
            0.32,       # pinky_pip
            0.32,       # pinky_dip
            0.5,       # thumb_cmc
            -0.53,     # thumb_mcp
            0.2,       # thumb_ip
        ]

def run_test():
    # Start PyBullet in GUI mode
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Increase solver iterations for better contact stability
    p.setPhysicsEngineParameter(numSolverIterations=100)
    p.setGravity(0, 0, -9.81)
    
    # Load Plane and Robot
    p.loadURDF("plane.urdf")
    # Set the hand slightly higher to give the ball room to fall if the grasp fails
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.5], useFixedBase=True)

    # 1. Identify "Revolute" joints
    revolute_joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[2] == p.JOINT_REVOLUTE:
            revolute_joint_indices.append(i)

    # 2. Apply the angles BEFORE unpausing
    if len(revolute_joint_indices) == len(test_angles):
        for i, joint_idx in enumerate(revolute_joint_indices):
            p.resetJointState(robot_id, joint_idx, test_angles[i])
            # Set motors to hold this position
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=test_angles[i])
    
    # 3. Load the Tennis Ball
    # Based on your script's logic: small_tennis_ball, mass ~0.05
    ball_radius = 0.04 # Standard tennis ball is ~6.7cm diameter, radius ~0.033m
    visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 1, 0, 1])
    collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    
    # Position the ball in the center of the palm
    # In many Allegro/Ruka setups, the palm is at the origin of the URDF
    ball_id = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0.01, 0.12, 0.6], # Slightly above the palm center
    )

    # Add friction to both hand and ball to simulate the real physics
    p.changeDynamics(robot_id, -1, lateralFriction=1.0)
    for j in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, j, lateralFriction=1.0)
    p.changeDynamics(ball_id, -1, lateralFriction=1.0, rollingFriction=0.01)

    print("Simulation running. If the ball stays in the hand, the grasp holds.")
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    run_test()