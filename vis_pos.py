import pybullet as p
import pybullet_data
import time
import numpy as np

# --- SETTINGS ---
URDF_PATH = "/Users/sissi/Downloads/hora/assets/robot.urdf"

# Paste one row from your .npy file here (20 joint angles)
# This matches your 20-DOF canonical_pose structure
test_angles = [0.01, -0.36, 0.02, 1.66, 0.0, 0.0, 1.54, 0.2, 0.2, -0.23, 1.65, 0.0, 0.0, -0.3, 1.72, 0.16, 0.16, 0.66, -0.76, 0.0]

def run_test():
    # Start PyBullet in GUI mode
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load Plane and Robot
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.1], useFixedBase=True)

    # 1. Identify "Revolute" (motorized) joints
    # URDFs often have fixed joints; we must skip them to match your 20-angle array
    revolute_joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_type = info[2]
        if joint_type == p.JOINT_REVOLUTE:
            revolute_joint_indices.append(i)

    print(f"Found {len(revolute_joint_indices)} revolute joints.")

    # 2. Apply the angles
    if len(revolute_joint_indices) != len(test_angles):
        print(f"ERROR: URDF has {len(revolute_joint_indices)} joints, but array has {len(test_angles)}")
    else:
        for i, joint_idx in enumerate(revolute_joint_indices):
            p.resetJointState(robot_id, joint_idx, test_angles[i])
        print("Pose applied successfully.")

    # Keep the window open
    print("Close the window or press Ctrl+C to exit.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    run_test()