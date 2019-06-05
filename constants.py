import numpy as np


home = {
    "shoulder_pan_joint": -0.346,
    "shoulder_lift_joint": -1.32,
    "elbow_joint": 2.29,
    "wrist_1_joint": -0.943,
    "wrist_2_joint": 1.13,
    "wrist_3_joint": 0.0,
    "gripperpalm_finger_1_joint": 0.0,
    "gripperfinger_1_joint_1": 0.0,
    "gripperfinger_1_joint_2": 1.12,
    "gripperfinger_1_joint_3": 0,
    "gripperpalm_finger_2_joint": -0.0445,
    "gripperfinger_2_joint_1": 0.196,
    "gripperfinger_2_joint_2": 1.15,
    "gripperfinger_2_joint_3": 0.602,
    "gripperpalm_finger_middle_joint": 0.0,
    "gripperfinger_middle_joint_1": 1.08,
    "gripperfinger_middle_joint_2": 0.346,
    "gripperfinger_middle_joint_3": 0.38,
}

random_reach = {
    "shoulder_pan_joint": -0.35 + np.random.uniform(-0.3, 0.3),
    "shoulder_lift_joint": -1.7 + np.random.uniform(-0.3, 0.3),
    "elbow_joint": 1.7 + np.random.uniform(-0.3, 0.3),
    "wrist_1_joint": 0.0 + np.random.uniform(-1, 1),
    "wrist_2_joint": 0.0 + np.random.uniform(-1, 1),
    "wrist_3_joint": 0.0 + np.random.uniform(-1, 1),
    "gripperpalm_finger_1_joint": 0.0,
    "gripperfinger_1_joint_1": 0.0,
    "gripperfinger_1_joint_2": 0.0,
    "gripperfinger_1_joint_3": 0.0,
    "gripperpalm_finger_2_joint": 0.0,
    "gripperfinger_2_joint_1": 0.0,
    "gripperfinger_2_joint_2": 0.0,
    "gripperfinger_2_joint_3": 0.0,
    "gripperpalm_finger_middle_joint": 0.0,
    "gripperfinger_middle_joint_1": 0.0,
    "gripperfinger_middle_joint_2": 0.0,
    "gripperfinger_middle_joint_3": 0.0,
}

start_qpos = home


gripper_close = {
    "gripperfinger_1_joint_1": 0.5,
    "gripperfinger_1_joint_2": 0.7,
    "gripperfinger_1_joint_3": 0.2,
    "gripperfinger_2_joint_1": 0.5,
    "gripperfinger_2_joint_2": 0.7,
    "gripperfinger_2_joint_3": 0.2,
    "gripperfinger_middle_joint_1": 0.5,
    "gripperfinger_middle_joint_2": 0.7,
    "gripperfinger_middle_joint_3": 0.2
}
