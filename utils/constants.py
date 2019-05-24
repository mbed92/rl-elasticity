very_close_position = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -0.5,
    "elbow_joint": 1.0,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0
}

base_position = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.7,
    "elbow_joint": 1.0,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0
}

start_qpos = base_position


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

initial_keep_random = 0.5
train_log = ('.', 'logs')
model_dir = ('.', 'saved')
output_path = ('.', 'saved')
model_nn = (*model_dir, 'ckpt')
mujoco_model = ('.', 'models', 'ur5', 'UR5gripper.xml')
x_range = (0.4, 0.6)
y_range = (-0.5, 0.5)
z_range = (0.5, 0.8)