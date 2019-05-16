import mujoco_py
import os
from agents import Core
import tensorflow.contrib as tfc
import tensorflow as tf
import numpy as np
from utils import *


# load scene in MuJoCo
path = os.path.join('.', 'models', 'ur5', 'UR5gripper.xml')
env = mujoco_py.load_model_from_path(path)
sim = mujoco_py.MjSim(env)
viewer = mujoco_py.MjRenderContextOffscreen(sim, 0)

# setup tf env
optimizer = tf.train.AdamOptimizer(0.0001)
l2_reg = tf.keras.regularizers.l2(1e-4)
train_log_path = os.path.join('.', 'logs')
train_writer = tfc.summary.create_file_writer(train_log_path)
model = Core(6)

for epoch in range(10):
    sim.reset()

    # for w in range(50):
    #     viewer.render(420, 380, 0)
    #     img = np.asarray(viewer.read_pixels(420, 380, depth=False)[::-1, :, :], dtype=np.uint8)
    #     tool = get_tool_pose(sim)
    #     target = get_target_pose(sim)
    #     reward = get_reward(sim, tool, target)
    #
    # with tf.GradientTape() as tape:
    #     controls = model(img, True)
    #     loss = ...
    #
    # grads = tape.gradient(loss, model.trainable_variables)
    # optimizer.apply_gradients(zip(grads, model.trainable_variables),
    #                           global_step=tf.train.get_or_create_global_step())
    #
    # sim.step()

# # to speed up computation we need the off screen rendering
# # viewer = mujoco_py.MjRenderContextOffscreen(sim, 0)
# for i in range(10000):
#     # viewer.render(420, 380, 0)
#     # data = np.asarray(viewer.read_pixels(420, 380, depth=False)[::-1, :, :], dtype=np.uint8)
#     # state = sim.get_state()
#     #
#     # compute reward
#     tool = get_tool_pose(sim)
#     target = get_target_pose(sim)
#     reward = get_reward(sim, tool, target)
#
#     # break
#     # # save data
#     # if data is not None:
#     #     cv2.imwrite("test{0}.png".format(i), data)
#     # reward = sim.data.userdata[0]
#     print(reward)
#
#     viewer.render()
#     sim.data.ctrl[4] = sim.data.ctrl[4] + 0.0001
#     sim.step()
