import os
import tensorflow.contrib as tfc
import mujoco_py
from .constants import model_nn, train_log, mujoco_model


def setup_writer():
    train_log_path = os.path.join(*train_log)
    checkpoint_prefix = os.path.join(*model_nn)
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(checkpoint_prefix, exist_ok=True)
    return tfc.summary.create_file_writer(train_log_path)


def setup_environment():
    path = os.path.join(*mujoco_model)
    scene = mujoco_py.load_model_from_path(path)
    env = mujoco_py.MjSim(scene)
    viewer = mujoco_py.MjRenderContextOffscreen(env, 0)
    return env, viewer
