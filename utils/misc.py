import os
import tensorflow as tf
import tensorflow.contrib as tfc
import mujoco_py


def setup_writer(log_path):
    os.makedirs(log_path, exist_ok=True)
    return tfc.summary.create_file_writer(log_path)


def setup_environment(mujoco_model_path):
    scene = mujoco_py.load_model_from_path(mujoco_model_path)
    env = mujoco_py.MjSim(scene)
    viewer = mujoco_py.MjRenderContextOffscreen(env, 0)
    return env, viewer


def setup_optimizer(restore_path, lr, model):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    if restore_path is not "":
        ckpt.restore(tf.train.latest_checkpoint(restore_path))
        print("Weight successfully restored")

    return optimizer, ckpt


def update_keep_random(initial_keep_random, epoch, epochs):
    return initial_keep_random + ((epoch / epochs) * (1 - initial_keep_random))
