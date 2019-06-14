from argparse import ArgumentParser

import cv2

from agents import ContinuousAgent
from environment import ManEnv
from utils import *

tf.enable_eager_execution()
tf.executing_eagerly()

# remove tensorflow warning "tried to deallocate nullptr"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(args):
    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)
    train_writer = setup_writer(args.logs_path)
    train_writer.set_as_default()

    # make the policy network
    model = ContinuousAgent(num_controls=env.num_actions)

    # setup optimizer
    eta = tfc.eager.Variable(args.learning_rate)
    eta_f = tf.train.exponential_decay(
        args.learning_rate,
        tf.train.get_or_create_global_step(),
        args.epochs,
        0.98)
    eta.assign(eta_f())
    optimizer, ckpt = setup_optimizer(args.restore_path, eta, model)
    l2_reg = tf.keras.regularizers.l2(1e-6)

    # run training
    for n in range(args.epochs):
        batch_rewards = []
        batch_sums = []
        ep_rewards = []       # list for rewards accrued throughout ep
        ep_log_grad = []      # list of log-likelihood gradients
        total_gradient = []   # list of gradients multiplied by rewards per epochs
        keep_random = update_keep_random(args.keep_random, n, args.epochs)

        # domain randomization after each epoch
        env.randomize_environment()
        env.reset()

        # start trajectory
        t, trajs = 0, 0
        while True:
            rgb, poses = env.get_observations()
            if rgb[0] is not None and poses is not None:
                pass
                # cv2.imshow("trajectory", rgb[0])
                # cv2.waitKey(1)
            else:
                continue

            # take action in the environment under the current policy
            with tf.GradientTape(persistent=True) as tape:
                ep_mean_act, ep_log_dev = model([rgb, poses], True)
                ep_std_dev, ep_variance = tf.exp(ep_log_dev) + 1e-05, tf.square(tf.exp(ep_log_dev)) + 1e-05
                actions = env.take_continuous_action(ep_mean_act, ep_std_dev, keep_random)
                env.step()
                ep_rew, distance = env.get_reward(actions)
                ep_rewards.append(ep_rew)

                # optimize a mean and a std_dev - compute gradient from logprob
                loss_value = tf.log((1 / ep_std_dev) + 1e-05) - (1 / ep_variance) * tf.losses.mean_squared_error(ep_mean_act, actions)
                loss_value = tf.reduce_mean(loss_value)

                # do not let to optimize when loss -> NaN
                if tf.is_nan(loss_value):
                    print("Loss is nan: ep_mean_act {0}, ep_log_dev: {1}".format(ep_mean_act, ep_log_dev))
                    break

                # apply regularization
                reg_value = tfc.layers.apply_regularization(l2_reg, model.trainable_variables)
                loss = loss_value + reg_value

            # compute and store gradients
            grads = tape.gradient(loss, model.trainable_variables)
            # grads = [tf.clip_by_value(g, -1., 1.) for g in grads]
            ep_log_grad.append(grads)
            t += 1

            # compute grad log-likelihood for a current episode
            if is_done(distance, ep_rew, t, args):
                if len(ep_rewards) > 5 and len(ep_rewards) == len(ep_log_grad):
                    ep_rewards = bound_to_nonzero(ep_rewards)
                    # ep_rewards = reward_to_go(ep_rewards)
                    ep_rewards = discount_rewards(ep_rewards)
                    ep_rewards = standardize_rewards(ep_rewards)
                    ep_reward_sum, ep_reward_mean = sum(ep_rewards), mean(ep_rewards)
                    batch_rewards.append(ep_reward_mean)
                    batch_sums.append(ep_reward_mean)

                    # gradient[i] * (reward - b)
                    for i, (grad, reward) in enumerate(zip(ep_log_grad, ep_rewards)):
                        # ep_log_grad[i] = [tf.multiply(g, (reward - ep_reward_mean)) for g in grad]
                        ep_log_grad[i] = [tf.multiply(g, reward) for g in grad]

                    # sum over one trajectory
                    trajectory_gradient = []
                    for i, grad in enumerate(ep_log_grad):
                        trajectory_gradient = [tf.add(a, b) for a, b in zip(trajectory_gradient, grad)] if i > 0 else grad

                    # sum over all trajectories
                    total_gradient = [tf.add(prob, prev_prob) for prob, prev_prob in zip(total_gradient, trajectory_gradient)] if len(total_gradient) > 0 else trajectory_gradient
                    print("Episode is done! Mean reward: {0}, keep random ratio: {1}, distance: {2}".format(ep_reward_mean, keep_random, distance))
                    trajs += 1
                    if trajs >= args.update_step:
                        break

                # reset episode-specific variables
                ep_rewards, ep_log_grad = [], []
                env.randomize_environment()
                env.reset()
                t = 0

        # get gradients and apply them to model's variables - gradient is computed as a mean from episodes
        total_gradient = [tf.divide(grad, trajs) for grad in total_gradient] if trajs > 0 else total_gradient
        if total_gradient:
            optimizer.apply_gradients(zip(total_gradient, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())
            print('Epoch {0} finished!'.format(n))

        # update parameters
        eta.assign(eta_f())

        # update summary
        with tfc.summary.always_record_summaries():
            tfc.summary.histogram('histogram/total_gradient', total_gradient)
            tfc.summary.scalar('metric/distance', distance, step=n)
            tfc.summary.scalar('metric/mean_reward', np.mean(batch_rewards), step=n)
            tfc.summary.scalar('metric/mean_sum', np.mean(batch_sums), step=n)
            tfc.summary.scalar('metric/last_reward', ep_rew, step=n)
            tfc.summary.scalar('metric/learning_rate', eta.value(), step=n)
            train_writer.flush()

        # save model
        if n % args.model_save_interval == 0:
            ckpt.save(os.path.join(args.save_path, 'ckpt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--model-save-interval', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--update-step', type=int, default=1)
    parser.add_argument('--sim-step', type=int, default=5)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--sim-cam-id', type=int, default=0)
    parser.add_argument('--sim-cam-img-w', type=int, default=640)
    parser.add_argument('--sim-cam-img-h', type=int, default=480)
    parser.add_argument('--sim-max-length', type=int, default=100)
    parser.add_argument('--sim-max-dist', type=float, default=1.5)
    parser.add_argument('--sim-min-dist', type=float, default=0.01)
    parser.add_argument('--restore-path', type=str, default='')
    parser.add_argument('--save-path', type=str, default='./saved')
    parser.add_argument('--logs-path', type=str, default='./log/1')
    parser.add_argument('--keep-random', type=float, default=0.8)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()

    os.makedirs(args.save_path, exist_ok=True)
    train(args)
