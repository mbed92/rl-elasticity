from argparse import ArgumentParser

import cv2

from agents import PolicyNetwork, ValueEstimator
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
    policy_network = PolicyNetwork(num_controls=env.num_actions)
    value_network = ValueEstimator()

    # setup optimizer
    eta = tfc.eager.Variable(args.learning_rate)
    eta_f = tf.train.exponential_decay(
        args.learning_rate,
        tf.train.get_or_create_global_step(),
        args.epochs,
        0.98)
    eta.assign(eta_f())
    policy_optimizer, policy_ckpt = setup_optimizer(args.policy_restore_path, eta, policy_network)
    value_optimizer, value_ckpt = setup_optimizer(args.value_restore_path, eta, value_network)
    policy_reg = tf.keras.regularizers.l2(1e-6)

    # run training
    for n in range(args.epochs):
        batch_rewards = []
        batch_sums = []
        batch_value_looses = []
        batch_baselines = []
        batch_advantages = []
        ep_rewards = []       # list for rewards accrued throughout ep
        ep_log_grad = []      # list of log-likelihood gradients
        total_policy_gradient = []   # list of gradients multiplied by rewards per epochs
        keep_random = update_keep_random(args.keep_random, n, args.epochs)

        # domain randomization after each epoch
        env.randomize_environment()
        env.reset()

        # start trajectory
        t, trajs = 0, 0
        while True:
            rgb, poses, joints = env.get_observations()
            if rgb[0] is not None and poses is not None:
                pass
                # cv2.imshow("trajectory", rgb[0])
                # cv2.waitKey(1)
            else:
                continue

            # take action in the environment under the current policy
            with tf.GradientTape(persistent=True) as policy_tape:
                ep_mean_act, ep_log_dev = policy_network([rgb, poses], True)
                ep_std_dev, ep_variance = tf.exp(ep_log_dev) + 1e-05, tf.square(tf.exp(ep_log_dev)) + 1e-05
                actions = env.take_continuous_action(ep_mean_act, ep_std_dev, keep_random)
                env.step()
                ep_rew, distance = env.get_reward(actions)
                ep_rewards.append(ep_rew)

                # optimize a mean and a std_dev - compute gradient from logprob
                policy_loss = policy_network.compute_loss(ep_std_dev, ep_variance, ep_mean_act, actions, policy_reg)

            # compute and store gradients
            policy_grads = policy_tape.gradient(policy_loss, policy_network.trainable_variables)
            ep_log_grad.append(policy_grads)
            t += 1

            # compute grad log-likelihood for a current episode
            if is_done(distance, ep_rew, t, args):
                if len(ep_rewards) > 5 and len(ep_rewards) == len(ep_log_grad):
                    ep_rewards = process_rewards(ep_rewards)
                    ep_reward_sum, ep_reward_mean = np.sum(ep_rewards), np.mean(ep_rewards)
                    batch_rewards.append(ep_reward_mean)
                    batch_sums.append(ep_reward_sum)

                    # compute baseline and update ValueNetwork
                    with tf.GradientTape(persistent=True) as value_tape:
                        baseline = value_network([poses, joints], training=True)
                        value_loss = tf.losses.mean_squared_error(ep_reward_sum, baseline)
                        advantage = ep_reward_sum - baseline
                    value_grads = value_tape.gradient(value_loss, value_network.trainable_variables)
                    value_network.update(value_grads, value_network, value_optimizer)
                    batch_value_looses.append(value_loss)
                    batch_advantages.append(advantage)
                    batch_baselines.append(baseline)

                    # update PolicyNetwork
                    trajectory_gradient = []
                    for i, (grad, reward) in enumerate(zip(ep_log_grad, ep_rewards)):
                        ep_log_grad[i] = [tf.multiply(g, (reward - advantage)) for g in grad]
                    for i, grad in enumerate(ep_log_grad):
                        trajectory_gradient = [tf.add(a, b) for a, b in zip(trajectory_gradient, grad)] if i > 0 else grad

                    # sum over all trajectories
                    total_policy_gradient = [tf.add(prob, prev_prob) for prob, prev_prob in zip(total_policy_gradient, trajectory_gradient)] if len(total_policy_gradient) > 0 else trajectory_gradient
                    print("Episode is done: keep random ratio: {0}, distance: {1}".format(keep_random, distance))
                    trajs += 1
                    if trajs >= args.update_step:
                        break

                # reset episode-specific variables
                ep_rewards, ep_log_grad = [], []
                env.randomize_environment()
                env.reset()
                t = 0

        # get gradients and apply them to model's variables - gradient is computed as a mean from episodes
        total_policy_gradient = [tf.divide(grad, trajs) for grad in total_policy_gradient] if trajs > 0 else total_policy_gradient
        policy_network.update(total_policy_gradient, policy_network, policy_optimizer)

        # update parameters
        eta.assign(eta_f())

        # update summary
        with tfc.summary.always_record_summaries():
            for layer, grad in enumerate(total_policy_gradient):
                tfc.summary.histogram('histogram/total_gradient_layer_{0}'.format(layer), grad)
            tfc.summary.scalar('metric/distance', distance, step=n)
            tfc.summary.scalar('metric/mean_reward', np.mean(batch_rewards), step=n)
            tfc.summary.scalar('metric/mean_sum', np.mean(batch_sums), step=n)
            tfc.summary.scalar('metric/last_reward', ep_rew, step=n)
            tfc.summary.scalar('metric/learning_rate', eta.value(), step=n)
            tfc.summary.scalar('metric/value_loss', np.mean(batch_value_looses), step=n)
            tfc.summary.scalar('metric/advantages', np.mean(batch_advantages), step=n)
            tfc.summary.scalar('metric/baselines', np.mean(batch_baselines), step=n)
            train_writer.flush()

        # save model
        if n % args.model_save_interval == 0:
            policy_ckpt.save(os.path.join(args.policy_save_path, 'policy'))
            value_ckpt.save(os.path.join(args.value_save_path, 'value'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--model-save-interval', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--update-step', type=int, default=2)
    parser.add_argument('--sim-step', type=int, default=5)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--sim-cam-id', type=int, default=0)
    parser.add_argument('--sim-cam-img-w', type=int, default=640)
    parser.add_argument('--sim-cam-img-h', type=int, default=480)
    parser.add_argument('--sim-max-length', type=int, default=100)
    parser.add_argument('--sim-max-dist', type=float, default=1.5)
    parser.add_argument('--sim-min-dist', type=float, default=0.01)
    parser.add_argument('--policy-restore-path', type=str, default='')
    parser.add_argument('--value-restore-path', type=str, default='')
    parser.add_argument('--policy-save-path', type=str, default='./saved/policy')
    parser.add_argument('--value-save-path', type=str, default='./saved/value')
    parser.add_argument('--logs-path', type=str, default='./log/1')
    parser.add_argument('--keep-random', type=float, default=0.8)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()

    os.makedirs(args.policy_save_path, exist_ok=True)
    os.makedirs(args.value_save_path, exist_ok=True)
    train(args)
