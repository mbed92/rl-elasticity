from argparse import ArgumentParser
from agents import PolicyNetwork, ValueEstimator
from environment import ManEnv
from utils import *

tf.enable_eager_execution()
tf.executing_eagerly()

# remove tensorflow warning "tried to deallocate nullptr"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(args):

    # create the environment
    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)
    train_writer = setup_writer(args.logs_path)
    train_writer.set_as_default()

    # create policy and value estimators
    policy_network = PolicyNetwork(num_controls=env.num_actions)
    value_network = ValueEstimator()

    # setup optimizer and learning parameters
    eta_p = tfc.eager.Variable(args.learning_rate)
    eta_policy = tf.train.exponential_decay(
        args.learning_rate,
        tf.train.get_or_create_global_step(),
        args.epochs,
        0.98)
    eta_p.assign(eta_policy())

    eta_v = tfc.eager.Variable(args.learning_rate)
    eta_value = tf.train.exponential_decay(
        args.learning_rate * 100,
        tf.train.get_or_create_global_step(),
        args.epochs,
        0.98)
    eta_v.assign(eta_value())

    policy_optimizer, policy_ckpt = setup_optimizer(args.policy_restore_path, eta_p, policy_network)
    value_optimizer, value_ckpt = setup_optimizer(args.value_restore_path, eta_v, value_network)
    policy_reg = tf.keras.regularizers.l2(1e-6)

    # run training
    for n in range(args.epochs):
        batch_rewards = []
        batch_sums = []
        batch_value_looses = []
        batch_baselines = []
        ep_rewards = []                 # list for rewards accrued throughout ep
        ep_log_grad = []                # list of log-likelihood gradients
        total_policy_gradient = []      # list of gradients weighted by rewards per each epochs
        keep_random = update_keep_random(args.keep_random, n, args.epochs)

        # domain randomization after each epoch
        env.randomize_environment()
        env.reset()

        # start trajectory
        t, trajs = 0, 0
        while True:
            poses, joints = env.get_observations()
            # if rgb[0] is not None and poses is not None:
            #     # cv2.imshow("trajectory", rgb[0])
            #     # cv2.waitKey(1)
            # else:
            #     continue

            # take action in the environment under the current policy
            with tf.GradientTape(persistent=True) as policy_tape:
                _, _, action_dist = policy_network([poses, joints], True)
                actions = env.take_continuous_action(action_dist, keep_random)
                env.step()
                ep_rew, distance = env.get_reward(actions)
                ep_rewards.append(ep_rew)
                policy_loss = policy_network.compute_loss(action_dist, actions, policy_reg)

            # compute and store gradients in PolicyNetwork
            policy_grads = policy_tape.gradient(policy_loss, policy_network.trainable_variables)
            ep_log_grad.append(policy_grads)
            t += 1

            # compute grad log-likelihood for a current episode
            if is_done(distance, ep_rew, t, args):
                if len(ep_rewards) > 5 and len(ep_rewards) == len(ep_log_grad):
                    ep_rewards = process_rewards(ep_rewards)
                    ep_reward_sum, ep_reward_mean = np.sum(ep_rewards), np.mean(ep_rewards)

                    # update ValueNetwork
                    with tf.GradientTape(persistent=True) as value_tape:
                        baseline = value_network([poses, joints], training=True)
                        value_loss = value_network.compute_loss(ep_reward_sum, baseline)
                    value_grads = value_tape.gradient(value_loss, value_network.trainable_variables)
                    value_network.update(value_grads, value_network, value_optimizer)

                    # update gradients of a PolicyNetwork
                    trajectory_gradient = []
                    for i, (grad, reward) in enumerate(zip(ep_log_grad, ep_rewards)):
                        weighted_gradient = [tf.multiply(g, ep_reward_sum - baseline) for g in grad]
                        trajectory_gradient = [tf.add(a, b) for a, b in zip(trajectory_gradient, weighted_gradient)] if i > 0 else weighted_gradient
                    total_policy_gradient = [tf.add(a, b) for a, b in zip(total_policy_gradient, trajectory_gradient)] if len(total_policy_gradient) > 0 else trajectory_gradient

                    # save metrics
                    batch_value_looses.append(value_loss)
                    batch_baselines.append(baseline)
                    batch_rewards.append(ep_reward_mean)
                    batch_sums.append(ep_reward_sum)
                    trajs += 1
                    if trajs >= args.update_step:
                        print("Episode {0} is done: keep random ratio: {1}, distance: {2}".format(n, keep_random, distance))
                        break

                # reset episode-specific variables
                ep_rewards, ep_log_grad, trajectory_gradient, ep_val_grad = [], [], [], []
                env.randomize_environment()
                env.reset()
                t = 0

        # update PolicyNetwork
        total_policy_gradient = [tf.divide(grad, trajs) for grad in total_policy_gradient] if trajs > 0 else total_policy_gradient
        policy_network.update(total_policy_gradient, policy_network, policy_optimizer)

        # update summary
        eta_p.assign(eta_policy())
        eta_v.assign(eta_value())
        with tfc.summary.always_record_summaries():
            for layer, grad in enumerate(total_policy_gradient):
                tfc.summary.histogram('histogram/total_gradient_layer_{0}'.format(layer), grad)
            tfc.summary.scalar('metric/distance', distance, step=n)
            tfc.summary.scalar('metric/mean_reward', np.mean(batch_rewards), step=n)
            tfc.summary.scalar('metric/mean_sum', np.mean(batch_sums), step=n)
            tfc.summary.scalar('metric/last_reward', ep_rew, step=n)
            tfc.summary.scalar('metric/value_loss', np.mean(batch_value_looses), step=n)
            tfc.summary.scalar('metric/baselines', np.mean(batch_baselines), step=n)
            train_writer.flush()

        # save model
        if n % args.model_save_interval == 0:
            policy_ckpt.save(os.path.join(args.policy_save_path, 'policy'))
            value_ckpt.save(os.path.join(args.value_save_path, 'value'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
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
    parser.add_argument('--sim-min-dist', type=float, default=0.05)
    parser.add_argument('--policy-restore-path', type=str, default='')
    parser.add_argument('--value-restore-path', type=str, default='')
    parser.add_argument('--policy-save-path', type=str, default='./saved/policy')
    parser.add_argument('--value-save-path', type=str, default='./saved/value')
    parser.add_argument('--logs-path', type=str, default='./log/1')
    parser.add_argument('--keep-random', type=float, default=0.5)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()

    os.makedirs(args.policy_save_path, exist_ok=True)
    os.makedirs(args.value_save_path, exist_ok=True)
    train(args)
