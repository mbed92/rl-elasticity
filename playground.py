from argparse import ArgumentParser
from agents import PolicyNetwork, ValueEstimator
from utils import *
import gym


tf.enable_eager_execution()
tf.executing_eagerly()

# remove tensorflow warning "tried to deallocate nullptr"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate(args):
    env = gym.make('MountainCarContinuous-v0')
    scaler, featurizer = get_fitter(env)
    policy_network = PolicyNetwork(1)
    policy_network.load_weights(os.path.join(args.policy_restore_path, 'policy_network.hdf5'))

    for t in range(args.epochs):
        observation = process_state(env.reset(), scaler, featurizer)
        while True:
            env.render()
            action_dist = policy_network(observation, training=True)
            actions = action_dist.sample(1)
            actions = tf.clip_by_value(actions, env.action_space.low[0], env.action_space.high[0])
            new_observation, ep_rew, done, info = env.step(actions)
            if done:
                print(ep_rew)
                env.reset()
                break
            observation = process_state(new_observation, scaler, featurizer)
    env.close()

def train(args):

    # create the environment
    env = gym.make('MountainCarContinuous-v0')
    train_writer = setup_writer(args.logs_path)
    train_writer.set_as_default()
    uniform_dist = tf.distributions.Uniform(env.action_space.low[0], env.action_space.high[0])

    # create policy and value estimators
    scaler, featurizer = get_fitter(env)
    policy_network = PolicyNetwork(num_controls=1)
    value_network = ValueEstimator()

    # setup optimizer and learning parameters
    eta_p = tfc.eager.Variable(args.policy_learning_rate)
    eta_policy = tf.train.exponential_decay(
        args.policy_learning_rate,
        tf.train.get_or_create_global_step(),
        args.epochs,
        0.99)
    eta_p.assign(eta_policy())

    eta_v = tfc.eager.Variable(args.value_learning_rate)
    eta_value = tf.train.exponential_decay(
        args.value_learning_rate,
        tf.train.get_or_create_global_step(),
        args.epochs,
        0.99)
    eta_v.assign(eta_value())

    policy_optimizer, policy_ckpt = setup_optimizer(args.policy_restore_path, eta_p, policy_network)
    value_optimizer, value_ckpt = setup_optimizer(args.value_restore_path, eta_v, value_network)

    # run training
    for n in range(args.epochs):
        ep_rewards = []
        total_policy_gradient = []
        batch_sums = []
        batch_rewards = []
        keep_random = update_keep_random(args.keep_random, n, args.epochs)

        # start trajectory
        t, trajs, value_loss = 0, 0, 0.0
        observation = process_state(env.reset(), scaler, featurizer)
        while True:
            # env.render()
            with tf.GradientTape() as policy_tape:
                action_dist = policy_network(observation, training=True)
                if np.random.uniform() < keep_random:
                    actions = action_dist.sample(1)
                else:
                    shape = tf.shape(action_dist.sample(1))
                    actions = tf.cast(uniform_dist.sample(shape), dtype=tf.float64)

                actions = tf.clip_by_value(actions, env.action_space.low[0], env.action_space.high[0])
                new_observation, ep_rew, done, info = env.step(actions)
                ep_rewards.append(ep_rew)
                new_observation = process_state(new_observation, scaler, featurizer)

                # compute gradients of a critic
                with tf.GradientTape() as value_tape:
                    value = value_network(observation, training=True)
                    target = ep_rew + 0.95 * value_network(new_observation, training=False)
                    advantage = target - value
                    value_loss += value_network.compute_loss(value, target)
                policy_loss = policy_network.compute_loss(action_dist, actions, advantage)

            policy_grads = policy_tape.gradient(policy_loss, policy_network.trainable_variables)
            total_policy_gradient = [a + b for a, b in zip(policy_grads, total_policy_gradient)] if t > 0 else policy_grads
            t += 1
            observation = new_observation

            if done:
                trajs += 1
                batch_rewards.append(np.mean(ep_rewards[-100:]))
                batch_sums.append(np.sum(ep_rewards[-100:]))

                if trajs >= args.update_step:
                    print("Episode {0} is done: keep random ratio: {1}".format(n, keep_random))
                    break

                # reset episode-specific variables
                ep_rewards, ep_log_grad, ep_val_grad = [], [], []
                env.reset()
                t = 0

        # update PolicyNetwork and ValueNetwork
        value_grads = value_tape.gradient(value_loss, value_network.trainable_variables)
        value_network.update(value_grads, value_network, value_optimizer)
        total_policy_gradient = [tf.divide(grad, trajs) for grad in total_policy_gradient] if trajs > 1 else total_policy_gradient
        policy_network.update(total_policy_gradient, policy_network, policy_optimizer)

        # update summary
        eta_p.assign(eta_policy())
        eta_v.assign(eta_value())
        with tfc.summary.always_record_summaries():
            for layer, grad in enumerate(total_policy_gradient):
                tfc.summary.histogram('histogram/total_gradient_layer_{0}'.format(layer), grad)
            tfc.summary.scalar('metric/batch_mean_rewards', np.mean(batch_rewards), step=n)
            tfc.summary.scalar('metric/batch_sums_rewards', np.mean(batch_sums), step=n)
            tfc.summary.scalar('metric/value_loss', value_loss, step=n)
            train_writer.flush()

        if n % args.model_save_interval == 0:
            policy_ckpt.save(os.path.join(args.policy_save_path, 'ckpt'))
            value_ckpt.save(os.path.join(args.value_save_path, 'ckpt'))
            policy_network.save_weights(os.path.join(args.policy_save_path, 'policy_network.hdf5'))
            value_network.save_weights(os.path.join(args.value_save_path, 'value_network.hdf5'))
    env.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--model-save-interval', type=int, default=20)
    parser.add_argument('--policy-learning-rate', type=float, default=5e-4)
    parser.add_argument('--value-learning-rate', type=float, default=2e-3)
    parser.add_argument('--update-step', type=int, default=1)
    parser.add_argument('--sim-step', type=int, default=10)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--sim-cam-id', type=int, default=0)
    parser.add_argument('--sim-cam-img-w', type=int, default=640)
    parser.add_argument('--sim-cam-img-h', type=int, default=480)
    parser.add_argument('--sim-max-length', type=int, default=100)
    parser.add_argument('--sim-max-dist', type=float, default=1.5)
    parser.add_argument('--sim-min-dist', type=float, default=0.1)
    # parser.add_argument('--policy-restore-path', type=str, default='')
    parser.add_argument('--policy-restore-path', type=str, default='./saved/actor')
    # parser.add_argument('--value-restore-path', type=str, default='')
    parser.add_argument('--value-restore-path', type=str, default='./saved/critic')
    parser.add_argument('--policy-save-path', type=str, default='./saved/actor')
    parser.add_argument('--value-save-path', type=str, default='./saved/critic')
    parser.add_argument('--logs-path', type=str, default='./log/actor_critic')
    parser.add_argument('--keep-random', type=float, default=1.0)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()

    os.makedirs(args.policy_save_path, exist_ok=True)
    os.makedirs(args.value_save_path, exist_ok=True)
    train(args)
    evaluate(args)