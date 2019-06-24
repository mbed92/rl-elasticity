from argparse import ArgumentParser

from agents import PolicyNetwork, ValueEstimator
from environment import ManEnv
from utils import *


tf.enable_eager_execution()
tf.executing_eagerly()

# remove tensorflow warning "tried to deallocate nullptr"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@exec_time
def train(args):

    # create the environment
    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)

    scaler, featurizer = get_fitter(env)
    train_writer = setup_writer(args.logs_path)
    train_writer.set_as_default()

    # create policy and value estimators
    policy_network = PolicyNetwork(num_controls=env.num_actions)
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

    policy_reg = tf.keras.regularizers.l2(1e-5)
    value_reg = tf.keras.regularizers.l2(1e-5)
    policy_optimizer, policy_ckpt = setup_optimizer(args.policy_restore_path, eta_p, policy_network, 'policy_network.hdf5')
    value_optimizer, value_ckpt = setup_optimizer(args.value_restore_path, eta_v, value_network, 'value_network.hdf5')

    # run training
    for n in range(args.epochs):
        ep_rewards = []
        ep_policy_looses = []
        total_policy_gradient = []
        total_policy_loss = []
        batch_sums = []
        batch_rewards = []
        keep_random = update_keep_random(args.keep_random, n, args.epochs)

        # start trajectory
        env.randomize_environment()
        env.reset()
        t, trajs, value_loss = 0, 0, 0.0
        state = process_state(env.get_observations(), scaler, featurizer)
        while True:
            with tf.GradientTape() as policy_tape:
                action_dist = policy_network(state, training=True)
                actions = env.take_continuous_action(action_dist, keep_random)
                ep_rew, distance, done = env.step(args.sim_step, actions, args.sim_min_dist)
                next_state = process_state(env.get_observations(), scaler, featurizer)

                # compute gradients of a critic
                with tf.GradientTape() as value_tape:
                    value = value_network(state, training=True)
                    target = ep_rew + 0.95 * value_network(next_state, training=True)
                    advantage = target - value
                    value_loss += value_network.compute_loss(value, target, value_reg)
                policy_loss = policy_network.compute_loss(action_dist, actions, advantage, policy_reg)

            policy_grads = policy_tape.gradient(policy_loss, policy_network.trainable_variables)
            total_policy_gradient = [a + b for a, b in zip(policy_grads, total_policy_gradient)] if t > 1 else policy_grads

            ep_policy_looses.append(policy_loss)
            ep_rewards.append(ep_rew)
            state = next_state
            t += 1

            # compute total gradient for a current episode
            if done or t > args.sim_max_length:
                total_policy_loss.append(np.mean(ep_policy_looses))
                batch_rewards.append(np.mean(ep_rewards))
                batch_sums.append(np.sum(ep_rewards))
                trajs += 1
                if trajs == args.update_step:
                    print("Episode {0} is done: keep random ratio: {1}".format(n, keep_random))
                    break

                # reset episode-specific variables
                ep_rewards, ep_log_grad, ep_val_grad, ep_policy_looses = [], [], [], []
                env.randomize_environment()
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
            tfc.summary.scalar('metric/distance', distance, step=n)
            tfc.summary.scalar('metric/mean_reward', np.mean(batch_rewards), step=n)
            tfc.summary.scalar('metric/sum_reward', np.mean(batch_sums), step=n)
            tfc.summary.scalar('metric/value_loss', value_loss, step=n)
            tfc.summary.scalar('metric/total_policy_loss', np.mean(total_policy_loss), step=n)
            train_writer.flush()

        # save model
        if n % args.model_save_interval == 0:
            policy_ckpt.save(os.path.join(args.policy_save_path, 'policy'))
            value_ckpt.save(os.path.join(args.value_save_path, 'value'))
            policy_network.save_weights(os.path.join(args.policy_save_path, 'policy_network.hdf5'))
            value_network.save_weights(os.path.join(args.value_save_path, 'value_network.hdf5'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100000000)
    parser.add_argument('--model-save-interval', type=int, default=100)
    parser.add_argument('--policy-learning-rate', type=float, default=1e-5)
    parser.add_argument('--value-learning-rate', type=float, default=1e-4)
    parser.add_argument('--update-step', type=int, default=1)
    parser.add_argument('--sim-step', type=int, default=1)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--sim-cam-id', type=int, default=0)
    parser.add_argument('--sim-cam-img-w', type=int, default=640)
    parser.add_argument('--sim-cam-img-h', type=int, default=480)
    parser.add_argument('--sim-max-length', type=int, default=150)
    parser.add_argument('--sim-max-dist', type=float, default=1.5)
    parser.add_argument('--sim-min-dist', type=float, default=0.1)
    # parser.add_argument('--policy-restore-path', type=str, default='')
    parser.add_argument('--policy-restore-path', type=str, default='./saved/actor')
    # parser.add_argument('--value-restore-path', type=str, default='')
    parser.add_argument('--value-restore-path', type=str, default='./saved/critic')
    parser.add_argument('--policy-save-path', type=str, default='./saved/actor2')
    parser.add_argument('--value-save-path', type=str, default='./saved/critic2')
    parser.add_argument('--logs-path', type=str, default='./log/actor_critic2')
    parser.add_argument('--keep-random', type=float, default=1.0)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()

    os.makedirs(args.policy_save_path, exist_ok=True)
    os.makedirs(args.value_save_path, exist_ok=True)
    train(args)
