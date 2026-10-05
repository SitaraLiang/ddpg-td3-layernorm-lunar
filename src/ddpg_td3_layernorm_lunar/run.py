import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from ddpg_td3_layernorm_lunar.ddpg import DDPG
from ddpg_td3_layernorm_lunar.replay_buffer import ReplayBuffer
from ddpg_td3_layernorm_lunar.td3 import TD3


def eval_env(
    policy: DDPG | TD3,
    env_name: str,
    eval_episodes: int,
):
    eval_env = gym.make(env_name, continuous=True)

    total_reward = 0.0
    biases = []

    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        terminated, truncated = False, False

        q_values = []
        rewards = []

        while not (terminated or truncated):
            action = policy.select_action(np.array(state))

            # Estimate Q(s_t, a_t)
            with torch.no_grad():
                state_tensor = (
                    torch.as_tensor(state, dtype=torch.float32)
                    .reshape(1, -1)
                    .to(policy.device)
                )
                action_tensor = (
                    torch.as_tensor(action, dtype=torch.float32)
                    .reshape(1, -1)
                    .to(policy.device)
                )

                if isinstance(policy, TD3):
                    q1 = policy.critic_1(state_tensor, action_tensor)
                    q2 = policy.critic_2(state_tensor, action_tensor)
                    q_value = torch.minimum(q1, q2).item()
                else:
                    q_value = policy.critic(state_tensor, action_tensor).item()

            q_values.append(q_value)

            state, reward, terminated, truncated, _ = eval_env.step(action)

            rewards.append(reward)
            total_reward += reward

        # Monte Carlo returns
        returns = []
        G = 0.0

        for reward in reversed(rewards):
            G = reward + policy.discount * G
            returns.append(G)

        returns.reverse()

        # Q(s_t, a_t) - G_t
        biases.extend(np.asarray(q_values) - np.asarray(returns))

    eval_env.close()

    average_reward = float(total_reward / eval_episodes)
    average_bias = float(np.mean(biases))

    print(
        f"Evaluation over {eval_episodes} episodes: "
        f"reward={average_reward:.3f}, "
        f"bias={average_bias:.3f}"
    )

    return average_reward, average_bias


def run_policy(
    Policy: DDPG | TD3,
    env_name: str,
    eval_episodes: int = 10,
    start_timesteps: int = 25_000,
    max_timesteps: int = 500_000,
    batch_size: int = 256,
    eval_freq: int = 5_000,
    expl_noise: float = 0.1,
    seed: int = 1,
    layer_normalization: bool = False,
) -> tuple[DDPG | TD3, list[float], list[float]]:
    # Create environment
    env = gym.make(env_name, continuous=True)

    # Set seeds
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create policy and replay buffer
    policy = Policy(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Use a separate TensorBoard run for each policy and seed.
    with SummaryWriter(
        comment=f"_{Policy.__name__}_seed_{seed}{'_LN' if layer_normalization else ''}"
    ) as writer:
        # Evaluate on the original policy
        evaluation, bias = eval_env(policy, env_name, eval_episodes)
        writer.add_scalar("reward/evaluation", evaluation, 0)
        writer.add_scalar("critic/bias", bias, 0)
        steps = [0]
        evaluations = [evaluation]
        biases = [bias]

        state, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1

        progress_bar = trange(1, max_timesteps + 1)
        for timestep in progress_bar:
            episode_timesteps += 1

            # Initially start with completely random actions
            if timestep < start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                    policy.select_action(np.asarray(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(env.action_space.low, env.action_space.high)

            # Perform action
            next_state, reward, terminated, truncated, _ = env.step(action)
            writer.add_scalar("reward/step", reward, timestep + 1)

            # Store data in replay buffer
            replay_buffer.add(
                torch.tensor(state),
                torch.tensor(action),
                torch.tensor(next_state),
                torch.tensor(reward),
                torch.tensor(terminated),
                torch.tensor(truncated),
            )

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if timestep >= start_timesteps:
                policy.train(replay_buffer, batch_size)

            if terminated or truncated:
                progress_bar.set_description(f"Reward: {episode_reward}")
                writer.add_scalar("reward/episode", episode_reward, timestep)

                # Reset environment
                state, _ = env.reset()
                terminated, truncated = False, False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if timestep % eval_freq == 0:
                evaluation, bias = eval_env(policy, env_name, eval_episodes)
                steps.append(timestep)
                evaluations.append(evaluation)
                biases.append(bias)

                writer.add_scalar("reward/evaluation", evaluation, timestep)
                writer.add_scalar("critic/bias", bias, timestep + 1)

    env.close()

    return (policy, steps, evaluations, biases)
