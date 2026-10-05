import copy
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import gymnasium as gym
import numpy as np
from rl_mind import Evaluator
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm
from rl_mind.collectors import TransitionCollector
from rl_mind.core import Action
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.env import VecEnv
from rl_mind.nn import soft_update
from evaluation import BiasEvaluator
from torch.utils.tensorboard import SummaryWriter
from networks import ContinuousDeterministicActor, ContinuousQNetwork, GaussianNoise
from config import DDPGConfig, TD3Config
from rl_mind.notebook import run_directory, setup_tensorboard, silence_known_warnings


def compute_critic_loss(
    gamma: float, batch: Transitions[Action], q_values: Tensor, next_q_values: Tensor
) -> Tensor:
    """Compute the DDPG critic loss from a batch of transitions

    :param gamma: The discount factor
    :param batch: The batch of transitions
    :param q_values: Q(s_t, a_t) from the critic (shape `[B]`)
    :param next_q_values: Q'(s_{t+1}, pi(s_{t+1})) from the target critic
        (shape `[B]`)
    :return: The critic loss (a scalar)
    """
    # Compute the target (do not bootstrap when `batch.terminated`), then the MSE loss
    with torch.no_grad():
        target = batch.reward + gamma * (1 - batch.terminated.float()) * next_q_values
    return F.mse_loss(q_values, target)


def compute_actor_loss(q_values: Tensor) -> Tensor:
    """Return the actor loss given Q(s_t, pi(s_t)) (shape `[B]`)"""
    # Compute the actor loss
    return -q_values.mean()


def run_ddpg(cfg: DDPGConfig) -> BiasEvaluator:
    """Notebook DDPG: current actor in backups, soft target critic, replay."""
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed, same_step_reset=True)
    actor = ContinuousDeterministicActor(
        env.observation_dim, cfg.actor_hidden, env.action_dim, cfg.actor_layer_norm
    )
    critic = ContinuousQNetwork(
        env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.layer_norm
    )
    target_critic = copy.deepcopy(critic).requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    # Data collection (with exploration noise), replay buffer and evaluation
    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    run_dir = run_directory(
        f"ddpg-{cfg.env_name}-LN{int(cfg.layer_norm)}-actorLN{int(cfg.actor_layer_norm)}-S{cfg.seed}"
    )
    evaluator = Evaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,  # number of training steps between two evaluations
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )
    """
    evaluator = BiasEvaluator( # do we need an evaluation seed?
                VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100), 
                critic, 
                every=cfg.eval_interval,
                run_dir=run_dir, 
                writer=writer)
    """
    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue

        # Sample a batch of transitions uniformly at random (with replacement)
        batch = buffer.sample(cfg.batch_size)

        # Update the critic

        # Q-values of the actions that were played (this is where gradients flow)
        # Current critic: evaluates stored historical actions in current states
        q_values = critic(batch.obs, batch.action.value)  # (64,)
        # Q-values of the *current* actor's actions in the next states,
        # estimated by the target critic (no gradient!)
        with torch.no_grad():
            # Target critic: evaluates current-policy actions in successor states
            next_actions = actor(batch.next_obs).value
            next_q_values = target_critic(batch.next_obs, next_actions)
        critic_loss = compute_critic_loss(cfg.gamma, batch, q_values, next_q_values)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update the actor (maximize Q(s, pi(s)))
        # evaluates actions the current actor proposes
        q_values = critic(batch.obs, actor(batch.obs).value)
        actor_loss = compute_actor_loss(q_values)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update the target critic
        soft_update(critic, target_critic, cfg.tau)

        evaluator.writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)
        evaluator.writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)
        if result := evaluator.run_if_needed(collector.steps, actor):
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )

    pbar.close()
    return evaluator


def run_td3(cfg: TD3Config | None = None) -> BiasEvaluator:
    """TD3: two critics, target actor with noise smoothing, delayed actor/target updates."""
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed, same_step_reset=True)
    actor = ContinuousDeterministicActor(
        env.observation_dim, cfg.actor_hidden, env.action_dim, cfg.actor_layer_norm
    )
    critic1 = ContinuousQNetwork(
        env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.layer_norm
    )
    critic2 = ContinuousQNetwork(
        env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.layer_norm
    )
    target_actor = copy.deepcopy(actor).requires_grad_(False)
    target_critic1 = copy.deepcopy(critic1).requires_grad_(False)
    target_critic2 = copy.deepcopy(critic2).requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.lr_critic
    )

    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    run_dir = run_directory(
        f"td3-{cfg.env_name}-LN{int(cfg.layer_norm)}-actorLN{int(cfg.actor_layer_norm)}-S{cfg.seed}"
    )
    evaluator = Evaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )
    """
    evaluator = BiasEvaluator( # do we need an evaluation seed?
                VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100), 
                critic, 
                every=cfg.eval_interval,
                run_dir=run_dir, 
                writer=writer)
    """

    updates = 0  # counter of critic-update round
    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue

        batch = buffer.sample(cfg.batch_size)

        # Implement the TD3 update

        # 1. Critic update: compute the common target with the target actor
        # (+ clipped noise) and the min of the two target critics, then
        # update both critics.
        # 2. Every `cfg.policy_delay` updates: update the actor using
        # critic_1, and softly update the three target networks.
        with torch.no_grad():
            next_actions = target_actor(batch.next_obs).value
            # randn_like: Returns a tensor with the same size as input that is filled with random numbers from a
            # normal distribution with mean 0 and variance 1.
            noise = (torch.randn_like(next_actions) * cfg.target_noise).clamp(
                -cfg.target_noise_clip, cfg.target_noise_clip
            )
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            # torch.minimum: takes the smaller value for each transition.
            next_values = torch.minimum(
                target_critic1(batch.next_obs, next_actions),
                target_critic2(batch.next_obs, next_actions),
            )
        loss_1 = compute_critic_loss(
            cfg.gamma, batch, critic1(batch.obs, batch.action.value), next_values
        )
        loss_2 = compute_critic_loss(
            cfg.gamma, batch, critic2(batch.obs, batch.action.value), next_values
        )
        critic_loss = loss_1 + loss_2
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()  # update both critics.
        evaluator.writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)

        updates += 1

        if updates % cfg.policy_delay == 0:
            # Do not optimize critic weights during actor update.
            critic1.requires_grad_(False)
            actor_loss = compute_actor_loss(critic1(batch.obs, actor(batch.obs).value))
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            critic1.requires_grad_(True)
            soft_update(actor, target_actor, cfg.tau)
            soft_update(critic1, target_critic1, cfg.tau)
            soft_update(critic2, target_critic2, cfg.tau)
            evaluator.writer.add_scalar(
                "loss/actor", actor_loss.item(), collector.steps
            )

        if result := evaluator.run_if_needed(collector.steps, actor):
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )

    pbar.close()
    return evaluator
