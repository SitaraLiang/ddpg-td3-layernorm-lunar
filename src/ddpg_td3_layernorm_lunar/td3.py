import copy

import torch
import torch.nn.functional as F
from torch import Tensor

from .models import Actor, Critic
from .replay_buffer import ReplayBuffer


class TD3:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        device: str = "cpu",
        layer_normalization: bool = False,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim, layer_normalization).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim, layer_normalization).to(device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device

        self.total_it = 0

    def select_action(self, state: Tensor):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, terminated, _truncated = (
            replay_buffer.sample(batch_size)
        )

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - terminated) * self.discount * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_1_loss = F.mse_loss(current_Q1, target_Q)
        critic_2_loss = F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic_1.parameters(), self.critic_1_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.critic_2.parameters(), self.critic_2_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename: str):
        torch.save(self.critic_1.state_dict(), filename + "_critic_1")
        torch.save(
            self.critic_1_optimizer.state_dict(), filename + "_critic_1_optimizer"
        )

        torch.save(self.critic_2.state_dict(), filename + "_critic_2")
        torch.save(
            self.critic_2_optimizer.state_dict(), filename + "_critic_2_optimizer"
        )

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename: str):
        self.critic_1.load_state_dict(torch.load(filename + "_critic_1"))
        self.critic_1_optimizer.load_state_dict(
            torch.load(filename + "_critic_1_optimizer")
        )
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(torch.load(filename + "_critic_2"))
        self.critic_2_optimizer.load_state_dict(
            torch.load(filename + "_critic_2_optimizer")
        )
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
