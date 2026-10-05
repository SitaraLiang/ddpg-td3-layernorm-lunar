import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state: Tensor):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, layer_normalization: bool = False
    ):
        super().__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.layer_normalization = layer_normalization
        if layer_normalization:
            self.ln1 = nn.LayerNorm(256)
            self.ln2 = nn.LayerNorm(256)

    def forward(self, state: Tensor, action: Tensor):
        if self.layer_normalization:
            q = F.relu(self.ln1(self.l1(torch.cat([state, action], 1))))
            q = F.relu(self.ln2(self.l2(q)))
        else:
            q = F.relu(self.l1(torch.cat([state, action], 1)))
            q = F.relu(self.l2(q))

        return self.l3(q)


# class CriticOriginalDDPG(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()

#         self.l1 = nn.Linear(state_dim, 400)
#         self.l2 = nn.Linear(400 + action_dim, 300)
#         self.l3 = nn.Linear(300, 1)

#     def forward(self, state, action):
#         q = F.relu(self.l1(state))
#         q = F.relu(self.l2(torch.cat([q, action], 1)))
#         return self.l3(q)
