import torch
import torch.nn as nn
from torch import Tensor
from rl_mind.core import Action, Actor
from rl_mind.nn import build_mlp

class LayerNormMLP(nn.Module):
    def __init__(self, sizes, output_activation=None):
        super().__init__()

        model = build_mlp(sizes, output_activation=output_activation)

        last_linear = max(i for i, layer in enumerate(model) if isinstance(layer, nn.Linear))

        layers = []
        for i, layer in enumerate(model):
            layers.append(layer)
            if isinstance(layer, nn.Linear) and i != last_linear:
                layers.append(nn.LayerNorm(layer.out_features))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ContinuousQNetwork(nn.Module):
    """The Q-network $Q(s, a)$ for continuous actions"""
    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int, layer_norm: bool = False):
        super().__init__()
        # The sizes of the layers, e.g. [obs_dim, 64, 64, action_dim]
        sizes = [obs_dim + action_dim, *hidden, 1]
        self.model = LayerNormMLP(sizes) if layer_norm else build_mlp(sizes)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """
        Compute $Q(s, a)$ for a batch: `[B, obs_dim] x [B, action_dim] -> [B]`
        obs.shape = (64, 4)
        action.shape = (64, 1)
        """
        # torch.cat(): Concatenates the given sequence of tensors in tensors in the given dimension. 
        # All tensors must either have the same shape (except in the concatenating dimension) 
        # or be a 1-D empty tensor with size (0,).
        return self.model(torch.cat([obs, action], dim=-1)).squeeze(-1)


class ContinuousDeterministicActor(Actor[Action]):
    """Deterministic: with fixed weights, same observation → same action"""
    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int, layer_norm: bool = False):
        super().__init__()
        sizes = [obs_dim, *hidden, action_dim]
        self.model = (
            LayerNormMLP(sizes, output_activation=nn.Tanh()) if layer_norm 
            else build_mlp(sizes, output_activation=nn.Tanh())
        )

    def forward(self, obs: Tensor) -> Action:
        """An actor returns a batch of actions."""
        return Action(value=self.model(obs)) # (64, ), also "value=..." needs us to use actor(obs).value to access the tensor.
                                            # or actor.act(obs) directly thansk to the parent class's method.


class GaussianNoise(Actor[Action]):
    """Adds Gaussian noise to the actions of another actor (at training time)"""
    def __init__(self, actor: Actor[Action], sigma: float):
        super().__init__()
        self.actor = actor
        self.sigma = sigma

    def forward(self, obs: Tensor) -> Action:
        """ 
        Noisy action, for training, exploration behavior.
        """
        action = self.actor(obs).value
        return Action(value=(action + self.sigma * torch.randn_like(action)).clamp(-1.0, 1.0))

    def act(self, obs: Tensor) -> Tensor:
        """
        Delegates to the original actor without adding exploration noise.
        For evaluation.
        """
        return self.actor.act(obs)

