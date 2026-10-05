import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DDPGConfig:
    env_name: str = "LunarLanderContinuous-v3"
    env_kwargs: dict = field(default_factory=dict)
    seed: int = 1

    #: Total number of environment steps
    max_steps: int = 30_000
    #: Number of parallel training environments
    n_envs: int = 1
    #: Environment steps between two gradient updates
    steps_per_update: int = 1
    #: Steps before learning starts
    learning_starts: int = 1_000

    #: Replay buffer capacity
    buffer_size: int = 200_000
    batch_size: int = 64

    #: Discount factor
    gamma: float = 0.98
    #: Target network update coefficient
    tau: float = 0.05
    #: Exploration noise
    action_noise: float = 0.1

    actor_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (64, 64)
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3

    #: Steps between two evaluations, and number of evaluation episodes
    eval_interval: int = 2_000
    n_eval_envs: int = 10

    output_dir: str | None = "outputs"
    progress: bool = True

    layer_norm: bool = False  # Critics only: isolates the critic intervention.
    actor_layer_norm: bool = False # may test later

    updates_per_transition: float = 1.0  # Critic rounds per new eligible transition.
    max_episode_steps: int = 1_000  # Training and performance-scoring horizon.
    #eval_seed: int = 10_000 do we need it
    mc_horizon: int = 5_000  # Extend evaluation for MC; exclude capped episodes.



@dataclass(frozen=True)
class TD3Config(DDPGConfig):
    #: Number of critic updates between two policy updates
    policy_delay: int = 2
    #: Std of the noise added to the target policy actions
    target_noise: float = 0.2
    #: Clipping of the target policy noise
    target_noise_clip: float = 0.5
