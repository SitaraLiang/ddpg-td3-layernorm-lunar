import argparse
from pathlib import Path

from ddpg_td3_layernorm_lunar.ddpg import DDPG
from ddpg_td3_layernorm_lunar.run import run_policy
from ddpg_td3_layernorm_lunar.td3 import TD3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG", choices=("TD3", "DDPG"))
    parser.add_argument(
        "--env_name", default="LunarLander-v3"
    )  # OpenAI gym environment name (only LunarLander-v3)
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--layer_normalization", default="false", type=str.lower, choices=("true", "false")
    )  # Layer normalization
    parser.add_argument(
        "--eval_episodes", default=10, type=int
    )  # Number of eval episodes
    parser.add_argument(
        "--start_timesteps", default=25_000, type=int
    )  # Time steps initial random policy is used
    parser.add_argument(
        "--max_timesteps", default=500_000, type=int
    )  # Max time steps to run environment
    parser.add_argument(
        "--batch_size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument(
        "--eval_freq", default=5_000, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--expl_noise", default=0.1, type=float
    )  # Std of Gaussian exploration noise
    # parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    # parser.add_argument(
    #     "--tau", default=0.005, type=float
    # )  # Target network update rate
    # parser.add_argument(
    #     "--policy_noise", default=0.2
    # )  # Noise added to target policy during critic update
    # parser.add_argument(
    #     "--noise_clip", default=0.5
    # )  # Range to clip target policy noise
    # parser.add_argument(
    #     "--policy_freq", default=2, type=int
    # )  # Frequency of delayed policy updates
    parser.add_argument("--results_dir", default=Path("./"), type=Path)

    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    # policy, steps, evaluations, biases = run_policy()
    if args.policy == "DDPG":
        Policy = DDPG
    else:
        Policy = TD3

    run_policy(
        Policy=Policy,
        env_name=args.env_name,
        eval_episodes=args.eval_episodes,
        start_timesteps=args.start_timesteps,
        max_timesteps=args.max_timesteps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        expl_noise=args.expl_noise,
        seed=args.seed,
        layer_normalization=args.layer_normalization == "true",
        save_results=args.results_dir
        / Path(f"{Policy.__name__}_seed_{args.seed}_ln_{args.layer_normalization == 'true'}"),
    )
