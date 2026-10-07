#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <seed>" >&2
	exit 1
fi

seed="$1"
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
	echo "Seed must be a non-negative integer." >&2
	exit 1
fi
command -v tmux >/dev/null || {
	echo "tmux is required." >&2
	exit 1
}
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session_name="lunarlander-seed-${seed}"

if [[ ! -f "$project_dir/.venv/bin/activate" ]]; then
	echo "Virtual environment not found: $project_dir/.venv" >&2
	exit 1
fi

printf -v session_command '%q ' bash -c '
set -euo pipefail
cd "$1"
source .venv/bin/activate
results_dir="$1/results"
mkdir -p "$results_dir"
for algorithm in DDPG TD3; do
    for layer_normalization in false true; do
        echo "Running $algorithm with seed $2, layer_normalization=$layer_normalization"
        experiment --policy "$algorithm" --seed "$2" --layer_normalization "$layer_normalization" --results_dir "$results_dir"
    done
done
' bash "$project_dir" "$seed"

tmux new-session -d -s "$session_name" "$session_command"
echo "Started session: $session_name"
echo "Results directory: $project_dir/results"
echo "Attach with: tmux attach -t $session_name"
