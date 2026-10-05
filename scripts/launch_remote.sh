#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/remote_config.sh"

if [[ $# -ne 0 ]]; then
	echo "Usage: $0" >&2
	exit 1
fi
if [[ ${#machines[@]} -ne ${#seeds[@]} ]]; then
	echo "Provide exactly one seed per machine." >&2
	exit 1
fi
for seed in "${seeds[@]}"; do
	if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
		echo "Seeds must be non-negative integers." >&2
		exit 1
	fi
done

status=0
for i in "${!machines[@]}"; do
	machine="${machines[$i]}"
	seed="${seeds[$i]}"
	echo "Launching seed $seed on $machine"
	if ssh -o BatchMode=yes -o ConnectTimeout=10 "$machine" "bash -s -- $seed" <<'REMOTE'
set -euo pipefail
# Set this to the project directory on your remote machines.
cd "$HOME/ddpg-td3-layernorm-lunar"
bash ./scripts/run_experiment.sh "$1"
REMOTE
	then
		echo "Launched on $machine"
	else
		echo "Failed to launch on $machine" >&2
		status=1
	fi
done
exit "$status"
