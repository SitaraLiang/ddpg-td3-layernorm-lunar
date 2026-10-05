#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/remote_config.sh"

status=0
for machine in "${machines[@]}"; do
    echo "Stopping experiments on $machine"
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$machine" 'bash -s' <<'REMOTE'
set -euo pipefail
command -v tmux >/dev/null
sessions="$(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)"
while IFS= read -r session; do
    case "$session" in
        lunarlander-seed-*)
            tmux kill-session -t "=$session"
            echo "Stopped $session"
            ;;
    esac
done <<< "$sessions"
REMOTE
    then
        echo "Finished on $machine"
    else
        echo "Failed to stop experiments on $machine" >&2
        status=1
    fi
done
exit "$status"
