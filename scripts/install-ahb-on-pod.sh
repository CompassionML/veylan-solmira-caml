#!/bin/bash
# install-ahb-on-pod.sh
#
# Installs the AHB inspect_evals task onto a freshly-provisioned pod.
# Required because AHB isn't in upstream inspect_evals (PyPI 0.11.0 or
# UKGovernmentBEIS GitHub main, as of 2026-05-08). The reference module
# lives in this repo at data/ahb/inspect-reference/ and gets shimmed into
# the pod's installed inspect_evals package directory.
#
# Run this once per fresh pod after provisioning, before any `inspect eval
# inspect_evals/ahb` invocation.
#
# Usage:
#   ./install-ahb-on-pod.sh <ip> <port> [<ssh-key-path>]
#
# Example:
#   ./scripts/install-ahb-on-pod.sh 194.68.245.89 22128 ~/.ssh/runpod_ed25519
#
# TODO (tracked in /caml/docs/2026-05-15-plan.md): bake AHB into the
# caml-env Docker image so this manual step disappears.

set -e

IP="${1:?usage: $0 <ip> <port> [<ssh-key-path>]}"
PORT="${2:?usage: $0 <ip> <port> [<ssh-key-path>]}"
SSH_KEY="${3:-$HOME/.ssh/runpod_ed25519}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AHB_FILES_DIR="$SCRIPT_DIR/../data/ahb/inspect-reference"

if [ ! -d "$AHB_FILES_DIR" ]; then
    echo "Error: AHB reference files not found at $AHB_FILES_DIR" >&2
    exit 1
fi

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$IP"

echo "[1/3] Copying AHB module files to pod..."
tar -C "$AHB_FILES_DIR" -czf - . | $SSH \
  'INSPECT_DIR=$(python3 -c "import inspect_evals, os; print(os.path.dirname(inspect_evals.__file__))") && \
   mkdir -p "$INSPECT_DIR/ahb" && \
   tar -xzf - -C "$INSPECT_DIR/ahb"'

echo "[2/3] Patching _registry.py to register the ahb task..."
$SSH \
  'INSPECT_DIR=$(python3 -c "import inspect_evals, os; print(os.path.dirname(inspect_evals.__file__))") && \
   REG="$INSPECT_DIR/_registry.py" && \
   if ! grep -q "from inspect_evals.ahb import ahb" "$REG"; then
       sed -i "s|^from inspect_evals.abstention_bench|from inspect_evals.ahb import ahb\nfrom inspect_evals.abstention_bench|" "$REG"
       echo "    registered ahb"
   else
       echo "    already registered"
   fi'

echo "[3/3] Verifying import..."
$SSH \
  'python3 -c "from inspect_evals.ahb.ahb import ahb; print(\"OK — AHB task fn:\", ahb)"'

echo
echo "Done. You can now: inspect eval inspect_evals/ahb -T '\''grader_models=[\"google/gemini-2.5-flash-lite\"]'\'' --model hf/<your-model>"
