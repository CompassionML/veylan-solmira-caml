#!/bin/bash
# StrongCompute container connection helper
#
# Usage: ./connect.sh <hostname> <port>
# Example: ./connect.sh 192.168.127.170 47180
#
# This script:
# 1. Updates your SSH config with the current connection details
# 2. Connects to the container
# 3. Sets up the environment

set -e

HOSTNAME="${1:-}"
PORT="${2:-}"
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa}"
SSH_CONFIG="$HOME/.ssh/config"

if [[ -z "$HOSTNAME" || -z "$PORT" ]]; then
    echo "Usage: ./connect.sh <hostname> <port>"
    echo "Example: ./connect.sh 192.168.127.170 47180"
    echo ""
    echo "Get these values from Control Plane after starting your container:"
    echo "https://cp.strongcompute.ai/workstations"
    exit 1
fi

echo "=== StrongCompute Connection Helper ==="
echo "Host: $HOSTNAME"
echo "Port: $PORT"
echo ""

# Update SSH config
echo "Updating SSH config..."
if grep -q "Host strongcompute" "$SSH_CONFIG" 2>/dev/null; then
    # Update existing entry using temp file
    awk -v host="$HOSTNAME" -v port="$PORT" '
        /^Host strongcompute$/ { in_block=1; print; next }
        in_block && /^Host / { in_block=0 }
        in_block && /HostName/ { print "    HostName " host; next }
        in_block && /Port/ { print "    Port " port; next }
        { print }
    ' "$SSH_CONFIG" > "$SSH_CONFIG.tmp" && mv "$SSH_CONFIG.tmp" "$SSH_CONFIG"
    echo "Updated existing strongcompute entry"
else
    # Add new entry
    cat >> "$SSH_CONFIG" << EOF

Host strongcompute
    HostName $HOSTNAME
    User root
    Port $PORT
    IdentityFile $SSH_KEY
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
    echo "Added new strongcompute entry"
fi

echo ""
echo "Connecting to container..."
echo "Run 'source ~/.venv/bin/activate' after connecting"
echo ""

ssh strongcompute
