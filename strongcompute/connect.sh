#!/bin/bash
# StrongCompute container connection helper
#
# Usage:
#   ./connect.sh <hostname> <port>           # Update config and connect
#   ./connect.sh <hostname> <port> --vscode  # Open in VSCode Remote SSH
#   ./connect.sh <hostname> <port> --no-connect  # Just update config
#   ./connect.sh --status                    # Check current connection
#
# Examples:
#   ./connect.sh 192.168.127.170 47180
#   ./connect.sh 192.168.127.170 47180 --vscode
#
# Environment variables:
#   SSH_KEY     - Path to SSH private key (default: ~/Desktop/ai_dev/caml/secure/caml)
#   SKIP_VPN    - Set to 1 to skip VPN check

set -e

# Configuration
SSH_KEY="${SSH_KEY:-/Users/infinitespire/Desktop/ai_dev/caml/secure/caml}"
SSH_CONFIG="$HOME/.ssh/config"
VPN_GATEWAY="192.168.127.1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Parse arguments
HOSTNAME=""
PORT=""
VSCODE=false
NO_CONNECT=false
STATUS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --vscode)
            VSCODE=true
            shift
            ;;
        --no-connect)
            NO_CONNECT=true
            shift
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./connect.sh [hostname] [port] [options]"
            echo ""
            echo "Options:"
            echo "  --vscode      Open in VSCode Remote SSH instead of terminal"
            echo "  --no-connect  Just update SSH config, don't connect"
            echo "  --status      Show current connection status"
            echo "  --help        Show this help"
            echo ""
            echo "Environment:"
            echo "  SSH_KEY       Path to SSH key (default: ~/Desktop/ai_dev/caml/secure/caml)"
            echo "  SKIP_VPN      Set to 1 to skip VPN check"
            exit 0
            ;;
        *)
            if [[ -z "$HOSTNAME" ]]; then
                HOSTNAME="$1"
            elif [[ -z "$PORT" ]]; then
                PORT="$1"
            else
                echo "Unknown option: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Function: Check VPN connectivity
check_vpn() {
    if [[ "${SKIP_VPN:-0}" == "1" ]]; then
        return 0
    fi

    if ping -c 1 -W 2 "$VPN_GATEWAY" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function: Get current config
get_current_config() {
    if grep -q "Host strongcompute" "$SSH_CONFIG" 2>/dev/null; then
        local host=$(grep -A5 "Host strongcompute" "$SSH_CONFIG" | grep HostName | awk '{print $2}')
        local port=$(grep -A5 "Host strongcompute" "$SSH_CONFIG" | grep Port | awk '{print $2}')
        echo "$host:$port"
    else
        echo "not configured"
    fi
}

# Function: Test container reachability
test_container() {
    # Use the SSH config entry which has all the right options
    if ssh -o ConnectTimeout=10 -o BatchMode=yes strongcompute "echo ok" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Status check mode
if [[ "$STATUS_ONLY" == true ]]; then
    echo "=== StrongCompute Status ==="
    echo ""

    # VPN
    echo -n "VPN: "
    if check_vpn; then
        echo -e "${GREEN}Connected${NC}"
    else
        echo -e "${RED}Not connected${NC}"
        echo "  Start Wireguard VPN first"
    fi

    # SSH Config
    echo -n "SSH Config: "
    current=$(get_current_config)
    if [[ "$current" != "not configured" ]]; then
        echo -e "${GREEN}$current${NC}"

        # Test connection
        echo -n "Container: "
        if check_vpn && test_container; then
            echo -e "${GREEN}Reachable${NC}"
        else
            echo -e "${YELLOW}Not reachable${NC} (stopped or IP changed?)"
        fi
    else
        echo -e "${YELLOW}Not configured${NC}"
        echo "  Run: ./connect.sh <hostname> <port>"
    fi

    exit 0
fi

# Require hostname and port for non-status modes
if [[ -z "$HOSTNAME" || -z "$PORT" ]]; then
    echo "Usage: ./connect.sh <hostname> <port> [--vscode|--no-connect]"
    echo ""
    echo "Get hostname and port from Control Plane after starting container:"
    echo "  https://cp.strongcompute.ai/workstations"
    echo ""
    echo "Or check status: ./connect.sh --status"
    exit 1
fi

# Validate port is numeric
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error:${NC} Port must be a number (got: $PORT)"
    exit 1
fi

echo "=== StrongCompute Connection Helper ==="
echo ""

# Check VPN first
echo -n "Checking VPN... "
if check_vpn; then
    echo -e "${GREEN}Connected${NC}"
else
    echo -e "${RED}Not connected${NC}"
    echo ""
    echo "Start Wireguard VPN first:"
    echo "  macOS: Open Wireguard app, connect to Sydney profile"
    echo "  Linux: sudo wg-quick up sydney"
    exit 1
fi

# Update SSH config
echo -n "Updating SSH config... "
if grep -q "Host strongcompute" "$SSH_CONFIG" 2>/dev/null; then
    # Update existing entry
    awk -v host="$HOSTNAME" -v port="$PORT" -v key="$SSH_KEY" '
        /^Host strongcompute$/ { in_block=1; print; next }
        in_block && /^Host / { in_block=0 }
        in_block && /HostName/ { print "    HostName " host; next }
        in_block && /Port/ { print "    Port " port; next }
        in_block && /IdentityFile/ { print "    IdentityFile " key; next }
        { print }
    ' "$SSH_CONFIG" > "$SSH_CONFIG.tmp" && mv "$SSH_CONFIG.tmp" "$SSH_CONFIG"
    echo -e "${GREEN}Updated${NC} (${HOSTNAME}:${PORT})"
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
    echo -e "${GREEN}Added${NC} (${HOSTNAME}:${PORT})"
fi

# Exit if no-connect mode
if [[ "$NO_CONNECT" == true ]]; then
    echo ""
    echo "SSH config updated. Connect later with:"
    echo "  ssh strongcompute"
    echo "  code --remote ssh-remote+strongcompute /workspace"
    exit 0
fi

# Test connectivity before connecting
echo -n "Testing container... "
if test_container; then
    echo -e "${GREEN}Reachable${NC}"
else
    echo -e "${RED}Not reachable${NC}"
    echo ""
    echo "Container may be stopped or IP/port changed."
    echo "Check Control Plane: https://cp.strongcompute.ai/workstations"
    exit 1
fi

echo ""

# VSCode mode
if [[ "$VSCODE" == true ]]; then
    echo "Opening in VSCode..."
    code --remote ssh-remote+strongcompute /workspace
    exit 0
fi

# Terminal mode - connect and setup
echo "Connecting to container..."
echo "─────────────────────────────────────"

# Connect with auto-setup command
ssh -t strongcompute '
    # Activate venv
    if [[ -f ~/.venv/bin/activate ]]; then
        source ~/.venv/bin/activate
        echo "✓ Virtual environment activated"
    fi

    # Show GPU
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "N/A")"
    fi

    # Show Python
    echo "Python: $(python --version 2>&1)"
    echo ""

    # Start interactive shell
    exec bash
'
