#!/bin/bash
# Check if StrongCompute container is reachable
# Useful before attempting to connect

set -e

echo "=== StrongCompute Status Check ==="

# Check VPN (ping Sydney cluster gateway)
echo -n "VPN connectivity: "
if ping -c 1 -W 2 192.168.127.1 &> /dev/null; then
    echo "✓ Connected"
else
    echo "✗ Not connected"
    echo ""
    echo "Start Wireguard VPN first:"
    echo "  macOS: Open Wireguard app, connect to Sydney profile"
    echo "  Linux: sudo wg-quick up sydney"
    exit 1
fi

# Check if SSH config exists
echo -n "SSH config: "
if grep -q "Host strongcompute" ~/.ssh/config 2>/dev/null; then
    HOST=$(grep -A1 "Host strongcompute" ~/.ssh/config | grep HostName | awk '{print $2}')
    PORT=$(grep -A2 "Host strongcompute" ~/.ssh/config | grep Port | awk '{print $2}')
    echo "✓ Found (${HOST}:${PORT})"

    # Try to connect
    echo -n "Container: "
    if ssh -o ConnectTimeout=5 -o BatchMode=yes strongcompute "echo connected" 2>/dev/null; then
        echo "✓ Running and reachable"
    else
        echo "✗ Not reachable (stopped or wrong IP/port?)"
        echo ""
        echo "Start container via Control Plane, then update with:"
        echo "  ./connect.sh <new-hostname> <new-port>"
    fi
else
    echo "✗ Not configured"
    echo ""
    echo "After starting container, run:"
    echo "  ./connect.sh <hostname> <port>"
fi
