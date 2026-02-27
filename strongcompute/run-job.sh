#!/bin/bash
# StrongCompute job runner with tmux persistence
#
# Usage:
#   ./run-job.sh <command>              # Run command in new tmux session
#   ./run-job.sh --attach [session]     # Attach to session (default: job)
#   ./run-job.sh --status               # List active sessions
#   ./run-job.sh --logs [session]       # Tail logs from session
#   ./run-job.sh --kill [session]       # Kill a session
#
# Examples:
#   ./run-job.sh "cd /workspace && python train.py --activations 'outputs/activations/*.pt'"
#   ./run-job.sh --attach
#   ./run-job.sh --logs

set -e

# Default session name
SESSION="${TMUX_SESSION:-job}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check SSH connection
check_connection() {
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes strongcompute "echo ok" &>/dev/null; then
        echo -e "${RED}Error:${NC} Cannot connect to StrongCompute container"
        echo "Run: ./connect.sh <hostname> <port>"
        exit 1
    fi
}

# List active tmux sessions
list_sessions() {
    check_connection
    echo -e "${CYAN}Active tmux sessions:${NC}"
    ssh strongcompute 'tmux list-sessions 2>/dev/null || echo "  No active sessions"'
}

# Attach to a session
attach_session() {
    local session="${1:-$SESSION}"
    check_connection
    echo -e "${CYAN}Attaching to session: ${session}${NC}"
    echo -e "${YELLOW}Detach with: Ctrl+B, then D${NC}"
    echo ""
    ssh -t strongcompute "tmux attach-session -t ${session}"
}

# Tail logs from a session
tail_logs() {
    local session="${1:-$SESSION}"
    check_connection
    echo -e "${CYAN}Tailing output from session: ${session}${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    # Capture the pane content and follow
    ssh strongcompute "tmux capture-pane -t ${session} -p -S -100"
}

# Kill a session
kill_session() {
    local session="${1:-$SESSION}"
    check_connection
    echo -e "${YELLOW}Killing session: ${session}${NC}"
    ssh strongcompute "tmux kill-session -t ${session} 2>/dev/null && echo 'Session killed' || echo 'Session not found'"
}

# Run a command in a new tmux session
run_command() {
    local cmd="$1"
    check_connection

    # Check if session already exists
    if ssh strongcompute "tmux has-session -t ${SESSION} 2>/dev/null"; then
        echo -e "${YELLOW}Session '${SESSION}' already exists.${NC}"
        echo "Options:"
        echo "  ./run-job.sh --attach    # Attach to existing session"
        echo "  ./run-job.sh --kill      # Kill existing session"
        echo "  TMUX_SESSION=job2 ./run-job.sh \"command\"  # Use different session name"
        exit 1
    fi

    echo -e "${GREEN}Starting job in tmux session: ${SESSION}${NC}"
    echo -e "${CYAN}Command:${NC} ${cmd}"
    echo ""

    # Create tmux session and run command
    # Using a wrapper that logs output and shows completion status
    ssh strongcompute "tmux new-session -d -s ${SESSION} 'echo \"=== Job started: \$(date) ===\"; echo \"Command: ${cmd}\"; echo \"\"; ${cmd}; EXIT_CODE=\$?; echo \"\"; echo \"=== Job finished: \$(date) ===\"; echo \"Exit code: \$EXIT_CODE\"; echo \"Press Enter to close...\"; read'"

    echo -e "${GREEN}Job started successfully!${NC}"
    echo ""
    echo "Monitor with:"
    echo "  ./run-job.sh --attach     # Attach to session (Ctrl+B, D to detach)"
    echo "  ./run-job.sh --logs       # View recent output"
    echo "  ./run-job.sh --status     # List sessions"
    echo "  ./run-job.sh --kill       # Stop the job"
}

# Parse arguments
case "${1:-}" in
    --status|-s)
        list_sessions
        ;;
    --attach|-a)
        attach_session "${2:-}"
        ;;
    --logs|-l)
        tail_logs "${2:-}"
        ;;
    --kill|-k)
        kill_session "${2:-}"
        ;;
    --help|-h)
        echo "Usage: ./run-job.sh <command>           Run command in tmux session"
        echo "       ./run-job.sh --attach [session]  Attach to session"
        echo "       ./run-job.sh --status            List active sessions"
        echo "       ./run-job.sh --logs [session]    View recent output"
        echo "       ./run-job.sh --kill [session]    Kill session"
        echo ""
        echo "Environment:"
        echo "  TMUX_SESSION    Session name (default: job)"
        ;;
    "")
        echo "Error: No command provided"
        echo "Usage: ./run-job.sh \"cd /workspace && python train.py ...\""
        exit 1
        ;;
    *)
        run_command "$1"
        ;;
esac
