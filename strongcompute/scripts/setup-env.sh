#!/bin/bash
# Run this INSIDE the StrongCompute container after connecting
# Sets up the environment for CaML work

set -e

echo "=== StrongCompute Environment Setup ==="

# Activate virtual environment
if [[ -f ~/.venv/bin/activate ]]; then
    source ~/.venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No venv found at ~/.venv"
fi

# Verify ISC CLI
if command -v isc &> /dev/null; then
    echo "✓ ISC CLI available"
    isc ping && echo "✓ ISC authenticated" || echo "⚠ ISC auth failed"
else
    echo "⚠ ISC CLI not found"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "⚠ nvidia-smi not available"
fi

# Check Python
echo ""
echo "=== Python Environment ==="
python --version
pip --version

echo ""
echo "=== Ready ==="
echo "Container is ready for work."
echo ""
echo "Useful commands:"
echo "  isc experiments          # List experiments"
echo "  isc container stop -s    # Stop + squash when done"
