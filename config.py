"""
CAML Research Configuration

Centralizes path configuration using environment variables.
Copy .env.example to .env and customize for your setup.
"""

import os
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base paths - set via environment or use defaults
CAML_ROOT = Path(os.environ.get("CAML_ROOT", Path(__file__).parent))
CAML_SECURE = Path(os.environ.get("CAML_SECURE", CAML_ROOT.parent / "secure"))
CAML_OUTPUTS = Path(os.environ.get("CAML_OUTPUTS", CAML_ROOT / "experiments/linear-probes/outputs"))

# SSH keys
SSH_KEY_PATH = Path(os.environ.get("CAML_SSH_KEY", Path.home() / ".ssh/runpod_ed25519"))

# HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_ACTIVATIONS_REPO = "VeylanSolmira/compassion-activations"

# API Keys (loaded from environment only)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Derived paths
ACTIVATIONS_DIR = CAML_OUTPUTS / "activations"
VISUALIZATIONS_DIR = CAML_OUTPUTS / "visualizations"
PROBES_DIR = CAML_OUTPUTS / "probes"

# RunPod configuration file path
RUNPOD_CONFIG_PATH = CAML_SECURE / "runpod_current.json"


def ensure_dirs():
    """Create output directories if they don't exist."""
    for dir_path in [ACTIVATIONS_DIR, VISUALIZATIONS_DIR, PROBES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"CAML_ROOT: {CAML_ROOT}")
    print(f"CAML_SECURE: {CAML_SECURE}")
    print(f"CAML_OUTPUTS: {CAML_OUTPUTS}")
    print(f"SSH_KEY_PATH: {SSH_KEY_PATH}")
    print(f"RUNPOD_CONFIG_PATH: {RUNPOD_CONFIG_PATH}")
