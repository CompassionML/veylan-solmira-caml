# Persona Vectors

Activation steering for trait expression using contrastive activation addition (CAA).

## Overview

Extract and apply persona vectors to steer LLM behavior toward specific traits (e.g., compassion, empathy).

## Usage

```bash
python -m persona_vectors --help
```

## Documentation

- [Remediation Checklist](persona-vectors-remediation-checklist.md)

## Files

```
persona-vectors/
├── __init__.py
├── __main__.py
├── artifacts.py      # Artifact management
├── cli.py           # Command-line interface
├── config.py        # Configuration
├── controls.py      # Experimental controls
├── evaluation.py    # Evaluation metrics
├── extraction.py    # Vector extraction
├── main.py          # Main entry point
├── model.py         # Model loading
├── stats.py         # Statistics utilities
└── steering.py      # Activation steering
```
