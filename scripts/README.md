# CaML Scripts

## hf-backup.py - HuggingFace Organization Backup

Backs up models and datasets from `CompassioninMachineLearning` to `Backup-CaML`.

### Safety Features

- **Dry-run by default** - Must explicitly enable writes with `--no-dry-run`
- **Read-only on source** - Uses read token for source org
- **Verbose logging** - See exactly what will happen
- **Incremental** - Skips repos that already exist in backup
- **Test single repos** - Try with one model before running on all

### Setup

```bash
# Install dependencies
pip install huggingface_hub tqdm

# Set your write token (NOT the read-only one)
export HF_TOKEN=$(cat ~/Desktop/ai_dev/caml/secure/hf-caml-write)
```

### Usage

```bash
cd /path/to/caml-research/scripts

# Step 1: Dry run to see what would happen (SAFE)
python hf-backup.py --dry-run --all

# Step 2: Test with one small model
python hf-backup.py --no-dry-run --model CompassioninMachineLearning/tenK_cleanChat_dataset_oct_18

# Step 3: If that works, backup all (will take a while for 2-3TB)
python hf-backup.py --no-dry-run --all-models
```

### Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Simulate only, no changes (default) |
| `--no-dry-run` | Actually perform backup |
| `--all` | Backup all models and datasets |
| `--all-models` | Backup models only |
| `--all-datasets` | Backup datasets only |
| `--model <id>` | Backup single model |
| `--dataset <id>` | Backup single dataset |
| `-v, --verbose` | Verbose output |

### What It Does

1. Lists all repos in source org
2. For each repo:
   - Checks if backup already exists (skips if yes)
   - Downloads to temp directory
   - Creates private repo in backup org
   - Uploads files
   - Cleans up temp directory

### Time Estimates

- ~197 models × ~15GB = ~3TB
- Download + upload at ~100MB/s = ~8-10 hours for full backup
- Incremental (only new models) = much faster

### Credentials

| Token | Purpose | Stored At |
|-------|---------|-----------|
| Read-only | Exploring, testing | `secure/hugging-face-read-only-token.txt` |
| Write | Actual backups | `secure/hf-caml-write` |

**Always use read-only for exploration. Only use write token for actual backups.**
