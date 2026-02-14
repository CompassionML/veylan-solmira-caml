#!/usr/bin/env python3
"""
HuggingFace Organization Backup Script

Backs up models and datasets from CompassioninMachineLearning to Backup-CaML.

SAFETY FEATURES:
- Dry-run mode by default (must explicitly enable writes)
- Read-only access to source org
- Verbose logging of all operations
- Progress reporting
- Can target single repos for testing

Usage:
    # Dry run (safe - no changes made)
    python hf-backup.py --dry-run

    # Backup single model (for testing)
    python hf-backup.py --model CompassioninMachineLearning/llama3-8b_20kshotnodilution

    # Backup all models (after testing)
    python hf-backup.py --all-models

    # Backup everything
    python hf-backup.py --all

Requirements:
    pip install huggingface_hub tqdm
"""

import argparse
import logging
import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download, HfFolder
    from huggingface_hub.utils import RepositoryNotFoundError
    from tqdm import tqdm
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install huggingface_hub tqdm")
    sys.exit(1)

# Configuration
SOURCE_ORG = "CompassioninMachineLearning"
BACKUP_ORG = "Backup-CaML"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class HFBackup:
    def __init__(self, dry_run: bool = True, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.api = HfApi()
        self.stats = {
            'models_checked': 0,
            'models_backed_up': 0,
            'models_skipped': 0,
            'models_failed': 0,
            'datasets_checked': 0,
            'datasets_backed_up': 0,
            'datasets_skipped': 0,
            'datasets_failed': 0,
        }

        if dry_run:
            logger.info("=" * 60)
            logger.info("DRY RUN MODE - No changes will be made")
            logger.info("=" * 60)

    def verify_access(self) -> bool:
        """Verify we have correct permissions on both orgs."""
        logger.info("Verifying HuggingFace access...")

        try:
            user = self.api.whoami()
            logger.info(f"Logged in as: {user['name']}")

            orgs = {org['name']: org.get('roleInOrg', 'unknown') for org in user.get('orgs', [])}

            # Check source org (need read)
            if SOURCE_ORG not in orgs:
                logger.error(f"Not a member of source org: {SOURCE_ORG}")
                return False
            logger.info(f"Source org ({SOURCE_ORG}): {orgs[SOURCE_ORG]} access")

            # Check backup org (need write/admin)
            if BACKUP_ORG not in orgs:
                logger.error(f"Not a member of backup org: {BACKUP_ORG}")
                return False
            if orgs[BACKUP_ORG] not in ['admin', 'write']:
                logger.warning(f"Backup org ({BACKUP_ORG}): {orgs[BACKUP_ORG]} access - may not be able to write")
            else:
                logger.info(f"Backup org ({BACKUP_ORG}): {orgs[BACKUP_ORG]} access")

            return True

        except Exception as e:
            logger.error(f"Access verification failed: {e}")
            return False

    def list_source_models(self) -> list:
        """List all models in source org."""
        logger.info(f"Listing models in {SOURCE_ORG}...")
        models = list(self.api.list_models(author=SOURCE_ORG))
        logger.info(f"Found {len(models)} models")
        return models

    def list_source_datasets(self) -> list:
        """List all datasets in source org."""
        logger.info(f"Listing datasets in {SOURCE_ORG}...")
        datasets = list(self.api.list_datasets(author=SOURCE_ORG))
        logger.info(f"Found {len(datasets)} datasets")
        return datasets

    def get_repo_size(self, repo_id: str, repo_type: str = "model") -> int:
        """Get total size of a repository in bytes."""
        try:
            if repo_type == "model":
                info = self.api.model_info(repo_id, files_metadata=True)
            else:
                info = self.api.dataset_info(repo_id, files_metadata=True)

            if info.siblings:
                return sum(f.size for f in info.siblings if f.size)
            return 0
        except Exception:
            return 0

    def backup_exists(self, repo_name: str, repo_type: str = "model") -> bool:
        """Check if backup already exists in backup org."""
        backup_id = f"{BACKUP_ORG}/{repo_name}"
        try:
            if repo_type == "model":
                self.api.model_info(backup_id)
            else:
                self.api.dataset_info(backup_id)
            return True
        except RepositoryNotFoundError:
            return False

    def backup_model(self, model_id: str) -> bool:
        """Backup a single model to the backup org."""
        model_name = model_id.split('/')[-1]
        backup_id = f"{BACKUP_ORG}/{model_name}"

        self.stats['models_checked'] += 1

        # Check if already backed up
        if self.backup_exists(model_name, "model"):
            logger.info(f"  ⏭️  {model_name}: Already exists in backup org, skipping")
            self.stats['models_skipped'] += 1
            return True

        # Get size for logging
        size_bytes = self.get_repo_size(model_id, "model")
        size_gb = size_bytes / (1024**3)

        logger.info(f"  📦 {model_name}: {size_gb:.1f} GB")

        if self.dry_run:
            logger.info(f"     [DRY RUN] Would download from {model_id}")
            logger.info(f"     [DRY RUN] Would upload to {backup_id}")
            self.stats['models_backed_up'] += 1
            return True

        # Actually perform the backup
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = Path(tmpdir) / model_name

                # Download from source
                logger.info(f"     Downloading from {model_id}...")
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_path),
                    repo_type="model",
                )

                # Create backup repo
                logger.info(f"     Creating backup repo {backup_id}...")
                self.api.create_repo(
                    repo_id=backup_id,
                    repo_type="model",
                    exist_ok=True,
                    private=True,  # Keep backups private
                )

                # Upload to backup
                logger.info(f"     Uploading to {backup_id}...")
                self.api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=backup_id,
                    repo_type="model",
                )

                logger.info(f"  ✅ {model_name}: Backup complete")
                self.stats['models_backed_up'] += 1
                return True

        except Exception as e:
            logger.error(f"  ❌ {model_name}: Backup failed - {e}")
            self.stats['models_failed'] += 1
            return False

    def backup_dataset(self, dataset_id: str) -> bool:
        """Backup a single dataset to the backup org."""
        dataset_name = dataset_id.split('/')[-1]
        backup_id = f"{BACKUP_ORG}/{dataset_name}"

        self.stats['datasets_checked'] += 1

        # Check if already backed up
        if self.backup_exists(dataset_name, "dataset"):
            logger.info(f"  ⏭️  {dataset_name}: Already exists in backup org, skipping")
            self.stats['datasets_skipped'] += 1
            return True

        # Get size for logging
        size_bytes = self.get_repo_size(dataset_id, "dataset")
        size_mb = size_bytes / (1024**2)

        logger.info(f"  📦 {dataset_name}: {size_mb:.1f} MB")

        if self.dry_run:
            logger.info(f"     [DRY RUN] Would download from {dataset_id}")
            logger.info(f"     [DRY RUN] Would upload to {backup_id}")
            self.stats['datasets_backed_up'] += 1
            return True

        # Actually perform the backup
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = Path(tmpdir) / dataset_name

                # Download from source
                logger.info(f"     Downloading from {dataset_id}...")
                snapshot_download(
                    repo_id=dataset_id,
                    local_dir=str(local_path),
                    repo_type="dataset",
                )

                # Create backup repo
                logger.info(f"     Creating backup repo {backup_id}...")
                self.api.create_repo(
                    repo_id=backup_id,
                    repo_type="dataset",
                    exist_ok=True,
                    private=True,
                )

                # Upload to backup
                logger.info(f"     Uploading to {backup_id}...")
                self.api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=backup_id,
                    repo_type="dataset",
                )

                logger.info(f"  ✅ {dataset_name}: Backup complete")
                self.stats['datasets_backed_up'] += 1
                return True

        except Exception as e:
            logger.error(f"  ❌ {dataset_name}: Backup failed - {e}")
            self.stats['datasets_failed'] += 1
            return False

    def print_summary(self):
        """Print backup summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Models checked:    {self.stats['models_checked']}")
        logger.info(f"Models backed up:  {self.stats['models_backed_up']}")
        logger.info(f"Models skipped:    {self.stats['models_skipped']} (already exist)")
        logger.info(f"Models failed:     {self.stats['models_failed']}")
        logger.info("")
        logger.info(f"Datasets checked:  {self.stats['datasets_checked']}")
        logger.info(f"Datasets backed up:{self.stats['datasets_backed_up']}")
        logger.info(f"Datasets skipped:  {self.stats['datasets_skipped']} (already exist)")
        logger.info(f"Datasets failed:   {self.stats['datasets_failed']}")
        logger.info("=" * 60)

        if self.dry_run:
            logger.info("")
            logger.info("This was a DRY RUN. To actually perform backup, remove --dry-run flag.")


def main():
    parser = argparse.ArgumentParser(
        description="Backup HuggingFace org models and datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Safe dry run first (recommended)
  python hf-backup.py --dry-run --all

  # Test with single small model
  python hf-backup.py --model CompassioninMachineLearning/some-small-model

  # Backup all models (after testing)
  python hf-backup.py --all-models

  # Backup everything
  python hf-backup.py --all

Environment:
  HF_TOKEN    HuggingFace token with appropriate permissions
              Or run: huggingface-cli login
        """
    )

    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Simulate backup without making changes (default: True)')
    parser.add_argument('--no-dry-run', action='store_true',
                        help='Actually perform the backup (disables dry-run)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    # What to backup
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true',
                       help='Backup all models and datasets')
    group.add_argument('--all-models', action='store_true',
                       help='Backup all models only')
    group.add_argument('--all-datasets', action='store_true',
                       help='Backup all datasets only')
    group.add_argument('--model', type=str,
                       help='Backup specific model (full repo ID)')
    group.add_argument('--dataset', type=str,
                       help='Backup specific dataset (full repo ID)')

    args = parser.parse_args()

    # Handle dry-run logic
    dry_run = not args.no_dry_run

    if not any([args.all, args.all_models, args.all_datasets, args.model, args.dataset]):
        parser.print_help()
        print("\n⚠️  No backup target specified. Use --all, --all-models, --model, etc.")
        sys.exit(1)

    # Check for token
    token = os.environ.get('HF_TOKEN') or HfFolder.get_token()
    if not token:
        logger.error("No HuggingFace token found!")
        logger.error("Set HF_TOKEN environment variable or run: huggingface-cli login")
        sys.exit(1)

    # Initialize backup
    backup = HFBackup(dry_run=dry_run, verbose=args.verbose)

    # Verify access
    if not backup.verify_access():
        sys.exit(1)

    logger.info("")

    # Perform backup based on args
    if args.model:
        logger.info(f"Backing up single model: {args.model}")
        backup.backup_model(args.model)

    elif args.dataset:
        logger.info(f"Backing up single dataset: {args.dataset}")
        backup.backup_dataset(args.dataset)

    elif args.all_models or args.all:
        models = backup.list_source_models()
        logger.info("")
        logger.info("Starting model backup...")
        for model in tqdm(models, desc="Models", disable=args.verbose):
            backup.backup_model(model.id)

    if args.all_datasets or args.all:
        datasets = backup.list_source_datasets()
        logger.info("")
        logger.info("Starting dataset backup...")
        for dataset in tqdm(datasets, desc="Datasets", disable=args.verbose):
            backup.backup_dataset(dataset.id)

    # Print summary
    backup.print_summary()


if __name__ == "__main__":
    main()
