"""
Configuration Hashing and Provenance Tracking

Generates deterministic hashes for configs and profiles to ensure reproducibility.
Creates hash-based directory structure: config_name/config_hash/profile_name/profile_hash/

This ensures:
1. Different configs/profiles never mix results
2. Same config/profile always uses same directory
3. Full provenance: config and profile copies saved in output dir
4. Easy to compare runs with different hyperparameters
"""

import hashlib
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import shutil
from datetime import datetime


def normalize_config(config: Dict[str, Any]) -> str:
    """
    Normalize config to canonical form for hashing.

    Removes non-deterministic fields (like paths, timestamps).
    Sorts keys for consistent ordering.
    """
    # Deep copy to avoid modifying original
    import copy
    normalized = copy.deepcopy(config)

    # Remove fields that shouldn't affect hash
    normalized.pop('paths', None)

    # Convert to JSON with sorted keys for deterministic ordering
    canonical = json.dumps(normalized, sort_keys=True, indent=2)
    return canonical


def hash_config(config: Dict[str, Any]) -> str:
    """
    Generate SHA256 hash of config (first 8 chars).

    Args:
        config: Configuration dictionary

    Returns:
        8-character hash string
    """
    canonical = normalize_config(config)
    hash_obj = hashlib.sha256(canonical.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def hash_profile(profile: Dict[str, Any]) -> str:
    """
    Generate SHA256 hash of profile (first 8 chars).

    Args:
        profile: Profile dictionary

    Returns:
        8-character hash string
    """
    # Profiles are already deterministic, just hash as-is
    canonical = json.dumps(profile, sort_keys=True, indent=2)
    hash_obj = hashlib.sha256(canonical.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def get_output_paths(config_name: str, config: Dict[str, Any],
                     profile_name: str, profile: Dict[str, Any],
                     run_id: str,
                     base_dir: str = '.') -> Dict[str, Path]:
    """
    Generate hash-based output directory paths with run_id subdirectory.

    Directory structure:
        {base_dir}/runs/
            {config_name}/{config_hash}/{profile_name}/{profile_hash}/
                config_archive/          # Shared across all runs
                {run_id}/               # Per-run directory
                    best_model.pt
                    checkpoint_ep{N}.pt
                    memory_ep{N}.pkl
                    run_summary.txt
                    run_log.jsonl

    Args:
        config_name: Name of config (e.g., 'fast_iteration', 'full_training')
        config: Configuration dictionary
        profile_name: Name of profile (e.g., 'day_trader', 'scalper')
        profile: Profile dictionary
        run_id: Run identifier (timestamp)
        base_dir: Base directory for all outputs

    Returns:
        Dictionary with 'base_dir', 'run_dir', 'config_archive_dir' paths
    """
    config_hash = hash_config(config)
    profile_hash = hash_profile(profile)

    base = Path(base_dir)

    # Build path: config_name/config_hash/profile_name/profile_hash
    subpath = f"{config_name}/{config_hash}/{profile_name}/{profile_hash}"
    base_path = base / 'runs' / subpath

    paths = {
        'base_dir': base_path,                          # Shared directory
        'run_dir': base_path / run_id,                  # This run's directory
        'config_archive_dir': base_path / 'config_archive',  # Shared archive
    }

    return paths


def save_provenance(config_name: str, config: Dict[str, Any],
                    profile_name: str, profile: Dict[str, Any],
                    archive_dir: Path,
                    run_id: str = None) -> None:
    """
    Save config and profile copies to config_archive directory for full provenance.

    Only saves if config_archive doesn't exist yet (first run with this config/profile combo).
    Multiple runs with same config/profile share the same archive.

    Creates:
        archive_dir/
            config.yaml          # Full config
            profile.yaml         # Full profile
            metadata.yaml        # Run metadata (timestamp, hashes, names)
            config_canonical.json  # Normalized config used for hashing
            profile_canonical.json # Normalized profile used for hashing
            runs.txt             # Log of all runs with this config/profile

    Args:
        config_name: Name of config
        config: Configuration dictionary
        profile_name: Name of profile
        profile: Profile dictionary
        archive_dir: Directory to save provenance files (config_archive)
        run_id: Optional run ID to log in runs.txt
    """
    archive_dir = Path(archive_dir)

    # Check if archive already exists (from previous run with same config/profile)
    is_first_run = not (archive_dir.exists() and (archive_dir / 'config.yaml').exists())

    if not is_first_run:
        print(f"\n[Provenance] Archive already exists at {archive_dir}")
        print(f"  Config Hash:  {hash_config(config)}")
        print(f"  Profile Hash: {hash_profile(profile)}")
        print(f"  (Using existing archive from previous run)")
    else:
        # Create archive directory
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Save full config and profile
        with open(archive_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        with open(archive_dir / 'profile.yaml', 'w') as f:
            yaml.dump(profile, f, default_flow_style=False, sort_keys=False)

        # Save canonical forms used for hashing
        with open(archive_dir / 'config_canonical.json', 'w') as f:
            f.write(normalize_config(config))

        with open(archive_dir / 'profile_canonical.json', 'w') as f:
            f.write(json.dumps(profile, sort_keys=True, indent=2))

        # Save metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'config_name': config_name,
            'config_hash': hash_config(config),
            'profile_name': profile_name,
            'profile_hash': hash_profile(profile),
            'description': {
                'config': f"{config_name} configuration",
                'profile': profile.get('description', f"{profile_name} trading strategy"),
            }
        }

        with open(archive_dir / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        print(f"\n[Provenance] Saved to {archive_dir}")
        print(f"  Config Hash:  {metadata['config_hash']}")
        print(f"  Profile Hash: {metadata['profile_hash']}")

    # Log this run (append to runs log)
    if run_id:
        runs_log = archive_dir / 'runs.txt'
        with open(runs_log, 'a') as f:
            f.write(f"{datetime.now().isoformat()}\t{run_id}\n")


def compare_configs(config1_path: str, config2_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare two configs and show differences.

    Args:
        config1_path: Path to first config
        config2_path: Path to second config

    Returns:
        (are_same, differences_dict)
    """
    with open(config1_path) as f:
        config1 = yaml.safe_load(f)
    with open(config2_path) as f:
        config2 = yaml.safe_load(f)

    hash1 = hash_config(config1)
    hash2 = hash_config(config2)

    are_same = (hash1 == hash2)

    # Find differences (simplified - just compare normalized forms)
    diff = {
        'hash1': hash1,
        'hash2': hash2,
        'same': are_same,
        'canonical1': normalize_config(config1),
        'canonical2': normalize_config(config2),
    }

    return are_same, diff


if __name__ == '__main__':
    # Test with example config
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"Config hash: {hash_config(config)}")
        print(f"\nCanonical form:")
        print(normalize_config(config))
