"""
Test reward loading from trader profiles.
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'agent'))

from rewards import create_reward_manager


def test_all_profiles_load():
    """pytest entry point: every profile YAML must load via create_reward_manager."""
    profiles_dir = Path(__file__).parent.parent / 'agent' / 'profiles'
    profile_files = sorted(profiles_dir.glob('*.yaml'))
    assert profile_files, f"no profile YAMLs found under {profiles_dir}"
    for profile_path in profile_files:
        assert _load_profile(profile_path), f"profile failed to load: {profile_path.name}"


def _load_profile(profile_path):
    """Test loading rewards from a profile"""
    print(f"\n{'='*60}")
    print(f"Testing: {profile_path.name}")
    print('='*60)

    # Load profile
    with open(profile_path, 'r') as f:
        profile = yaml.safe_load(f)

    print(f"Profile: {profile['name']}")
    print(f"Description: {profile['description']}")

    # Check if reward_config exists
    if 'reward_config' not in profile:
        print("[ERROR] No reward_config in profile!")
        return False

    # Create reward manager
    try:
        reward_manager = create_reward_manager(profile['reward_config'])
        print(f"\n[SUCCESS] Reward manager created")
        print(f"Active calculators: {len(reward_manager.calculators)}")

        for calc in reward_manager.list_calculators():
            print(f"  - {calc}")

        # Test reward calculation
        context = {
            'action': 5,
            'price': 110,
            'cash': 1100,
            'crypto_quantity': 0,
            'portfolio_value': 1100,
            'starting_cash': 1000,
            'average_buy_price': 100,
            'fee_pct': 0.0015,
            'step': 5,
            'trade_info': {
                'type': 'SELL',
                'quantity': 1.0,
                'fee': 1.65,
                'profit': 100.0,
                'profit_pct': 0.10
            }
        }

        total_reward = reward_manager.calculate_reward(context)
        breakdown = reward_manager.get_breakdown(context)

        print(f"\nTest reward calculation:")
        for name, value in breakdown.items():
            print(f"  {name:30s}: {value:8.2f}")

        print(f"\n[PASS] Profile loaded successfully")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to create reward manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("Testing Reward Loading from Trader Profiles")
    print("="*60)

    profiles_dir = Path(__file__).parent.parent / 'agent' / 'profiles'

    if not profiles_dir.exists():
        print(f"[ERROR] Profiles directory not found: {profiles_dir}")
        return

    # Test all profiles
    profile_files = list(profiles_dir.glob('*.yaml'))
    results = {}

    for profile_path in sorted(profile_files):
        success = _load_profile(profile_path)
        results[profile_path.name] = success

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for profile_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {profile_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n[SUCCESS] All profiles loaded successfully!")
    else:
        print("\n[FAILURE] Some profiles failed to load")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
