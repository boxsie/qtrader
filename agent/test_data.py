"""
Quick data validation script - checks your CSV format without loading all data
"""

import sys
import pandas as pd
from pathlib import Path

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def test_data_format(filepath='data/coinbase.csv', sample_size=1000):
    """Test if data format is compatible"""

    print("\n" + "="*60)
    print("Data Format Validation")
    print("="*60)

    filepath = Path(filepath)

    # Check if file exists
    if not filepath.exists():
        print(f"\n❌ ERROR: File not found: {filepath}")
        print(f"   Please ensure your BTC data is at: {filepath}")
        return False

    print(f"\n✅ File found: {filepath}")
    print(f"   File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    # Load just a sample (don't load entire file)
    print(f"\n📊 Loading sample ({sample_size:,} rows)...")
    try:
        df = pd.read_csv(filepath, nrows=sample_size)
    except Exception as e:
        print(f"\n❌ ERROR reading CSV: {e}")
        return False

    # Check columns
    print(f"\n📋 Columns found: {df.columns.tolist()}")

    required_cols = ['High', 'Low', 'Close']
    volume_col = 'Volume' if 'Volume' in df.columns else 'Volume_(Currency)'

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\n❌ ERROR: Missing required columns: {missing}")
        return False

    if volume_col not in df.columns:
        print(f"\n❌ ERROR: No volume column found (need 'Volume' or 'Volume_(Currency)')")
        return False

    print(f"✅ All required columns present")

    # Check data types and values
    print(f"\n📈 Data sample:")
    print(df.head(3).to_string())

    print(f"\n📊 Statistics:")
    print(f"   Rows in sample: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    if 'Timestamp' in df.columns:
        print(f"   Time range (sample): {pd.to_datetime(df['Timestamp'].min(), unit='s')} to {pd.to_datetime(df['Timestamp'].max(), unit='s')}")

    print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    print(f"   Volume range: {df[volume_col].min():.4f} - {df[volume_col].max():.4f}")

    # Check for issues
    print(f"\n🔍 Data quality:")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"   ⚠️  Null values found:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"      {col}: {count}")
    else:
        print(f"   ✅ No null values")

    zero_volume = (df[volume_col] == 0).sum()
    print(f"   Zero volume rows: {zero_volume:,} ({zero_volume/len(df)*100:.1f}%)")

    # Test with ticker
    print(f"\n🧪 Testing with TickerModern...")
    try:
        from qtrader.ticker_modern import TickerModern
        ticker = TickerModern(filepath)
        print(f"   ✅ Successfully loaded by TickerModern")
        print(f"   Total data points: {len(ticker):,}")

        # Test getting a few ticks
        tick1, _ = ticker.get_next()
        tick2, _ = ticker.get_next()
        print(f"   ✅ Successfully read ticks")
        print(f"   Sample tick: {tick1.to_dict()}")

    except Exception as e:
        print(f"   ❌ ERROR in TickerModern: {e}")
        return False

    # Estimate memory requirements
    print(f"\n💾 Memory estimates:")

    # Get actual row count efficiently
    print(f"   Counting total rows (may take a moment)...")
    row_count = sum(1 for _ in open(filepath)) - 1  # Subtract header
    print(f"   Total rows: {row_count:,}")

    estimated_mb = row_count * len(df.columns) * 8 / 1024 / 1024  # 8 bytes per float
    print(f"   Estimated RAM needed: ~{estimated_mb:.0f} MB")

    if estimated_mb > 4000:
        print(f"   ⚠️  Large dataset! Consider:")
        print(f"      - Using only recent data (last 1-2 years)")
        print(f"      - Increasing tick_size in config.yaml")

    # Success!
    print("\n" + "="*60)
    print("✅ DATA FORMAT VALIDATION PASSED")
    print("="*60)
    print("\nYour data is ready for training!")
    print("\nNext steps:")
    print("  1. Review config.yaml settings")
    print("  2. Start dashboard: cd ../dashboard && python server.py")
    print("  3. Run training: python train.py --dashboard http://localhost:8765")
    print()

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test data format')
    parser.add_argument('--file', type=str, default='data/btcusd_1-min_data.csv',
                        help='Path to CSV file')
    parser.add_argument('--sample', type=int, default=1000,
                        help='Number of rows to sample for testing')

    args = parser.parse_args()

    success = test_data_format(args.file, args.sample)

    if not success:
        print("\n❌ Data validation failed. Please check the errors above.")
        exit(1)
