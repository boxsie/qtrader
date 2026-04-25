"""Download BTC/USD 1-minute OHLCV data from Binance Vision bulk archives.

Produces a CSV in the schema expected by agent/ticker.py:
    Timestamp(unix-seconds), Open, High, Low, Close, Volume

Binance Vision serves free monthly zip archives at
https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/
with no rate limit and no API key. BTCUSDT was listed on 2017-08, so the
earliest reproducible month is 2017-08.

Usage:
    python scripts/fetch_btc_data.py --start 2017-08 --end 2025-12
    python scripts/fetch_btc_data.py --start 2024-01 --end 2024-03 --output data/sample.csv
    python scripts/fetch_btc_data.py --start 2026-01 --end 2026-03 --append
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BINANCE_VISION_URL = (
    "https://data.binance.vision/data/spot/monthly/klines"
    "/{symbol}/1m/{symbol}-1m-{year:04d}-{month:02d}.zip"
)

UPSTREAM_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

OUTPUT_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class Month:
    year: int
    month: int

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def parse_month(value: str) -> Month:
    try:
        year_str, month_str = value.split("-")
        year, month = int(year_str), int(month_str)
        if not 1 <= month <= 12:
            raise ValueError
        return Month(year, month)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM, got {value!r}"
        ) from exc


def iter_months(start: Month, end: Month):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield Month(y, m)
        m += 1
        if m > 12:
            m = 1
            y += 1


def fetch_month(symbol: str, month: Month, session: requests.Session) -> pd.DataFrame | None:
    url = BINANCE_VISION_URL.format(symbol=symbol, year=month.year, month=month.month)
    resp = session.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"empty archive for {month}")
        with zf.open(names[0]) as fh:
            df = pd.read_csv(fh, header=None, names=UPSTREAM_COLUMNS)

    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    # Binance ships open_time in milliseconds; Ticker expects seconds.
    df["open_time"] = (df["open_time"] // 1000).astype("int64")
    df.columns = OUTPUT_COLUMNS
    return df


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("no rows fetched")
    if df["Timestamp"].duplicated().any():
        raise RuntimeError("duplicate timestamps in fetched data")
    if not df["Timestamp"].is_monotonic_increasing:
        raise RuntimeError("timestamps not monotonic")
    if df.isna().any().any():
        raise RuntimeError("NaN values in fetched data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=parse_month, required=True, help="first month, YYYY-MM (inclusive)")
    parser.add_argument("--end", type=parse_month, required=True, help="last month, YYYY-MM (inclusive)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol (default: BTCUSDT)")
    parser.add_argument("--output", type=Path, default=Path("data/btcusd_1-min_data.csv"),
                        help="output CSV path (default: data/btcusd_1-min_data.csv)")
    parser.add_argument("--append", action="store_true",
                        help="append to existing output, only writing rows newer than its max Timestamp")
    parser.add_argument("--allow-missing", action="store_true",
                        help="skip months that return 404 instead of failing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("fetch_btc_data")

    if (args.start.year, args.start.month) > (args.end.year, args.end.month):
        parser.error("--start must be <= --end")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_max_ts: int | None = None
    if args.append and args.output.exists():
        existing = pd.read_csv(args.output, usecols=["Timestamp"])
        if not existing.empty:
            existing_max_ts = int(existing["Timestamp"].max())
            log.info("appending: existing max Timestamp = %s", existing_max_ts)

    session = requests.Session()
    chunks: list[pd.DataFrame] = []

    for month in iter_months(args.start, args.end):
        log.info("fetching %s %s", args.symbol, month)
        df = fetch_month(args.symbol, month, session)
        if df is None:
            msg = f"no archive for {args.symbol} {month}"
            if args.allow_missing:
                log.warning("%s (skipping)", msg)
                continue
            log.error("%s; pass --allow-missing to skip", msg)
            return 2
        if existing_max_ts is not None:
            df = df[df["Timestamp"] > existing_max_ts]
        if not df.empty:
            chunks.append(df)
            log.info("  %d rows", len(df))

    if not chunks:
        log.info("no new rows to write")
        return 0

    fresh = pd.concat(chunks, ignore_index=True)
    validate(fresh)

    if args.append and args.output.exists() and existing_max_ts is not None:
        fresh.to_csv(args.output, mode="a", header=False, index=False)
        total = sum(1 for _ in args.output.open()) - 1
        log.info("appended %d rows; %s now has %d rows", len(fresh), args.output, total)
    else:
        fresh.to_csv(args.output, index=False)
        log.info("wrote %d rows to %s", len(fresh), args.output)

    log.info(
        "date range: %s -> %s",
        pd.to_datetime(int(fresh["Timestamp"].iloc[0]), unit="s"),
        pd.to_datetime(int(fresh["Timestamp"].iloc[-1]), unit="s"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
