# Data acquisition

The agent loads BTC/USD 1-minute OHLCV data from a local CSV (default
`data/btcusd_1-min_data.csv`). The CSV is **not** committed — fetch it with
the bundled script.

## Quick start

```bash
# Full history from BTCUSDT listing on Binance Spot.
python scripts/fetch_btc_data.py --start 2017-08 --end 2025-12

# Smaller slice for experiments.
python scripts/fetch_btc_data.py --start 2024-01 --end 2024-03 \
    --output data/sample.csv

# Append new months without re-downloading the rest.
python scripts/fetch_btc_data.py --start 2026-01 --end 2026-03 --append
```

## Source

`scripts/fetch_btc_data.py` downloads monthly zip archives from
[Binance Vision](https://data.binance.vision/) — free, no API key, no rate
limit. Each archive holds one month of 1-minute klines; the script unzips,
projects to the schema below, and concatenates.

URL pattern:
`https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM}.zip`

## Output schema

The script writes a CSV that `agent/ticker.py` consumes directly:

| Column     | Type            | Notes                              |
|------------|-----------------|------------------------------------|
| Timestamp  | int64 (Unix s)  | Bar open time, seconds since epoch |
| Open       | float           |                                    |
| High       | float           |                                    |
| Low        | float           |                                    |
| Close      | float           |                                    |
| Volume     | float           | Base-asset volume                  |

Ticker normalises `Volume` → `Volume_(Currency)` internally, so legacy
Bitstamp-style CSVs (with `Volume_(Currency)` already present) are also
accepted.

## CLI reference

```
--start YYYY-MM          first month, inclusive (required)
--end   YYYY-MM          last month, inclusive (required)
--symbol SYMBOL          Binance pair (default: BTCUSDT)
--output PATH            CSV destination (default: data/btcusd_1-min_data.csv)
--append                 append rows newer than the existing file's max Timestamp
--allow-missing          skip months that 404 instead of failing
```

## Notes

- BTCUSDT was listed on Binance Spot in August 2017; archives before then 404.
- A full 2017-08 → present fetch is ~4 GB compressed and produces a CSV of
  several million rows. Plan disk space accordingly.
- Pre-2017 data (e.g. Bitstamp from 2012) is not available from this source.
  If you need it, sources like Kaggle's
  *bitstampUSD_1-min_data* dataset provide a compatible schema.
