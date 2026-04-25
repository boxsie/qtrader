# DataService

> **Status: experimental.** Not required to train the agent. The agent loads
> OHLCV from a local CSV (see [`scripts/fetch_btc_data.py`](../scripts/fetch_btc_data.py)).
> DataService is plumbing for richer state inputs in a future version —
> sentiment scoring, LLM analysis, live tickers — and is included in the
> repo so that work can be picked up without re-discovering the design.

## What it is

A standalone FastAPI microservice (port `8766`) that aggregates several data
sources behind one HTTP API:

- **OHLCV candles** from Binance REST (or CSV fallback)
- **News & sentiment** placeholders (Twitter, Reddit — needs API keys to be useful)
- **Web search** via DuckDuckGo
- **LLM-driven market analysis** via a local Ollama server

It also exposes a WebSocket (`/ws/stream`) for live tick streaming.

The companion client lives at [`agent/data_client.py`](../agent/data_client.py)
(`DataServiceClient`) — also untriggered today, kept ready for the v0.2
integration.

## Run it locally

```bash
cd dataservice
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8766
```

Or build the image with the bundled `Dockerfile` and run it however suits
your environment.

## Endpoints

| Method | Path                                       | Notes                                |
|--------|--------------------------------------------|--------------------------------------|
| GET    | `/health`                                  | Status + active stream count         |
| GET    | `/api/data/ohlcv/{symbol}`                 | Candles, params: `timeframe`, `limit`, `start`, `end`, `source` |
| GET    | `/api/data/ohlcv/{symbol}/price`           | Current price                        |
| GET    | `/api/data/ohlcv/{symbol}/latest`          | Most recent candle                   |
| GET    | `/api/news/sentiment/{symbol}`             | Aggregated sentiment (placeholder)   |
| GET    | `/api/news/sentiment/{symbol}/features`    | 5-element feature vector for RL     |
| POST   | `/api/analysis/market`                     | LLM analysis (requires Ollama)       |
| GET    | `/api/search`                              | DuckDuckGo proxy                     |
| WS     | `/ws/stream`                               | Live broadcast channel               |

`source=auto|binance|csv` controls the OHLCV provider. With `auto` the
service tries Binance first and falls back to CSV.

## Configuration

Environment variables (see `dataservice/core/config.py` for defaults):

| Var                       | Default                      | Purpose                           |
|---------------------------|------------------------------|-----------------------------------|
| `DATABASE_URL`            | `sqlite:///storage/dataservice.db` | SQLite cache             |
| `OLLAMA_URL`              | `http://localhost:11434`     | LLM endpoint                      |
| `OLLAMA_MODEL`            | `llama3.1:8b`                | LLM model name                    |
| `BINANCE_API_KEY` / `_SECRET` | unset                    | Optional Binance auth             |
| `TWITTER_BEARER_TOKEN`    | unset                        | Sentiment scraping                |
| `REDDIT_CLIENT_ID` / `_SECRET` | unset                   | Sentiment scraping                |
| `SERPER_API_KEY`          | unset                        | Search backend (alternative)      |
| `DATA_PATH`               | `/app/data`                  | CSV fallback location             |
| `CACHE_TTL`               | `300`                        | In-memory cache TTL (seconds)     |

## Roadmap

- Wire `DataServiceClient` into `agent/ticker.py` so live training can stream
  from Binance instead of loading the full CSV.
- Replace the sentiment placeholders with real provider implementations.
- Surface LLM bias scores as state features in the DQN's input vector.

Open an issue if you'd like to drive any of these.
