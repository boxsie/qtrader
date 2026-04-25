# Deployment

QTrader is structured as three services that talk over HTTP and WebSocket:

```
┌─────────────────┐  WS commands  ┌────────────────┐
│ Dashboard       │◄──────────────│ Training Agent │ (GPU)
│ FastAPI + UI    │  events       │ PyTorch DQN    │
│ port 8765       │               │                │
└─────────────────┘               └────────────────┘
        │                                   │
        ▼                                   ▼
┌──────────────────────────────────────────────────┐
│ DataService (experimental, optional)             │
│ FastAPI: OHLCV, sentiment, LLM analysis          │
│ port 8766                                        │
└──────────────────────────────────────────────────┘
```

The agent stays in **standby mode** by default and the dashboard launches
training runs over WebSocket. You can also run the agent directly from the
CLI without the dashboard.

The repo only ships the source code and the `Dockerfile` for each service.
Container orchestration (docker-compose / Kubernetes manifests / cluster
deploy scripts) is out of scope here — that's environment-specific and lives
elsewhere.

## Run it locally (no containers)

```bash
# Terminal 1: dashboard
cd dashboard && pip install -r requirements.txt && python server.py

# Terminal 2: data service (optional)
cd dataservice && pip install -r requirements.txt && \
    uvicorn server:app --host 0.0.0.0 --port 8766

# Terminal 3: agent in standby (waits for dashboard commands)
cd agent && pip install -r requirements.txt && \
    python train.py --mode standby --dashboard http://localhost:8765
```

Open <http://localhost:8765>, wait for "Agent Connected", pick a profile/config,
click **Launch Training**.

To skip the dashboard:

```bash
cd agent && python train.py --config fast --profile day_trader
```

## Build the container images

Each service ships a `Dockerfile`. From the repo root:

```bash
docker build -f agent/Dockerfile       -t qtrader-agent:latest       .
docker build -f dashboard/Dockerfile   -t qtrader-dashboard:latest   .
docker build -f dataservice/Dockerfile -t qtrader-dataservice:latest .
```

Run them however suits your environment — `docker run`, your own
docker-compose file, Kubernetes, Nomad, etc.

## Container summary

| Image                  | Base                                            | GPU | Notes                                     |
|------------------------|-------------------------------------------------|-----|-------------------------------------------|
| `qtrader-agent`        | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` | yes | torch 2.5.1+cu121                         |
| `qtrader-dashboard`    | `python:3.11-slim`                              | no  | FastAPI + WebSocket UI                    |
| `qtrader-dataservice`  | `python:3.11-slim`                              | no  | Experimental, see `dataservice/README.md` |

## Environment variables

| Var                            | Where       | Default                   | Purpose                       |
|--------------------------------|-------------|---------------------------|-------------------------------|
| `DASHBOARD_URL`                | agent       | required for standby      | WebSocket target              |
| `DATA_SERVICE_URL`             | agent       | `http://dataservice:8766` | DataService base URL          |
| `CUDA_VISIBLE_DEVICES`         | agent       | `0`                       | GPU index                     |
| `LOG_LEVEL`                    | all         | `INFO`                    | Python logging level          |
| `BINANCE_API_KEY` / `_SECRET`  | dataservice | unset                     | Live Binance auth (optional)  |
| `TWITTER_BEARER_TOKEN`         | dataservice | unset                     | Sentiment scraping (optional) |
| `REDDIT_CLIENT_ID` / `_SECRET` | dataservice | unset                     | Sentiment scraping (optional) |
| `OLLAMA_URL`                   | dataservice | `http://ollama:11434`     | LLM endpoint (optional)       |

## Health checks

```bash
curl http://localhost:8765/health      # dashboard
curl http://localhost:8766/health      # dataservice
```

The dashboard surfaces agent-connection state in the UI header.

## Troubleshooting

- **Agent shows disconnected.** Confirm `DASHBOARD_URL` resolves from inside
  the agent container; the WebSocket path is `/ws/agent`.
- **GPU not detected.** Verify `nvidia-smi` on the host, then test the
  container runtime with
  `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`.
- **Out of disk on the agent host.** `agent/runs/` grows fast — point that
  directory at a large volume, or prune old runs.
