# QTrader Dashboard

Standalone real-time dashboard for monitoring QTrader training and inference.

## Architecture

- **Completely independent** from qtrader
- QTrader publishes events via HTTP POST
- Dashboard broadcasts to web clients via WebSocket
- Can be deployed separately on different machines

## Quick Start

```bash
cd dashboard
pip install -r requirements.txt
python server.py
```

Open browser: http://localhost:8765

## API Endpoints

- `POST /api/events` - Receive events from qtrader
- `GET /api/state` - Get current state snapshot
- `WS /ws` - WebSocket for browser clients
- `GET /health` - Health check

## Event Types

Events that qtrader can publish:

- `episode_start` - Training episode started
- `step_update` - Step update with stats
- `trade_executed` - Trade executed
- `episode_end` - Episode completed
- `training_metrics` - Model metrics (loss, epsilon, etc.)
- `system_status` - GPU/memory usage

## Configuration

```bash
python server.py --host 0.0.0.0 --port 8765
```

## Integration with QTrader

From qtrader, use the event publisher client:

```python
from qtrader.client import Client

client = Client("http://localhost:8765")
client.emit_step_update(episode=1, step=100, ...)
```
