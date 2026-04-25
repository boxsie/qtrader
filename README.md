# QTrader — Deep Reinforcement Learning Trading Agent

> Cryptocurrency trading research system using PyTorch Double DQN with LSTM,
> multi-timeframe analysis, and a real-time dashboard. **Research project, not
> trading advice — see the warnings at the bottom before doing anything with
> real money.**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 What is QTrader?

QTrader is a deep reinforcement learning system that learns to trade cryptocurrencies by:
- **Analyzing** multiple timeframes simultaneously (1m to 4h)
- **Learning** from historical price data and technical indicators
- **Optimizing** for portfolio growth with intelligent risk management
- **Adapting** strategy based on market conditions

### Key Innovation: Multi-Timeframe HODL Strategy

Unlike traditional RL trading agents that try to time every move, QTrader implements a **HODL-based approach** where it:
- Learns **when to enter** positions (buying the dip)
- Learns **position sizing** based on market volatility (10%, 25%, 50%, 100%)
- Learns **when to exit** (taking profits or cutting losses)
- Uses **trailing stops** and **take-profit** levels automatically

This matches real-world crypto trading better than high-frequency approaches.

## 📊 Architecture

```
┌─────────────────┐         HTTP/WS          ┌─────────────────┐
│                 │ ◄────────────────────────┤                 │
│  Training Agent │  Real-time Events       │    Dashboard    │
│  (PyTorch DQN)  │                         │  (Web UI)       │
│                 │ ─────────────────────── │                 │
└────────┬────────┘                          └─────────────────┘
         │
         ├─ Double DQN + LSTM
         ├─ Prioritized Experience Replay
         ├─ Multi-Timeframe State (5m, 15m, 30m, 1h, 4h)
         ├─ Advanced Technical Indicators (RSI, MACD, BB, etc.)
         └─ Risk Management (Position Sizing, Stop Loss, Take Profit)
```

**Fully Decoupled**: Agent and Dashboard can run independently on different machines!

## ✨ Features

### Training Agent
- 🧠 **Double DQN with Dueling Architecture** - Stable, efficient learning
- 📈 **LSTM Networks** - Captures temporal patterns in price movements
- 🎯 **Prioritized Experience Replay** - Learns from important transitions
- 📊 **Multi-Timeframe Analysis** - Sees market from multiple perspectives
- 🛡️ **Advanced Risk Management** - Dynamic position sizing, stop-loss, take-profit
- ⚡ **GPU Accelerated** - CUDA support with mixed precision training
- 💾 **Automatic Checkpointing** - Saves best models during training
- 📁 **Config-Driven** - No code changes for different strategies

### Real-Time Dashboard
- 📊 **Live Portfolio Tracking** - Balance, P&L, ROI updated every second
- 📈 **Performance Metrics** - Win rate, Sharpe ratio, max drawdown
- 🎨 **Trading Exchange UI** - Professional Bootstrap 5 interface
- 📉 **Real-Time Charts** - Price and reward visualization
- 💸 **Trade History Feed** - Live buy/sell activity
- 🔌 **WebSocket Streaming** - Sub-second latency
- 🌐 **Remote Monitoring** - Deploy dashboard on separate machine/cloud

## 🚀 Quick Start

### 1. Install Training Agent

```bash
git clone https://github.com/boxsie/qtrader.git
cd qtrader

# GPU build — installs torch 2.5.1+cu121 from the PyTorch index.
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r agent/requirements.txt

# CPU-only (works for tests + smoke runs):
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install -r agent/requirements.txt
```

### 2. Fetch market data

The BTC/USD 1-minute CSV is **not** committed. Download it from Binance Vision
(free, no API key) into `data/`:

```bash
python scripts/fetch_btc_data.py --start 2017-08 --end 2025-12
```

See [`docs/DATA.md`](docs/DATA.md) for CLI options and the output schema.

### 3. Start Training

```bash
cd agent

# Basic training (50 episodes, fast config)
python train.py --config fast --profile day_trader

# Full training with dashboard monitoring
python train.py --config full --profile day_trader --dashboard http://localhost:8765

# Custom episode count
python train.py --config fast --profile scalper --episodes 200
```

### 4. Launch Dashboard (Optional)

```bash
# In a new terminal
cd dashboard
pip install -r requirements.txt
python server.py
```

Open browser to **http://localhost:8765** and watch your agent train in real-time!

## 📁 Project Structure

```
qtrader/
├── agent/                      # Training Agent (Self-Contained)
│   ├── config/                 # Training configurations
│   │   ├── fast.yaml          # Quick training (50 episodes)
│   │   └── full.yaml          # Full training (1000 episodes)
│   ├── profiles/               # Trading strategies
│   │   ├── scalper.yaml       # 1m-15m timeframes
│   │   ├── day_trader.yaml    # 15m-4h timeframes
│   │   └── swing_trader.yaml  # 1h-1d timeframes
│   ├── qlearn/                 # Deep RL models
│   ├── train.py               # Main training script
│   └── requirements.txt       # Agent dependencies
│
├── dashboard/                  # Dashboard Server (Standalone)
│   ├── server.py              # FastAPI WebSocket server
│   ├── index.html             # Web UI with playback controls
│   ├── history/               # Saved run recordings (auto-generated)
│   └── requirements.txt       # Dashboard dependencies
│
├── dataservice/                # Experimental: HTTP API for OHLCV + sentiment
│   └── README.md              # (not required for training)
│
├── shared/                     # Shared event contracts
│   └── events.py              # Event models
│
├── scripts/
│   └── fetch_btc_data.py      # Download BTCUSDT 1-min data from Binance Vision
│
├── tests/                      # pytest suite
├── docs/                       # DATA.md, REWARDS.md, DEPLOYMENT.md
├── examples/                   # Reward configuration examples
│
├── data/                       # Market data (gitignored, populated by fetch script)
│   └── btcusd_1-min_data.csv
│
├── runs/                       # Training outputs (gitignored, auto-generated)
│   └── [config]/[hash]/[profile]/[hash]/[timestamp]/
│       ├── best_model.pt
│       ├── run_log.jsonl
│       └── run_summary.txt
│
└── README.md                   # This file
```

## ⚙️ Configuration

### Training Configs (`agent/config/`)

**fast.yaml** - Quick experimentation (50 episodes)
```yaml
training:
  num_episodes: 50
  batch_size: 256
memory:
  capacity: 10000
model:
  hidden_size: 128
  lstm_layers: 2
```

**full.yaml** - Production training (1000 episodes)
```yaml
training:
  num_episodes: 1000
  batch_size: 512
memory:
  capacity: 100000
model:
  hidden_size: 256
  lstm_layers: 3
```

### Trading Profiles (`agent/profiles/`)

**day_trader.yaml** - Intraday swings (recommended)
```yaml
timeframe_weights:
  5m: 0.15
  15m: 0.45   # Primary
  30m: 0.15
  1h: 0.20
  4h: 0.05
```

**scalper.yaml** - Short-term trades
```yaml
timeframe_weights:
  1m: 0.30
  5m: 0.40   # Primary
  15m: 0.20
  30m: 0.10
```

**swing_trader.yaml** - Multi-day holds
```yaml
timeframe_weights:
  1h: 0.20
  4h: 0.45   # Primary
  1d: 0.35
```

## 🎮 Actions

The agent can take 9 discrete actions per step:

| Action | Description |
|--------|-------------|
| 0 | **HOLD** - Do nothing |
| 1-4 | **BUY** 10% / 25% / 50% / 100% of cash |
| 5-8 | **SELL** 10% / 25% / 50% / 100% of position |

Position sizing is **dynamic** based on volatility (higher vol = smaller positions).

## 📈 Reward Shaping

QTrader uses a sophisticated multi-component reward system:

1. **Immediate Rewards** (30%)
   - Trading fees (small penalty)
   - Realized P&L on position closes

2. **Timing Rewards** (20%)
   - Retrospective evaluation of past decisions
   - "Did buying 15 steps ago turn out good?"

3. **Portfolio Growth Rewards** (50%)
   - New portfolio highs = BIG rewards
   - Encourages long-term growth over quick wins

This prevents overfitting to individual trades and focuses on **total portfolio performance**.

## 🛡️ Risk Management

Built-in risk controls:

- **Position Sizing**: Volatility-adjusted (max 15% of portfolio)
- **Stop Loss**: Dynamic 2% trailing stop
- **Take Profit**: Automatic 4% profit taking
- **Max Drawdown**: Training stops at 20% drawdown
- **Fee Modeling**: 0.15% per trade (Coinbase standard)

## 📊 Dashboard Features

The web dashboard provides real-time monitoring with **full playback capabilities**:

### Live Monitoring

**Portfolio Panel:**
- Portfolio value with color-coded P&L
- Cash vs crypto allocation
- ROI percentage
- Unrealized P&L on open positions

**Performance Metrics:**
- Total trades (buys/sells breakdown)
- Win rate (decision quality, not just profitable trades)
- Average profit per trade
- Total fees paid
- Current market price

**Training Metrics:**
- Episode progress bar
- Epsilon decay (exploration → exploitation)
- Neural network loss
- Average reward per step

**Live Charts:**
- Price chart with real-time updates
- Reward chart showing agent performance
- Smooth animations, auto-scaling

**Trade Feed:**
- Live trade history (last 50 trades)
- Color-coded BUY (green) / SELL (red)
- Price, quantity, and total value

### 🎬 Playback System

**Record, replay, and analyze training runs:**

**Auto-Recording:**
- Every training run automatically recorded with unique ID
- All events saved to `dashboard/history/` as JSON
- Auto-saves when new run starts (run_id change detected)
- User notified with toast message on save

**History Browser:**
- Click **History** button to browse all saved runs
- See metadata: timestamp, duration, episodes, ROI, event count
- Load any run for instant replay

**Playback Controls:**
- ▶️ **Play/Pause** - Control playback speed
- 🛑 **Stop** - Exit playback, return to live mode
- 📊 **Timeline Slider** - Seek to any event (rebuilds state)
- ⚡ **Speed Control** - 0.5x to 10x playback speed
- 📍 **Position Display** - Current event / total events

**Mode Indicators:**
- **LIVE** mode - Green dot, WebSocket connected
- **PLAYBACK** mode - Yellow indicator, disconnected from live
- Clear visual distinction between modes

**Smart Features:**
- Non-blocking recording (zero performance impact)
- State reconstruction on timeline seek (accurate historical replay)
- Auto-save on run change (no manual saves needed)
- Handles corrupted files and edge cases gracefully

**Example Workflow:**
```bash
# 1. Start training with dashboard
python train.py --dashboard http://localhost:8765

# 2. Training auto-records all events

# 3. Start new run → previous run auto-saves
python train.py --dashboard http://localhost:8765

# 4. Click History → Browse → Load → Play!
# Replay entire run at any speed, seek anywhere
```

**Storage Format:**
```json
{
  "run_id": "20251005_143022",
  "metadata": {
    "start_time": "2025-10-05T14:30:22",
    "duration_seconds": 2725,
    "total_episodes": 100,
    "final_stats": {"roi": 15.5, ...}
  },
  "events": [...],  // All events with timestamps
  "saved_at": "2025-10-05T15:15:47"
}
```

Files saved to: `dashboard/history/run_{run_id}_{timestamp}.json`

## 🔧 Advanced Usage

### Training from Checkpoint

```bash
python train.py --resume runs/fast/.../20251005_123456/best_model.pt
```

### Evaluation Mode

```bash
python train.py --mode eval --model runs/.../best_model.pt --episodes 10
```

### Remote Dashboard

```bash
# On the monitoring machine
cd dashboard
python server.py --host 0.0.0.0 --port 8765

# On the GPU training machine
cd agent
python train.py --dashboard http://<dashboard-host>:8765
```

### Cloud Dashboard Deployment

Deploy dashboard to Heroku/Railway, then:
```bash
python train.py --dashboard https://your-dashboard.railway.app
```

## 📚 Documentation

- **[docs/DATA.md](docs/DATA.md)** — fetching market data and the CSV schema
- **[docs/REWARDS.md](docs/REWARDS.md)** — modular reward system and trader profiles
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — Docker Compose / Kubernetes / bare-metal
- **[dataservice/README.md](dataservice/README.md)** — experimental HTTP data API

## 🧪 Development

```bash
pytest tests/ -v            # reward + profile sanity checks
python agent/test_data.py   # validate your CSV's columns
```

## 🎯 Current Performance

**Latest Training Run** (Day Trader, 41 episodes):
- **ROI**: 11.45%
- **Win Rate**: 65%
- **Trades**: 19 total
- **Sharpe Ratio**: 1.82
- **Max Drawdown**: -4.2%

See `runs/` directory for detailed logs and training metrics. Monitor training in real-time via the web dashboard at `http://localhost:8765`.

## ⚠️ Important Notes

### Before Live Trading

**🚨 DO NOT TRADE LIVE WITHOUT:**

1. ✅ **Extended Training** - Run 200+ episodes minimum
2. ✅ **Out-of-Sample Testing** - Test on 2023, 2024 data (different market regimes)
3. ✅ **Paper Trading** - 30 days on real-time data
4. ✅ **Risk Metrics** - Verify Sharpe ratio > 1.5, max drawdown < 10%
5. ✅ **Baseline Comparison** - Must beat buy-and-hold
6. ✅ **Stress Testing** - Flash crashes, high volatility periods
7. ✅ **Fee Reconciliation** - Verify exchange fees match model (0.15%)
8. ✅ **Position Limits** - Set max exposure per trade
9. ✅ **Circuit Breakers** - Auto-stop on excessive drawdown

### Known Limitations

- **Sample Size**: Most experiments to date are small (need 200+ episodes for stability).
- **Slippage Modeling**: Configurable but conservative defaults; doesn't capture book impact at scale.
- **No Liquidity Constraints**: Assumes orders always fill at the modelled price.
- **Single Asset**: BTC/USD only.
- **Training Data**: As far back as the fetch script can pull from Binance Vision (BTCUSDT listed 2017-08).

Open an issue if you'd like to help with any of these.

## 🤝 Contributing

Contributions welcome! Areas needing work:

- Multi-asset portfolio support
- Additional technical indicators
- Alternative reward functions
- Hyperparameter optimization
- Live trading connectors (Coinbase, Binance)
- Ensemble methods

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Dashboard backend
- [Bootstrap 5](https://getbootstrap.com/) - UI framework
- [Chart.js](https://www.chartjs.org/) - Charting library
- [TA-Lib](https://github.com/mrjbq7/ta-lib) - Technical indicators

---

**⚡ Start training:** `cd agent && python train.py --config fast --profile day_trader`

**📊 Monitor progress:** Open http://localhost:8765 in your browser

**🚀 Happy trading!**
