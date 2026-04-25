# Dashboard Quick Start Guide

Get the QTrader dashboard up and running in 3 minutes! 🚀

## Step 1: Install Dashboard Dependencies

```bash
cd dashboard
pip install -r requirements.txt
```

This installs:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- WebSockets
- Pydantic (data validation)

## Step 2: Start the Dashboard Server

```bash
python server.py
```

You should see:
```
============================================================
QTrader Dashboard Server
============================================================
Dashboard UI:    http://127.0.0.1:8765
WebSocket:       ws://127.0.0.1:8765/ws
Events endpoint: POST http://127.0.0.1:8765/api/events
============================================================
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8765
```

## Step 3: Open Dashboard in Browser

Open http://localhost:8765 in your browser.

You should see the dashboard with "Disconnected" status - that's normal! It's waiting for QTrader to start.

## Step 4: Test Dashboard (Optional)

In a **new terminal**, test the dashboard with mock data:

```bash
cd dashboard
python test_dashboard.py
```

You'll see simulated trading activity appear in the dashboard! This verifies everything is working.

## Step 5: Run QTrader with Dashboard

In a **new terminal**, run training with dashboard enabled:

```bash
# Make sure you're in the qtrader root directory
cd ..

# Install requests dependency if needed
pip install requests

# Run training with dashboard
python train.py --config fast --profile day_trader --dashboard http://localhost:8765
```

## What You'll See

The dashboard will show:

### Portfolio Panel (Left)
- **Portfolio Value**: Total value of cash + crypto
- **Cash**: Available cash balance
- **Crypto**: Value of crypto holdings
- **ROI**: Return on investment %
- **Unrealized P&L**: Profit/loss on open positions

### Performance Metrics (Left)
- **Trades**: Total, buys, sells
- **Win Rate**: % of good decisions
- **Avg Profit**: Average profit per trade
- **Total Fees**: Cumulative trading fees
- **Current Price**: Latest market price

### Training Metrics (Left)
- **Episode Progress**: Current episode / total
- **Epsilon**: Exploration vs exploitation
- **Loss**: Neural network training loss
- **Avg Reward**: Average reward per step

### Charts (Middle)
- **Price Chart**: Real-time price movements
- **Reward Chart**: Agent reward over time

### Recent Trades (Right)
- Live feed of executed trades
- Color-coded: Green (BUY), Red (SELL)
- Shows price, quantity, and total value

## Configuration

### Custom Port

```bash
# Dashboard on port 9000
python server.py --port 9000

# Then run training with:
python train.py --dashboard http://localhost:9000
```

### Remote Dashboard

```bash
# Dashboard server (allow external connections)
python server.py --host 0.0.0.0 --port 8765

# From another machine
python train.py --dashboard http://<dashboard-host>:8765
```

## Troubleshooting

### Dashboard won't start?
```bash
# Check if port is already in use
netstat -an | grep 8765

# Or use a different port
python server.py --port 9000
```

### QTrader can't connect?
```bash
# Test dashboard endpoint
curl http://localhost:8765/health

# Should return: {"status":"healthy",...}
```

### Dashboard shows "Disconnected"?
- Make sure you started training with `--dashboard` flag
- Check dashboard server logs for connection attempts
- Verify the URL matches: `http://localhost:8765`

### No data showing up?
- Check browser console for errors (F12)
- Verify WebSocket connection is established
- Try refreshing the page

## Next Steps

- Read [DASHBOARD.md](../DASHBOARD.md) for full documentation
- Customize the UI by editing `index.html`
- Add custom events in `dashboard_client.py`
- Deploy to cloud for remote monitoring

## Tips

1. **Keep dashboard running** between training sessions - it remembers state
2. **Use two monitors** - dashboard on one, terminal on the other
3. **Throttle events** - step updates sent every 10 steps to reduce load
4. **Test first** - use `test_dashboard.py` before running real training
5. **Dashboard is optional** - QTrader works fine without it!

Enjoy your awesome real-time trading dashboard! 📊💰🚀
