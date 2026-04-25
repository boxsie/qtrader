"""
Test Dashboard - Send Mock Events

Quick test script to verify dashboard is working without running full training.
Sends simulated trading events to the dashboard server.

Usage:
    python test_dashboard.py
"""

import time
import random
import requests
from datetime import datetime


def send_event(event_type, data, dashboard_url="http://localhost:8765"):
    """Send event to dashboard"""
    event = {
        "type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

    try:
        response = requests.post(
            f"{dashboard_url}/api/events",
            json=event,
            timeout=1.0
        )
        if response.status_code == 200:
            print(f"✓ Sent {event_type}")
        else:
            print(f"✗ Failed {event_type}: {response.status_code}")
    except Exception as e:
        print(f"✗ Error sending {event_type}: {e}")


def simulate_trading_session(dashboard_url="http://localhost:8765"):
    """Simulate a training session with mock data"""
    print("QTrader Dashboard Test")
    print("=" * 60)
    print(f"Dashboard URL: {dashboard_url}")
    print("=" * 60)

    # Check dashboard health
    try:
        response = requests.get(f"{dashboard_url}/health", timeout=1.0)
        if response.status_code == 200:
            print("✓ Dashboard is running\n")
        else:
            print("✗ Dashboard not responding\n")
            return
    except Exception as e:
        print(f"✗ Cannot connect to dashboard: {e}\n")
        return

    # Simulate training
    num_episodes = 3
    starting_balance = 1000.0

    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---")

        # Episode start
        send_event("episode_start", {
            "episode": episode,
            "total_episodes": num_episodes,
            "config": {"epsilon": 0.5, "batch_size": 64}
        })

        # Simulate 50 steps
        balance = starting_balance
        crypto_quantity = 0.0
        price = 40000.0
        step = 0

        for step in range(1, 51):
            # Simulate price movement
            price += random.uniform(-100, 100)
            price = max(price, 30000)  # Floor

            # Random action
            action = random.choice([0, 0, 0, 1, 2, 5, 6])  # Mostly HOLD

            # Simulate trades
            if action in [1, 2] and balance > 100:  # BUY
                percent = {1: 0.10, 2: 0.25}.get(action, 0.1)
                cash_used = balance * percent
                bought = cash_used / price
                crypto_quantity += bought
                balance -= cash_used

                send_event("trade_executed", {
                    "episode": episode,
                    "step": step,
                    "trade_type": "BUY",
                    "price": price,
                    "quantity": bought,
                    "total_value": cash_used
                })

            elif action in [5, 6] and crypto_quantity > 0:  # SELL
                percent = {5: 0.10, 6: 0.25}.get(action, 0.1)
                crypto_sold = crypto_quantity * percent
                cash_received = crypto_sold * price
                crypto_quantity -= crypto_sold
                balance += cash_received

                send_event("trade_executed", {
                    "episode": episode,
                    "step": step,
                    "trade_type": "SELL",
                    "price": price,
                    "quantity": crypto_sold,
                    "total_value": cash_received
                })

            # Portfolio value
            portfolio_value = balance + (crypto_quantity * price)
            roi = (portfolio_value - starting_balance) / starting_balance * 100

            # Step update (every 5 steps)
            if step % 5 == 0:
                send_event("step_update", {
                    "episode": episode,
                    "step": step,
                    "action": action,
                    "reward": random.uniform(-1, 2),
                    "price": price,
                    "stats": {
                        "portfolio_value": portfolio_value,
                        "cash": balance,
                        "crypto_quantity": crypto_quantity,
                        "crypto_value": crypto_quantity * price,
                        "roi": roi,
                        "num_trades": random.randint(0, 10),
                        "num_buys": random.randint(0, 5),
                        "num_sells": random.randint(0, 5),
                        "win_rate": random.uniform(0.4, 0.7),
                        "avg_profit": random.uniform(-5, 10),
                        "total_fees": random.uniform(0, 50),
                        "unrealized_pnl": (crypto_quantity * price - crypto_quantity * 40000) if crypto_quantity > 0 else 0,
                        "unrealized_pnl_pct": random.uniform(-5, 5)
                    }
                })

            time.sleep(0.1)  # Throttle

        # Episode end
        final_portfolio = balance + (crypto_quantity * price)
        send_event("episode_end", {
            "episode": episode,
            "final_stats": {
                "portfolio_value": final_portfolio,
                "roi": (final_portfolio - starting_balance) / starting_balance * 100,
                "num_trades": random.randint(5, 15)
            },
            "duration_seconds": 5.0
        })

        # Training metrics
        send_event("training_metrics", {
            "episode": episode,
            "loss": random.uniform(0.001, 0.1),
            "epsilon": max(0.01, 0.5 - episode * 0.1),
            "avg_reward": random.uniform(-1, 3)
        })

        time.sleep(1)

    print("\n" + "=" * 60)
    print("✓ Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    simulate_trading_session()
