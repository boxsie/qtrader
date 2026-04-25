"""
Quick integration test - check if dashboard client sends events properly
"""
import time
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from qtrader.client import Client

print("Testing Dashboard Client Integration")
print("=" * 60)

# Create client
print("\n1. Creating dashboard client...")
client = Client(
    dashboard_url="http://localhost:8765",
    enabled=True
)

print(f"   Client created: enabled={client.enabled}, connected={client.is_connected}")

# Send episode start
print("\n2. Sending episode_start event...")
client.emit_episode_start(
    episode=1,
    total_episodes=5,
    config={"test": True}
)

# Wait for background thread to process
print("   Waiting 2 seconds for background thread...")
time.sleep(2)

# Send step update
print("\n3. Sending step_update event...")
client.emit_step_update(
    episode=1,
    step=10,
    action=1,
    reward=0.5,
    price=42000.0,
    stats={
        "portfolio_value": 1050.0,
        "cash": 500.0,
        "roi": 5.0
    }
)

print("   Waiting 2 seconds...")
time.sleep(2)

# Send trade executed
print("\n4. Sending trade_executed event...")
client.emit_trade_executed(
    episode=1,
    step=15,
    trade_type="BUY",
    price=42000.0,
    quantity=0.01,
    total_value=420.0
)

print("   Waiting 2 seconds...")
time.sleep(2)

# Check queue status
print(f"\n5. Queue size: {client.event_queue.qsize()}")
print(f"   Connection status: {client.is_connected}")
print(f"   Failed attempts: {client.failed_attempts}")

print("\n" + "=" * 60)
print("Test complete! Check dashboard in browser.")
print("=" * 60)

# Keep alive a bit longer
time.sleep(3)
client.stop()
