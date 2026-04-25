"""
Minimal test to verify dashboard integration in training flow
"""
import sys
print(f"Python: {sys.executable}")
print(f"Args: {sys.argv}")

# Test 1: Can we import the dashboard client?
print("\n[Test 1] Importing Client...")
try:
    from qtrader.client import Client
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Can we create a client?
print("\n[Test 2] Creating Client...")
try:
    client = Client("http://localhost:8765", enabled=True)
    print(f"✓ Client created")
    print(f"  Connected: {client.is_connected}")
except Exception as e:
    print(f"✗ Client creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Can we send an event?
print("\n[Test 3] Sending test event...")
try:
    client.emit_episode_start(
        episode=1,
        total_episodes=10,
        config={"test": True}
    )
    print("✓ Event queued")

    import time
    print("  Waiting 2 seconds for background thread...")
    time.sleep(2)

    print(f"  Queue size: {client.event_queue.qsize()}")
    print(f"  Connection: {client.is_connected}")
    print(f"  Failed attempts: {client.failed_attempts}")
except Exception as e:
    print(f"✗ Event send failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Simulate what train.py does
print("\n[Test 4] Simulating train.py flow...")
try:
    dashboard_url = "http://localhost:8765"
    dashboard_client = None

    if dashboard_url:
        print(f"  Dashboard URL is set: {dashboard_url}")
        dashboard_client = Client(dashboard_url=dashboard_url, enabled=True)
        print(f"  ✓ Dashboard client initialized")
        print(f"    Connected: {dashboard_client.is_connected}")
    else:
        print(f"  Dashboard URL is None - client would not be created")

    if dashboard_client:
        print(f"  Client object exists: {dashboard_client}")
    else:
        print(f"  ERROR: Client is None!")

except Exception as e:
    print(f"✗ Train.py simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)

client.stop()
