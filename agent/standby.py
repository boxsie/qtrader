"""
QTrader Standby Mode Controller

Connects to the dashboard via WebSocket and waits for commands.
When a training command is received, spawns training in a subprocess.
"""

import asyncio
import json
import signal
import subprocess
import sys
from typing import Optional, Dict, Any

import websockets
from websockets.exceptions import ConnectionClosed

from logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class StandbyController:
    """Controller for standby mode - connects to dashboard and handles commands"""

    def __init__(self, dashboard_url: str):
        """
        Initialize standby controller.

        Args:
            dashboard_url: Dashboard base URL (e.g., http://qtrader-dashboard:8765)
        """
        # Convert HTTP URL to WebSocket URL
        ws_url = dashboard_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{ws_url}/ws/agent"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.training_process: Optional[subprocess.Popen] = None
        self.running = True
        self.reconnect_delay = 5  # seconds

        logger.info(f"Standby controller initialized")
        logger.info(f"  Dashboard URL: {dashboard_url}")
        logger.info(f"  WebSocket URL: {self.ws_url}")

    async def connect(self) -> bool:
        """Connect to dashboard WebSocket"""
        try:
            logger.info(f"Connecting to dashboard at {self.ws_url}...")
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10
            )
            logger.info("Connected to dashboard successfully")

            # Send initial status
            await self.send_status("ready", "Agent connected and ready")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to dashboard: {e}")
            self.ws = None
            return False

    async def send_status(self, status: str, message: str = ""):
        """Send status update to dashboard"""
        if self.ws:
            try:
                await self.ws.send(json.dumps({
                    "type": "status",
                    "data": {
                        "status": status,
                        "message": message,
                        "training_active": self.training_process is not None and self.training_process.poll() is None
                    }
                }))
            except Exception as e:
                logger.error(f"Failed to send status: {e}")

    async def handle_command(self, command: Dict[str, Any]):
        """Handle command from dashboard"""
        cmd_type = command.get("type")
        cmd_data = command.get("data", {})

        logger.info(f"Received command: {cmd_type}")
        logger.info(f"  Data: {cmd_data}")

        if cmd_type == "start_training":
            await self.start_training(cmd_data)
        elif cmd_type == "stop_training":
            await self.stop_training()
        else:
            logger.warning(f"Unknown command type: {cmd_type}")

    async def start_training(self, config: Dict[str, Any]):
        """Start training subprocess"""
        # Check if already training
        if self.training_process is not None and self.training_process.poll() is None:
            logger.warning("Training already in progress")
            await self.send_status("busy", "Training already in progress")
            return

        # Build command
        profile = config.get("profile", "balanced")
        config_name = config.get("config", "fast")
        episodes = config.get("episodes")
        dashboard_url = config.get("dashboard_url", self.ws_url.replace("/ws/agent", "").replace("ws://", "http://"))

        cmd = [
            sys.executable, "train.py",
            "--mode", "train",
            "--profile", profile,
            "--config", config_name,
            "--dashboard", dashboard_url
        ]

        if episodes:
            cmd.extend(["--episodes", str(episodes)])

        logger.info(f"Starting training: {' '.join(cmd)}")

        try:
            # Start training subprocess
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            await self.send_status("training", f"Training started with profile={profile}, config={config_name}")

            # Monitor process in background
            asyncio.create_task(self._monitor_training())

        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            await self.send_status("error", f"Failed to start training: {e}")

    async def _monitor_training(self):
        """Monitor training subprocess and log output"""
        if not self.training_process:
            return

        logger.info("Monitoring training process...")

        try:
            # Stream output
            while self.training_process.poll() is None:
                if self.training_process.stdout:
                    line = self.training_process.stdout.readline()
                    if line:
                        print(line.rstrip())  # Print to container logs
                await asyncio.sleep(0.1)

            # Process finished
            exit_code = self.training_process.returncode
            logger.info(f"Training process finished with exit code: {exit_code}")

            if exit_code == 0:
                await self.send_status("ready", "Training completed successfully")
            else:
                await self.send_status("error", f"Training failed with exit code {exit_code}")

        except Exception as e:
            logger.error(f"Error monitoring training: {e}")
        finally:
            self.training_process = None

    async def stop_training(self):
        """Stop training subprocess gracefully"""
        if self.training_process is None or self.training_process.poll() is not None:
            logger.info("No training process to stop")
            await self.send_status("ready", "No training process running")
            return

        logger.info("Stopping training process...")

        try:
            # Send SIGTERM for graceful shutdown
            self.training_process.terminate()

            # Wait up to 30 seconds for graceful shutdown
            for _ in range(30):
                if self.training_process.poll() is not None:
                    break
                await asyncio.sleep(1)

            # Force kill if still running
            if self.training_process.poll() is None:
                logger.warning("Training process didn't stop gracefully, forcing kill")
                self.training_process.kill()
                await asyncio.sleep(1)

            logger.info("Training process stopped")
            await self.send_status("ready", "Training stopped")

        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            await self.send_status("error", f"Error stopping training: {e}")
        finally:
            self.training_process = None

    async def run(self):
        """Main loop - connect, listen for commands, reconnect on disconnect"""
        logger.info("Starting standby controller main loop")

        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self.running:
            # Connect to dashboard
            if not await self.connect():
                logger.info(f"Retrying in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                continue

            # Listen for commands
            try:
                async for message in self.ws:
                    if not self.running:
                        break

                    data = json.loads(message)

                    if data.get("type") == "pong":
                        continue
                    elif data.get("type") in ["start_training", "stop_training"]:
                        await self.handle_command(data)
                    else:
                        logger.debug(f"Received message: {data.get('type')}")

            except ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            # Reconnect after disconnect
            if self.running:
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)

        # Cleanup
        logger.info("Shutting down standby controller")
        await self.stop_training()
        if self.ws:
            await self.ws.close()


async def main(dashboard_url: str):
    """Entry point for standby mode"""
    controller = StandbyController(dashboard_url)
    await controller.run()


def run_standby(dashboard_url: str):
    """Run standby mode (blocking)"""
    asyncio.run(main(dashboard_url))
