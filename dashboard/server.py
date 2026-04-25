"""
QTrader Dashboard Server - Standalone Service

Completely independent from qtrader. Receives events via HTTP POST,
stores state, broadcasts to web clients via WebSocket.

To run:
    cd dashboard
    pip install -r requirements.txt
    python server.py

Default: http://localhost:8765
"""

import sys
from pathlib import Path

# Add parent directory to path to import shared module
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import logging
from datetime import datetime
from typing import Set, Dict, Any, Optional, List
from collections import deque
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Import shared event models
from shared.events import Event as EventModel, EventType, CommandType, Command

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

# Pydantic model for FastAPI validation (compatible with shared.events.Event)
class Event(BaseModel):
    """Event from qtrader (Pydantic wrapper for FastAPI validation)"""
    type: str
    data: Dict[str, Any]
    timestamp: Optional[str] = None
    run_id: Optional[str] = None


class DashboardState:
    """Maintains current state for new client connections"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history

        # Current session state
        self.current_run_id: Optional[str] = None
        self.current_episode: Optional[int] = None
        self.total_episodes: Optional[int] = None
        self.session_start: Optional[str] = None
        self.latest_stats: Dict[str, Any] = {}
        self.latest_metrics: Dict[str, Any] = {}
        self.system_status: Dict[str, Any] = {}
        self.is_active: bool = False
        self.current_action: Optional[int] = None
        self.current_step: Optional[int] = None
        self.current_price: Optional[float] = None
        self.current_run_metadata: Dict[str, Any] = {}
        self.max_episode_steps: Optional[int] = None

        # Episode history for current run (kept in state snapshot)
        self.episode_history: List[Dict[str, Any]] = []
        self.max_episode_history: int = 50  # Keep last 50 episodes

        # Recent events for state snapshot (not persisted)
        self.recent_events: deque = deque(maxlen=max_history)

    def update(self, event: Event):
        """Update state based on incoming event."""
        self.recent_events.append(event)

        # Track run_id changes
        new_run_id = event.run_id
        if new_run_id and new_run_id != self.current_run_id:
            self._handle_run_change(new_run_id, event.timestamp)

        # Update state based on event type
        if event.type == "episode_start":
            self.current_episode = event.data.get("episode")
            self.total_episodes = event.data.get("total_episodes")
            self.is_active = True
            if not self.session_start:
                self.session_start = event.timestamp
            # Store config in metadata
            if event.data.get("config"):
                self.current_run_metadata["config"] = event.data["config"]
                # Extract max_episode_steps from config
                self.max_episode_steps = event.data["config"].get("max_episode_steps")
            logger.info(f"Episode started: {self.current_episode}/{self.total_episodes}")

        elif event.type == "step_update":
            self.latest_stats = event.data.get("stats", {})
            self.current_action = event.data.get("action")
            self.current_step = event.data.get("step")
            self.current_price = event.data.get("price")
            # Keep episode info updated from step updates too
            if event.data.get("episode") is not None:
                old_episode = self.current_episode
                self.current_episode = event.data.get("episode")
                if old_episode != self.current_episode:
                    logger.info(f"Episode updated from step_update: {old_episode} -> {self.current_episode}")
            if event.data.get("total_episodes") is not None:
                self.total_episodes = event.data.get("total_episodes")

        elif event.type == "episode_end":
            self.latest_stats = event.data.get("final_stats", {})
            self.is_active = False

            # Add to episode history
            # Get total reward from reward_breakdown.total, or calculate from components
            reward_breakdown = self.latest_stats.get("reward_breakdown", {})
            if "total" in reward_breakdown:
                total_reward = reward_breakdown["total"]
            else:
                # Calculate total from components
                total_reward = (
                    reward_breakdown.get("immediate", 0) +
                    reward_breakdown.get("timing", 0) +
                    reward_breakdown.get("portfolio_growth", 0)
                )

            episode_stats = {
                "episode": event.data.get("episode"),
                "reward": total_reward,
                "roi": self.latest_stats.get("roi", 0),
                "num_trades": self.latest_stats.get("num_trades", 0),
                "win_rate": self.latest_stats.get("win_rate", 0),
                "portfolio_value": self.latest_stats.get("portfolio_value", 0),
                "duration": event.data.get("duration_seconds", 0)
            }
            self.episode_history.insert(0, episode_stats)  # Add to front

            # Keep only last N episodes
            if len(self.episode_history) > self.max_episode_history:
                self.episode_history = self.episode_history[:self.max_episode_history]

            logger.info(f"Episode ended: {self.current_episode}, reward={total_reward:.2f}, added to history (total: {len(self.episode_history)} episodes)")

        elif event.type == "training_metrics":
            self.latest_metrics = event.data
            logger.debug(f"Training metrics updated for episode {event.data.get('episode')}")

        elif event.type == "system_status":
            self.system_status = event.data

    def _handle_run_change(self, new_run_id: str, timestamp: str):
        """Handle run_id change - start new run"""
        logger.info(f"Run changed: {self.current_run_id} -> {new_run_id}")
        self.current_run_id = new_run_id
        self.session_start = timestamp
        self.episode_history = []  # Clear episode history for new run
        self.max_episode_steps = None  # Reset max episode steps for new run

    def get_snapshot(self) -> Dict[str, Any]:
        """Get current state snapshot"""
        snapshot = {
            "type": "state_snapshot",
            "data": {
                "run_id": self.current_run_id,
                "current_episode": self.current_episode,
                "total_episodes": self.total_episodes,
                "session_start": self.session_start,
                "is_active": self.is_active,
                "latest_stats": self.latest_stats,
                "latest_metrics": self.latest_metrics,
                "system_status": self.system_status,
                "current_action": self.current_action,
                "current_step": self.current_step,
                "current_price": self.current_price,
                "episode_history": self.episode_history,
                "max_episode_steps": self.max_episode_steps,
            },
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Sending state snapshot: run_id={self.current_run_id}, episode={self.current_episode}/{self.total_episodes}, step={self.current_step}, active={self.is_active}, history={len(self.episode_history)} episodes")
        return snapshot


# ============================================================================
# Global State
# ============================================================================

dashboard_state = DashboardState()
active_connections: Set[WebSocket] = set()  # Browser clients
agent_connection: Optional[WebSocket] = None  # Single agent connection
pending_commands: deque = deque(maxlen=100)  # Commands waiting for agent


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="QTrader Dashboard",
    description="Standalone real-time trading dashboard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# WebSocket Broadcasting
# ============================================================================

async def broadcast_to_clients(event: Event):
    """Broadcast event to all connected clients"""
    if not active_connections:
        logger.debug(f"No active connections to broadcast {event.type}")
        return

    logger.info(f"Broadcasting {event.type} to {len(active_connections)} client(s)")
    disconnected = set()
    for connection in active_connections:
        try:
            await connection.send_json(event.dict())
        except Exception as e:
            logger.warning(f"Error sending to client: {e}")
            disconnected.add(connection)

    for connection in disconnected:
        active_connections.discard(connection)


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/events")
async def receive_event(event: Event, background_tasks: BackgroundTasks):
    """
    Receive events from qtrader via HTTP POST.
    QTrader publishes to this endpoint.
    """
    if not event.timestamp:
        event.timestamp = datetime.now().isoformat()

    # Update state
    dashboard_state.update(event)

    # Broadcast the event
    background_tasks.add_task(broadcast_to_clients, event)

    return {"status": "received", "type": event.type}


@app.get("/api/state")
async def get_state():
    """Get current state snapshot"""
    return dashboard_state.get_snapshot()


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "active_connections": len(active_connections),
        "agent_connected": agent_connection is not None,
        "session_active": dashboard_state.is_active,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Agent Control Endpoints
# ============================================================================

class CommandRequest(BaseModel):
    """Request to send command to agent"""
    command: str  # CommandType value
    data: Optional[Dict[str, Any]] = None


@app.post("/api/command")
async def send_command(request: CommandRequest):
    """Send a command to the connected agent"""
    global agent_connection

    command = Command(command_type=request.command, data=request.data or {})
    logger.info(f"Received command request: {command.type} with data: {command.data}")

    if agent_connection is None:
        logger.warning("No agent connected, queuing command")
        pending_commands.append(command)
        return {
            "status": "queued",
            "message": "Agent not connected, command queued",
            "command": command.type
        }

    try:
        await agent_connection.send_json(command.to_dict())
        logger.info(f"Command sent to agent: {command.type}")
        return {
            "status": "sent",
            "command": command.type
        }
    except Exception as e:
        logger.error(f"Failed to send command to agent: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/api/agent/status")
async def get_agent_status():
    """Get agent connection status"""
    return {
        "connected": agent_connection is not None,
        "training_active": dashboard_state.is_active,
        "current_run_id": dashboard_state.current_run_id,
        "current_episode": dashboard_state.current_episode,
        "total_episodes": dashboard_state.total_episodes,
        "pending_commands": len(pending_commands),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/profiles")
async def list_profiles():
    """List available trading profiles"""
    profiles_dir = Path(__file__).parent.parent / "agent" / "profiles"
    profiles = []

    if profiles_dir.exists():
        for f in profiles_dir.glob("*.yaml"):
            profiles.append(f.stem)

    # Fallback to known profiles if directory not accessible
    if not profiles:
        profiles = [
            "scalper", "day_trader", "swing_trader",
            "balanced", "risk_averse", "contrarian", "trend_follower"
        ]

    return {"profiles": sorted(profiles)}


@app.get("/api/configs")
async def list_configs():
    """List available training configs"""
    configs_dir = Path(__file__).parent.parent / "agent" / "config"
    configs = []

    if configs_dir.exists():
        for f in configs_dir.glob("*.yaml"):
            configs.append(f.stem)

    # Fallback to known configs if directory not accessible
    if not configs:
        configs = ["fast", "medium", "full"]

    return {"configs": sorted(configs)}


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for browser clients"""
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"Client connected. Active: {len(active_connections)}")

    try:
        # Send current state on connect
        snapshot = dashboard_state.get_snapshot()
        await websocket.send_json(snapshot)

        # Keep alive
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "request_snapshot":
                await websocket.send_json(dashboard_state.get_snapshot())

    except WebSocketDisconnect:
        active_connections.discard(websocket)
        logger.info(f"Client disconnected. Active: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_connections.discard(websocket)


@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """WebSocket for agent connection - receives commands, sends events"""
    global agent_connection

    await websocket.accept()

    # Only one agent at a time
    if agent_connection is not None:
        logger.warning("Agent already connected, rejecting new connection")
        await websocket.close(code=4000, reason="Agent already connected")
        return

    agent_connection = websocket
    logger.info("Agent connected via WebSocket")

    # Notify browser clients of agent connection
    await broadcast_to_clients(Event(
        type="agent_status",
        data={"connected": True},
        timestamp=datetime.now().isoformat()
    ))

    try:
        # Send any pending commands
        while pending_commands:
            command = pending_commands.popleft()
            await websocket.send_json(command.to_dict())
            logger.info(f"Sent pending command to agent: {command.type}")

        # Main loop - receive events from agent
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "event":
                # Agent is sending an event - process and broadcast
                event_data = message.get("data", {})
                event = Event(
                    type=event_data.get("type", "unknown"),
                    data=event_data.get("data", {}),
                    timestamp=event_data.get("timestamp", datetime.now().isoformat())
                )
                if hasattr(event, 'run_id'):
                    event.run_id = event_data.get("run_id")
                dashboard_state.update(event)
                await broadcast_to_clients(event)
            elif message.get("type") == "status":
                # Agent status update
                logger.info(f"Agent status: {message.get('data')}")

    except WebSocketDisconnect:
        logger.info("Agent disconnected")
    except Exception as e:
        logger.error(f"Agent WebSocket error: {e}")
    finally:
        agent_connection = None
        # Notify browser clients of agent disconnection
        try:
            await broadcast_to_clients(Event(
                type="agent_status",
                data={"connected": False},
                timestamp=datetime.now().isoformat()
            ))
        except Exception:
            pass
        logger.info("Agent connection closed")


# ============================================================================
# Serve Dashboard UI
# ============================================================================

@app.get("/")
async def serve_dashboard():
    """Serve dashboard HTML"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())

    return HTMLResponse(content="""
        <html>
            <head><title>QTrader Dashboard</title></head>
            <body style="font-family: Arial; padding: 20px;">
                <h1>QTrader Dashboard</h1>
                <p>Dashboard UI not found. Create <code>index.html</code> in this directory.</p>
                <h2>API Endpoints:</h2>
                <ul>
                    <li>POST /api/events - Receive events from qtrader</li>
                    <li>GET /api/state - Get current state</li>
                    <li>WS /ws - WebSocket connection</li>
                    <li>GET /health - Health check</li>
                </ul>
            </body>
        </html>
    """)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QTrader Dashboard Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("QTrader Dashboard Server")
    logger.info("="*60)
    logger.info(f"Dashboard UI:    http://{args.host}:{args.port}")
    logger.info(f"WebSocket:       ws://{args.host}:{args.port}/ws")
    logger.info(f"Events endpoint: POST http://{args.host}:{args.port}/api/events")
    logger.info("="*60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
