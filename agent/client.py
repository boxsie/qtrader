"""
Dashboard Event Publisher Client

Publishes events from qtrader to the standalone dashboard server via HTTP.
Completely decoupled - fails gracefully if dashboard is unavailable.
"""

import sys
from pathlib import Path

# Add parent directory to path to import shared module
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from threading import Thread
from queue import Queue, Empty

# Import shared event types for type safety
from shared.events import EventType

logger = logging.getLogger(__name__)


class Client:
    """
    HTTP client for publishing events to dashboard server.

    Features:
    - Non-blocking event publishing (uses queue + background thread)
    - Graceful degradation if dashboard unavailable
    - Automatic retries with backoff
    - No impact on training performance
    """

    def __init__(
        self,
        dashboard_url: str = "http://localhost:8765",
        enabled: bool = True,
        timeout: float = 0.05,  # Very short timeout - fire and forget
        queue_size: int = 1000,
        run_id: Optional[str] = None
    ):
        self.dashboard_url = dashboard_url.rstrip('/')
        self.events_endpoint = f"{self.dashboard_url}/api/events"
        self.enabled = enabled
        self.timeout = timeout
        self.run_id = run_id  # Run ID for this training session

        # Event queue for non-blocking publishing
        self.event_queue: Queue = Queue(maxsize=queue_size)

        # Background worker thread
        self.worker_thread: Optional[Thread] = None
        self.is_running = False

        # Connection status
        self.is_connected = False
        self.failed_attempts = 0
        self.max_failed_attempts = 5

        if self.enabled:
            self._start_worker()
            self._check_connection()

    def _check_connection(self):
        """Check if dashboard is reachable"""
        try:
            response = requests.get(
                f"{self.dashboard_url}/health",
                timeout=self.timeout
            )
            if response.status_code == 200:
                self.is_connected = True
                self.failed_attempts = 0
                return True
        except Exception as e:
            pass  # Silent - no logging for connection checks

        self.is_connected = False
        return False

    def _start_worker(self):
        """Start background worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            return

        self.is_running = True
        self.worker_thread = Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Background worker that processes event queue"""
        while self.is_running:
            try:
                # Get event from queue (blocking with timeout)
                event = self.event_queue.get(timeout=1.0)

                # Try to send event
                self._send_event(event)

            except Empty:
                continue
            except Exception as e:
                pass  # Silent - no logging for worker errors

    def _send_event(self, event: Dict[str, Any]):
        """Send event to dashboard via HTTP POST"""
        # For critical events (episode_start, episode_end), always try to send
        # even if we've had previous failures
        is_critical_event = event.get('type') in ['episode_start', 'episode_end', 'training_metrics']

        if not self.is_connected and self.failed_attempts >= self.max_failed_attempts and not is_critical_event:
            # Skip sending if we've failed too many times (but still send critical events)
            return

        try:
            # Convert event to JSON-serializable format
            import json
            json_event = json.loads(json.dumps(event, default=str))

            # Use longer timeout for critical events, shorter for step updates
            timeout = 1.0 if is_critical_event else self.timeout

            response = requests.post(
                self.events_endpoint,
                json=json_event,
                timeout=timeout
            )

            if response.status_code == 200:
                self.is_connected = True
                self.failed_attempts = 0
                # Silent - no logging during training
            else:
                self.failed_attempts += 1

        except requests.exceptions.Timeout:
            # Timeouts don't mean connection is lost - just slow response
            # Don't increment failed_attempts for timeouts
            pass  # Silent - no logging for timeouts during training
        except requests.exceptions.ConnectionError:
            # Silent - no logging for connection errors during training
            self.is_connected = False
            self.failed_attempts += 1
        except Exception as e:
            # Silent - no logging for general errors during training
            self.failed_attempts += 1

    def emit(self, event_type: str, data: Dict[str, Any]):
        """
        Emit an event to the dashboard (non-blocking).

        Args:
            event_type: Type of event (e.g., 'step_update', 'episode_start')
            data: Event data dictionary
        """
        if not self.enabled:
            return

        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id  # Include run_id in every event
        }

        try:
            self.event_queue.put_nowait(event)
        except Exception:
            # Queue full - drop event silently
            logger.debug(f"Event queue full, dropped {event_type} event")
            pass

    def set_run_id(self, run_id: str):
        """Update the run_id for this client"""
        self.run_id = run_id
        logger.info(f"Dashboard client run_id set to: {run_id}")

    # ========================================================================
    # Convenience methods for common events
    # ========================================================================

    def emit_episode_start(self, episode: int, total_episodes: int, config: Dict[str, Any] = None):
        """Emit episode start event"""
        self.emit("episode_start", {
            "episode": episode,
            "total_episodes": total_episodes,
            "config": config or {}
        })

    def emit_step_update(
        self,
        episode: int,
        step: int,
        action: int,
        reward: float,
        price: float,
        stats: Dict[str, Any],
        total_episodes: int = None
    ):
        """Emit step update event"""
        self.emit("step_update", {
            "episode": episode,
            "step": step,
            "action": action,
            "reward": reward,
            "price": price,
            "stats": stats,
            "total_episodes": total_episodes
        })

    def emit_trade_executed(
        self,
        episode: int,
        step: int,
        trade_type: str,
        price: float,
        quantity: float,
        total_value: float
    ):
        """Emit trade execution event"""
        self.emit("trade_executed", {
            "episode": episode,
            "step": step,
            "trade_type": trade_type,
            "price": price,
            "quantity": quantity,
            "total_value": total_value
        })

    def emit_episode_end(
        self,
        episode: int,
        final_stats: Dict[str, Any],
        duration_seconds: float
    ):
        """Emit episode end event"""
        self.emit("episode_end", {
            "episode": episode,
            "final_stats": final_stats,
            "duration_seconds": duration_seconds
        })

    def emit_training_metrics(
        self,
        episode: int,
        loss: Optional[float],
        epsilon: float,
        avg_reward: float,
        learning_rate: Optional[float] = None,
        memory_size: Optional[int] = None
    ):
        """Emit training metrics"""
        self.emit("training_metrics", {
            "episode": episode,
            "loss": loss,
            "epsilon": epsilon,
            "avg_reward": avg_reward,
            "learning_rate": learning_rate,
            "memory_size": memory_size
        })

    def emit_system_status(
        self,
        gpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        gpu_memory_used: Optional[float] = None,
        gpu_memory_total: Optional[float] = None
    ):
        """Emit system status"""
        self.emit("system_status", {
            "gpu_usage": gpu_usage,
            "memory_usage": memory_usage,
            "gpu_memory_used": gpu_memory_used,
            "gpu_memory_total": gpu_memory_total
        })

    def stop(self):
        """Stop the background worker (call on shutdown)"""
        logger.info("Stopping dashboard client...")
        self.is_running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

        logger.info("Dashboard client stopped")

    def __del__(self):
        """Cleanup on destruction"""
        self.stop()
