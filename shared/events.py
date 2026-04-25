"""
Event Models for QTrader Dashboard Communication

Defines the contract between QTrader agent and Dashboard server.
Both systems import these models to ensure compatibility.
"""

from typing import Dict, Any, Optional
from enum import Enum


class EventType(str, Enum):
    """Event types for dashboard communication"""
    EPISODE_START = "episode_start"
    STEP_UPDATE = "step_update"
    TRADE_EXECUTED = "trade_executed"
    EPISODE_END = "episode_end"
    TRAINING_METRICS = "training_metrics"
    SYSTEM_STATUS = "system_status"
    STATE_SNAPSHOT = "state_snapshot"


class CommandType(str, Enum):
    """Command types for controlling the agent from dashboard"""
    START_TRAINING = "start_training"
    STOP_TRAINING = "stop_training"


class Command:
    """Command from dashboard to agent"""
    def __init__(self, command_type: str, data: Optional[Dict[str, Any]] = None):
        self.type = command_type
        self.data = data or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.type,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Command":
        """Create Command from dictionary"""
        return cls(command_type=d["type"], data=d.get("data", {}))


class Event:
    """Base event model"""
    def __init__(self, event_type: str, data: Dict[str, Any], timestamp: Optional[str] = None):
        self.type = event_type
        self.data = data
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp
        }


# Data Models (plain dicts for simplicity, but typed for clarity)

class EpisodeStartData:
    """Data for episode_start event"""
    episode: int
    total_episodes: int
    config: Dict[str, Any]


class StepUpdateData:
    """Data for step_update event"""
    episode: int
    step: int
    action: int
    reward: float
    price: float
    stats: Dict[str, Any]


class TradeExecutedData:
    """Data for trade_executed event"""
    episode: int
    step: int
    trade_type: str  # "BUY" or "SELL"
    price: float
    quantity: float
    total_value: float


class EpisodeEndData:
    """Data for episode_end event"""
    episode: int
    final_stats: Dict[str, Any]
    duration_seconds: float


class TrainingMetricsData:
    """Data for training_metrics event"""
    episode: int
    loss: Optional[float]
    epsilon: float
    avg_reward: float
    learning_rate: Optional[float]


class SystemStatusData:
    """Data for system_status event"""
    gpu_usage: Optional[float]
    memory_usage: Optional[float]
    gpu_memory_used: Optional[float]
    gpu_memory_total: Optional[float]
