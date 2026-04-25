"""
Shared library for QTrader and Dashboard

Contains event models and contracts that are used by both systems.
"""

from .events import (
    Event,
    EventType,
    EpisodeStartData,
    StepUpdateData,
    TradeExecutedData,
    EpisodeEndData,
    TrainingMetricsData,
    SystemStatusData
)

__all__ = [
    'Event',
    'EventType',
    'EpisodeStartData',
    'StepUpdateData',
    'TradeExecutedData',
    'EpisodeEndData',
    'TrainingMetricsData',
    'SystemStatusData'
]
