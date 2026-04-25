"""
Reward Manager - combines multiple reward calculators.
"""

from typing import List, Dict, Any
from .base import RewardCalculator


class RewardManager:
    """
    Manages multiple reward calculators and combines their outputs.

    Easy plug-and-play system for mixing different reward strategies.
    """

    def __init__(self, calculators: List[RewardCalculator]):
        """
        Args:
            calculators: List of RewardCalculator instances
        """
        self.calculators = calculators
        self._last_breakdown = {}  # Cache for efficiency

    def reset(self):
        """Reset all calculators for new episode"""
        for calc in self.calculators:
            calc.reset()

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate combined reward from all calculators.

        Also caches breakdown for efficiency (avoids recalculating in get_breakdown).

        Args:
            context: Current context

        Returns:
            Total weighted reward
        """
        total_reward = 0.0
        self._last_breakdown = {}

        for calc in self.calculators:
            reward = calc.calculate_reward(context)
            weighted_reward = reward * calc.weight
            total_reward += weighted_reward
            self._last_breakdown[calc.name] = weighted_reward

        self._last_breakdown['total'] = total_reward
        return total_reward

    def get_breakdown(self, context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Get reward breakdown by calculator (for debugging/logging).

        Returns cached breakdown from last calculate_reward() call.
        If context is provided and no cached breakdown exists, calculates fresh.

        Args:
            context: Current context (optional, only needed if no cache)

        Returns:
            Dictionary mapping calculator name to reward value
        """
        # Return cached breakdown if available
        if self._last_breakdown:
            return self._last_breakdown.copy()

        # Fallback: calculate if context provided
        if context is not None:
            breakdown = {}
            for calc in self.calculators:
                reward = calc.calculate_reward(context)
                weighted_reward = reward * calc.weight
                breakdown[calc.name] = weighted_reward
            breakdown['total'] = sum(breakdown.values())
            return breakdown

        # No cache and no context - return empty
        return {'total': 0.0}

    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get diagnostic info from all calculators.

        Returns:
            Dictionary mapping calculator name to its info dict
        """
        info = {}
        for calc in self.calculators:
            info[calc.name] = calc.get_info()
        return info

    def add_calculator(self, calculator: RewardCalculator):
        """Add a new calculator to the manager"""
        self.calculators.append(calculator)

    def remove_calculator(self, calculator_name: str):
        """Remove a calculator by name"""
        self.calculators = [c for c in self.calculators if c.name != calculator_name]

    def list_calculators(self) -> List[str]:
        """Get list of active calculator names"""
        return [f"{c.name} (weight={c.weight})" for c in self.calculators]
