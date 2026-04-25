"""
Buy and Hold Strategy - The simplest baseline.

This is the most important baseline to beat. If an RL agent
can't beat buy-and-hold after fees, it's not useful.

Strategy: Buy 100% on step 1, hold forever.
"""

import numpy as np
from .base import BaselineStrategy


class BuyAndHoldStrategy(BaselineStrategy):
    """
    Buy and hold Bitcoin for the entire episode.

    This is the benchmark every trading strategy must beat.
    In crypto bull markets, this is surprisingly hard to beat.
    In bear markets, it's easy to beat (by just holding cash).

    The RL agent should ideally beat this in BOTH conditions.
    """

    def __init__(self):
        super().__init__("Buy and Hold")
        self._has_bought = False

    def reset(self):
        super().reset()
        self._has_bought = False

    def get_action(self, state: np.ndarray, price: float, **kwargs) -> int:
        """
        Buy 100% on first step, then hold forever.

        Returns:
            4 (BUY 100%) on first step with cash
            0 (HOLD) thereafter
        """
        cash = kwargs.get('cash', 0)

        # Buy 100% if we haven't bought yet and have cash
        if not self._has_bought and cash > 1:
            self._has_bought = True
            return 4  # BUY 100%

        # Hold forever
        return 0  # HOLD
