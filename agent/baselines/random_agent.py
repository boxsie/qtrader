"""
Random Strategy - The floor baseline.

If your RL agent can't beat random actions, it's not learning
any useful signal from the data. This is more common than you'd think!

Strategy: Take random actions weighted toward HOLD to avoid
excessive trading fees.
"""

import numpy as np
from .base import BaselineStrategy


class RandomStrategy(BaselineStrategy):
    """
    Random action selection with configurable action distribution.

    By default, heavily weighted toward HOLD to avoid fee destruction.
    This makes it a fairer comparison - pure uniform random would
    lose everything to fees quickly.
    """

    def __init__(
        self,
        hold_probability: float = 0.7,
        buy_probability: float = 0.15,
        sell_probability: float = 0.15,
        seed: int = None
    ):
        """
        Initialize random strategy.

        Args:
            hold_probability: Probability of holding (default 70%)
            buy_probability: Probability of buying (split across buy sizes)
            sell_probability: Probability of selling (split across sell sizes)
            seed: Random seed for reproducibility
        """
        super().__init__("Random")
        self.hold_prob = hold_probability
        self.buy_prob = buy_probability
        self.sell_prob = sell_probability
        self.rng = np.random.default_rng(seed)

        # Validate probabilities sum to 1
        total = hold_probability + buy_probability + sell_probability
        if not np.isclose(total, 1.0):
            # Normalize
            self.hold_prob /= total
            self.buy_prob /= total
            self.sell_prob /= total

    def reset(self):
        super().reset()

    def get_action(self, state: np.ndarray, price: float, **kwargs) -> int:
        """
        Select random action with configured distribution.

        Returns:
            Random action (0-8)
        """
        crypto_quantity = kwargs.get('crypto_quantity', 0)

        # Determine action category
        rand = self.rng.random()

        if rand < self.hold_prob:
            return 0  # HOLD

        elif rand < self.hold_prob + self.buy_prob:
            # Random buy size (1-4)
            return self.rng.integers(1, 5)

        else:
            # Random sell size (5-8), but only if we have crypto
            if crypto_quantity > 0:
                return self.rng.integers(5, 9)
            else:
                return 0  # Can't sell, so hold


class UniformRandomStrategy(BaselineStrategy):
    """
    Uniformly random action selection (no weighting).

    This is a stricter baseline - if you can't beat this,
    there's definitely no signal being learned.

    Warning: This will incur massive fees and likely lose money.
    """

    def __init__(self, seed: int = None):
        super().__init__("Uniform Random")
        self.rng = np.random.default_rng(seed)

    def get_action(self, state: np.ndarray, price: float, **kwargs) -> int:
        """Select uniformly random action."""
        crypto_quantity = kwargs.get('crypto_quantity', 0)

        # If no position, can only HOLD or BUY
        if crypto_quantity == 0:
            return self.rng.integers(0, 5)  # HOLD or BUY variants

        # Otherwise, any action
        return self.rng.integers(0, 9)
