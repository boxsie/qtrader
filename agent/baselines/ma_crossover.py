"""
Moving Average Crossover Strategy - Classic technical analysis baseline.

This is the "competent human trader" baseline. If your RL agent
can't beat simple MA crossover, it's not learning market dynamics.

Strategy:
- BUY when fast MA crosses above slow MA (golden cross)
- SELL when fast MA crosses below slow MA (death cross)
- HOLD otherwise
"""

import numpy as np
from collections import deque
from .base import BaselineStrategy


class MACrossoverStrategy(BaselineStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Uses two moving averages (fast and slow) to generate signals:
    - Golden Cross (fast > slow): Buy signal
    - Death Cross (fast < slow): Sell signal

    This is a trend-following strategy that performs well in
    trending markets but poorly in sideways/choppy markets.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        position_size: float = 1.0
    ):
        """
        Initialize MA crossover strategy.

        Args:
            fast_period: Period for fast moving average (default 10)
            slow_period: Period for slow moving average (default 30)
            position_size: Position size as fraction (1.0 = 100%)
        """
        super().__init__(f"MA Crossover ({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size

        # Map position size to action
        if position_size <= 0.10:
            self.buy_action = 1   # BUY 10%
            self.sell_action = 5  # SELL 10%
        elif position_size <= 0.25:
            self.buy_action = 2   # BUY 25%
            self.sell_action = 6  # SELL 25%
        elif position_size <= 0.50:
            self.buy_action = 3   # BUY 50%
            self.sell_action = 7  # SELL 50%
        else:
            self.buy_action = 4   # BUY 100%
            self.sell_action = 8  # SELL 100%

        # Price history for MA calculation
        self.price_history = deque(maxlen=slow_period)
        self._last_signal = None  # 'buy', 'sell', or None

    def reset(self):
        super().reset()
        self.price_history.clear()
        self._last_signal = None

    def _calculate_ma(self, prices: list, period: int) -> float:
        """Calculate simple moving average."""
        if len(prices) < period:
            return None
        return np.mean(list(prices)[-period:])

    def get_action(self, state: np.ndarray, price: float, **kwargs) -> int:
        """
        Get action based on MA crossover signal.

        Returns:
            Action based on MA crossover
        """
        cash = kwargs.get('cash', 0)
        crypto_quantity = kwargs.get('crypto_quantity', 0)

        # Add price to history
        self.price_history.append(price)

        # Need enough history for slow MA
        if len(self.price_history) < self.slow_period:
            return 0  # HOLD until we have enough data

        # Calculate MAs
        fast_ma = self._calculate_ma(self.price_history, self.fast_period)
        slow_ma = self._calculate_ma(self.price_history, self.slow_period)

        if fast_ma is None or slow_ma is None:
            return 0

        # Determine signal
        current_signal = 'buy' if fast_ma > slow_ma else 'sell'

        # Only act on signal CHANGES (crossovers)
        if self._last_signal is None:
            self._last_signal = current_signal
            return 0  # First signal, just record it

        # Check for crossover
        if current_signal != self._last_signal:
            self._last_signal = current_signal

            if current_signal == 'buy' and cash > 1:
                # Golden cross - buy
                return self.buy_action

            elif current_signal == 'sell' and crypto_quantity > 0:
                # Death cross - sell
                return self.sell_action

        # No crossover, hold
        return 0


class RSIMeanReversionStrategy(BaselineStrategy):
    """
    RSI Mean Reversion Strategy - Another classic approach.

    Contrarian strategy that buys oversold conditions and
    sells overbought conditions.

    Strategy:
    - BUY when RSI < 30 (oversold)
    - SELL when RSI > 70 (overbought)
    - HOLD otherwise
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70
    ):
        """
        Initialize RSI strategy.

        Args:
            period: RSI calculation period
            oversold: RSI level to trigger buy
            overbought: RSI level to trigger sell
        """
        super().__init__(f"RSI Mean Reversion ({period})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.price_history = deque(maxlen=period + 1)

    def reset(self):
        super().reset()
        self.price_history.clear()

    def _calculate_rsi(self) -> float:
        """Calculate RSI from price history."""
        if len(self.price_history) < self.period + 1:
            return 50  # Neutral

        prices = list(self.price_history)
        changes = np.diff(prices)

        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])

        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_action(self, state: np.ndarray, price: float, **kwargs) -> int:
        """
        Get action based on RSI signal.

        Returns:
            Action based on RSI levels
        """
        cash = kwargs.get('cash', 0)
        crypto_quantity = kwargs.get('crypto_quantity', 0)

        self.price_history.append(price)

        if len(self.price_history) < self.period + 1:
            return 0

        rsi = self._calculate_rsi()

        # Oversold - buy signal
        if rsi < self.oversold and cash > 1:
            return 3  # BUY 50%

        # Overbought - sell signal
        if rsi > self.overbought and crypto_quantity > 0:
            return 7  # SELL 50%

        return 0  # HOLD
