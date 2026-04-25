"""
Base class for baseline trading strategies.

All baselines use the same Broker interface as the RL agent,
making direct comparison fair and straightforward.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaselineStrategy(ABC):
    """
    Abstract base class for baseline strategies.

    Strategies implement get_action() which returns an action
    compatible with the Broker's action space:
    - 0: HOLD
    - 1-4: BUY (10/25/50/100%)
    - 5-8: SELL (10/25/50/100%)
    """

    def __init__(self, name: str):
        self.name = name
        self._step = 0

    @abstractmethod
    def get_action(self, state: np.ndarray, price: float, **kwargs) -> int:
        """
        Get action given current state and price.

        Args:
            state: Current state vector from broker
            price: Current BTC price
            **kwargs: Additional info (position, cash, etc.)

        Returns:
            Action integer (0-8)
        """
        pass

    def reset(self):
        """Reset strategy state for new episode."""
        self._step = 0

    def step(self):
        """Increment step counter."""
        self._step += 1

    def evaluate(self, broker, max_steps: int = None) -> dict:
        """
        Evaluate strategy on broker's data.

        Args:
            broker: Broker instance with data loaded
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with evaluation metrics
        """
        self.reset()
        broker.reset()

        state = broker.get_state()
        if state is None:
            return {'error': 'No initial state'}

        total_reward = 0
        steps = 0
        prices = []
        portfolio_values = []

        while True:
            # Get current price from broker's last tick
            price = broker._last_tick['Close'] if broker._last_tick is not None else 0
            prices.append(price)

            # Get current portfolio value
            portfolio_value = broker._cash + (broker._crypto_quantity * price)
            portfolio_values.append(portfolio_value)

            # Get action from strategy
            action = self.get_action(
                state=state,
                price=price,
                cash=broker._cash,
                crypto_quantity=broker._crypto_quantity,
                portfolio_value=portfolio_value
            )

            # Execute action
            done, next_state, reward, info = broker.update(action)

            total_reward += reward
            steps += 1
            self.step()

            if done:
                break

            if max_steps and steps >= max_steps:
                break

            state = next_state

        # Get final stats
        final_price = broker._last_tick['Close'] if broker._last_tick is not None else prices[-1]
        final_stats = broker.get_stats(final_price)

        # Calculate additional metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Sharpe ratio (annualized, assuming 1-minute data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(525600)  # Minutes per year
        else:
            sharpe = 0

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        result = {
            'strategy': self.name,
            'steps': steps,
            'total_reward': total_reward,
            'final_portfolio_value': final_stats.get('portfolio_value', 0),
            'roi': final_stats.get('roi', 0),
            'num_trades': final_stats.get('num_trades', 0),
            'win_rate': final_stats.get('win_rate', 0),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,  # As percentage
            'total_fees': final_stats.get('total_fees', 0),
        }

        return result
