"""
Coinbase Broker

Multi-timeframe trading broker with modular reward system.
Agent sees weighted combination of all timeframes based on trader profile.
"""

import numpy as np
import pandas as pd
from collections import deque

from rewards.factory import create_reward_manager
from rewards.manager import RewardManager


class SlippageModel:
    """
    Realistic slippage simulation for crypto markets.

    Slippage = base_bps + (order_size / avg_volume) * impact_factor

    Real-world effects modeled:
    - Base spread (even small orders have some slippage)
    - Market impact (larger orders move price more)
    - Volatility impact (high volatility = more slippage)
    """

    def __init__(
        self,
        base_slippage_bps: float = 5.0,      # 0.05% base spread
        impact_factor: float = 10.0,          # Market impact multiplier
        volatility_factor: float = 2.0,       # How much volatility affects slippage
        max_slippage_bps: float = 100.0       # Cap at 1% slippage
    ):
        """
        Args:
            base_slippage_bps: Base slippage in basis points (100 bps = 1%)
            impact_factor: Multiplier for size-based market impact
            volatility_factor: Multiplier for volatility-based slippage
            max_slippage_bps: Maximum slippage cap
        """
        self.base_slippage_bps = base_slippage_bps
        self.impact_factor = impact_factor
        self.volatility_factor = volatility_factor
        self.max_slippage_bps = max_slippage_bps

    def calculate_slippage(
        self,
        order_value: float,
        avg_volume: float,
        volatility: float = 0.02,
        direction: int = 1  # 1 for buy (pay more), -1 for sell (receive less)
    ) -> float:
        """
        Calculate slippage in basis points.

        Args:
            order_value: Value of order in dollars
            avg_volume: Average volume in dollars
            volatility: Recent price volatility (as decimal, e.g., 0.02 = 2%)
            direction: 1 for buy, -1 for sell

        Returns:
            Slippage as decimal (e.g., 0.001 = 0.1%)
        """
        # Base slippage
        slippage_bps = self.base_slippage_bps

        # Size impact: larger orders relative to volume = more slippage
        if avg_volume > 0:
            size_ratio = order_value / avg_volume
            slippage_bps += size_ratio * self.impact_factor * 100  # Convert to bps

        # Volatility impact
        slippage_bps += volatility * self.volatility_factor * 100

        # Cap slippage
        slippage_bps = min(slippage_bps, self.max_slippage_bps)

        # Convert to decimal
        slippage_decimal = slippage_bps / 10000

        return slippage_decimal * direction

    def get_execution_price(
        self,
        intended_price: float,
        order_value: float,
        avg_volume: float,
        volatility: float = 0.02,
        is_buy: bool = True
    ) -> tuple:
        """
        Get actual execution price after slippage.

        Args:
            intended_price: Price you wanted
            order_value: Order size in dollars
            avg_volume: Average market volume
            volatility: Recent volatility
            is_buy: True for buy, False for sell

        Returns:
            (execution_price, slippage_amount)
        """
        direction = 1 if is_buy else -1
        slippage = self.calculate_slippage(order_value, avg_volume, volatility, direction)

        execution_price = intended_price * (1 + slippage)
        slippage_amount = abs(execution_price - intended_price) * (order_value / intended_price)

        return execution_price, slippage_amount


class Broker:
    """
    Coinbase-style broker with multi-timeframe state representation.

    Uses RewardManager for modular, configurable rewards.
    """

    def __init__(
        self,
        ticker,
        reward_config,
        starting_cash=1000,
        fee_pct=0.0015,
        slippage_enabled=True,
        slippage_config=None
    ):
        """
        Args:
            ticker: Ticker instance for market data
            reward_config: List of reward calculator configs (from YAML profile)
            starting_cash: Initial capital
            fee_pct: Trading fee percentage (0.0015 = 0.15%)
            slippage_enabled: Whether to simulate slippage
            slippage_config: Dict of SlippageModel params (optional)
        """
        self._ticker = ticker
        self._starting_cash = starting_cash
        self._fee_pct = fee_pct

        # Create reward manager from config
        self._reward_manager = create_reward_manager(reward_config)

        # Slippage model
        self._slippage_enabled = slippage_enabled
        if slippage_enabled:
            slippage_config = slippage_config or {}
            self._slippage = SlippageModel(**slippage_config)
        else:
            self._slippage = None

        # Track slippage costs
        self._total_slippage_cost = 0.0

        # Portfolio state
        self._cash = starting_cash
        self._crypto_quantity = 0.0
        self._average_buy_price = 0.0

        # Actions: HOLD, BUY (10/25/50/100%), SELL (10/25/50/100%)
        self.num_actions = 9
        self.num_states = ticker.num_features

        # State tracking
        self._step_count = 0
        self._episode_returns = []

        # Trade history
        self._trade_history = []

        # Price history for reward calculators
        self._price_history = deque(maxlen=100)

        # Portfolio tracking
        self._peak_portfolio_value = starting_cash
        self._portfolio_value_history = deque(maxlen=100)

        self.reset()

    def reset(self):
        """Reset broker to initial state"""
        # Reset portfolio
        self._cash = self._starting_cash
        self._crypto_quantity = 0.0
        self._average_buy_price = 0.0

        # Reset tracking
        self._step_count = 0
        self._episode_returns = []
        self._trade_history = []
        self._price_history.clear()
        self._peak_portfolio_value = self._starting_cash
        self._portfolio_value_history.clear()
        self._last_reward_breakdown = {}
        self._total_slippage_cost = 0.0

        # Reset reward manager
        self._reward_manager.reset()

        # Reset ticker
        self._ticker.reset_index()

        # Record episode start timestamp
        self._episode_start_timestamp = self._ticker.get_current_timestamp()

        # Get initial state
        self._last_tick = None
        self._last_trade_info = None

    def get_state(self):
        """Get current multi-timeframe state representation"""
        position_info = None
        if self._crypto_quantity > 0:
            current_price = self._last_tick['Close'] if self._last_tick is not None else 0
            current_value = self._crypto_quantity * current_price
            cost_basis = self._crypto_quantity * self._average_buy_price
            unrealized_pnl = current_value - cost_basis

            position_info = {
                'quantity': self._crypto_quantity,
                'buy_sell': 'buy',
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl / cost_basis if cost_basis > 0 else 0
            }

        current_price = self._last_tick['Close'] if self._last_tick is not None else 0
        portfolio_value = self._cash + (self._crypto_quantity * current_price)

        state = self._ticker.get_current_state(position_info, portfolio_value)
        return state

    def _build_reward_context(self, action, price, trade_info=None):
        """Build context dict for reward calculators"""
        portfolio_value = self._cash + (self._crypto_quantity * price)

        return {
            'action': action,
            'price': price,
            'cash': self._cash,
            'crypto_quantity': self._crypto_quantity,
            'portfolio_value': portfolio_value,
            'starting_cash': self._starting_cash,
            'average_buy_price': self._average_buy_price,
            'fee_pct': self._fee_pct,
            'step': self._step_count,
            'trade_info': trade_info,
            'price_history': list(self._price_history),
            'portfolio_history': list(self._portfolio_value_history),
            'peak_portfolio_value': self._peak_portfolio_value,
        }

    def update(self, action):
        """
        Execute action and return next state, reward, done, info.

        Actions:
        0: HOLD
        1-4: BUY (10/25/50/100%)
        5-8: SELL (10/25/50/100%)
        """
        # Get next tick
        self._last_tick, is_complete = self._ticker.get_next()

        if self._last_tick is None or self._last_tick.empty or 'Close' not in self._last_tick:
            return True, None, 0, None

        price = self._last_tick['Close']
        self._price_history.append(price)

        # Calculate current portfolio value
        portfolio_value = self._cash + (self._crypto_quantity * price)

        # Check termination
        if is_complete:
            return True, None, 0, self.get_stats(price)

        # Bankruptcy check
        if portfolio_value < self._starting_cash * 0.1:
            return True, None, -100, self.get_stats(price)

        # Execute action (updates portfolio, records trade)
        trade_info = self._execute_action(action, price)

        # Penalize invalid actions (trying to buy with no cash, sell with no crypto)
        invalid_action_penalty = 0.0
        if trade_info and trade_info.get('type') in ['FAILED_BUY', 'INVALID_SELL']:
            invalid_action_penalty = -10.0  # Significant penalty to discourage wasted actions

        # Update peak tracking
        new_portfolio_value = self._cash + (self._crypto_quantity * price)
        if new_portfolio_value > self._peak_portfolio_value:
            self._peak_portfolio_value = new_portfolio_value

        # Build context and calculate reward via RewardManager
        context = self._build_reward_context(action, price, trade_info)
        reward = self._reward_manager.calculate_reward(context) + invalid_action_penalty

        # Store breakdown for logging/dashboard
        self._last_reward_breakdown = self._reward_manager.get_breakdown()

        # Get next state
        state = self.get_state()
        if state is None:
            return True, None, 0, None

        self._step_count += 1
        self._episode_returns.append(reward)
        self._portfolio_value_history.append(new_portfolio_value)

        return False, state, reward, self.get_stats(price)

    def _execute_action(self, action, price):
        """Execute trading action, return trade_info dict or None"""
        if action == 0:  # HOLD
            return None

        elif action in [1, 2, 3, 4]:  # BUY
            percent = {1: 0.10, 2: 0.25, 3: 0.50, 4: 1.0}[action]
            return self._buy_crypto(price, percent)

        elif action in [5, 6, 7, 8]:  # SELL
            if self._crypto_quantity == 0:
                return {'type': 'INVALID_SELL', 'penalty': -1.0}
            percent = {5: 0.10, 6: 0.25, 7: 0.50, 8: 1.0}[action]
            return self._sell_crypto(price, percent)

        return None

    def _get_volatility(self):
        """Calculate recent price volatility"""
        if len(self._price_history) < 10:
            return 0.02  # Default 2%
        prices = list(self._price_history)[-20:]
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0.02

    def _get_avg_volume(self):
        """Get average volume from ticker data"""
        # Use a default based on typical BTC volume
        # In real implementation, this would come from ticker data
        return 1_000_000  # $1M default volume per candle

    def _buy_crypto(self, price, percent_of_cash):
        """Execute buy with slippage, return trade_info"""
        if self._cash < 1:
            return {'type': 'FAILED_BUY', 'reason': 'insufficient_cash'}

        cash_to_use = self._cash * percent_of_cash

        # Calculate slippage
        slippage_cost = 0.0
        execution_price = price
        if self._slippage_enabled and self._slippage:
            volatility = self._get_volatility()
            avg_volume = self._get_avg_volume()
            execution_price, slippage_cost = self._slippage.get_execution_price(
                intended_price=price,
                order_value=cash_to_use,
                avg_volume=avg_volume,
                volatility=volatility,
                is_buy=True
            )
            self._total_slippage_cost += slippage_cost

        fee = cash_to_use * self._fee_pct
        crypto_bought = (cash_to_use - fee) / execution_price

        if crypto_bought <= 0:
            return {'type': 'FAILED_BUY', 'reason': 'zero_quantity'}

        # Update cost basis
        total_cost = (self._crypto_quantity * self._average_buy_price) + cash_to_use
        self._crypto_quantity += crypto_bought
        self._average_buy_price = total_cost / self._crypto_quantity if self._crypto_quantity > 0 else 0
        self._cash -= cash_to_use

        trade_info = {
            'type': 'BUY',
            'price': price,
            'execution_price': execution_price,
            'quantity': crypto_bought,
            'cash_used': cash_to_use,
            'fee': fee,
            'slippage_cost': slippage_cost,
            'percent': percent_of_cash,
            'step': self._step_count
        }

        self._trade_history.append(trade_info)
        return trade_info

    def _sell_crypto(self, price, percent_of_position):
        """Execute sell with slippage, return trade_info"""
        if self._crypto_quantity == 0:
            return {'type': 'INVALID_SELL'}

        crypto_to_sell = self._crypto_quantity * percent_of_position
        order_value = crypto_to_sell * price

        # Calculate slippage
        slippage_cost = 0.0
        execution_price = price
        if self._slippage_enabled and self._slippage:
            volatility = self._get_volatility()
            avg_volume = self._get_avg_volume()
            execution_price, slippage_cost = self._slippage.get_execution_price(
                intended_price=price,
                order_value=order_value,
                avg_volume=avg_volume,
                volatility=volatility,
                is_buy=False  # Sell
            )
            self._total_slippage_cost += slippage_cost

        cash_received = crypto_to_sell * execution_price
        fee = cash_received * self._fee_pct
        net_cash = cash_received - fee

        cost_basis = crypto_to_sell * self._average_buy_price
        profit = net_cash - cost_basis
        profit_pct = profit / cost_basis if cost_basis > 0 else 0

        self._crypto_quantity -= crypto_to_sell
        self._cash += net_cash

        trade_info = {
            'type': 'SELL',
            'price': price,
            'execution_price': execution_price,
            'quantity': crypto_to_sell,
            'cash_received': net_cash,
            'fee': fee,
            'slippage_cost': slippage_cost,
            'profit': profit,
            'profit_pct': profit_pct,
            'percent': percent_of_position,
            'step': self._step_count
        }

        self._trade_history.append(trade_info)
        return trade_info

    def get_stats(self, price):
        """Get trading statistics"""
        portfolio_value = self._cash + (self._crypto_quantity * price)
        roi = (portfolio_value - self._starting_cash) / self._starting_cash * 100

        num_trades = len(self._trade_history)
        num_buys = len([t for t in self._trade_history if t.get('type') == 'BUY'])
        num_sells = len([t for t in self._trade_history if t.get('type') == 'SELL'])

        profits = [t['profit'] for t in self._trade_history if t.get('type') == 'SELL' and 'profit' in t]
        winning_trades = [p for p in profits if p > 0]

        win_rate = len(winning_trades) / len(profits) if profits else 0.0
        avg_profit = np.mean(profits) if profits else 0
        total_fees = sum(t.get('fee', 0) for t in self._trade_history)

        position_value = self._crypto_quantity * price if self._crypto_quantity > 0 else 0
        position_pct = position_value / portfolio_value * 100 if portfolio_value > 0 else 0

        unrealized_pnl = 0
        unrealized_pnl_pct = 0
        if self._crypto_quantity > 0 and self._average_buy_price > 0:
            cost_basis = self._crypto_quantity * self._average_buy_price
            unrealized_pnl = position_value - cost_basis
            unrealized_pnl_pct = unrealized_pnl / cost_basis * 100

        total_slippage = sum(t.get('slippage_cost', 0) for t in self._trade_history)

        stats = {
            'portfolio_value': portfolio_value,
            'cash': self._cash,
            'crypto_quantity': self._crypto_quantity,
            'crypto_value': position_value,
            'position_pct': position_pct,
            'roi': roi,
            'num_trades': num_trades,
            'num_buys': num_buys,
            'num_sells': num_sells,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_fees': total_fees,
            'total_slippage': total_slippage,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct
        }

        start_ts = getattr(self, '_episode_start_timestamp', None)
        end_ts = self._ticker.get_current_timestamp()
        if start_ts and end_ts:
            stats['episode_start_date'] = start_ts
            stats['episode_end_date'] = end_ts

        return stats

    @property
    def reward_breakdown(self):
        """Get last reward breakdown by component with legacy mappings for dashboard"""
        breakdown = self._last_reward_breakdown.copy()

        # Map new component names to legacy dashboard categories
        # immediate: Direct profit/loss from trades
        breakdown['immediate'] = (
            breakdown.get('ProfitLossReward', 0) +
            breakdown.get('DrawdownPenaltyReward', 0)
        )

        # timing: Entry/exit quality and market timing
        breakdown['timing'] = (
            breakdown.get('BuyTheDipReward', 0) +
            breakdown.get('MomentumAlignmentReward', 0) +
            breakdown.get('TradingFrequencyReward', 0)
        )

        # portfolio_growth: Long-term portfolio health
        breakdown['portfolio_growth'] = (
            breakdown.get('PortfolioGrowthReward', 0) +
            breakdown.get('SharpeRatioReward', 0) +
            breakdown.get('PositionSizingReward', 0)
        )

        return breakdown

    def list_reward_calculators(self):
        """List active reward calculators"""
        return self._reward_manager.list_calculators()
