from position import Position
from trader import Trader

class Broker:
    def __init__(self, ticker, starting_balance, trade_pct, fee_pct, tick_size):
        self._trader = Trader(starting_balance)
        self._ticker = ticker
        self._starting_balance = starting_balance
        self._trade_pct = trade_pct
        self._fee_pct = fee_pct
        self._tick_size = tick_size
        self.reset()

        self.num_states = self._trader.num_states
        self.num_actions = 4

    def reset(self):
        self._position = None
        self._balance = self._starting_balance
        self._closed_trades = []
        self._last_tick = self._ticker.last_tick
        self._current_reward = 0

    def update(self, action):
        self._current_reward = 0
        self._last_tick = self._ticker.get_next(groups=self._tick_size)

        if self._balance < 1 or not self._last_tick.any():
            return True, None, 0, None

        price = self._last_tick['Close']
        stats = self.get_stats(price)

        if action > 1: # 2=BUY 3=SELL
            quantity = (1 / price) * (self._trade_pct * self._balance)
            self._trade(quantity, price, 'buy' if action == 2 else 'sell')
        elif action == 1: # 1=CLOSE
            self._close(price)
        elif action == 0: # 2=HOLD
            self._hold(price)

        reward = self._current_reward + (1 - (self._balance / self._starting_balance))
        state = self._trader.get_state(price, self._position, self._last_tick, self._balance)

        return False, state, self._current_reward, stats

    def get_stats(self, price):
        profits = [i.profit for i in self._closed_trades]
        pcts = [i.current_profit_pct(i.close) for i in self._closed_trades]

        avg_profit = sum(pcts) / len(pcts) if len(pcts) > 0 else 0
        max_profit = max(profits) if len(profits) > 0 else 0
        min_profit = min(profits) if len(profits) > 0 else 0

        current_pct = self._position.current_profit_pct(price) if self._position else 0

        currentPos = {
            'pos': self._position.buy_sell,
            'open': self._position.open,
            'quantity': self._position.quantity,
            'pnlPct': current_pct
        } if self._position else None

        return {
            'maxWin': max_profit,
            'maxLoss': min_profit,
            'avgProfitPct': avg_profit,
            'balance': self._balance,
            'currentPosition': currentPos,
            'currentPrice': price
        }

    def _trade(self, quantity, price, buy_sell):
        if self._position:
            if self._position.buy_sell != buy_sell:
                self._close(price)
            else:
                return

        self._position = Position(quantity, price, buy_sell, self._fee_pct)
        self._balance -= self._position.cost

    def _close(self, price):
        if not self._position:
            self._current_reward = -1.0
            return

        self._current_reward = self._position.current_profit_pct(price) * 10.0

        self._position.close_position(price)
        self._balance += self._position.revenue
        self._closed_trades.append(self._position)
        self._position = None

    def _hold(self, price):
        if not self._position:
            return

        profitPct = self._position.current_profit_pct(price)

        if profitPct < 0.1:
            self._current_reward = profitPct * 5.0
        else:
            self._current_reward = profitPct
