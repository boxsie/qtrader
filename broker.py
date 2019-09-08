import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from position import Position
import indicators

class Broker:
    def __init__(self, ticker, starting_balance, trade_pct, fee_pct, tick_size):
        self.num_states = 8
        self.num_actions = 4
        self._ticker = ticker
        self._starting_balance = starting_balance
        self._trade_pct = trade_pct
        self._fee_pct = fee_pct
        self._tick_size = tick_size

        self.reset()

    def reset(self):
        self._position = None
        self._balance = self._starting_balance
        self._closed_trades = []
        self._last_tick = self._ticker.last_tick
        self._current_reward = 0
        self._previous_closes = []

    def update(self, action):
        price = self._last_tick['Close']

        if action > 1: # 2=BUY 3=SELL
            quantity = (1 / price) * (self._trade_pct * self._balance)
            self._trade(quantity, price, 'buy' if action == 2 else 'sell')
        elif action == 1: # 1=CLOSE
            self._close(price)
        elif action == 0: # 2=HOLD
            self._hold(price)

        self._previous_closes.append(math.log(price))
        self._last_tick = self._ticker.get_next(groups=self._tick_size)
        self._current_reward += ((self._balance - self._starting_balance) / self._starting_balance)

        is_complete = len(self._last_tick) == 0
        reward = self._current_reward
        state = None if is_complete else self.get_state(price)

        self._current_reward = 0

        profits = [i.profit for i in self._closed_trades]
        pcts = [i.current_profit_pct(i.close) for i in self._closed_trades]
        current_pct = self._position.current_profit_pct(price) if self._position else 0

        avg_profit = sum(pcts) / len(pcts) if len(pcts) > 0 else 0
        max_profit = max(profits) if len(profits) > 0 else 0
        min_profit = min(profits) if len(profits) > 0 else 0
        pos = f'{self._position.buy_sell.upper()}:{self._position.quantity:,.2f}@{self._position.open:,.2f}' if self._position else 'None     '
        print(f'Biggest Win:{max_profit:,.2f} | Biggest Loss:{min_profit:,.2f} | Avg. Profit:{avg_profit * 100:,.2f}% | Balance:{self._balance:,.2f} | Position P&L:{current_pct * 100:,.2f}% | Position:{pos} | Price:{price}                 ', end='\r')

        return is_complete, state, reward

    def get_state(self, price):
        pos_buy_sell = 0
        pos_profit = 0

        if self._position:
            pos_buy_sell = 1 if self._position.buy_sell == 'buy' else -1
            pos_profit = self._position.current_profit_pct(price)

        return np.array([
            self._balance / self._starting_balance,
            pos_buy_sell,
            pos_profit,
            self._get_ema(5),
            self._get_ema(10),
            self._get_ema(20),
            self._get_ema(50),
            self._get_ema(100)
        ])

    def _get_ema(self, period):
        df = pd.Series(self._previous_closes)
        ema = pd.Series.ewm(df[::-1], span=period).mean()
        return ema.iloc[-1]

    def _trade(self, quantity, price, buy_sell):
        if self._position:
            if self._position.buy_sell == buy_sell:
                self._current_reward -= 1.0
                return
            else:
                self._close(price)

        self._position = Position(quantity, price, buy_sell, self._fee_pct)
        self._balance -= self._position.cost

    def _close(self, price):
        if not self._position:
            self._current_reward -= 1.0
            return

        self._position.close_position(price)
        self._balance += self._position.revenue
        self._closed_trades.append(self._position)

        self._current_reward += -0.1 + self._position.current_profit_pct(price) if self._position.profit > 0 else -1

        col = ''
        col_end = '\033[0m'
        if self._position.profit > 0:
            col = '\033[92m'
        elif self._position.profit < 0:
            col = '\033[91m'
        print(f'Closed trade {self._position.buy_sell.upper()}:{self._position.quantity:,.2f}@{price:,.2f} {col}{self._position.profit:,.2f}({self._position.current_profit_pct(price) * 100:,.2f}%){col_end}                                                                                                        ')

        self._position = None

    def _hold(self, price):
        if not self._position:
            return

        self._current_reward += self._position.current_profit_pct(price) * 0.1