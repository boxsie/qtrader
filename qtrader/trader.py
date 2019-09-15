import ta
import math
import pandas as pd
import numpy as np

class Trader:
    def __init__(self):
        self.num_states = 5
        self._previous_candles = pd.DataFrame()

    def get_state(self, price, position, last_tick):
        self._previous_candles = self._previous_candles.append(last_tick, ignore_index=True)

        if self._previous_candles.shape[0] > 1000:
            self._previous_candles = self._previous_candles.drop(self._previous_candles.index[0])

        pos_buy_sell = 0
        pos_profit = 0

        if position:
            pos_buy_sell = 1 if position.buy_sell == 'buy' else -1
            pos_profit = position.current_profit_pct(price)

        h, l, c = self._previous_candles['High'], self._previous_candles['Low'], self._previous_candles['Close']

        rsi = ta.momentum.rsi(c, fillna=True).iloc[-1]
        rsi = (rsi / 50) - 1

        atr = ta.volatility.average_true_range(h, l, c, fillna=True).iloc[-1]
        atr = (atr / last_tick['Close']) * 100.0

        ao = ta.momentum.ao(h, l, fillna=True).iloc[-1]
        ao = ao * 0.1

        return np.array([
            pos_buy_sell,
            pos_profit,
            rsi,
            atr,
            ao
        ])