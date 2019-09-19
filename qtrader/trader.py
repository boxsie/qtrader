import ta
import math
import pandas as pd
import numpy as np

class Trader:
    def __init__(self, starting_candles):
        self.num_states = (5 * len(starting_candles)) + 2
        self._previous_candles = pd.DataFrame(starting_candles)

    def get_state(self, position, last_tick):
        self._previous_candles = self._previous_candles.append(last_tick, ignore_index=True)

        if self._previous_candles.shape[0] > 1000:
            self._previous_candles = self._previous_candles.drop(self._previous_candles.index[0])

        pos_buy_sell = 0
        pos_profit = 0

        if position:
            pos_buy_sell = 1 if position.buy_sell == 'buy' else -1
            pos_profit = position.current_profit_pct(last_tick['Close'])
            pos_profit = pos_profit if not pd.isnull(pos_profit) else 0

        h = self._previous_candles['High']
        l = self._previous_candles['Low']
        c = self._previous_candles['Close']
        v = self._previous_candles['Volume_(Currency)']

        rsi = ta.momentum.rsi(c) \
            .tail(14) \
            .apply(lambda x: (x / 50) - 1) \
            .replace({np.nan:None}) \
            .values

        atr = ta.volatility.average_true_range(h, l, c) \
            .tail(14) \
            .apply(lambda x: (x / last_tick['Close']) * 100.0) \
            .replace({np.nan:None}) \
            .values

        ao = ta.momentum.ao(h, l) \
            .tail(14) \
            .apply(lambda x: x * 0.12) \
            .replace({np.nan:None}) \
            .values

        mf = ta.momentum.money_flow_index(h, l, c, v) \
            .tail(14) \
            .apply(lambda x: (x / 50) - 1) \
            .replace({np.nan:None}) \
            .values

        tsi = ta.momentum.tsi(c) \
            .tail(14) \
            .apply(lambda x: x * 0.02) \
            .replace({np.nan:None}) \
            .values

        base = np.array([
            pos_buy_sell,
            pos_profit
        ])

        return np.concatenate((base, rsi, atr, ao, mf, tsi), axis=0)