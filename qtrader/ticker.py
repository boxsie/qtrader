import pandas as pd
import numpy as np
import threading
import os
import sys

class TickerLocal:
    def __init__(self, filename):
        self._filename = filename
        self.reset()

    def reset(self):
        self._data = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), self._filename))
        self.last_tick = self._data.iloc[0]
        self._idx = 1
        self._last_close = 0

    def get_next(self, groups=1):
        if self._idx >= self._data.shape[0]:
            return {}, True

        rows = self._data.iloc[self._idx:self._idx + groups]
        self._idx += groups

        o = rows['Open'].loc[~rows['Open'].isnull()]
        c = rows['Close'].loc[~rows['Close'].isnull()]

        if c.any():
            self._last_close = c.iloc[-1]

        self.last_tick = pd.Series({
            'High': rows['High'].max(),
            'Low': rows['Low'].min(),
            'Open': o.iloc[0] if o.any() else np.NaN,
            'Close': self._last_close,
            'Volume_(Currency)': rows['Volume_(Currency)'].sum()
        })

        return self.last_tick, False

    def get_bulk(self, count, groups=1):
        return [self.get_next(groups) for i in range(count)]