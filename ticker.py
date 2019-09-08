import pandas as pd
import threading

class TickerLocal:
    def __init__(self, filename):
        self._data = pd.read_csv(filename)
        self.last_tick = self._data.iloc[0]
        self._idx = 1

    def get_next(self, groups=1):
        if self._idx >= self._data.shape[0]:
            return {}

        rows = self._data.iloc[self._idx:self._idx + groups]

        if not rows.iloc[-1].isnull().values.any():
            self.last_tick = {
                'Close': rows.iloc[-1]['Close']
            }

        self._idx += groups
        return self.last_tick