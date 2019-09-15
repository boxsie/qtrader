import pandas as pd
import threading
import os
import sys

class TickerLocal:
    def __init__(self, filename):
        self._filename = filename
        self.reset()

    def get_next(self, groups=1):
        if self._idx >= self._data.shape[0]:
            return {}

        rows = self._data.iloc[self._idx:self._idx + groups]

        if not rows.iloc[-1].isnull().values.any():
            self.last_tick = rows.iloc[-1]

        self._idx += groups
        return self.last_tick

    def reset(self):
        self._data = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), self._filename))
        self.last_tick = self._data.iloc[0]
        self._idx = 1