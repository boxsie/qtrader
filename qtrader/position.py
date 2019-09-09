
class Position:
    def __init__(self, quantity, price, buy_sell, fee_pct):
        self.quantity = quantity
        self.open = price
        self.buy_sell = buy_sell
        self.fee_pct = fee_pct
        self.cost = self.quantity * self.open
        self.close = None
        self.profit = None
        self.revenue = None

    def current_profit(self, current_price):
        return self._calculate_profit(current_price)

    def current_profit_pct(self, current_price):
        return self._calculate_profit(current_price) / self.cost if self.quantity > 0 else 0

    def close_position(self, price):
        self.close = price
        self.profit = self._calculate_profit(self.close)
        self.revenue = self.profit + self.cost

    def _calculate_profit(self, price):
        profit = 0
        if self.buy_sell == 'buy':
            profit = (self.quantity * price) - self.cost
        elif self.buy_sell == 'sell':
            profit = self.cost - (self.quantity * price)
        return profit - (self.cost * self.fee_pct)