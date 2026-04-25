# Reward system

QTrader's reward signal is built from small, composable calculators rather
than a single hardcoded function. Each calculator returns a scalar; the
`RewardManager` combines them as a weighted sum and the result is what the
DQN sees.

## How it's wired

`agent/broker.py` builds a context dict each step (action, price, cash,
portfolio value, trade info, etc.) and hands it to the `RewardManager`.
`agent/train.py` constructs the manager from the active profile YAML's
`reward_config` block.

```python
# agent/train.py (simplified)
if 'reward_config' in profile:
    reward_manager = create_reward_manager(profile['reward_config'])
else:
    reward_manager = RewardManager([
        ProfitLossReward(weight=1.0),
        PortfolioGrowthReward(weight=2.0),
    ])
```

Add or remove calculators in YAML — no code changes.

## Available calculators

| Type                 | Best for                | Behaviour                                         |
|----------------------|-------------------------|---------------------------------------------------|
| `profit_loss`        | All traders             | Reward realized profit, penalize loss + fees.     |
| `portfolio_growth`   | Growth-focused          | Reward new portfolio peaks and growth above MA.   |
| `opportunity_cost`   | Active traders          | Penalize sitting flat through favourable moves.   |
| `drawdown_penalty`   | Risk-averse             | Penalize unrealized drawdown from peak.           |
| `momentum_alignment` | Trend followers         | Reward trades that go with the prevailing trend.  |
| `buy_the_dip`        | Contrarians             | Reward buying weakness, selling strength.         |
| `trade_frequency`    | Selective traders       | Penalize overtrading.                             |
| `position_sizing`    | Risk managers           | Reward holding near a target position fraction.   |
| `sharpe_ratio`       | Consistent returns      | Reward risk-adjusted PnL over a rolling window.   |

Each lives in `agent/rewards/<name>.py`. Read the constructor signature for
the available `params`.

## Profile bundles

Trader profiles in `agent/profiles/` ship with a `reward_config` that
matches the trading style.

| Profile          | Primary signal       | Highlights                                |
|------------------|----------------------|-------------------------------------------|
| `scalper`        | profit_loss (3.0)    | Aggressive realized PnL, short lookback.  |
| `day_trader`     | portfolio_growth (2) | Balanced profits + growth.                |
| `swing_trader`   | portfolio_growth (3) | Let winners run, long history window.     |
| `balanced`       | mixed                | All calculators, well-rounded.            |
| `risk_averse`    | drawdown_penalty (3) | Heavy loss penalty, sharpe-aware.         |
| `contrarian`     | buy_the_dip          | Counter-trend bias.                       |
| `trend_follower` | momentum_alignment   | Pro-trend bias.                           |

Run with: `python agent/train.py --config fast --profile day_trader`.

## YAML format

```yaml
# agent/profiles/<name>.yaml
reward_config:
  - type: profit_loss          # calculator name
    weight: 1.5                # multiplier on the calculator's output
    params:                    # forwarded to the calculator's __init__
      profit_multiplier: 10.0
      loss_multiplier: 3.0

  - type: portfolio_growth
    weight: 2.0
    params:
      peak_multiplier: 2.5
```

Validate that a profile loads cleanly:

```bash
pytest tests/test_profile_rewards.py -v
```

## Custom calculators

Subclass `RewardCalculator` and register the type with the factory.

```python
# agent/rewards/my_reward.py
from .base import RewardCalculator

class MyReward(RewardCalculator):
    def __init__(self, weight: float = 1.0, my_param: float = 1.0):
        super().__init__(weight)
        self.my_param = my_param

    def reset(self):
        ...

    def calculate_reward(self, context: dict) -> float:
        # context: action, price, cash, crypto_quantity, portfolio_value,
        #          starting_cash, average_buy_price, fee_pct, step, trade_info
        return 0.0
```

Then add it in `agent/rewards/factory.py` `CALCULATOR_REGISTRY` and reference
it from any profile YAML by `type: my_reward`.

## Debugging

The trainer logs a per-calculator breakdown each episode:

```
Reward Breakdown:
  ProfitLossReward         :   149.75
  PortfolioGrowthReward    :    42.30
  OpportunityCostReward    :    -5.20
  Total                    :   186.85
```

Use it to spot calculators that dominate or contribute nothing — usually a
sign that weights need tuning.
