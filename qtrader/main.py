import gevent.monkey
gevent.monkey.patch_all()
from requests.packages.urllib3.util.ssl_ import create_urllib3_context
create_urllib3_context()

import os
from qlearn.agent import Agent
from ticker import TickerLocal
from broker import Broker

if __name__ == "__main__":
    ticker = TickerLocal(os.path.join('data', 'coinbase.csv'))

    broker = Broker(
        ticker=ticker,
        starting_balance=100,
        trade_pct=0.1,
        fee_pct=0.0015,
        tick_size=5
    )

    agent = Agent(
        name='btc-broker',
        num_states=broker.num_states,
        num_actions=broker.num_actions,
        batch_size=32,
        max_memory=10000,
        save_cnt=1000,
        model_path='models'
    )

    agent.train(
        broker,
        max_eps=0.1,
        min_eps=1e-3,
        decay=1e-4,
        gamma=0.5,
        learning_rate=1e-4
    )
