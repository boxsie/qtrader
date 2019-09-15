import os
from qlearn.agent import Agent
from ticker import TickerLocal
from broker import Broker

if __name__ == "__main__":
    ticker = TickerLocal(os.path.join('data', 'coinbase-tidy.csv'))

    broker = Broker(
        ticker=ticker,
        starting_balance=1000,
        trade_pct=0.15,
        fee_pct=0.0015,
        tick_size=15
    )

    agent = Agent(
        name='btc-broker',
        num_states=broker.num_states,
        num_actions=broker.num_actions,
        batch_size=64,
        max_memory=10000,
        save_cnt=1000,
        model_path='models',
        hub_address='localhost',
        hub_port=5001
    )

    agent.train(
        broker,
        max_eps=0.1,
        min_eps=1e-3,
        decay=1e-3,
        gamma=0.85,
        learning_rate=1e-4
    )
