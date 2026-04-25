import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import deque
import json
from datetime import datetime


class Trainer:
    """
    Reinforcement learning trainer.

    Features:
    - Double DQN with Prioritized Experience Replay
    - Adaptive epsilon decay
    - Model checkpointing with best model tracking
    - Episode statistics and analytics
    - JSON and text log files
    """

    def __init__(
        self,
        agent,
        broker,
        memory,
        batch_size=64,
        max_epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=0.995,
        update_frequency=4,
        save_dir='runs',
        save_frequency=100,
        max_episode_steps=None,
        run_id=None,
        dashboard_client=None
    ):
        self.agent = agent
        self.broker = broker
        self.max_episode_steps = max_episode_steps
        self.memory = memory
        self.batch_size = batch_size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency

        # Epsilon tracking
        self.epsilon = max_epsilon

        # Run ID for tracking (timestamp if not provided)
        self.run_id = run_id if run_id else datetime.now().strftime("%Y%m%d_%H%M%S")

        # Directory for this run (all files go here)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create run-specific log files (no run_id in filename since they're in run_id directory)
        self.run_log_file = self.save_dir / "run_log.jsonl"
        self.run_summary_file = self.save_dir / "run_summary.txt"

        # Write run header
        with open(self.run_summary_file, 'w') as f:
            f.write(f"Training Run: {self.run_id}\n")
            f.write(f"="*80 + "\n")
            f.write(f"Save Dir: {self.save_dir}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Epsilon: {self.epsilon:.3f}\n")
            f.write(f"="*80 + "\n\n")

        # Tracking
        self.episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.losses = deque(maxlen=100)

        # Dashboard integration (optional)
        self.dashboard = dashboard_client
        self.total_episodes = None  # Will be set when train() is called

        print(f"[Trainer] Initialized")
        print(f"[Trainer] Run ID: {self.run_id}")
        print(f"[Trainer] Save dir: {self.save_dir}")
        print(f"[Trainer] Batch size: {self.batch_size}")
        print(f"[Trainer] Epsilon: {self.epsilon:.3f}")
        if self.dashboard:
            print(f"[Trainer] Dashboard enabled")

    def train_episode(self):
        """
        Train for one complete episode.

        Returns:
            episode_stats: Dictionary with episode statistics
        """
        # Emit episode start to dashboard
        if self.dashboard:
            print(f"[Dashboard] Emitting episode_start for episode {self.episode}")
            self.dashboard.emit_episode_start(
                episode=self.episode,
                total_episodes=self.total_episodes or 0,
                config={
                    "epsilon": self.epsilon,
                    "batch_size": self.batch_size,
                    "max_episode_steps": self.max_episode_steps
                }
            )

        self.broker.reset()
        state = self.broker.get_state()

        if state is None:
            return None

        episode_reward = 0
        episode_loss = []
        steps = 0
        import time
        episode_start_time = time.time()

        # Track action distribution and reward breakdown
        # Actions: 0=HOLD, 1-4=BUY(10/25/50/100%), 5-8=SELL(10/25/50/100%)
        action_counts = {i: 0 for i in range(9)}
        reward_breakdown = {}  # Dynamic - populated from RewardManager
        q_values_history = []

        # Track previous trade count to detect new trades
        previous_trade_count = 0

        # Progress bar for this episode
        pbar = tqdm(desc=f"Episode {self.episode}", leave=False)

        # Initialize LSTM hidden state for this episode
        # This allows the agent to remember patterns across the episode
        hidden_state = None

        while True:
            # Select action (maintaining LSTM memory across steps)
            action, hidden_state = self.agent.policy_net.get_action(
                state,
                epsilon=self.epsilon,
                hidden_state=hidden_state
            )

            # Track action distribution
            action_counts[action] += 1

            # Execute action
            done, next_state, reward, info = self.broker.update(action)

            # Track reward breakdown from broker's modular reward system
            if hasattr(self.broker, '_last_reward_breakdown'):
                rb = self.broker._last_reward_breakdown
                for key, value in rb.items():
                    if key not in reward_breakdown:
                        reward_breakdown[key] = 0.0
                    reward_breakdown[key] += value

            episode_reward += reward
            steps += 1
            self.total_steps += 1

            # Store in replay memory
            if next_state is not None:
                self.memory.add(
                    (state, action, reward, next_state, float(done))
                )

            # Train agent (do multiple updates per step for more GPU work)
            # Start training as soon as we have enough samples (use smaller threshold)
            min_samples = min(self.batch_size, 1024)  # Start training with at least 1024 samples
            if len(self.memory) >= min_samples and self.total_steps % self.update_frequency == 0:
                # Multiple gradient steps per update for MAXIMUM GPU utilization
                num_updates = 4  # Do 4 gradient updates per step - GPU goes BRRRR

                for _ in range(num_updates):
                    loss = self._train_step()
                    if loss is not None:
                        episode_loss.append(loss)

            # Update progress bar
            pbar.update(1)
            if info:
                pbar.set_postfix({
                    'portfolio': f"{info['portfolio_value']:.2f}",
                    'reward': f"{episode_reward:.2f}",
                    'epsilon': f"{self.epsilon:.3f}"
                })

                # Emit step update to dashboard (every step, no throttling)
                if self.dashboard:
                    current_price = self.broker._last_tick['Close'] if self.broker._last_tick is not None and 'Close' in self.broker._last_tick else 0

                    # Include reward breakdown in step update
                    step_data = info.copy()
                    if hasattr(self.broker, '_last_reward_breakdown'):
                        step_data['reward_breakdown'] = self.broker._last_reward_breakdown

                    self.dashboard.emit_step_update(
                        episode=self.episode,
                        step=steps,
                        action=action,
                        reward=reward,
                        price=current_price,
                        stats=step_data,
                        total_episodes=self.total_episodes
                    )

                # Detect and emit new trades
                if self.dashboard:
                    current_trade_count = len(self.broker._trade_history)
                    if current_trade_count > previous_trade_count:
                        # A new trade was executed
                        latest_trade = self.broker._trade_history[-1]
                        current_price = self.broker._last_tick['Close'] if self.broker._last_tick is not None and 'Close' in self.broker._last_tick else 0

                        self.dashboard.emit_trade_executed(
                            episode=self.episode,
                            step=steps,
                            trade_type=latest_trade['type'],
                            price=latest_trade['price'],
                            quantity=latest_trade['quantity'],
                            total_value=latest_trade.get('cash_used', latest_trade.get('cash_received', 0))
                        )
                        previous_trade_count = current_trade_count

            if done:
                break

            # Check max episode steps
            if self.max_episode_steps and steps >= self.max_episode_steps:
                break

            state = next_state

        pbar.close()

        # Get FINAL stats from broker (critical for accurate summary)
        current_price = self.broker._last_tick['Close'] if self.broker._last_tick is not None and 'Close' in self.broker._last_tick else 0
        final_info = self.broker.get_stats(current_price)

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Compile episode statistics
        episode_stats = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'episode_steps': steps,
            'episode_reward': episode_reward,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(episode_loss) if episode_loss else 0,
            'action_counts': action_counts,
            'reward_breakdown': reward_breakdown,
        }

        # Use FINAL stats, not stale info
        if final_info:
            episode_stats.update(final_info)

        # Track for moving averages
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        if episode_loss:
            self.losses.extend(episode_loss)

        # Save to log files
        self._save_episode_log(episode_stats)

        # Save checkpoint if improved
        self._save_checkpoint(episode_stats)

        # Emit episode end and training metrics to dashboard
        if self.dashboard:
            episode_duration = time.time() - episode_start_time
            # Add reward_breakdown to final_stats for dashboard
            dashboard_final_stats = (final_info or {}).copy()
            dashboard_final_stats['reward_breakdown'] = reward_breakdown

            self.dashboard.emit_episode_end(
                episode=self.episode,
                final_stats=dashboard_final_stats,
                duration_seconds=episode_duration
            )
            # Get learning rate from agent's optimizer
            learning_rate = None
            if hasattr(self.agent, 'optimizer'):
                learning_rate = self.agent.optimizer.param_groups[0]['lr']

            self.dashboard.emit_training_metrics(
                episode=self.episode,
                loss=episode_stats.get('avg_loss'),
                epsilon=self.epsilon,
                avg_reward=episode_reward / max(steps, 1),
                learning_rate=learning_rate,
                memory_size=len(self.memory)
            )

        self.episode += 1

        return episode_stats

    def _train_step(self):
        """Perform one training step with PER"""
        # Sample batch from replay memory (larger batches = more GPU work)
        batch_data, indices, weights = self.memory.sample(self.batch_size)

        # Unpack batch - use vstack for proper array stacking
        states = np.vstack([x[0] for x in batch_data])
        actions = np.array([x[1] for x in batch_data])
        rewards = np.array([x[2] for x in batch_data])
        next_states = np.vstack([x[3] for x in batch_data])
        dones = np.array([x[4] for x in batch_data])

        batch = (states, actions, rewards, next_states, dones)

        # Update agent
        loss, td_errors = self.agent.update(batch, weights)

        # Update priorities in memory
        self.memory.update_priorities(indices, td_errors)

        return loss

    def _save_episode_log(self, stats):
        """Save episode statistics to JSON log file and summary text file"""
        # Save JSON line (for programmatic analysis)
        with open(self.run_log_file, 'a') as f:
            json.dump(stats, f, default=str)
            f.write('\n')

        # Save human-readable summary (append mode) - use same format as console
        with open(self.run_summary_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Episode {stats['episode']} Summary\n")
            f.write(f"{'-'*60}\n")

            # Episode date range
            if 'episode_start_date' in stats and 'episode_end_date' in stats:
                try:
                    import pandas as pd
                    start = pd.to_datetime(stats['episode_start_date'])
                    end = pd.to_datetime(stats['episode_end_date'])
                    f.write(f"  Date Range: {start.strftime('%Y-%m-%d %H:%M')} -> {end.strftime('%Y-%m-%d %H:%M')}\n")
                except:
                    f.write(f"  Date Range: {stats['episode_start_date']} -> {stats['episode_end_date']}\n")

            # Basic metrics
            f.write(f"  Steps: {stats['episode_steps']}\n")
            f.write(f"  Reward: {stats['episode_reward']:.2f}\n")
            f.write(f"  Portfolio Value: ${stats.get('portfolio_value', 0):.2f}\n")
            f.write(f"  Cash: ${stats.get('cash', 0):.2f}  Crypto: {stats.get('crypto_quantity', 0):.6f} BTC (${stats.get('crypto_value', 0):.2f})\n")
            f.write(f"  ROI: {stats.get('roi', 0):.2f}%\n")
            f.write(f"  Trades: {stats.get('num_trades', 0)} (Buys: {stats.get('num_buys', 0)}, Sells: {stats.get('num_sells', 0)})\n")
            f.write(f"  Win Rate: {stats.get('win_rate', 0):.2%}\n")
            f.write(f"  Fees: ${stats.get('total_fees', 0):.2f}  Slippage: ${stats.get('total_slippage', 0):.2f}\n")

            # Action distribution
            if 'action_counts' in stats:
                actions = stats['action_counts']
                total = sum(actions.values())
                buy_total = sum(actions[i] for i in [1, 2, 3, 4])
                sell_total = sum(actions[i] for i in [5, 6, 7, 8])
                f.write(f"\n  Action Distribution:\n")
                f.write(f"    HOLD:     {actions[0]:4d} ({actions[0]/total*100:5.1f}%)\n")
                f.write(f"    BUY (all):  {buy_total:4d} ({buy_total/total*100:5.1f}%)  [10%:{actions[1]}, 25%:{actions[2]}, 50%:{actions[3]}, 100%:{actions[4]}]\n")
                f.write(f"    SELL (all): {sell_total:4d} ({sell_total/total*100:5.1f}%)  [10%:{actions[5]}, 25%:{actions[6]}, 50%:{actions[7]}, 100%:{actions[8]}]\n")

            # Reward breakdown (dynamic from modular system)
            if 'reward_breakdown' in stats:
                rb = stats['reward_breakdown']
                f.write(f"\n  Reward Sources:\n")
                for name, value in sorted(rb.items()):
                    if name != 'total':
                        f.write(f"    {name:30s}: {value:8.2f}\n")
                if 'total' in rb:
                    f.write(f"    {'─'*40}\n")
                    f.write(f"    {'TOTAL':30s}: {rb['total']:8.2f}\n")

            # Learning metrics
            f.write(f"\n  Learning:\n")
            f.write(f"    Avg Loss:  {stats['avg_loss']:.4f}\n")
            f.write(f"    Epsilon:   {stats['epsilon']:.3f}\n")
            f.write(f"    Memory:    {len(self.memory):,} / {self.memory.capacity:,}\n")

            # GPU metrics (if CUDA)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem_used = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    f.write(f"\n  GPU:\n")
                    f.write(f"    Memory Used: {gpu_mem_used:.2f} / {gpu_mem_total:.2f} GB ({gpu_mem_used/gpu_mem_total*100:.1f}%)\n")
            except:
                pass

            f.write(f"{'='*60}\n")

    def _save_checkpoint(self, stats):
        """Save checkpoint if performance improved"""
        episode_reward = stats['episode_reward']

        # Save based on reward (no run_id in filename since we're in run_id directory)
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            filepath = self.save_dir / 'best_model.pt'
            self.agent.save(filepath)
            print(f"\n[Checkpoint] New best reward: {episode_reward:.2f}")

        # Regular checkpoint (no run_id in filename since we're in run_id directory)
        if self.episode % self.save_frequency == 0:
            filepath = self.save_dir / f'checkpoint_ep{self.episode}.pt'
            self.agent.save(filepath)

            # Save memory
            mem_filepath = self.save_dir / f'memory_ep{self.episode}.pkl'
            self.memory.save(mem_filepath)

    def train(self, num_episodes=1000):
        """
        Train for multiple episodes.

        Args:
            num_episodes: Number of episodes to train
        """
        # Store total episodes for dashboard
        self.total_episodes = num_episodes

        print(f"\n{'='*60}")
        print(f"Starting training for {num_episodes} episodes")
        print(f"{'='*60}\n")

        for _ in range(num_episodes):
            stats = self.train_episode()

            if stats:
                # Print summary after each episode (always useful for analysis)
                self._print_summary(stats)

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Total episodes: {self.episode}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Best reward: {self.best_reward:.2f}")
        print(f"{'='*60}\n")

        # Write final summary to file
        with open(self.run_summary_file, 'a') as f:
            f.write(f"\n\n{'='*80}\n")
            f.write(f"TRAINING COMPLETE\n")
            f.write(f"{'='*80}\n")
            f.write(f"  Total episodes: {self.episode}\n")
            f.write(f"  Total steps: {self.total_steps}\n")
            f.write(f"  Best reward: {self.best_reward:.2f}\n")
            f.write(f"\n  Best model saved to: {self.save_dir / 'best_model.pt'}\n")
            f.write(f"\n  Log files:\n")
            f.write(f"    JSON data: {self.run_log_file}\n")
            f.write(f"    Summary:   {self.run_summary_file}\n")
            f.write(f"{'='*80}\n")

        print(f"\nLogs saved to:")
        print(f"  JSON: {self.run_log_file}")
        print(f"  Summary: {self.run_summary_file}")

    def _print_summary(self, stats):
        """Print training summary"""
        print(f"\n{'='*60}")
        print(f"Episode {stats['episode']} Summary")
        print(f"{'-'*60}")

        # Episode date range
        if 'episode_start_date' in stats and 'episode_end_date' in stats:
            import pandas as pd
            start = pd.to_datetime(stats['episode_start_date'])
            end = pd.to_datetime(stats['episode_end_date'])
            print(f"  Date Range: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')}")

        # Basic metrics
        print(f"  Steps: {stats['episode_steps']}")
        print(f"  Reward: {stats['episode_reward']:.2f}")
        print(f"  Portfolio Value: ${stats.get('portfolio_value', 0):.2f}")
        print(f"  Cash: ${stats.get('cash', 0):.2f}  Crypto: {stats.get('crypto_quantity', 0):.6f} BTC (${stats.get('crypto_value', 0):.2f})")
        print(f"  ROI: {stats.get('roi', 0):.2f}%")
        print(f"  Trades: {stats.get('num_trades', 0)} (Buys: {stats.get('num_buys', 0)}, Sells: {stats.get('num_sells', 0)})")
        print(f"  Win Rate: {stats.get('win_rate', 0):.2%}")
        print(f"  Fees: ${stats.get('total_fees', 0):.2f}  Slippage: ${stats.get('total_slippage', 0):.2f}")

        # Action distribution
        if 'action_counts' in stats:
            actions = stats['action_counts']
            total = sum(actions.values())
            buy_total = sum(actions[i] for i in [1, 2, 3, 4])
            sell_total = sum(actions[i] for i in [5, 6, 7, 8])
            print(f"\n  Action Distribution:")
            print(f"    HOLD:     {actions[0]:4d} ({actions[0]/total*100:5.1f}%)")
            print(f"    BUY (all):  {buy_total:4d} ({buy_total/total*100:5.1f}%)  [10%:{actions[1]}, 25%:{actions[2]}, 50%:{actions[3]}, 100%:{actions[4]}]")
            print(f"    SELL (all): {sell_total:4d} ({sell_total/total*100:5.1f}%)  [10%:{actions[5]}, 25%:{actions[6]}, 50%:{actions[7]}, 100%:{actions[8]}]")

        # Reward breakdown (dynamic from modular system)
        if 'reward_breakdown' in stats:
            rb = stats['reward_breakdown']
            print(f"\n  Reward Sources:")
            for name, value in sorted(rb.items()):
                if name != 'total':
                    print(f"    {name:30s}: {value:8.2f}")
            if 'total' in rb:
                print(f"    {'─'*40}")
                print(f"    {'TOTAL':30s}: {rb['total']:8.2f}")

        # Learning metrics
        print(f"\n  Learning:")
        print(f"    Avg Loss:  {stats['avg_loss']:.4f}")
        print(f"    Epsilon:   {stats['epsilon']:.3f}")
        print(f"    Memory:    {len(self.memory):,} / {self.memory.capacity:,}")

        # GPU metrics (if CUDA is available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated(0) / 1024**3
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"\n  GPU:")
                print(f"    Memory Used: {gpu_mem_used:.2f} / {gpu_mem_total:.2f} GB ({gpu_mem_used/gpu_mem_total*100:.1f}%)")
        except:
            pass

        print(f"{'='*60}\n")

    def load_checkpoint(self, model_path, memory_path=None):
        """
        Load a saved checkpoint.

        Args:
            model_path: Path to model checkpoint
            memory_path: Optional path to memory checkpoint
        """
        self.agent.load(model_path)
        print(f"Loaded model from {model_path}")

        if memory_path and Path(memory_path).exists():
            self.memory.load(memory_path)
            print(f"Loaded memory from {memory_path}")

    def evaluate(self, num_episodes=10):
        """
        Evaluate the agent without training.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            List of episode statistics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating for {num_episodes} episodes")
        print(f"{'='*60}\n")

        old_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation

        eval_stats = []

        for i in range(num_episodes):
            self.broker.reset()
            state = self.broker.get_state()

            if state is None:
                continue

            episode_reward = 0
            steps = 0

            while True:
                action, _ = self.agent.policy_net.get_action(state, epsilon=0.0)
                done, next_state, reward, info = self.broker.update(action)

                episode_reward += reward
                steps += 1

                if done:
                    break

                state = next_state

            stats = {
                'episode': i,
                'steps': steps,
                'reward': episode_reward
            }
            if info:
                stats.update(info)

            eval_stats.append(stats)

            print(f"Eval Episode {i+1}/{num_episodes} - "
                  f"Reward: {episode_reward:.2f}, "
                  f"Balance: {stats.get('balance', 0):.2f}, "
                  f"Win Rate: {stats.get('win_rate', 0):.2%}")

        self.epsilon = old_epsilon

        # Print evaluation summary
        avg_reward = np.mean([s['reward'] for s in eval_stats])
        avg_balance = np.mean([s.get('balance', 0) for s in eval_stats])
        avg_win_rate = np.mean([s.get('win_rate', 0) for s in eval_stats])

        print(f"\n{'='*60}")
        print(f"Evaluation Summary")
        print(f"{'-'*60}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Balance: {avg_balance:.2f}")
        print(f"  Avg Win Rate: {avg_win_rate:.2%}")
        print(f"{'='*60}\n")

        return eval_stats
