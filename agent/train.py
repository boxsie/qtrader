"""
Modern QTrader Training Script

Improvements over original:
- PyTorch-based Double DQN with Dueling architecture
- LSTM for temporal pattern recognition
- Prioritized Experience Replay
- Comprehensive technical indicators
- Advanced reward shaping with risk management
- Real-time dashboard monitoring (WebSocket streaming)
- Configuration management
- Model checkpointing
"""

import argparse
import yaml
from pathlib import Path
import torch
from datetime import datetime

# Import components
from ticker import Ticker
from broker import Broker
from qlearn.model_pytorch import DoubleDQNAgent
from qlearn.memory_prioritized import Memory
from qlearn.trainer_pytorch import Trainer
from hashing import get_output_paths, save_provenance, hash_config, hash_profile
from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config, profile):
    """Setup trading environment"""
    print("\n" + "="*60)
    print("Setting up environment")
    print("="*60)

    # Multi-timeframe mode (required)
    if 'timeframe_weights' not in profile:
        raise ValueError("Profile must include timeframe_weights")

    profile_name = profile['name']
    timeframe_weights = profile['timeframe_weights']

    print(f"\n[Mode] Multi-Timeframe Trading")
    print(f"[Profile] {profile_name.upper()} - {profile['description']}")
    print(f"\n[Timeframe Weights]")
    for tf, weight in timeframe_weights.items():
        if weight > 0:
            print(f"  {tf:4s}: {weight:.2f}")

    data_file = config['data']['data_file']
    use_recent_data = config['data'].get('use_recent_data', False)
    recent_data_rows = config['data'].get('recent_data_rows', None)
    random_start = config['data'].get('random_start', False)
    episode_length = config['data'].get('episode_length', None)

    print(f"\n[Data] Loading from {data_file}")
    if random_start:
        print(f"[Data] Random start enabled - will vary episode windows")
    ticker = Ticker(
        data_file,
        timeframe_weights=timeframe_weights,
        use_recent_data=use_recent_data,
        recent_data_rows=recent_data_rows,
        random_start=random_start,
        episode_length=episode_length
    )

    print(f"\n[Broker] Initializing Multi-Timeframe Broker")
    print(f"[Broker] Actions: HOLD, BUY (10/25/50/100%), SELL (10/25/50/100%)")

    # Get reward config from profile
    reward_config = profile.get('reward_config')
    if not reward_config:
        raise ValueError("Profile must include reward_config")

    print(f"[Broker] Reward calculators: {len(reward_config)}")
    for rc in reward_config:
        print(f"  - {rc['type']} (weight={rc.get('weight', 1.0)})")

    # Slippage config (optional)
    slippage_config = config.get('trading', {}).get('slippage', {})
    slippage_enabled = slippage_config.get('enabled', True)

    broker = Broker(
        ticker=ticker,
        reward_config=reward_config,
        starting_cash=config['trading']['starting_balance'],
        fee_pct=config['trading']['fee_pct'],
        slippage_enabled=slippage_enabled,
        slippage_config=slippage_config.get('params', None)
    )

    print(f"[Broker] State space size: {broker.num_states}")
    print(f"[Broker] Action space size: {broker.num_actions}")

    return ticker, broker


def setup_agent(config, broker):
    """Setup RL agent"""
    print(f"\n[Agent] Initializing Double DQN agent")

    # Get device from config, or auto-detect
    device = config['model'].get('device', None)
    if device == 'cuda':
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device == 'cpu':
        import torch
        device = torch.device('cpu')
    else:
        device = None  # Let agent auto-detect

    agent = DoubleDQNAgent(
        num_states=broker.num_states,
        num_actions=broker.num_actions,
        hidden_size=config['model']['hidden_size'],
        lstm_layers=config['model']['lstm_layers'],
        learning_rate=config['model']['learning_rate'],
        gamma=config['model']['gamma'],
        tau=config['model']['tau'],
        device=device
    )

    return agent


def setup_memory(config):
    """Setup replay memory"""
    print(f"\n[Memory] Initializing Prioritized Experience Replay")

    memory = Memory(
        capacity=config['memory']['capacity'],
        alpha=config['memory']['alpha'],
        beta=config['memory']['beta'],
        beta_increment=config['memory']['beta_increment'],
        epsilon=config['memory']['epsilon']
    )
    print(f"[Memory] Capacity: {config['memory']['capacity']:,}")

    return memory


def setup_trainer(config, profile, agent, broker, memory, run_id, dashboard_url=None):
    """Setup trainer"""
    print(f"\n[Trainer] Initializing modern trainer")

    # Get episode length from config
    max_episode_steps = config['data'].get('episode_length')

    # Setup dashboard client if URL provided
    dashboard_client = None
    if dashboard_url:
        try:
            from client import Client
            print(f"[Trainer] Initializing dashboard client for {dashboard_url}...")
            dashboard_client = Client(
                dashboard_url=dashboard_url,
                enabled=True,
                run_id=run_id  # Pass run_id to dashboard client
            )
            print(f"[Trainer] ✓ Dashboard client initialized successfully")
            print(f"[Trainer]   - URL: {dashboard_url}")
            print(f"[Trainer]   - Run ID: {run_id}")
            print(f"[Trainer]   - Connected: {dashboard_client.is_connected}")

            # Test emission
            import time
            dashboard_client.emit("training_start", {"message": "Training starting soon..."})
            time.sleep(1)  # Give background thread time to send
            print(f"[Trainer]   - Test event sent")
        except ImportError as e:
            logger.error("could not import Client: %s; install with: pip install requests", e)
        except Exception as e:
            logger.exception("could not initialize dashboard client: %s", e)

    trainer = Trainer(
        agent=agent,
        broker=broker,
        memory=memory,
        batch_size=config['training']['batch_size'],
        max_epsilon=config['training']['max_epsilon'],
        min_epsilon=config['training']['min_epsilon'],
        epsilon_decay=config['training']['epsilon_decay'],
        update_frequency=config['training']['update_frequency'],
        save_dir=config['paths']['run_dir'],  # Use run_dir instead of model_dir
        save_frequency=config['training']['save_frequency'],
        max_episode_steps=max_episode_steps,
        run_id=run_id,
        dashboard_client=dashboard_client
    )

    return trainer


def print_system_info(agent, config, trainer):
    """Print system and GPU information"""
    import torch
    print("\n" + "="*60)
    print("System Information")
    print("="*60)

    # Device info
    print(f"\n[Device] {agent.device}")
    if agent.device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Memory Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"  Total Memory:     {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  TF32 Enabled:     {torch.backends.cuda.matmul.allow_tf32}")

    # Model info
    total_params = sum(p.numel() for p in agent.policy_net.parameters())
    trainable_params = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
    print(f"\n[Model]")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Hidden Size (actual): {agent.policy_net.hidden_size}")
    print(f"  LSTM Layers (actual): {agent.policy_net.lstm_layers}")
    print(f"  Mixed Precision:      {'Enabled' if agent.scaler else 'Disabled'}")

    # Verify network size matches config
    if agent.policy_net.hidden_size != config['model']['hidden_size']:
        print(f"\n  ⚠️  WARNING: Hidden size mismatch!")
        print(f"      Config: {config['model']['hidden_size']}")
        print(f"      Actual: {agent.policy_net.hidden_size}")

    print()


def train(config, config_name, profile, profile_name, resume_from=None, dashboard_url=None):
    """Train the agent"""
    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save provenance (config and profile copies)
    save_provenance(config_name, config, profile_name, profile,
                   config['paths']['config_archive_dir'], run_id)

    # Setup
    ticker, broker = setup_environment(config, profile)
    agent = setup_agent(config, broker)
    memory = setup_memory(config)
    trainer = setup_trainer(config, profile, agent, broker, memory, run_id, dashboard_url=dashboard_url)

    # Print system information and verify config
    print_system_info(agent, config, trainer)

    # Resume from checkpoint if specified
    if resume_from:
        print(f"\n[Resume] Loading checkpoint from {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Train
    print("\n" + "="*60)
    print("Starting training")
    print("="*60)
    print()

    num_episodes = config['training']['num_episodes']
    trainer.train(num_episodes)

    return trainer


def evaluate(config, profile, model_path):
    """Evaluate a trained model"""
    # Generate run ID for eval
    run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Setup
    ticker, broker = setup_environment(config, profile)
    agent = setup_agent(config, broker)
    memory = setup_memory(config)
    trainer = setup_trainer(config, profile, agent, broker, memory, run_id)

    # Load model
    print(f"\n[Evaluate] Loading model from {model_path}")
    trainer.load_checkpoint(model_path)

    # Evaluate
    num_episodes = config['evaluation']['num_episodes']
    eval_stats = trainer.evaluate(num_episodes)

    return eval_stats


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Modern QTrader Training')
    parser.add_argument('--config', type=str,
                        choices=['fast', 'medium', 'full'],
                        default='fast',
                        help='Config preset to use (default: fast)')
    parser.add_argument('--profile', type=str,
                        choices=['scalper', 'day_trader', 'swing_trader', 'balanced', 'risk_averse', 'contrarian', 'trend_follower'],
                        default='balanced',
                        help='Trading profile preset (default: balanced)')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'standby'], default='train',
                        help='Mode: train, eval, or standby (wait for dashboard commands)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--model', type=str, default=None,
                        help='Model path for evaluation')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override number of episodes')
    parser.add_argument('--dashboard', type=str, default=None,
                        help='Dashboard URL (e.g., http://localhost:8765)')

    args = parser.parse_args()

    logger.debug("dashboard URL argument: %s", args.dashboard)

    # Load config (system settings) - now in agent/config/
    config_path = f'config/{args.config}.yaml'
    print(f"\nLoading config: {args.config}")
    config = load_config(config_path)
    config_name = args.config

    # Load profile (trading strategy) - now in agent/profiles/
    profile_path = f'profiles/{args.profile}.yaml'
    print(f"Loading trader profile: {args.profile}")
    profile = load_config(profile_path)
    profile_name = profile['name']

    # Generate run ID first (needed for paths)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate hash-based paths for provenance
    paths = get_output_paths(config_name, config, profile_name, profile, run_id)
    config['paths'] = {
        'base_dir': str(paths['base_dir']),
        'run_dir': str(paths['run_dir']),
        'config_archive_dir': str(paths['config_archive_dir']),
    }

    # Print hash information
    print(f"\n[Provenance] Config hash: {hash_config(config)}")
    print(f"[Provenance] Profile hash: {hash_profile(profile)}")
    print(f"[Provenance] Run ID: {run_id}")
    print(f"[Provenance] Output directory: {paths['run_dir']}")

    # Override episodes if specified
    if args.episodes:
        if args.mode == 'train':
            config['training']['num_episodes'] = args.episodes
        else:
            config['evaluation']['num_episodes'] = args.episodes

    # Print header
    print("\n" + "="*60)
    print("MODERN QTRADER - Deep Reinforcement Learning Trading Agent")
    print("="*60)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Config: {args.config}")

    if args.mode == 'standby':
        # Standby mode - connect to dashboard and wait for commands
        if not args.dashboard:
            logger.error("--dashboard URL required for standby mode")
            return

        print("\n" + "="*60)
        print("STANDBY MODE")
        print("="*60)
        print(f"\nConnecting to dashboard: {args.dashboard}")
        print("Waiting for training commands...")
        print()

        from standby import run_standby
        run_standby(args.dashboard)
        return

    elif args.mode == 'train':
        trainer = train(config, config_name, profile, profile_name, resume_from=args.resume, dashboard_url=args.dashboard)

        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"\nRun directory: {config['paths']['run_dir']}/")
        print(f"  best_model.pt")
        print(f"  run_summary.txt")
        print(f"  run_log.jsonl")
        print()

    elif args.mode == 'eval':
        if not args.model:
            # Use best model by default
            args.model = str(Path(config['paths']['run_dir']) / 'best_model.pt')

        if not Path(args.model).exists():
            logger.error("model not found at %s", args.model)
            return

        eval_stats = evaluate(config, profile, args.model)


if __name__ == "__main__":
    main()
