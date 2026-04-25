import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler


class Indicators:
    """
    Technical indicators and feature engineering for trading.

    Improvements over original:
    - More comprehensive indicator suite (MACD, Bollinger, Stochastic, etc.)
    - Proper normalization and scaling
    - Multi-timeframe analysis
    - Volume profile and order flow
    - Market regime detection
    """

    def __init__(self, normalize=True):
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.fitted = False

    def calculate_all_indicators(self, df, position=None, balance=None):
        """
        Calculate comprehensive set of technical indicators.

        Args:
            df: DataFrame with OHLCV data
            position: Current position (optional)
            balance: Current balance (optional)

        Returns:
            Feature vector as numpy array
        """
        if df.shape[0] < 50:
            # Need enough data for indicators
            return None

        features = {}

        # Position features
        if position:
            # Handle both dict and Position object
            if isinstance(position, dict):
                features['pos_type'] = 1 if position.get('buy_sell') == 'buy' else -1 if position.get('buy_sell') == 'sell' else 0
                features['pos_profit_pct'] = position.get('unrealized_pnl_pct', 0)
                crypto_value = position.get('quantity', 0) * df['Close'].iloc[-1]
                features['pos_size'] = crypto_value / balance if balance else 0
                # Cash ratio = 1 - position_size (how much of portfolio is cash)
                features['cash_ratio'] = 1.0 - features['pos_size']
            else:
                features['pos_type'] = 1 if position.buy_sell == 'buy' else -1 if position.buy_sell == 'sell' else 0
                features['pos_profit_pct'] = position.current_profit_pct(df['Close'].iloc[-1])
                crypto_value = position.quantity * df['Close'].iloc[-1]
                features['pos_size'] = crypto_value / balance if balance else 0
                features['cash_ratio'] = 1.0 - features['pos_size']
        else:
            features['pos_type'] = 0
            features['pos_profit_pct'] = 0
            features['pos_size'] = 0
            features['cash_ratio'] = 1.0  # 100% cash when no position

        # Price action features
        features.update(self._price_action_features(df))

        # Momentum indicators
        features.update(self._momentum_indicators(df))

        # Volatility indicators
        features.update(self._volatility_indicators(df))

        # Volume indicators
        features.update(self._volume_indicators(df))

        # Trend indicators
        features.update(self._trend_indicators(df))

        # Market structure
        features.update(self._market_structure(df))

        # Convert to array
        feature_vector = np.array(list(features.values()), dtype=np.float32)

        # Handle NaN values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize if enabled
        if self.normalize:
            if not self.fitted:
                feature_vector = feature_vector.reshape(1, -1)
                self.scaler.fit(feature_vector)
                self.fitted = True
                feature_vector = self.scaler.transform(feature_vector).flatten()
            else:
                feature_vector = feature_vector.reshape(1, -1)
                feature_vector = self.scaler.transform(feature_vector).flatten()

        return feature_vector

    def _price_action_features(self, df):
        """Price action and candlestick patterns"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_price = df['Open']

        # Returns at different windows (use pre-computed if available)
        if 'return_1' in df.columns:
            features = {
                'return_1': df['return_1'].iloc[-1],
                'return_5': df['return_5'].iloc[-1],
                'return_10': df['return_10'].iloc[-1],
                'return_20': df['return_20'].iloc[-1],
            }
        else:
            features = {
                'return_1': close.pct_change(1).iloc[-1],
                'return_5': close.pct_change(5).iloc[-1],
                'return_10': close.pct_change(10).iloc[-1],
                'return_20': close.pct_change(20).iloc[-1],
            }

        # Price position in range
        high_20 = high.rolling(20).max().iloc[-1]
        low_20 = low.rolling(20).min().iloc[-1]
        features['price_position'] = (close.iloc[-1] - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5

        # Candlestick patterns
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low

        features['body_size'] = (body / close).iloc[-1]
        features['upper_shadow'] = (upper_shadow / close).iloc[-1]
        features['lower_shadow'] = (lower_shadow / close).iloc[-1]

        return features

    def _momentum_indicators(self, df):
        """Momentum-based indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume_(Currency)']

        features = {}

        # RSI
        rsi = ta.momentum.rsi(close, window=14)
        features['rsi'] = (rsi.iloc[-1] - 50) / 50  # Normalize to [-1, 1]

        # Stochastic Oscillator
        stoch = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        features['stoch'] = (stoch.iloc[-1] - 50) / 50

        # MACD
        macd_indicator = ta.trend.MACD(close)
        macd = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_diff = macd_indicator.macd_diff()

        features['macd'] = macd.iloc[-1] / close.iloc[-1]
        features['macd_signal'] = macd_signal.iloc[-1] / close.iloc[-1]
        features['macd_diff'] = macd_diff.iloc[-1] / close.iloc[-1]

        # Awesome Oscillator
        ao = ta.momentum.awesome_oscillator(high, low)
        features['ao'] = ao.iloc[-1] * 0.01  # Scale down

        # True Strength Index
        tsi = ta.momentum.tsi(close)
        features['tsi'] = tsi.iloc[-1] * 0.01

        # Williams %R
        williams = ta.momentum.williams_r(high, low, close)
        features['williams_r'] = (williams.iloc[-1] + 50) / 50  # Normalize

        # ROC (Rate of Change)
        roc = ta.momentum.roc(close, window=10)
        features['roc'] = roc.iloc[-1] * 0.01

        # Money Flow Index
        mfi = ta.volume.money_flow_index(high, low, close, volume, window=14)
        features['mfi'] = (mfi.iloc[-1] - 50) / 50

        return features

    def _volatility_indicators(self, df):
        """Volatility and risk indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        features = {}

        # Average True Range
        atr = ta.volatility.average_true_range(high, low, close, window=14)
        features['atr'] = (atr.iloc[-1] / close.iloc[-1]) * 100  # Percentage

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        features['bb_high'] = (bb.bollinger_hband().iloc[-1] - close.iloc[-1]) / close.iloc[-1]
        features['bb_low'] = (close.iloc[-1] - bb.bollinger_lband().iloc[-1]) / close.iloc[-1]
        features['bb_width'] = bb.bollinger_wband().iloc[-1]
        features['bb_position'] = bb.bollinger_pband().iloc[-1]  # Position within bands

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(high, low, close, window=20)
        features['kc_position'] = (close.iloc[-1] - kc.keltner_channel_mband().iloc[-1]) / close.iloc[-1]

        # Donchian Channel
        dc = ta.volatility.DonchianChannel(high, low, close, window=20)
        features['dc_position'] = (close.iloc[-1] - dc.donchian_channel_mband().iloc[-1]) / close.iloc[-1]

        # Ulcer Index (downside volatility)
        ulcer = ta.volatility.ulcer_index(close, window=14)
        features['ulcer'] = ulcer.iloc[-1]

        # Historical volatility
        returns = close.pct_change()
        features['volatility_10'] = returns.rolling(10).std().iloc[-1]
        features['volatility_30'] = returns.rolling(30).std().iloc[-1]

        return features

    def _volume_indicators(self, df):
        """Volume-based indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume_(Currency)']

        features = {}

        # On-Balance Volume
        obv = ta.volume.on_balance_volume(close, volume)
        obv_change = obv.pct_change(5).iloc[-1]
        features['obv_change'] = obv_change if not np.isnan(obv_change) else 0

        # Chaikin Money Flow
        cmf = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
        features['cmf'] = cmf.iloc[-1]

        # Volume Price Trend
        vpt = ta.volume.volume_price_trend(close, volume)
        vpt_change = vpt.pct_change(5).iloc[-1]
        features['vpt_change'] = vpt_change if not np.isnan(vpt_change) else 0

        # Ease of Movement
        eom = ta.volume.ease_of_movement(high, low, volume, window=14)
        features['eom'] = eom.iloc[-1]

        # Force Index
        force = ta.volume.force_index(close, volume, window=13)
        features['force_index'] = force.iloc[-1] / 1e6  # Scale down

        # Volume ratio
        volume_sma = volume.rolling(20).mean()
        features['volume_ratio'] = volume.iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1

        # Accumulation/Distribution Index
        adi = ta.volume.acc_dist_index(high, low, close, volume)
        adi_change = adi.pct_change(5).iloc[-1]
        features['adi_change'] = adi_change if not np.isnan(adi_change) else 0

        return features

    def _trend_indicators(self, df):
        """Trend-following indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        features = {}

        # Moving averages
        sma_10 = ta.trend.sma_indicator(close, window=10)
        sma_20 = ta.trend.sma_indicator(close, window=20)
        sma_50 = ta.trend.sma_indicator(close, window=50) if len(close) >= 50 else close

        features['price_to_sma10'] = (close.iloc[-1] - sma_10.iloc[-1]) / sma_10.iloc[-1]
        features['price_to_sma20'] = (close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
        features['price_to_sma50'] = (close.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]

        # EMA
        ema_12 = ta.trend.ema_indicator(close, window=12)
        ema_26 = ta.trend.ema_indicator(close, window=26)
        features['ema_diff'] = (ema_12.iloc[-1] - ema_26.iloc[-1]) / ema_26.iloc[-1]

        # ADX (trend strength)
        adx = ta.trend.adx(high, low, close, window=14)
        features['adx'] = adx.iloc[-1] / 50  # Normalize

        # Positive/Negative Directional Indicators
        pdi = ta.trend.adx_pos(high, low, close, window=14)
        ndi = ta.trend.adx_neg(high, low, close, window=14)
        features['pdi'] = pdi.iloc[-1] / 50
        features['ndi'] = ndi.iloc[-1] / 50

        # Aroon
        aroon = ta.trend.AroonIndicator(high=high, low=low, window=25)
        features['aroon_up'] = aroon.aroon_up().iloc[-1] / 100
        features['aroon_down'] = aroon.aroon_down().iloc[-1] / 100
        features['aroon_diff'] = aroon.aroon_indicator().iloc[-1] / 100

        # CCI (Commodity Channel Index)
        cci = ta.trend.cci(high, low, close, window=20)
        features['cci'] = cci.iloc[-1] / 100  # Normalize

        # DPO (Detrended Price Oscillator)
        dpo = ta.trend.dpo(close, window=20)
        features['dpo'] = dpo.iloc[-1] / close.iloc[-1]

        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(high, low)
        features['ichimoku_conv'] = (ichimoku.ichimoku_conversion_line().iloc[-1] - close.iloc[-1]) / close.iloc[-1]
        features['ichimoku_base'] = (ichimoku.ichimoku_base_line().iloc[-1] - close.iloc[-1]) / close.iloc[-1]

        return features

    def _market_structure(self, df):
        """Market structure and regime features"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        features = {}

        # Higher highs / Lower lows
        hh = (high.iloc[-1] > high.rolling(20).max().iloc[-2])
        ll = (low.iloc[-1] < low.rolling(20).min().iloc[-2])
        features['higher_high'] = 1.0 if hh else 0.0
        features['lower_low'] = 1.0 if ll else 0.0

        # Support/Resistance proximity
        support = low.rolling(20).min().iloc[-1]
        resistance = high.rolling(20).max().iloc[-1]
        features['dist_to_support'] = (close.iloc[-1] - support) / close.iloc[-1]
        features['dist_to_resistance'] = (resistance - close.iloc[-1]) / close.iloc[-1]

        # Trend consistency (consecutive ups/downs)
        changes = close.diff().iloc[-10:]
        features['trend_consistency'] = (changes > 0).sum() / len(changes)

        # Pivot points
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        features['pivot_position'] = (close.iloc[-1] - pivot) / close.iloc[-1]

        return features

    def get_feature_names(self):
        """Return list of all feature names in order"""
        # This should match the order in calculate_all_indicators
        names = [
            'pos_type', 'pos_profit_pct', 'pos_size',
            # Price action
            'return_1', 'return_5', 'return_10', 'return_20', 'price_position',
            'body_size', 'upper_shadow', 'lower_shadow',
            # Momentum
            'rsi', 'stoch', 'macd', 'macd_signal', 'macd_diff', 'ao', 'tsi',
            'williams_r', 'roc', 'mfi',
            # Volatility
            'atr', 'bb_high', 'bb_low', 'bb_width', 'bb_position',
            'kc_position', 'dc_position', 'ulcer', 'volatility_10', 'volatility_30',
            # Volume
            'obv_change', 'cmf', 'vpt_change', 'eom', 'force_index',
            'volume_ratio', 'adi_change',
            # Trend
            'price_to_sma10', 'price_to_sma20', 'price_to_sma50', 'ema_diff',
            'adx', 'pdi', 'ndi', 'aroon_up', 'aroon_down', 'aroon_diff',
            'cci', 'dpo', 'ichimoku_conv', 'ichimoku_base',
            # Market structure
            'higher_high', 'lower_low', 'dist_to_support', 'dist_to_resistance',
            'trend_consistency', 'pivot_position'
        ]
        return names

    def get_num_features(self):
        """Get total number of features"""
        return len(self.get_feature_names())
