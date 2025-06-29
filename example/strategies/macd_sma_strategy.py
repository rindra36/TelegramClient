"""
MACD-SMA Strategy Implementation

This module implements a hybrid strategy combining MACD and SMA(14)
for binary options trading with 5-second intervals.
"""

from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from .base_strategy import BaseStrategy

# Mute pd warnings
pd.set_option('future.no_silent_downcasting', True)

class MACDSMAStrategy(BaseStrategy):
    """MACD-SMA hybrid trading strategy implementation."""

    def __init__(self) -> None:
        """Initialize MACD-SMA strategy with default parameters."""
        super().__init__(
            name="MACD-SMA Strategy",
            description="Hybrid strategy using MACD crossovers confirmed by SMA(14) with 5-second intervals"
        )
        
        # Set strategy-specific timeframe
        self.strategy_timeframe = {
            'candles': 5,  # For 5-second intervals
            'resample': '5s',  # For resampling to 5-second candles
            'min_points': 5,  # Minimum points for resampling
            'candles_history': 36000, # Amount of historical candle data needed
            'dropna': 'all' # Configuration for dropping NaN values
        }

        # Technical parameters
        self.parameters = {
            'window_slow': 24,    # Slow moving average window for MACD
            'window_fast': 14,     # Fast moving average window for MACD
            'window_sign': 8,     # Signal line window for MACD
            'sma_period': 14,     # SMA period
            'ema50_period': 50,   # EMA 50 period
            'ema100_period': 100, # EMA 100 period
            'ema200_period': 200, # EMA 200 period
            'bb_window': 20,         # Window size for moving average for bands
            'bb_window_dev': 2,      # Number of standard deviations for bands
            'atr_period': 14,          # ATR window for volatility
            'adx_window': 14,        # ADX window for trend confirmation
        }
        
        # Trade parameters
        self.trade_parameters = {
            'expiration': 30,  # 30-second expiry
            'amount': 1.0,     # Conservative $1 base amount
        }

    # def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Generate trading signals based on MACD and SMA(14).
        
    #     Args:
    #         df (pd.DataFrame): Candlestick data with OHLC prices
        
    #     Returns:
    #         pd.DataFrame: Enhanced dataframe with trading signals
    #     """
    #     # Heikin Ashi candlesticks and Japanese candlesitcks
    #     ha_close_prices = df['ha_close'].astype(float)
    #     close_prices = df['close'].astype(float)

    #     # Calculate MACD
    #     macd = MACD(
    #         ha_close_prices, 
    #         window_slow=self.parameters['window_slow'],
    #         window_fast=self.parameters['window_fast'],
    #         window_sign=self.parameters['window_sign']
    #     )
    #     df['macd_line'] = macd.macd()
    #     df['signal_line'] = macd.macd_signal()
    #     df['macd_diff'] = macd.macd_diff()

    #     # Detect crossovers
    #     df['macd_above'] = df['macd_line'] > df['signal_line']
    #     df['prev_macd_above'] = df['macd_above'].shift(1)

    #     df['cross_up'] = (df['macd_above'] & ~df['prev_macd_above']
    #         .fillna(False)
    #         .infer_objects()
    #         .astype(bool))
    #     df['cross_down'] = (~df['macd_above'] & df['prev_macd_above'])

    #     # MACD Histogram Widening (Momentum confirmation)
    #     df['macd_diff_prev'] = df['macd_diff'].shift(1)
    #     df["histogram_widening_call"] = df["macd_diff"] > df["macd_diff_prev"]  # Histogram increasing
    #     df["histogram_widening_put"] = df["macd_diff"] < df["macd_diff_prev"]  # Histogram decreasing

    #     # Check next candle's direction
    #     df['next_close'] = df['ha_close'].shift(-1)
    #     df['next_open'] = df['ha_open'].shift(-1)
    #     df['next_bullish'] = df['next_close'] > df['next_open']
    #     df['next_bearish'] = df['next_close'] < df['next_open']
        
    #     # Calculate SMA(14)
    #     sma = SMAIndicator(ha_close_prices, window=self.parameters['sma_period'])
    #     df['sma14'] = sma.sma_indicator()

    #     # Detect closing upper and lower than SMA(14)
    #     df['close_above_sma'] = df['ha_close'] > df['sma14']
    #     df['close_below_sma'] = df['ha_close'] < df['sma14']

    #     # Calculate EMA(100)
    #     ema100 = EMAIndicator(ha_close_prices, window=self.parameters['ema100_period'])
    #     df['ema100'] = ema100.ema_indicator()

    #     # Detect closing upper and lower than EMA(100)
    #     df['close_above_ema100'] = df['ha_close'] > df['ema100']
    #     df['close_below_ema100'] = df['ha_close'] < df['ema100']

    #     # Calculate EMA(50)
    #     ema50 = EMAIndicator(ha_close_prices, window=self.parameters['ema50_period'])
    #     df['ema50'] = ema50.ema_indicator()

    #     # Detect closing upper and lower than EMA(50)
    #     df['close_above_ema50'] = df['ha_close'] > df['ema50']
    #     df['close_below_ema50'] = df['ha_close'] < df['ema50']

    #     # Calculate ADX
    #     adx = ADXIndicator(
    #         high=df['ha_high'],
    #         low=df['ha_low'],
    #         close=df['ha_close'],
    #         window=self.parameters['adx_window']
    #     )
    #     df['adx'] = adx.adx()
    #     df['adx_positive'] = adx.adx_pos()
    #     df['adx_negative'] = adx.adx_neg()
    #     df['adx_trend'] = df['adx'] > 25
    #     df['adx_positive_trend'] = df['adx_positive'] > df['adx_negative']
    #     df['adx_negative_trend'] = df['adx_negative'] > df['adx_positive']

    #     # Calculate ATR
    #     atr = AverageTrueRange(
    #         high=df['ha_high'],
    #         low=df['ha_low'],
    #         close=df['ha_close'],
    #         window=self.parameters['atr_period']
    #     )
    #     df['atr'] = atr.average_true_range()
    #     df['atr_ma20'] = df['atr'].rolling(20).mean()
    #     df['atr_condition'] = df['atr'] <= 1.5 * df['atr_ma20']
        
    #     # This will be implemented based on your specific rules
    #     df['call_signal'] = (
    #         df['cross_up']
    #         & df['close_above_sma']
    #         & df['close_above_ema100']
    #         & df['next_bullish']
    #         & ~df['next_close'].isna()
    #         & ~df['next_open'].isna()
    #         & df['adx_trend']
    #         & df['adx_positive_trend']
    #         & df['atr_condition']
    #         & df['histogram_widening_call']
    #     )
    #     df['put_signal'] = (
    #         df['cross_down']
    #         & df['close_below_sma']
    #         & df['close_below_ema100']
    #         & df['next_bearish']
    #         & ~df['next_close'].isna()
    #         & ~df['next_open'].isna()
    #         & df['adx_trend']
    #         & df['adx_negative_trend']
    #         & df['atr_condition']
    #         & df['histogram_widening_put']
    #     )

    #     # # Add market condition analysis
    #     # # Volatility Check using Bollinger Bands
    #     # bb = BollingerBands(close_prices, window=self.parameters['bb_window'], window_dev=self.parameters['bb_window_dev'])
    #     # df['bb_width'] = bb.bollinger_wband()
    #     # df['volatility_suitable'] = (df['bb_width'] > 0.001) & (df['bb_width'] < 0.005)  # Adjust thresholds

    #     # # Trend Strength
    #     # df['strong_trend'] = (df['adx'] > 30) & (abs(df['macd_line']) > abs(df['macd_line'].rolling(20).mean()))

    #     # # Market Phase Detection
    #     # df['price_std'] = df['ha_close'].rolling(20).std()
    #     # df['is_ranging'] = df['price_std'] < df['price_std'].rolling(100).mean()
        
    #     # # Modify signal conditions
    #     # df['call_signal'] = (
    #     #     df['call_signal']  # Previous conditions
    #     #     & df['volatility_suitable']
    #     #     & df['strong_trend']
    #     #     & ~df['is_ranging']  # Avoid ranging markets
    #     # )
    #     # df['put_signal'] = (
    #     #     df['put_signal']   # Previous conditions
    #     #     & df['volatility_suitable']
    #     #     & df['strong_trend']
    #     #     & ~df['is_ranging']  # Avoid ranging markets
    #     # )

    #     # Calculate support and resistance levels
    #     df = self.calculate_pivot_points(df)

    #     # # Check price action against support/resistance
    #     # df['near_resistance'] = (
    #     #     (np.abs(df['ha_close'] - df['resistance1']) < df['atr']) |
    #     #     (np.abs(df['ha_close'] - df['resistance2']) < df['atr'])
    #     # )
    #     # df['near_support'] = (
    #     #     (np.abs(df['ha_close'] - df['support1']) < df['atr']) |
    #     #     (np.abs(df['ha_close'] - df['support2']) < df['atr'])
    #     # )

    #     # Detect market regime
    #     df = self.detect_market_regime(df)

    #     # Modify signal conditions using enhanced regime detection
    #     # df['call_signal'] = (
    #     #     df['call_signal']  # Previous conditions
    #     #     & (df['regime'] != 'ranging')  # Not in ranging market
    #     #     & (df['trend_strength'] > 50)  # Strong enough trend
    #     #     & (
    #     #         ~df['near_resistance']  # Not too close to resistance
    #     #         | (df['in_resistance_zone'] & (df['trend_strength'] > 70))  # Or in resistance zone with very strong trend
    #     #     )
    #     #     & (df['pivot_strength'] > 0.5)  # Reduced pivot strength requirement
    #     # )
    #     df['call_signal'] = (
    #         df['call_signal']  # Previous conditions
    #         & (df['regime'] == 'trending')  # In trending market
    #         & (df['trend_strength'] > 40)  # Moderately strong trend
    #         & ~(df['near_resistance'] & (df['trend_strength'] < 60))  # Avoid resistance unless very strong trend
    #     )
        
    #     # df['put_signal'] = (
    #     #     df['put_signal']   # Previous conditions
    #     #     & (df['regime'] != 'ranging')  # Not in ranging market
    #     #     & (df['trend_strength'] > 50)  # Strong enough trend
    #     #     & (
    #     #         ~df['near_support']  # Not too close to support
    #     #         | (df['in_support_zone'] & (df['trend_strength'] > 70))  # Or in support zone with very strong trend
    #     #     )
    #     #     & (df['pivot_strength'] > 0.5)  # Reduced pivot strength requirement
    #     # )
    #     df['put_signal'] = (
    #         df['put_signal']   # Previous conditions
    #         & (df['regime'] == 'trending')  # In trending market
    #         & (df['trend_strength'] > 40)  # Moderately strong trend
    #         & ~(df['near_support'] & (df['trend_strength'] < 60))  # Avoid support unless very strong trend
    #     )

    #     # # Add new analysis
    #     # df = self.calculate_fibonacci_levels(df)
    #     # df = self.analyze_price_action(df)
    #     # df = self.enhanced_market_regime(df)
        
    #     # # Additional signal conditions
    #     # df['call_signal'] = (
    #     #     df['call_signal']  # Previous conditions
    #     #     & ~df['near_fib_level']  # Not near Fibonacci levels
    #     #     # & (df['strong_bullish'] | df['trend_strength'] > 1.5)  # Strong bullish PA or trend
    #     #     & (df['dm_positive_smooth'] > df['dm_negative_smooth'])  # Confirmed uptrend
    #     # )
        
    #     # df['put_signal'] = (
    #     #     df['put_signal']  # Previous conditions
    #     #     & ~df['near_fib_level']  # Not near Fibonacci levels
    #     #     # & (df['strong_bearish'] | df['trend_strength'] < -1.5)  # Strong bearish PA or trend
    #     #     & (df['dm_negative_smooth'] > df['dm_positive_smooth'])  # Confirmed downtrend
    #     # )
        
    #     return df

    def generate_signals_old(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MACD and SMA(14).
        """
        # Create or open debug log file
        with open('strategy_debug.log', 'a') as f:
            try:
                # Initialize signal columns with False
                df['call_signal'] = False
                df['put_signal'] = False

                # Calculate all indicators first
                ha_close_prices = df['ha_close'].astype(float)
                close_prices = df['close'].astype(float)

                # Calculate MACD
                macd = MACD(
                    ha_close_prices, 
                    window_slow=self.parameters['window_slow'],
                    window_fast=self.parameters['window_fast'],
                    window_sign=self.parameters['window_sign']
                )
                df['macd_line'] = macd.macd()
                df['signal_line'] = macd.macd_signal()
                df['macd_diff'] = macd.macd_diff()

                # Calculate crossovers (with explicit boolean conversion)
                df['macd_above'] = (df['macd_line'] > df['signal_line']).astype(bool)
                df['prev_macd_above'] = df['macd_above'].shift(1).fillna(False).astype(bool)
                df['cross_up'] = (df['macd_above'] & (~df['prev_macd_above'])).astype(bool)
                df['cross_down'] = ((~df['macd_above']) & df['prev_macd_above']).astype(bool)

                # Calculate MACD Histogram
                df['macd_diff_prev'] = df['macd_diff'].shift(1)
                df["histogram_widening_call"] = (df["macd_diff"] > df["macd_diff_prev"]).fillna(False).astype(bool)
                df["histogram_widening_put"] = (df["macd_diff"] < df["macd_diff_prev"]).fillna(False).astype(bool)

                # Calculate other indicators
                df['next_close'] = df['ha_close'].shift(-1)
                df['next_open'] = df['ha_open'].shift(-1)
                df['next_bullish'] = (df['next_close'] > df['next_open']).fillna(False).astype(bool)
                df['next_bearish'] = (df['next_close'] < df['next_open']).fillna(False).astype(bool)

                # Calculate SMA
                sma = SMAIndicator(ha_close_prices, window=self.parameters['sma_period'])
                df['sma14'] = sma.sma_indicator()
                df['close_above_sma'] = (df['ha_close'] > df['sma14']).fillna(False).astype(bool)
                df['close_below_sma'] = (df['ha_close'] < df['sma14']).fillna(False).astype(bool)
                df['prev_close_above_sma'] = df['close_above_sma'].shift(1).fillna(False).astype(bool)
                df['cross_up_sma'] = (df['close_above_sma'] & ~df['prev_close_above_sma']).fillna(False).astype(bool)
                df['cross_down_sma'] = (~df['close_above_sma'] & df['prev_close_above_sma']).fillna(False).astype(bool)

                # Calculate EMA
                ema100 = EMAIndicator(ha_close_prices, window=self.parameters['ema100_period'])
                df['ema100'] = ema100.ema_indicator()
                df['close_above_ema100'] = (df['ha_close'] > df['ema100']).fillna(False).astype(bool)
                df['close_below_ema100'] = (df['ha_close'] < df['ema100']).fillna(False).astype(bool)

                # Calculate ADX
                adx = ADXIndicator(
                    high=df['ha_high'],
                    low=df['ha_low'],
                    close=df['ha_close'],
                    window=self.parameters['adx_window']
                )
                df['adx'] = adx.adx()
                df['adx_positive'] = adx.adx_pos()
                df['adx_negative'] = adx.adx_neg()
                df['adx_trend'] = (df['adx'] > 25).fillna(False).astype(bool)
                df['adx_positive_trend'] = (df['adx_positive'] > df['adx_negative']).fillna(False).astype(bool)
                df['adx_negative_trend'] = (df['adx_negative'] > df['adx_positive']).fillna(False).astype(bool)

                # Calculate ATR
                atr = AverageTrueRange(
                    high=df['ha_high'],
                    low=df['ha_low'],
                    close=df['ha_close'],
                    window=self.parameters['atr_period']
                )
                df['atr'] = atr.average_true_range()
                df['atr_ma20'] = df['atr'].rolling(20).mean()
                df['atr_condition'] = (df['atr'] <= 1.5 * df['atr_ma20']).fillna(False).astype(bool)

                # Basic conditions check with explicit boolean conversion
                df['basic_call_conditions'] = (
                    df['cross_up']
                    & df['close_above_sma']
                    & df['cross_up_sma']
                    & df['close_above_ema100']
                    # & df['next_bullish']
                    # & ~df['next_close'].isna()
                    # & ~df['next_open'].isna()
                    # & df['adx_trend']
                    # & df['adx_positive_trend']
                    # & df['atr_condition']
                    # & df['histogram_widening_call']
                    # Add momentum confirmation
                    # & (df['macd_line'] > df['macd_line'].shift(1))
                    # & (df['ha_close'] > df['ha_close'].shift(1))
                ).astype(bool)

                df['basic_put_conditions'] = (
                    df['cross_down']
                    & df['close_below_sma']
                    & df['cross_down_sma']
                    & df['close_below_ema100']
                    # & df['next_bearish']
                    # & ~df['next_close'].isna()
                    # & ~df['next_open'].isna()
                    # & df['adx_trend']
                    # & df['adx_negative_trend']
                    # & df['atr_condition']
                    # & df['histogram_widening_put']
                    # Add momentum confirmation
                    # & (df['macd_line'] < df['macd_line'].shift(1))
                    # & (df['ha_close'] < df['ha_close'].shift(1))
                ).astype(bool)

                # Calculate pivot points and support/resistance
                df = self.calculate_pivot_points(df)

                # Add volatility filter
                df['volatility_suitable'] = (
                    (df['atr'] > df['atr'].rolling(100).mean() * 0.5) &  # Not too low volatility
                    (df['atr'] < df['atr'].rolling(100).mean() * 2.0)    # Not too high volatility
                ).astype(bool)

                # Add trend consistency check
                df['trend_consistent'] = (
                    (df['ha_close'].rolling(5).mean() > df['ha_close'].rolling(20).mean()) ==
                    (df['ha_close'].rolling(20).mean() > df['ha_close'].rolling(50).mean())
                ).astype(bool)

                # Final signal generation
                df['trend_strength'] = df['adx']
                df['regime'] = np.where(df['adx'] > 20, 'trending', 'ranging')

                df['call_signal'] = (
                    df['basic_call_conditions']
                    # & (df['regime'] == 'trending')
                    # & (~df['near_resistance'] | (df['adx'] >= 30))
                    # & df['volatility_suitable']
                    # & df['trend_consistent']
                    # Add time-based momentum
                    # & (df['ha_close'] > df['ha_close'].shift(3))
                ).astype(bool)

                df['put_signal'] = (
                    df['basic_put_conditions']
                    # & (df['regime'] == 'trending')
                    # & (~df['near_support'] | (df['adx'] >= 30))
                    # & df['volatility_suitable']
                    # & df['trend_consistent']
                    # Add time-based momentum
                    # & (df['ha_close'] < df['ha_close'].shift(3))
                ).astype(bool)

                # Add new analysis
                df = self.calculate_fibonacci_levels(df)
                df = self.analyze_price_action(df)
                df = self.enhanced_market_regime(df)
                
                # Additional signal conditions
                df['call_signal'] = (
                    df['call_signal']  # Previous conditions
                    # & ~df['near_fib_level']  # Not near Fibonacci levels
                    # & (df['strong_bullish'] | d9f['trend_strength'] > 1.5)  # Strong bullish PA or trend
                    # & (df['dm_positive_smooth'] > df['dm_negative_smooth'])  # Confirmed uptrend
                )
                
                df['put_signal'] = (
                    df['put_signal']  # Previous conditions
                    # & ~df['near_fib_level']  # Not near Fibonacci levels
                    # & (df['strong_bearish'] | df['trend_strength'] < -1.5)  # Strong bearish PA or trend
                    # & (df['dm_negative_smooth'] > df['dm_positive_smooth'])  # Confirmed downtrend
                )

                # Write signal summary
                f.write("\nSignal Summary:\n")
                total_basic_calls = int(df['basic_call_conditions'].sum())
                total_basic_puts = int(df['basic_put_conditions'].sum())
                final_calls = int(df['call_signal'].sum())
                final_puts = int(df['put_signal'].sum())
                
                f.write(f"Total Basic Call Signals: {total_basic_calls}\n")
                f.write(f"Total Basic Put Signals: {total_basic_puts}\n")
                f.write(f"Final Call Signals: {final_calls}\n")
                f.write(f"Final Put Signals: {final_puts}\n")

            except Exception as e:
                f.write(f"\nError in signal generation: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
                # Initialize signals as False in case of error
                df['call_signal'] = False
                df['put_signal'] = False

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MACD and SMA(14).
        """
        # Create or open debug log file
        with open('strategy_debug.log', 'a') as f:
            try:
                # Initialize signal columns with False
                df['call_signal'] = False
                df['put_signal'] = False

                # Calculate all indicators first
                ha_close_prices = df['ha_close'].astype(float)
                close_prices = df['close'].astype(float)

                # Calculate MACD
                macd = MACD(
                    ha_close_prices, 
                    window_slow=self.parameters['window_slow'],
                    window_fast=self.parameters['window_fast'],
                    window_sign=self.parameters['window_sign']
                )
                df['macd_line'] = macd.macd()
                df['signal_line'] = macd.macd_signal()
                df['macd_diff'] = macd.macd_diff()

                # Calculate crossovers (with explicit boolean conversion)
                df['macd_above'] = (df['macd_line'] > df['signal_line']).astype(bool)
                df['prev_macd_above'] = df['macd_above'].shift(1).fillna(False).astype(bool)
                df['cross_up'] = (df['macd_above'] & (~df['prev_macd_above'])).astype(bool)
                df['cross_down'] = ((~df['macd_above']) & df['prev_macd_above']).astype(bool)

                # Calculate SMA
                sma = SMAIndicator(ha_close_prices, window=self.parameters['sma_period'])
                df['sma14'] = sma.sma_indicator()
                df['close_above_sma'] = (df['ha_close'] > df['sma14']).fillna(False).astype(bool)
                df['close_below_sma'] = (df['ha_close'] < df['sma14']).fillna(False).astype(bool)
                df['prev_close_above_sma'] = df['close_above_sma'].shift(1).fillna(False).astype(bool)
                df['cross_up_sma'] = (df['close_above_sma'] & ~df['prev_close_above_sma']).fillna(False).astype(bool)
                df['cross_down_sma'] = (~df['close_above_sma'] & df['prev_close_above_sma']).fillna(False).astype(bool)

                # Calculate EMA
                ema100 = EMAIndicator(ha_close_prices, window=self.parameters['ema100_period'])
                df['ema100'] = ema100.ema_indicator()
                df['close_above_ema100'] = (df['ha_close'] > df['ema100']).fillna(False).astype(bool)
                df['close_below_ema100'] = (df['ha_close'] < df['ema100']).fillna(False).astype(bool)

                # Check if candle is bullish or bearish
                df['bullish_candle'] = (df['ha_close'] > df['ha_open']).fillna(False).astype(bool)
                df['bearish_candle'] = (df['ha_close'] < df['ha_open']).fillna(False).astype(bool)

                # Calculate candle sizes and identify big candles
                df['candle_size'] = abs(df['close'] - df['open'])
                # Calculate average candle size over the last 10 candles
                df['avg_candle_size'] = df['candle_size'].rolling(window=10).mean()
                # Define big candle as 1.5 times the average size
                df['is_big_candle'] = df['candle_size'] > (df['avg_candle_size'] * 1.5)

                # Basic conditions check with explicit boolean conversion
                df['call_signal'] = (
                    df['cross_up']
                    & df['close_above_sma']
                    & df['cross_up_sma']
                    & df['close_above_ema100']
                    & df['is_big_candle']
                    & df['bullish_candle']
                ).astype(bool)

                df['put_signal'] = (
                    df['cross_down']
                    & df['close_below_sma']
                    & df['cross_down_sma']
                    & df['close_below_ema100']
                    & df['is_big_candle']
                    & df['bearish_candle']
                ).astype(bool)

            except Exception as e:
                f.write(f"\nError in signal generation: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
                # Initialize signals as False in case of error
                df['call_signal'] = False
                df['put_signal'] = False

        return df

    def backtest(self, df: pd.DataFrame, symbol: str, start_time=None, end_time=None) -> Tuple[float, int, int, int]:
        """
        Performs backtesting of the MACD-SMA strategy on historical data.
        
        Args:
            df (pd.DataFrame): Historical price data with signals
            symbol (str): Trading symbol being tested
        
        Returns:
            tuple: (win_rate, total_trades, wins, losses)
        """
        print(f'Backtesting MACD-SMA strategy with: {symbol}')
        # Filter signals based on time range if provided
        if start_time is not None and end_time is not None:
            df = df[
                (df['time'] >= start_time) & 
                (df['time'] <= end_time)
            ]

        wins = 0
        losses = 0
        total_trades = 0

        with open('failed_trades_analysis.log', 'a') as f:
            f.write(f"\nAnalyzing trades for {symbol}\n")
            f.write("="*50 + "\n")

        close_prices = df['close']

        # Get the last valid index (length - 6 to ensure we have 6 candles ahead)
        max_index = len(df) - 7
        
        for i in range(max_index + 1):
            # Skip if no signal
            if not df['call_signal'].iloc[i] and not df['put_signal'].iloc[i]:
                continue
                
            # Validate we have enough future candles
            if i + 7 >= len(close_prices):
                print(f"Warning: Skipping trade at index {i} - not enough future candles")
                continue

            opening_price = close_prices.iloc[i + 1]
            closing_price = close_prices.iloc[i + 7]
            
            # Validate prices are valid numbers
            if pd.isna(opening_price) or pd.isna(closing_price):
                print(f"Warning: Skipping trade at index {i} - invalid prices")
                continue
                
            total_trades += 1

            # Analyze failed trades
            # if ((df['call_signal'].iloc[i] and closing_price <= opening_price) or
            #     (df['put_signal'].iloc[i] and closing_price >= opening_price)):
            #     self.analyze_failed_trade(df, i)

            if df['call_signal'].iloc[i]:
                if closing_price > opening_price:
                    print(f"Date: {df['time'].iloc[i]}, Opening Price: {opening_price}, Closing Price: {closing_price}, Signal: Call, Status: Win")
                    wins += 1
                else:
                    print(f"Date: {df['time'].iloc[i]}, Opening Price: {opening_price}, Closing Price: {closing_price}, Signal: Call, Status: Loss")
                    losses += 1
            elif df['put_signal'].iloc[i]:
                if closing_price < opening_price:
                    print(f"Date: {df['time'].iloc[i]}, Opening Price: {opening_price}, Closing Price: {closing_price}, Signal: Put, Status: Win")
                    wins += 1
                else:
                    print(f"Date: {df['time'].iloc[i]}, Opening Price: {opening_price}, Closing Price: {closing_price}, Signal: Put, Status: Loss")
                    losses += 1
            else:
                total_trades -= 1  # If no closing price, do not count this trade
                continue

        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        print(f'Backtest results for {symbol}: Total Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%')

        return win_rate, total_trades, wins, losses

    def get_required_candle_history(self) -> int:
        """Get the amount of historical candle data required for the strategy."""
        return self.parameters['candle_history']

    def get_trade_parameters(self, asset: str) -> Dict[str, Any]:
        """Get the trading parameters for this strategy."""
        return self.trade_parameters.copy()

    def check_trade_entry(self, df_clean: pd.DataFrame) -> Tuple[bool, bool, str, str]:
        """Check for MACD-SMA based trade entry signals."""
        if df_clean.empty:
            return False, False, ""
            
        # Get the last row for signal checking
        last_row = df_clean.iloc[-1]
        signal_row = df_clean.iloc[-1]
        
        call_signal = bool(signal_row['call_signal'])
        put_signal = bool(signal_row['put_signal'])
        
        # Get the signal time
        signal_time = str(signal_row['time']) if 'time' in signal_row else ""
        
        # Get the trade time
        trade_time = str(last_row['time']) if 'time' in last_row else ""
        
        return call_signal, put_signal, signal_time, trade_time

    # def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Calculate pivot points and support/resistance levels with validation and smoothing.
    #     """
    #     try:
    #         # Use multiple timeframes for pivot points
    #         short_window = 20
    #         medium_window = 40
    #         long_window = 60
            
    #         # Calculate pivot points for different timeframes
    #         df['pivot_short'] = (df['ha_high'].rolling(window=short_window).max() + 
    #                         df['ha_low'].rolling(window=short_window).min() + 
    #                         df['ha_close'].rolling(window=short_window).mean()) / 3
            
    #         df['pivot_medium'] = (df['ha_high'].rolling(window=medium_window).max() + 
    #                             df['ha_low'].rolling(window=medium_window).min() + 
    #                             df['ha_close'].rolling(window=medium_window).mean()) / 3
            
    #         df['pivot_long'] = (df['ha_high'].rolling(window=long_window).max() + 
    #                         df['ha_low'].rolling(window=long_window).min() + 
    #                         df['ha_close'].rolling(window=long_window).mean()) / 3
            
    #         # Calculate dynamic range sizes
    #         range_short = df['ha_high'].rolling(window=short_window).max() - df['ha_low'].rolling(window=short_window).min()
    #         range_medium = df['ha_high'].rolling(window=medium_window).max() - df['ha_low'].rolling(window=medium_window).min()
            
    #         # Calculate support and resistance with multiple timeframes
    #         df['resistance1'] = df['pivot_medium'] + (range_short * 0.382)
    #         df['resistance2'] = df['pivot_medium'] + (range_medium * 0.618)
    #         df['support1'] = df['pivot_medium'] - (range_short * 0.382)
    #         df['support2'] = df['pivot_medium'] - (range_medium * 0.618)
            
    #         # Forward fill NaN values
    #         for col in ['pivot_short', 'pivot_medium', 'pivot_long', 'resistance1', 'resistance2', 'support1', 'support2']:
    #             df[col] = df[col].ffill()
            
    #         # Calculate dynamic ATR multiplier based on volatility
    #         df['atr_multiplier'] = np.where(
    #             df['atr'] > df['atr'].rolling(100).mean(),
    #             2.0,  # More room during high volatility
    #             1.5   # Tighter during normal volatility
    #         )
            
    #         # Modify near support/resistance detection
    #         df['near_resistance'] = (
    #             (np.abs(df['ha_close'] - df['resistance1']) < df['atr'] * df['atr_multiplier']) |
    #             (np.abs(df['ha_close'] - df['resistance2']) < df['atr'] * df['atr_multiplier'])
    #         )
    #         df['near_support'] = (
    #             (np.abs(df['ha_close'] - df['support1']) < df['atr'] * df['atr_multiplier']) |
    #             (np.abs(df['ha_close'] - df['support2']) < df['atr'] * df['atr_multiplier'])
    #         )
            
    #         # Calculate level strength with more nuance
    #         df['pivot_strength'] = (
    #             (np.abs(df['ha_close'] - df['pivot_medium']) / df['atr']) +
    #             (np.abs(df['ha_close'] - df['pivot_long']) / df['atr'])
    #         ) / 2
            
    #         # Add zone classification
    #         df['in_resistance_zone'] = (df['ha_close'] > df['pivot_medium']) & (df['ha_close'] < df['resistance1'])
    #         df['in_support_zone'] = (df['ha_close'] < df['pivot_medium']) & (df['ha_close'] > df['support1'])
            
    #         return df
                
    #     except Exception as e:
    #         print(f"Error in pivot points calculation: {str(e)}")
    #         return df

    # def detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Enhanced market regime detection using multiple indicators.
    #     """
    #     try:
    #         # Calculate returns and volatility
    #         df['returns'] = df['ha_close'].pct_change()
    #         df['volatility'] = df['returns'].rolling(window=20).std()
            
    #         # Trend detection using multiple timeframes
    #         short_period = 20
    #         long_period = 60
            
    #         # Price-based trend detection
    #         df['short_ma'] = df['ha_close'].rolling(window=short_period).mean()
    #         df['long_ma'] = df['ha_close'].rolling(window=long_period).mean()
    #         df['ma_trend'] = df['short_ma'] > df['long_ma']
            
    #         # Volatility-based trend detection
    #         df['vol_ratio'] = df['volatility'] / df['volatility'].rolling(window=long_period).mean()
            
    #         # ADX-based trend strength
    #         df['adx_trending'] = df['adx'] > 25
            
    #         # MACD trend confirmation
    #         df['macd_trend'] = df['macd_line'] > df['signal_line']
            
    #         # Combined trend indicators
    #         df['is_trending'] = (
    #             (df['vol_ratio'] > 1.0) &  # Higher than average volatility
    #             df['adx_trending'] &        # Strong trend via ADX
    #             (df['macd_trend'] == df['ma_trend'])  # MACD confirms MA trend
    #         )
            
    #         # Add trend strength metric (0-100)
    #         df['trend_strength'] = (
    #             (df['vol_ratio'] * 20) +  # Volatility component (0-20)
    #             (df['adx'] * 0.8) +       # ADX component (0-40)
    #             (np.abs(df['macd_line'] - df['signal_line']) * 40)  # MACD component (0-40)
    #         ).clip(0, 100)
            
    #         # Market regime classification
    #         df['regime'] = np.where(
    #             df['is_trending'] & (df['trend_strength'] > 60),
    #             'strong_trend',
    #             np.where(
    #                 df['is_trending'] & (df['trend_strength'] > 30),
    #                 'weak_trend',
    #                 'ranging'
    #             )
    #         )
            
    #         return df
    #     except Exception as e:
    #         print(f"Error in market_regime_detection: {e}")

    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pivot points and support/resistance levels with validation and smoothing.
        """
        try:
            # Use shorter timeframes for more dynamic levels
            window = 20  # Back to shorter window
            
            # Calculate simple moving averages for trend context
            df['sma20'] = df['ha_close'].rolling(window=20).mean()
            df['sma50'] = df['ha_close'].rolling(window=50).mean()
            
            # Calculate pivot points
            df['pivot'] = (df['ha_high'].rolling(window=window).max() + 
                        df['ha_low'].rolling(window=window).min() + 
                        df['ha_close'].rolling(window=window).mean()) / 3
            
            # Calculate dynamic range size
            range_size = df['ha_high'].rolling(window=window).max() - df['ha_low'].rolling(window=window).min()
            
            # Calculate support and resistance levels
            df['resistance1'] = df['pivot'] + (range_size * 0.382)
            df['resistance2'] = df['pivot'] + (range_size * 0.618)
            df['support1'] = df['pivot'] - (range_size * 0.382)
            df['support2'] = df['pivot'] - (range_size * 0.618)
            
            # Forward fill NaN values
            for col in ['pivot', 'resistance1', 'resistance2', 'support1', 'support2']:
                df[col] = df[col].ffill()
            
            # More lenient support/resistance zones
            df['near_resistance'] = (
                (np.abs(df['ha_close'] - df['resistance1']) < df['atr'] * 2.0) |
                (np.abs(df['ha_close'] - df['resistance2']) < df['atr'] * 2.0)
            )
            df['near_support'] = (
                (np.abs(df['ha_close'] - df['support1']) < df['atr'] * 2.0) |
                (np.abs(df['ha_close'] - df['support2']) < df['atr'] * 2.0)
            )
            
            # Simplified pivot strength
            df['pivot_strength'] = np.abs(df['ha_close'] - df['pivot']) / df['atr']
            
            return df
                
        except Exception as e:
            print(f"Error in pivot points calculation: {str(e)}")
            return df

    def detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified market regime detection focusing on key indicators.
        """
        try:
            # Basic trend detection
            df['short_ma'] = df['ha_close'].rolling(window=20).mean()
            df['long_ma'] = df['ha_close'].rolling(window=50).mean()
            
            # Trend direction
            df['trend_direction'] = np.where(df['short_ma'] > df['long_ma'], 1, -1)
            
            # Simplified trend strength calculation
            df['trend_strength'] = (
                (df['adx'] * 0.5) +  # ADX component (0-50)
                (np.abs(df['macd_line'] - df['signal_line']) * 50)  # MACD component (0-50)
            ).clip(0, 100)
            
            # Market regime classification
            df['regime'] = np.where(
                df['trend_strength'] > 40,  # Lower threshold
                'trending',
                'ranging'
            )
            
            return df
        except Exception as e:
            print(f"Error in market regime detection: {e}")
            return df

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci retracement levels."""
        # Calculate swing high and low over a rolling window
        window = 20
        df['rolling_max'] = df['ha_high'].rolling(window=window).max()
        df['rolling_min'] = df['ha_low'].rolling(window=window).min()
        
        # Calculate Fibonacci levels
        diff = df['rolling_max'] - df['rolling_min']
        df['fib_236'] = df['rolling_max'] - (diff * 0.236)
        df['fib_382'] = df['rolling_max'] - (diff * 0.382)
        df['fib_618'] = df['rolling_max'] - (diff * 0.618)
        
        # Check if price is near Fibonacci levels
        df['near_fib_level'] = (
            (np.abs(df['ha_close'] - df['fib_236']) < df['atr']) |
            (np.abs(df['ha_close'] - df['fib_382']) < df['atr']) |
            (np.abs(df['ha_close'] - df['fib_618']) < df['atr'])
        )
        return df

    def analyze_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze price action patterns."""
        # Calculate candle sizes
        df['body_size'] = np.abs(df['ha_close'] - df['ha_open'])
        df['upper_shadow'] = df['ha_high'] - np.maximum(df['ha_close'], df['ha_open'])
        df['lower_shadow'] = np.minimum(df['ha_close'], df['ha_open']) - df['ha_low']
        
        # Identify strong candles
        avg_body = df['body_size'].rolling(20).mean()
        df['strong_bullish'] = (
            (df['ha_close'] > df['ha_open']) &  # Bullish candle
            (df['body_size'] > 1.2 * avg_body) &  # Larger than average
            (df['lower_shadow'] < df['body_size'] * 0.3)  # Small lower shadow
        )
        df['strong_bearish'] = (
            (df['ha_close'] < df['ha_open']) &  # Bearish candle
            (df['body_size'] > 1.2 * avg_body) &  # Larger than average
            (df['upper_shadow'] < df['body_size'] * 0.3)  # Small upper shadow
        )
        return df

    def enhanced_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced market regime detection using multiple indicators."""
        # Existing trend detection
        df['returns'] = df['ha_close'].pct_change()
        rolling_vol = df['returns'].rolling(window=20).std()
        
        # Add Directional Movement
        df['up_move'] = df['ha_high'] - df['ha_high'].shift(1)
        df['down_move'] = df['ha_low'].shift(1) - df['ha_low']
        
        # Calculate trend consistency
        df['dm_positive'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['dm_negative'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Smooth the directional movement
        df['dm_positive_smooth'] = df['dm_positive'].rolling(window=14).mean()
        df['dm_negative_smooth'] = df['dm_negative'].rolling(window=14).mean()
        
        # Combined trend indication
        df['is_trending'] = (
            (rolling_vol > rolling_vol.rolling(window=60).mean()) &  # Volatility-based
            ((df['dm_positive_smooth'] > df['dm_negative_smooth'] * 1.5) |  # Strong uptrend
            (df['dm_negative_smooth'] > df['dm_positive_smooth'] * 1.5))    # Strong downtrend
        )
        
        # Market phase identification
        df['trend_strength'] = np.where(
            df['dm_positive_smooth'] > df['dm_negative_smooth'],
            df['dm_positive_smooth'] / df['dm_negative_smooth'],
            -df['dm_negative_smooth'] / df['dm_positive_smooth']
        )
        
        return df

    def analyze_failed_trade(self, df: pd.DataFrame, index: int) -> None:
        """Analyze why a trade failed and log the conditions."""
        with open('failed_trades_analysis.log', 'a') as f:
            row = df.iloc[index]
            next_row = df.iloc[index + 1]
            prev_row = df.iloc[index - 1] if index > 0 else row
            
            f.write(f"\n=== Failed Trade Analysis at {row['time']} ===\n")
            f.write(f"Price Action:\n")
            f.write(f"Previous Close: {prev_row['ha_close']:.5f}\n")
            f.write(f"Current Close: {row['ha_close']:.5f}\n")
            f.write(f"Next Close: {next_row['ha_close']:.5f}\n")
            
            f.write(f"\nIndicator Values:\n")
            f.write(f"ADX: {row['adx']:.2f} (Trend: {row['adx_trend']})\n")
            f.write(f"MACD Line: {row['macd_line']:.5f}\n")
            f.write(f"Signal Line: {row['signal_line']:.5f}\n")
            f.write(f"MACD Histogram: {row['macd_diff']:.5f}\n")
            
            f.write(f"\nTrend Analysis:\n")
            f.write(f"Close vs SMA: {row['close_above_sma']}\n")
            f.write(f"Close vs EMA100: {row['close_above_ema100']}\n")
            f.write(f"ADX Positive Trend: {row['adx_positive_trend']}\n")
            f.write(f"ADX Negative Trend: {row['adx_negative_trend']}\n")
            
            f.write(f"\nVolatility and Momentum:\n")
            f.write(f"ATR: {row['atr']:.5f}\n")
            f.write(f"ATR Condition Met: {row['atr_condition']}\n")
            f.write(f"Histogram Widening Call: {row['histogram_widening_call']}\n")
            f.write(f"Histogram Widening Put: {row['histogram_widening_put']}\n")
            
            f.write(f"\nSupport/Resistance:\n")
            f.write(f"Near Resistance: {row['near_resistance']}\n")
            f.write(f"Near Support: {row['near_support']}\n")
            
            f.write(f"\nMarket Regime:\n")
            f.write(f"Regime: {row['regime']}\n")
            f.write(f"Trend Strength: {row['trend_strength']:.2f}\n")
            
            f.write("\n" + "="*50 + "\n")