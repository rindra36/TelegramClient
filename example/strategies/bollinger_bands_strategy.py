"""
Bollinger Bands Strategy Implementation

This module implements a trading strategy based on Bollinger Bands
for binary options trading.
"""

from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from .base_strategy import BaseStrategy
from technical_analysis import candles

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands trading strategy implementation."""

    def __init__(self) -> None:
        """Initialize Bollinger Bands strategy with default parameters."""
        super().__init__(
            name="Bollinger Bands Strategy",
            description="Trading strategy based on price touches of Bollinger Bands with trend confirmation"
        )

        # Set strategy-specific timeframe
        self.strategy_timeframe = {
            'candles': 60, # For 1-minute in seconds
            'resample': '1min', # For resampling to 1-minute candles
            'min_points': 60,  # Minimum points for resampling
            'candles_history': 39600,  # Amount of historical candle data needed
        }
        
        # Technical parameters
        self.parameters = {
            'bb_window': 20,         # Window size for moving average for bands
            'bb_window_dev': 2,      # Number of standard deviations for bands
            'adx_window': 14,        # ADX window for trend confirmation
        }
        
        # Trade parameters
        self.trade_parameters = {
            'expiration': 300,  # 5-minute expiry for mean reversion
            'amount': 1.0,      # Conservative $1 base amount
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            df (pd.DataFrame): Candlestick data with OHLC prices
        
        Returns:
            pd.DataFrame: Enhanced dataframe with trading signals
            
        Signal Generation:
        1. Calculates Bollinger Bands
        2. Identifies price touches of upper/lower bands
        3. Confirms trend direction with next candle
        4. Generates call/put signals based on band bounces
        """
        # Calculate Bollinger Bands
        bb_indicator = BollingerBands(
            close=df["close"], 
            window=self.parameters['bb_window'], 
            window_dev=self.parameters['bb_window_dev']
        )
        
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()

        # Get ADX for trend confirmation
        adx_indicator = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.parameters['adx_window']
        )

        df['adx'] = adx_indicator.adx()
        
        # Check for price touches
        df['touch_upper'] = df['high'] >= df['bb_upper']
        df['touch_lower'] = df['low'] <= df['bb_lower']

        # Check for price close above upper band or below lower band
        df['close_above_upper'] = df['close'] > df['bb_upper']
        df['close_below_lower'] = df['close'] < df['bb_lower']
        
        # Check next candle's direction
        df['next_close'] = df['close'].shift(-1)
        df['next_open'] = df['open'].shift(-1)
        df['next_bullish'] = df['next_close'] > df['next_open']
        df['next_bearish'] = df['next_close'] < df['next_open']

        df['hammer'] = candles.hammer(df['open'], df['high'], df['low'], df['close'])
        df['inverted_hammer'] = candles.inverted_hammer(df['open'], df['high'], df['low'], df['close'])
        df['bullish_engulfing'] = candles.bullish_engulfing(df['open'], df['high'], df['low'], df['close'])
        df['bearish_engulfing'] = candles.bearish_engulfing(df['open'], df['high'], df['low'], df['close'])
        df['bullish_star'] = candles.bullish_star(df['open'], df['high'], df['low'], df['close'])
        df['bearish_star'] = candles.bearish_star(df['open'], df['high'], df['low'], df['close'])
        
        # Generate signals
        # df['call_signal'] = (df['touch_lower'] & df['next_bullish']) & ((~df['next_close'].isna() & ~df['next_open'].isna()))
        df['call_signal'] = ((df['close_below_lower'].shift(1)) & (df['close'] > df['bb_lower']) & (df['adx'] < 20))
        # df['put_signal'] = (df['touch_upper'] & df['next_bearish']) & ((~df['next_close'].isna() & ~df['next_open'].isna()))
        df['put_signal'] = ((df['close_above_upper'].shift(1)) & (df['close'] < df['bb_upper']) & (df['adx'] < 20))
        
        return df

    def backtest(self, df: pd.DataFrame, symbol: str, start_time=None, end_time=None) -> Tuple[float, int, int, int]:
        """
        Performs backtesting of the Bollinger Bands strategy on historical data.
        
        Args:
            df (pd.DataFrame): Historical price data with signals
            symbol (str): Trading symbol being tested
        
        Returns:
            tuple: (win_rate, total_trades, wins, losses)
                - win_rate: Percentage of winning trades
                - total_trades: Total number of trades executed
                - wins: Number of winning trades
                - losses: Number of losing trades
        """
        print(f'Backtesting Bollinger Bands strategy with: {symbol}')
        wins = 0
        losses = 0
        total_trades = 0

        close_prices = df['close']
        max_index = len(df) - 7  # Ensure we have 7 candles ahead for validation
        
        for i in range(max_index + 1):
            if not df['call_signal'].iloc[i] and not df['put_signal'].iloc[i]:
                continue

            total_trades += 1
            opening_price = close_prices.iloc[i + 1]
            closing_price = close_prices.iloc[i + 6]

            if df['call_signal'].iloc[i]:
                if closing_price > opening_price:
                    wins += 1
                else:
                    losses += 1
            elif df['put_signal'].iloc[i]:
                if closing_price < opening_price:
                    wins += 1
                else:
                    losses += 1
            else:
                total_trades -= 1
                continue

        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        print(f'Backtest results for {symbol}: Total Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%')

        return win_rate, total_trades, wins, losses

    def get_required_candle_history(self) -> int:
        """Get the amount of historical candle data required for the strategy."""
        return self.parameters['candle_history']

    def get_trade_parameters(self, asset: str) -> Dict[str, Any]:
        """
        Get the trading parameters for this strategy.
        
        Args:
            asset: The trading asset symbol
            
        Returns:
            Dict[str, Any]: A dictionary containing the trade parameters
        """
        # Return a copy of trade parameters to prevent modification of internal state
        return self.trade_parameters.copy()

    def check_trade_entry(self, df_clean: pd.DataFrame) -> Tuple[bool, bool, str]:
        """Check for Bollinger Bands-based trade entry signals."""
        if df_clean.empty:
            return False, False, ""
            
        # Get the last row for signal checking
        last_row = df_clean.iloc[-1]
        
        # Check conditions specific to Bollinger Bands strategy:
        # - Price crossing back above lower band for calls
        # - Price crossing back below upper band for puts
        # - ADX below 20 for ranging market confirmation
        call_signal = bool(last_row['call_signal'])  # ((close_below_lower.shift(1)) & (close > bb_lower) & (adx < 20))
        put_signal = bool(last_row['put_signal'])    # ((close_above_upper.shift(1)) & (close < bb_upper) & (adx < 20))
        
        # Get the signal time
        signal_time = str(last_row['time']) if 'time' in last_row else ""
        
        return call_signal, put_signal, signal_time
