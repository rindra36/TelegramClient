"""
Williams %R Strategy Implementation

This module implements a trading strategy based on the Williams %R indicator
for binary options trading. The Williams %R (or Williams Percent Range) is a momentum 
indicator that measures overbought and oversold levels.
"""

from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
from ta.momentum import WilliamsRIndicator
from ta.trend import SMAIndicator, EMAIndicator
from .base_strategy import BaseStrategy

class WilliamsRStrategy(BaseStrategy):
    """Williams %R trading strategy implementation."""

    def __init__(self) -> None:
        """Initialize Williams %R strategy with default parameters."""
        super().__init__(
            name="Williams %R Strategy",
            description="Trading strategy using Williams %R indicator with trend confirmation"
        )
        
        # Set strategy-specific timeframe
        self.strategy_timeframe = {
            'candles': 30,  # 1-minute intervals
            'resample': '30s',  # For resampling to 1-minute candles
            'min_points': 30,  # Minimum points for resampling
            'candles_history': 36000  # Amount of historical candle data needed
        }

        # Technical parameters
        self.parameters = {
            'wr_period': 14,  # Williams %R period
            'wr_overbought': -20,  # Overbought threshold
            'wr_oversold': -80,  # Oversold threshold
            'ema_period': 50,  # EMA period for trend confirmation
            'volume_ma_period': 20,  # Volume moving average period
            'exit_wr_level': -50,  # Exit level for mean reversion
        }
        
        # Trade parameters
        self.trade_parameters = {
            'expiration': 120,  # Default 3-minute expiry
            'amount': 1.0,  # Default $1 trade amount
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Williams %R and trend confirmation.
        
        Args:
            df (pd.DataFrame): Candlestick data with OHLC prices
        
        Returns:
            pd.DataFrame: Enhanced dataframe with trading signals
        """
        # Calculate Williams %R
        wr = WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            lbp=self.parameters['wr_period']
        )
        df['wr'] = wr.williams_r()
        
        # Calculate moving averages for trend confirmation
        ema = EMAIndicator(df['close'], window=self.parameters['ema_period'])
        df['ema'] = ema.ema_indicator()

        # Detect closing upper and lower than EMA(100)
        df['close_above_ema'] = df['close'] > df['ema']
        df['close_below_ema'] = df['close'] < df['ema']

        # Check current candle's state
        df['current_bullish'] = df['close'] > df['open']
        df['current_bearish'] = df['close'] < df['open']

        # Check next candle's direction
        df['next_close'] = df['close'].shift(-1)
        df['next_open'] = df['open'].shift(-1)
        df['next_bullish'] = df['next_close'] > df['next_open']
        df['next_bearish'] = df['next_close'] < df['next_open']

        # Check if next candle's close is above or below the previous one
        df['next_above'] = df['next_close'] > df['high']
        df['next_below'] = df['next_close'] < df['low']
        
        # Generate trading signals
        df['call_signal'] = (
            (df['wr'] < self.parameters['wr_oversold']) &  # Oversold condition
            df['close_above_ema'] &  # Price above EMA
            df['next_bullish'] & # Next candle need to be bullish
            df['current_bearish'] & # Current candle need to be bearish
            df['next_above'] &  # Next candle need to close above the previous one
            ~df['next_close'].isna() & # Next candle close need to be available
            ~df['next_open'].isna() # Next candle open need to be available
        )
        
        df['put_signal'] = (
            (df['wr'] > self.parameters['wr_overbought']) &  # Overbought condition
            df['close_below_ema'] &  # Price below EMA
            df['next_bearish'] & # Next candle need to be bearish
            df['current_bullish'] & # Current candle need to be bullish
            df['next_below'] &  # Next candle need to close above the previous one
            ~df['next_close'].isna() & # Next candle close need to be available
            ~df['next_open'].isna() # Next candle open need to be available
        )
        
        return df

    def backtest(self, df: pd.DataFrame, symbol: str, start_time=None, end_time=None) -> Tuple[float, int, int, int]:
        """
        Performs backtesting of the Williams %R strategy on historical data.
        
        Args:
            df (pd.DataFrame): Historical price data with signals
            symbol (str): Trading symbol being tested
        
        Returns:
            tuple: (win_rate, total_trades, wins, losses)
        """
        print(f'Backtesting Williams %R strategy with: {symbol}')
        wins = 0
        losses = 0
        total_trades = 0
        
        close_prices = df['close']
        
        # Get the last valid index (length - 6 to ensure we have 6 candles ahead)
        max_index = len(df) - 6
        
        for i in range(max_index + 1):
            # Skip if no signal
            if not df['call_signal'].iloc[i] and not df['put_signal'].iloc[i]:
                continue
                
            # Validate we have enough future candles
            if i + 7 >= len(close_prices):
                print(f"Warning: Skipping trade at index {i} - not enough future candles")
                continue

            opening_price = close_prices.iloc[i + 1]
            closing_price = close_prices.iloc[i + 5]
            
            # Validate prices are valid numbers
            if pd.isna(opening_price) or pd.isna(closing_price):
                print(f"Warning: Skipping trade at index {i} - invalid prices")
                continue
                
            total_trades += 1

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
                    
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        print(f'Backtest results for {symbol}: Total Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%')
        
        return win_rate, total_trades, wins, losses

    def get_required_candle_history(self) -> int:
        """Get the amount of historical candle data required for the strategy."""
        return self.strategy_timeframe['candles_history']

    def get_trade_parameters(self, asset: str) -> Dict[str, Any]:
        """Get the trading parameters for this strategy."""
        return self.trade_parameters.copy()

    def check_trade_entry(self, df_clean: pd.DataFrame) -> Tuple[bool, bool, str, str]:
        """Check for Williams %R based trade entry signals."""
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
