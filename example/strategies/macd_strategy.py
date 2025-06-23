"""
MACD Strategy Implementation

This module implements the Moving Average Convergence Divergence (MACD)
strategy for binary options trading.
"""

from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
from ta.trend import MACD, EMAIndicator
from .base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    """MACD trading strategy implementation."""

    def __init__(self) -> None:
        """Initialize MACD strategy with default parameters."""
        super().__init__(
            name="MACD Strategy",
            description="Trading strategy based on MACD crossovers with next-candle confirmation"
        )
        # Set strategy-specific timeframe
        self.strategy_timeframe = {
            'candles': 60, # For 1-minute in seconds
            'resample': '1min', # For resampling to 1-minute candles
            'min_points': 60,  # Minimum points for resampling
            'candles_history': 36000,  # Amount of historical candle data needed
            'dropna': 'all', # Configuration for dropping NaN values
            'full_candles_history': 10800
        }

        # Technical parameters
        self.parameters = {
            'window_slow': 20,  # Slow moving average window
            'window_fast': 9,   # Fast moving average window
            'window_sign': 3,   # Signal line window
            'ema100_period': 100, # EMA 100 period
        }
        
        # Trade parameters
        self.trade_parameters = {
            'expiration': 120,  # 2-minute expiry for trend following
            'amount': 1.0,      # Conservative $1 base amount
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MACD crossovers.
        
        Args:
            df (pd.DataFrame): Candlestick data with OHLC prices
        
        Returns:
            pd.DataFrame: Enhanced dataframe with trading signals
            
        Signal Generation:
        - Calculates MACD with configured period settings 
        - Detects MACD line and signal line crossovers
        - Validates signals with next candle confirmation
        - Generates 'call_signal' and 'put_signal' columns
        """
        # Calculate MACD
        close_prices = df['close'].astype(float)
        macd = MACD(
            close_prices, 
            window_slow=self.parameters['window_slow'],
            window_fast=self.parameters['window_fast'],
            window_sign=self.parameters['window_sign']
        )
        df['macd_line'] = macd.macd()
        df['signal_line'] = macd.macd_signal()
        
        # Detect crossovers
        df['macd_above'] = df['macd_line'] > df['signal_line']
        df['prev_macd_above'] = df['macd_above'].shift(1)
        
        df['cross_up'] = (df['macd_above'] & ~df['prev_macd_above']
            .fillna(False)
            .infer_objects()
            .astype(bool))
        df['cross_down'] = (~df['macd_above'] & df['prev_macd_above'])
        
        # Check next candle's direction
        df['next_close'] = df['close'].shift(-1)
        df['next_open'] = df['open'].shift(-1)
        df['next_bullish'] = df['next_close'] > df['next_open']
        df['next_bearish'] = df['next_close'] < df['next_open']

        # TODO: ADD CHECK FOR SUPPORT AND RESISTANCE
        
        # TODO: ADD CHECK OF CANDLESTICK PATTERNS ON THE 2 LAST CANDLES

        # # Calculate EMA(100)
        # ema100 = EMAIndicator(close_prices, window=self.parameters['ema100_period'])
        # df['ema100'] = ema100.ema_indicator()

        # # Detect closing upper and lower than EMA(100)
        # df['close_above_ema'] = df['close'] > df['ema100']
        # df['close_below_ema'] = df['close'] < df['ema100']
        
        # Generate final signals
        df['call_signal'] = (df['cross_up'] & df['next_bullish']) & ((~df['next_close'].isna() & ~df['next_open'].isna()))
        df['put_signal'] = (df['cross_down'] & df['next_bearish']) & ((~df['next_close'].isna() & ~df['next_open'].isna()))

        return df

    def backtest(self, df: pd.DataFrame, symbol: str, start_time=None, end_time=None) -> Tuple[float, int, int, int]:
        """
        Performs backtesting of the MACD strategy on historical data.
        
        Args:
            df (pd.DataFrame): Historical price data with signals
            symbol (str): Trading symbol being tested
        
        Returns:
            tuple: (win_rate, total_trades, wins, losses)
                - win_rate: Percentage of winning trades
                - total_trades: Total number of trades executed
                - wins: Number of winning trades
                - losses: Number of losing trades
                
        The backtesting:
        1. Simulates trades based on historical signals
        2. Evaluates each trade outcome
        3. Calculates overall win rate
        4. Uses 3-candle lookahead for trade result validation
        """
        print(f'Backtesting MACD strategy with: {symbol}')
        wins = 0
        losses = 0
        total_trades = 0

        close_prices = df['close']

        # Get the last valid index (length - 4 to ensure we have 3 candles ahead)
        max_index = len(df) - 4
        
        for i in range(max_index + 1):
            if not df['call_signal'].iloc[i] and not df['put_signal'].iloc[i]:
                continue

            total_trades += 1
            opening_price = close_prices.iloc[i + 1]
            closing_price = close_prices.iloc[i + 3]

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
        """Check for MACD-based trade entry signals."""
        if df_clean.empty:
            return False, False, ""
            
        # Get the row before the last one for signal checking
        if len(df_clean) < 2:
            return False, False, ""
        
        last_row = df_clean.iloc[-2]
        
        # Check for call signal - MACD crosses above signal line
        call_signal = bool(last_row['call_signal'])
        
        # Check for put signal - MACD crosses below signal line
        put_signal = bool(last_row['put_signal'])
        
        # Get the signal time
        signal_time = str(last_row['time']) if 'time' in last_row else ""
        
        return call_signal, put_signal, signal_time
