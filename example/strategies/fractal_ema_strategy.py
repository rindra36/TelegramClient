"""
Fractal-EMA Strategy Implementation

This module implements a strategy combining fractal patterns with dual EMA (3 and 7 periods)
for binary options trading with 1-minute intervals.
"""

from typing import Dict, Tuple, Any
import pandas as pd
from stock_indicators import indicators, Quote, CandleProperties
from technical_analysis import candles
from ta.trend import EMAIndicator
from .base_strategy import BaseStrategy

class FractalEMAStrategy(BaseStrategy):
    """Strategy using fractal patterns combined with EMA crossovers."""

    def __init__(self) -> None:
        """Initialize Fractal-EMA strategy with default parameters."""
        super().__init__(
            name="Fractal-EMA Strategy",
            description="Trading strategy using Fractal patterns with dual EMA (3 and 7) crossovers"
        )
        
        # Strategy-specific timeframe - 1 minute
        self.strategy_timeframe = {
            'candles': 60,  # 1 minute candles (in seconds)
            'resample': '1min',
            'min_points': 60,  # Minimum points needed for resampling
            'candles_history': 36000,    # Amount of historical candle data needed
            'dropna': 'all' # Configuration for dropping NaN values
        }

        # Technical parameters
        self.parameters = {
            'ema_fast': 3,     # Fast EMA period
            'ema_slow': 7,     # Slow EMA period
            'ema_50': 50,    # EMA for trend
            'fractal_periods': 3  # Number of periods for fractal pattern
        }
        
        # Trade parameters
        self.trade_parameters = {
            'expiration': 60,  # 1-minute expiry
            'amount': 1.0     # Base trade amount
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Fractal patterns and EMA crossovers.
        
        Args:
            df (pd.DataFrame): Candlestick data with OHLC prices
        
        Returns:
            pd.DataFrame: Enhanced dataframe with trading signals
        """
        if df.empty:
            return df

        # Calculate candle sizes and identify big candles
        df['candle_size'] = abs(df['close'] - df['open'])
        # Calculate average candle size over the last 10 candles
        df['avg_candle_size'] = df['candle_size'].rolling(window=10).mean()
        # Define big candle as 1.5 times the average size
        df['is_big_candle'] = df['candle_size'] > (df['avg_candle_size'] * 1.5)

        quotes = self.pandas_dataframe_to_quote(df)
        close_prices = df['close'].astype(float)

        # Calculate EMAs
        ema3 = EMAIndicator(close_prices, window=self.parameters['ema_fast'])
        df['ema3'] = ema3.ema_indicator()
        ema7 = EMAIndicator(close_prices, window=self.parameters['ema_slow'])
        df['ema7'] = ema7.ema_indicator()
        ema50 = EMAIndicator(close_prices, window=self.parameters['ema_50'])
        df['ema50'] = ema50.ema_indicator()

        # Detect EMA crossovers
        df['ema_above'] = df['ema3'] > df['ema7']
        df['prev_ema_above'] = df['ema_above'].shift(1)
        
        df['ema_cross_up'] = (df['ema_above'] & ~df['prev_ema_above']
            .fillna(False)
            .astype(bool))
        df['ema_cross_down'] = (~df['ema_above'] & df['prev_ema_above']
            .fillna(False)
            .astype(bool))
        
        # Detect closing upper and lower than EMA(50)
        df['close_above_ema'] = df['close'] > df['ema50']
        df['close_below_ema'] = df['close'] < df['ema50']

        # Calculate fractal
        fractals_results = indicators.get_fractal(quotes, self.parameters['fractal_periods'])

        fractals_bear = []
        fractals_bull = []

        # Map fractal results to DataFrame
        for result in fractals_results:
            fractals_bear.append(result.fractal_bear)
            fractals_bull.append(result.fractal_bull)
        
        # Add into DataFrame
        df['fractal_bear'] = pd.Series(fractals_bear).values
        df['fractal_bull'] = pd.Series(fractals_bull).values

        # Checking candlestick pattern
        df['hammer'] = candles.hammer(df['open'], df['high'], df['low'], df['close'])
        df['inverted_hammer'] = candles.inverted_hammer(df['open'], df['high'], df['low'], df['close'])
        df['bullish_engulfing'] = candles.bullish_engulfing(df['open'], df['high'], df['low'], df['close'])
        df['bearish_engulfing'] = candles.bearish_engulfing(df['open'], df['high'], df['low'], df['close'])
        df['bullish_star'] = candles.bullish_star(df['open'], df['high'], df['low'], df['close'])
        df['bearish_star'] = candles.bearish_star(df['open'], df['high'], df['low'], df['close'])
        df['dragonfly_doji'] = candles.is_dragonfly_doji(df['open'], df['high'], df['low'], df['close'])
        df['gravestone_doji'] = candles.is_gravestone_doji(df['open'], df['high'], df['low'], df['close'])

        # Checking bullish or bearish candlestick pattern
        df['bullish_candlestick'] = df['hammer'] | df['bullish_engulfing'] | df['bullish_star'] | df['dragonfly_doji']
        df['bearish_candlestick'] = df['bearish_engulfing'] | df['inverted_hammer'] | df['bearish_star'] | df['gravestone_doji']
        
        # Initialize signal columns
        df['call_signal'] = False
        df['put_signal'] = False
        
        # Track active fractal signals and their appearance time
        active_bull_fractal = False
        active_bear_fractal = False
        bull_fractal_index = None
        bear_fractal_index = None
        
        # Process each candle for signal generation
        for i in range(len(df)):
            current_row = df.iloc[i]
            
            # Handle bull fractal signals
            if current_row['fractal_bull']:
                # New bull fractal cancels previous unconfirmed bull signal
                active_bull_fractal = True
                bull_fractal_index = i
            
            # Handle bear fractal signals
            if current_row['fractal_bear']:
                # New bear fractal cancels previous unconfirmed bear signal
                active_bear_fractal = True
                bear_fractal_index = i
            
            # Check for signal confirmations - must be at least 2 candles after fractal
            # if active_bull_fractal and current_row['ema_cross_up'] and current_row['close_above_ema']:
            if active_bull_fractal and i > 0 and df.iloc[i-1]['ema_cross_up'] and not df.iloc[i-1]['bearish_candlestick']:
                if i >= bull_fractal_index + 2:  # Ensure 2 candles have passed
                    # Check if previous candle was not a big candle
                    if not df.iloc[i-1]['is_big_candle']:
                        df.at[i, 'call_signal'] = True
                        active_bull_fractal = False
                        bull_fractal_index = None
            
            # if active_bear_fractal and current_row['ema_cross_down'] and current_row['close_below_ema']:
            if active_bear_fractal and i > 0 and df.iloc[i-1]['ema_cross_down'] and not df.iloc[i-1]['bullish_candlestick']:
                if i >= bear_fractal_index + 2:  # Ensure 2 candles have passed
                    if not df.iloc[i-1]['is_big_candle']:
                        df.at[i, 'put_signal'] = True
                        active_bear_fractal = False
                        bear_fractal_index = None
            
            # Cancel signals if a new opposite fractal appears before confirmation
            if active_bull_fractal and current_row['fractal_bear']:
                active_bull_fractal = False
                bull_fractal_index = None
            
            if active_bear_fractal and current_row['fractal_bull']:
                active_bear_fractal = False
                bear_fractal_index = None
            
            # Cancel unconfirmed signals if more than 5 candles have passed
            if active_bull_fractal and i > bull_fractal_index + 5:
                active_bull_fractal = False
                bull_fractal_index = None
                
            if active_bear_fractal and i > bear_fractal_index + 5:
                active_bear_fractal = False
                bear_fractal_index = None
        
        return df

    def backtest(self, df: pd.DataFrame, symbol: str, start_time=None, end_time=None) -> Tuple[float, int, int, int]:
        """
        Performs backtesting of the Fractal-EMA strategy on historical data.
        
        Args:
            df (pd.DataFrame): Historical price data with signals
            symbol (str): Trading symbol being tested
        
        Returns:
            tuple: (win_rate, total_trades, wins, losses)
        """
        print(f'Backtesting Fractal-EMA strategy with: {symbol}')
        wins = 0
        losses = 0
        total_trades = 0

        close_prices = df['close']

        # Get the last valid index (length - 2 to ensure we have 1 future candle)
        max_index = len(df) - 2
        
        for i in range(max_index + 1):
            # Skip if no signal
            if not df['call_signal'].iloc[i] and not df['put_signal'].iloc[i]:
                continue
                
            # Validate we have enough future candles
            if i + 1 >= len(close_prices):
                print(f"Warning: Skipping trade at index {i} - not enough future candles")
                continue

            opening_price = close_prices.iloc[i]
            closing_price = close_prices.iloc[i + 1]  # 1-minute expiry
            
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
            else:
                total_trades -= 1  # If no closing price, do not count this trade
                continue

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
        """Check for Fractal-EMA based trade entry signals."""
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
    
    def pandas_dataframe_to_quote(self, df: pd.DataFrame):
        """Transforming the pandas dataframe into quote to work with stock_indicators."""
        if df.empty:
            return False

        quotes_list = [
            Quote(d, o, h, l, c) 
            for d, o, h, l, c 
            in zip(df['time'], df['open'], df['high'], df['low'], df['close'])
        ]

        return quotes_list