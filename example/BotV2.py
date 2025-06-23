import pandas as pd
import asyncio
import sys
import json
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.utils import dropna
import talib
import numpy as np

# PocketOption API integration
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

class AdaptiveTradingBot:
    def __init__(self, ssid):
        self.api = PocketOptionAsync(ssid)
        self.min_payout = 70
        self.candlestick_patterns = {
            'engulfing': 0.7,
            'doji': 0.65,
            'hammer': 0.6,
            'shooting_star': 0.6,
            'piercing': 0.55
        }
        
    async def run(self, asset, action):
        """Main entry point with adaptive strategy selection"""
        # Market regime detection
        df = await self.get_candles(asset)
        df = self.classify_market_regime(df)
        
        # Pattern recognition and strategy selection
        df = self.identify_candlestick_patterns(df)
        df = self.generate_adaptive_signals(df)
        
        # Strategy execution based on market regime
        if action == '1':  # Single asset trading
            await self.execute_single_asset(df, asset)
        elif action == '2':  # Multi-asset trading
            await self.execute_multi_asset(df)
        elif action == '3':  # Backtesting
            self.backtest_strategy(df, asset)
            
    def classify_market_regime(self, df):
        """Classify market into 4 regimes using ADX and volatility"""
        # Calculate ADX for trend strength
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Calculate Bollinger Band width for volatility
        bb_indicator = BollingerBands(df['close'])
        df['bb_width'] = bb_indicator.bollinger_mavg() - bb_indicator.bollinger_lband()
        
        # Classify market regimes
        df['market_regime'] = np.select([
            (df['adx'] > 25) & (df['bb_width'] > df['bb_width'].quantile(0.75)),
            (df['adx'] > 25) & (df['bb_width'] < df['bb_width'].quantile(0.25)),
            (df['adx'] < 15) & (df['bb_width'] < df['bb_width'].quantile(0.25)),
            (df['adx'] < 15) & (df['bb_width'] > df['bb_width'].quantile(0.75))
        ], [
            'strong_trend', 'weak_trend', 'sideways', 'volatility_breakout'
        ], default='neutral')
        
        return df
    
    def identify_candlestick_patterns(self, df):
        """Identify and validate candlestick patterns using TA-Lib"""
        # Convert prices to numpy arrays for TA-Lib
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        # Recognize candlestick patterns
        df['engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
        df['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
        df['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
        df['piercing'] = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)
        
        # Calculate pattern reliability scores
        df['pattern_score'] = (
            (df['engulfing'] * 0.7) + 
            (df['doji'] * 0.65) + 
            (df['hammer'] * 0.6) +
            (df['shooting_star'] * 0.6) +
            (df['piercing'] * 0.55)
        )
        
        return df
    
    def generate_adaptive_signals(self, df):
        """Generate trading signals based on market regime and pattern reliability"""
        # Strong trend regime strategy
        strong_trend_mask = df['market_regime'] == 'strong_trend'
        df.loc[strong_trend_mask, 'signal'] = np.select([
            (df['engulfing'] > 0) & (df['pattern_score'] >= 0.7),
            (df['engulfing'] < 0) & (df['pattern_score'] >= 0.7)
        ], [1, -1], default=0)
        
        # Volatility breakout strategy
        vol_breakout_mask = df['market_regime'] == 'volatility_breakout'
        df.loc[vol_breakout_mask, 'signal'] = np.select([
            (df['piercing'] > 0) & (df['pattern_score'] >= 0.55),
            (df['shooting_star'] < 0) & (df['pattern_score'] >= 0.6)
        ], [1, -1], default=0)
        
        # Sideways market strategy
        sideways_mask = df['market_regime'] == 'sideways'
        df.loc[sideways_mask, 'signal'] = np.select([
            (df['doji'] > 0) & (df['pattern_score'] >= 0.65),
            (df['hammer'] > 0) & (df['pattern_score'] >= 0.6)
        ], [1, -1], default=0)
        
        return df
    
    async def get_candles(self, asset):
        """Fetch candlestick data with optimized timeframe selection"""
        # Get current time and calculate next candle start
        now = datetime.now()
        next_candle_start = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Wait until next candle start
        while datetime.now() < next_candle_start:
            await asyncio.sleep(1)
            
        # Fetch historical data
        time = 39600
        frame = 60
        try:
            candles = await asyncio.wait_for(self.api.get_candles(asset, frame, time), timeout=2)
            return pd.DataFrame.from_dict(candles)
        except asyncio.TimeoutError:
            print("Error fetching candles")
            return pd.DataFrame()
    
    async def execute_single_asset(self, df, asset):
        """Execute trades for single asset with dynamic timeframe selection"""
        last_row = df.iloc[-1]
        
        # Dynamic timeframe selection
        timeframe = self.select_timeframe(df)
        
        # Execute trade based on signal
        if last_row['signal'] == 1:
            await self.trade(asset, 0, timeframe)
        elif last_row['signal'] == -1:
            await self.trade(asset, 1, timeframe)
    
    def select_timeframe(self, df):
        """Select optimal timeframe based on market regime"""
        latest = df.iloc[-1]
        if latest['market_regime'] == 'strong_trend':
            return 60  # 1-minute expiry
        elif latest['market_regime'] == 'volatility_breakout':
            return 300  # 5-minute expiry
        else:
            return 900  # 15-minute expiry
    
    async def trade(self, asset, command, timeframe):
        """Execute trade with dynamic position sizing"""
        # Calculate position size based on drawdown
        position_size = self.calculate_position_size()
        
        try:
            print(f"Executing {['CALL', 'PUT'][command]} trade on {asset} with {timeframe}s timeframe")
            if command == 0:  # CALL
                trade_id = await self.api.buy(asset, position_size, timeframe, check_win=False)
            else:  # PUT
                trade_id = await self.api.sell(asset, position_size, timeframe, check_win=False)
            print(f"Trade placed ID: {trade_id}")
        except Exception as e:
            print(f"Trade execution failed: {e}")
    
    def calculate_position_size(self):
        """Dynamic position sizing based on account drawdown"""
        # Placeholder for actual drawdown calculation
        account_drawdown = 0.05  # Example value
        
        if account_drawdown < 0.05:
            return 1.0  # 1% of balance
        elif account_drawdown < 0.1:
            return 0.5  # 0.5% of balance
        else:
            return 0.25  # 0.25% of balance
    
    def backtest_strategy(self, df, asset):
        """Backtest strategy performance across all market regimes"""
        results = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'regime_performance': {
                'strong_trend': {'trades': 0, 'wins': 0},
                'weak_trend': {'trades': 0, 'wins': 0},
                'sideways': {'trades': 0, 'wins': 0},
                'volatility_breakout': {'trades': 0, 'wins': 0}
            }
        }
        
        for i in range(len(df) - 3):
            row = df.iloc[i]
            if row['signal'] == 0:
                continue
                
            # Count trade
            results['total_trades'] += 1
            regime = row['market_regime']
            results['regime_performance'][regime]['trades'] += 1
            
            # Check trade outcome
            opening_price = df.iloc[i+1]['open']
            closing_price = df.iloc[i+3]['close']
            
            if (row['signal'] == 1 and closing_price > opening_price) or \
               (row['signal'] == -1 and closing_price < opening_price):
                results['wins'] += 1
                results['regime_performance'][regime]['wins'] += 1
            else:
                results['losses'] += 1
                
        # Calculate win rates
        win_rate = (results['wins'] / results['total_trades']) * 100 if results['total_trades'] > 0 else 0
        print(f"\nOverall Win Rate: {win_rate:.2f}%")
        
        for regime, data in results['regime_performance'].items():
            if data['trades'] > 0:
                regime_win_rate = (data['wins'] / data['trades']) * 100
                print(f"{regime.capitalize()} Win Rate: {regime_win_rate:.2f}% ({data['trades']} trades)")
                
        return win_rate

# Main execution
if __name__ == '__main__':
    ssid = sys.argv[1] if len(sys.argv) > 1 else input('Enter your SSID: ')
    asset = sys.argv[2] if len(sys.argv) > 2 else input('Enter asset symbol: ')
    action = sys.argv[3] if len(sys.argv) > 3 else input('Choose action (1: Single, 2: Multi, 3: Backtest): ')
    
    bot = AdaptiveTradingBot(ssid)
    asyncio.run(bot.run(asset, action))