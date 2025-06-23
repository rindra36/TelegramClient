"""
Fast Price Action Strategy Implementation

This module implements a strategy focused on real-time price movements
analyzing price action within 30-second intervals.
"""

from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
import numpy as np
import asyncio
import os
import logging
from ta.volatility import AverageTrueRange
from datetime import datetime, timedelta
from .base_strategy import BaseStrategy
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from logging.handlers import RotatingFileHandler

class FastPriceActionStrategy(BaseStrategy):
    """Price action strategy implementation for quick movements."""

    def __init__(self) -> None:
        """Initialize price action strategy with default parameters."""
        super().__init__(
            name="Fast Price Action Strategy",
            description="Monitors price action in 30-second intervals using real-time data"
        )

        # Setup logging
        self.setup_logging()
        
        # Set strategy-specific timeframe
        self.strategy_timeframe = {
            'candles': 30,        # 30-second candles
            'resample': '1s',    # For resampling to 30-second candles
            'min_points': 30,     # Minimum points for resampling
            'candles_history': 3600,  # Amount of historical candle data needed (1 hour)
            'dropna': 'any',       # How to handle NaN values
            'force_candles': 1 # Force candles to be 1 second
        }

        # Technical parameters
        self.parameters = {
            'momentum_threshold': 0.00015,  # Increased from 0.0002 to 0.0008 for stronger moves
            'reversal_threshold': 0.0002,  # Increased from 0.0001 to 0.0002
            'monitoring_seconds': 7,       # Seconds to monitor at start of candle
            'min_candles': 3,             # Minimum candles needed for confirmation
            'acceleration_threshold': 0.00002,  # Minimum price acceleration
            'volume_threshold': 0.7,      # Minimum volume percentile (0-1)
            'trend_strength': 0.55         # Minimum trend strength (0-1)
        }
        
        # Trade parameters
        self.trade_parameters = {
            'expiration': 15,     # 15-second expiry
            'amount': 1.0,        # Default $1 base amount
        }

        # Initialize monitoring state
        self.current_candle_start = None
        self.monitoring_data = []
        self.is_monitoring = False
        self.last_processed_time = None

    def reset_monitoring(self):
        """Reset monitoring state."""
        self.monitoring_data = []
        self.is_monitoring = False
        self.current_candle_start = None
        self.last_processed_time = None

    def should_start_monitoring(self, candle_time: datetime) -> bool:
        """
        Check if we should start monitoring based on candle time.
        
        Args:
            candle_time: Time of the current candle
            
        Returns:
            bool: True if we should start monitoring
        """
        try:
            # Convert string time to datetime if needed
            if isinstance(candle_time, str):
                candle_time = datetime.fromisoformat(candle_time.replace('Z', '+00:00'))
            
            # Get seconds within the current minute
            seconds = candle_time.second
            
            # Check if we're at the start of a 30-second interval
            is_interval_start = seconds % 30 == 0 or seconds % 30 == 1
            
            # Don't start new monitoring if we're already monitoring
            if self.is_monitoring:
                return False
                
            # Start monitoring if we're at the beginning of a 30s interval
            return is_interval_start
            
        except Exception as e:
            print(f"Error in should_start_monitoring: {e}")
            return False

    def process_realtime_candle(self, candle: Dict) -> Tuple[bool, bool, str]:
        """
        Process a real-time candle and check for trading signals.
        
        Args:
            candle: Dictionary containing candle data
                   Format: {'time': str, 'open': float, 'close': float, 'high': float, 'low': float}
                   
        Returns:
            Tuple[bool, bool, str]: (call_signal, put_signal, signal_time)
        """
        try:
            # Convert candle time to datetime
            candle_time = datetime.fromisoformat(candle['time'].replace('Z', '+00:00'))
            
            # Check if we should start monitoring
            if self.should_start_monitoring(candle_time):
                self.current_candle_start = candle_time
                self.is_monitoring = True
                self.monitoring_data = []
                print(f"\nStarted monitoring at {candle_time}")

            # If we're monitoring an outcome
            if hasattr(self, 'outcome_monitoring') and self.outcome_monitoring:
                current_time = datetime.now()
                if current_time <= self.outcome_monitoring['expiry_time']:
                    # Collect price data during monitoring period
                    self.outcome_monitoring['price_points'].append({
                        'time': current_time,
                        'price': candle['close']
                    })
                else:
                    # Expiry reached - analyze outcome
                    self.analyze_trade_outcome()
                    self.outcome_monitoring = None
            
            # If we're monitoring, collect data
            if self.is_monitoring:
                # Calculate time elapsed since monitoring started
                elapsed = (candle_time - self.current_candle_start).total_seconds()
                
                # Add candle to monitoring data
                self.monitoring_data.append({
                    'time': candle_time,
                    'price': candle['close'],
                    'high': candle['high'],
                    'low': candle['low']
                })
                
                # Only analyze if we have enough data points
                if len(self.monitoring_data) >= self.parameters['min_candles']:
                    # Check if monitoring period is complete
                    if elapsed >= self.parameters['monitoring_seconds']:
                        # Analyze collected data
                        call_signal, put_signal = self.analyze_monitoring_data()
                        
                        # Reset monitoring after analysis
                        self.reset_monitoring()
                        
                        if call_signal or put_signal:
                            return call_signal, put_signal, str(candle_time)
                            
            return False, False, ""
            
        except Exception as e:
            print(f"Error processing realtime candle: {e}")
            self.reset_monitoring()
            return False, False, ""

    def setup_logging(self):
        """Setup logging configuration for both file and console output."""
        # Create logs directory if it doesn't exist
        log_dir = 'price_action_logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f'price_action_strategy_{timestamp}.log')
        
        # Setup logger
        self.logger = logging.getLogger('PriceActionStrategy')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers if any
        self.logger.handlers = []
        
        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_and_print(self, message: str, level: str = 'info'):
        """Log message to both file and console."""
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)

    def analyze_monitoring_data(self) -> Tuple[bool, bool]:
        """
        Analyze collected monitoring data for trading signals.
        
        Returns:
            Tuple[bool, bool]: (call_signal, put_signal)
        """
        try:
            # Store current monitoring data for outcome validation
            self.last_signal_data = {
                'time': self.monitoring_data[-1]['time'],
                'price': self.monitoring_data[-1]['price'],
                'movement_type': None,
                'signal_conditions': {}
            }

            if len(self.monitoring_data) < self.parameters['min_candles']:
                self.log_and_print("\nSignal Analysis: Insufficient data points")
                self.log_and_print(f"Required: {self.parameters['min_candles']}, Available: {len(self.monitoring_data)}")
                return False, False
                
            # Get price movement
            initial_price = self.monitoring_data[0]['price']
            current_price = self.monitoring_data[-1]['price']
            total_movement = current_price - initial_price
            
            # Get price series
            prices = [d['price'] for d in self.monitoring_data]
            highs = [d['high'] for d in self.monitoring_data]
            lows = [d['low'] for d in self.monitoring_data]
            
            # Calculate price acceleration
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            if len(price_changes) > 1:
                acceleration = sum(price_changes[i] - price_changes[i-1] for i in range(1, len(price_changes))) / len(price_changes)
            else:
                acceleration = 0

            # Calculate movement consistency
            price_diffs = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            if price_diffs:
                trend_consistency = sum(1 for diff in price_diffs if (total_movement > 0 and diff > 0) or 
                                                (total_movement < 0 and diff < 0)) / len(price_diffs)
            else:
                trend_consistency = 0
            
            # Calculate maximum reversal
            max_reversal = 0
            cumulative_reversal = 0
            for i in range(1, len(prices)):
                if total_movement > 0:
                    reversal = max(highs[:i]) - lows[i]
                    cumulative_reversal += max(0, prices[i-1] - prices[i])
                else:
                    reversal = highs[i] - min(lows[:i])
                    cumulative_reversal += max(0, prices[i] - prices[i-1])
                max_reversal = max(max_reversal, reversal)
            
            # Normalize cumulative reversal
            avg_reversal = cumulative_reversal / (len(prices) - 1) if len(prices) > 1 else float('inf')
            
            # Log detailed analysis
            self.log_and_print("\n=== Signal Analysis Report ===")
            self.log_and_print(f"Initial price: {initial_price:.6f}")
            self.log_and_print(f"Current price: {current_price:.6f}")
            self.log_and_print(f"Total movement: {total_movement:.6f} (Threshold: ±{self.parameters['momentum_threshold']:.6f})")
            self.log_and_print(f"Max reversal: {max_reversal:.6f} (Threshold: {self.parameters['reversal_threshold']:.6f})")
            self.log_and_print(f"Average reversal: {avg_reversal:.6f}")
            self.log_and_print(f"Acceleration: {acceleration:.6f} (Threshold: ±{self.parameters['acceleration_threshold']:.6f})")
            self.log_and_print(f"Trend consistency: {trend_consistency:.2f} (Threshold: {self.parameters['trend_strength']:.2f})")
            
            # Check movement conditions
            price_sequence_up = all(prices[i] >= prices[i-1] for i in range(1, len(prices)))
            price_sequence_down = all(prices[i] <= prices[i-1] for i in range(1, len(prices)))
            
            # Detailed condition logging for upward movement
            if total_movement > 0:
                self.log_and_print("\nUpward Movement Analysis:")
                self.log_and_print(f"✓ Total movement > threshold: {total_movement:.6f} > {self.parameters['momentum_threshold']:.6f}" if 
                      total_movement > self.parameters['momentum_threshold'] else
                      f"✗ Total movement <= threshold: {total_movement:.6f} <= {self.parameters['momentum_threshold']:.6f}")
                self.log_and_print(f"✓ Max reversal < threshold: {max_reversal:.6f} < {self.parameters['reversal_threshold']:.6f}" if
                      max_reversal < self.parameters['reversal_threshold'] else
                      f"✗ Max reversal >= threshold: {max_reversal:.6f} >= {self.parameters['reversal_threshold']:.6f}")
                self.log_and_print(f"✓ Avg reversal < half threshold: {avg_reversal:.6f} < {self.parameters['reversal_threshold']/2:.6f}" if
                      avg_reversal < self.parameters['reversal_threshold']/2 else
                      f"✗ Avg reversal >= half threshold: {avg_reversal:.6f} >= {self.parameters['reversal_threshold']/2:.6f}")
                self.log_and_print(f"✓ Consistent price sequence" if price_sequence_up else "✗ Inconsistent price sequence")
                self.log_and_print(f"✓ Trend consistency >= threshold: {trend_consistency:.2f} >= {self.parameters['trend_strength']:.2f}" if
                      trend_consistency >= self.parameters['trend_strength'] else
                      f"✗ Trend consistency < threshold: {trend_consistency:.2f} < {self.parameters['trend_strength']:.2f}")
            
            # Detailed condition logging for downward movement
            if total_movement < 0:
                self.log_and_print("\nDownward Movement Analysis:")
                self.log_and_print(f"✓ Total movement < -threshold: {total_movement:.6f} < -{self.parameters['momentum_threshold']:.6f}" if
                      total_movement < -self.parameters['momentum_threshold'] else
                      f"✗ Total movement >= -threshold: {total_movement:.6f} >= -{self.parameters['momentum_threshold']:.6f}")
                self.log_and_print(f"✓ Max reversal < threshold: {max_reversal:.6f} < {self.parameters['reversal_threshold']:.6f}" if
                      max_reversal < self.parameters['reversal_threshold'] else
                      f"✗ Max reversal >= threshold: {max_reversal:.6f} >= {self.parameters['reversal_threshold']:.6f}")
                self.log_and_print(f"✓ Avg reversal < half threshold: {avg_reversal:.6f} < {self.parameters['reversal_threshold']/2:.6f}" if
                      avg_reversal < self.parameters['reversal_threshold']/2 else
                      f"✗ Avg reversal >= half threshold: {avg_reversal:.6f} >= {self.parameters['reversal_threshold']/2:.6f}")
                self.log_and_print(f"✓ Consistent price sequence" if price_sequence_down else "✗ Inconsistent price sequence")
                self.log_and_print(f"✓ Trend consistency >= threshold: {trend_consistency:.2f} >= {self.parameters['trend_strength']:.2f}" if
                      trend_consistency >= self.parameters['trend_strength'] else
                      f"✗ Trend consistency < threshold: {trend_consistency:.2f} < {self.parameters['trend_strength']:.2f}")

            # Store conditions for outcome validation
            self.last_signal_data['signal_conditions'] = {
                'total_movement': total_movement,
                'max_reversal': max_reversal,
                'avg_reversal': avg_reversal,
                'trend_consistency': trend_consistency,
                'price_sequence_up': price_sequence_up,
                'price_sequence_down': price_sequence_down,
                'thresholds': {
                    'momentum': self.parameters['momentum_threshold'],
                    'reversal': self.parameters['reversal_threshold'],
                    'trend_strength': self.parameters['trend_strength']
                }
            }
                
            # Signal determination with conditions
            is_straight_up = (
                total_movement > self.parameters['momentum_threshold'] and
                max_reversal < self.parameters['reversal_threshold'] and
                avg_reversal < self.parameters['reversal_threshold'] / 2 and
                price_sequence_up and
                trend_consistency >= self.parameters['trend_strength']
            )
            
            is_straight_down = (
                total_movement < -self.parameters['momentum_threshold'] and
                max_reversal < self.parameters['reversal_threshold'] and
                avg_reversal < self.parameters['reversal_threshold'] / 2 and
                price_sequence_down and
                trend_consistency >= self.parameters['trend_strength']
            )
            
            # Store movement type for outcome validation
            if is_straight_up:
                self.last_signal_data['movement_type'] = 'UP'
            elif is_straight_down:
                self.last_signal_data['movement_type'] = 'DOWN'
            
            # Final signal decision with logging
            self.log_and_print("\nFinal Decision:")
            if is_straight_up:
                self.log_and_print("CALL SIGNAL GENERATED")
            elif is_straight_down:
                self.log_and_print("PUT SIGNAL GENERATED")
            else:
                self.log_and_print("NO SIGNAL")
                # Monitor what would have happened
                self.monitor_no_signal_outcome('UP' if total_movement > 0 else 'DOWN')
            
            return is_straight_up, is_straight_down
            
        except Exception as e:
            print(f"Error analyzing monitoring data: {e}")
            return False, False

    def monitor_no_signal_outcome(self,  direction: str):
        """
        Monitor what would have happened if a trade was taken.
        """
        self.outcome_monitoring = {
            'start_time': datetime.now(),
            'start_price': self.monitoring_data[-1]['price'],
            'signal_type': 'NO SIGNAL',
            'expiry_time': datetime.now() + timedelta(seconds=self.trade_parameters['expiration']),
            'price_points': [],
            'direction': direction
        }

    def analyze_trade_outcome(self):
        """
        Analyze the outcome of a monitored trade at expiry.
        """
        if not self.outcome_monitoring or not self.outcome_monitoring['price_points']:
            return

        start_price = self.outcome_monitoring['start_price']
        final_price = self.outcome_monitoring['price_points'][-1]['price']
        price_movement = final_price - start_price
        signal_type = self.outcome_monitoring['signal_type']
        direction = self.outcome_monitoring['direction']

        self.log_and_print("\n=== Trade Outcome Analysis ===")
        self.log_and_print(f"Signal Type: {signal_type}")
        self.log_and_print(f"Start Price: {start_price:.6f}")
        self.log_and_print(f"Final Price: {final_price:.6f}")
        self.log_and_print(f"Price Movement: {price_movement:.6f}")

    
        self.log_and_print("No Signal Outcome Analysis:")
        self.log_and_print(f"Direction of price movement is {direction}:")
        if direction == "UP":
            self.log_and_print(f"If CALL was taken: Would be {'WIN' if price_movement > 0 else 'LOSS'}")
            self.log_and_print(f"If PUT was taken: Would be {'WIN' if price_movement < 0 else 'LOSS'}")
        elif direction == "DOWN":
            self.log_and_print(f"If PUT was taken: Would be {'WIN' if price_movement < 0 else 'LOSS'}")
            self.log_and_print(f"If CALL was taken: Would be {'WIN' if price_movement > 0 else 'LOSS'}")
        

        # Log price movement details
        self.log_and_print("\nPrice Movement Details:")
        for point in self.outcome_monitoring['price_points']:
            movement = point['price'] - start_price
            self.log_and_print(f"Time: {point['time']}, Price: {point['price']:.6f}, Movement: {movement:.6f}")

    def check_trade_entry(self, df_clean: pd.DataFrame) -> Tuple[bool, bool, str]:
        """Not used for real-time operation."""
        return False, False, ""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Not used for real-time operation."""
        df['call_signal'] = False
        df['put_signal'] = False
        return df

    def get_required_candle_history(self) -> int:
        """Get required historical candle data amount."""
        return self.strategy_timeframe['candles_history']

    def get_trade_parameters(self, asset: str) -> Dict[str, Any]:
        """Get trading parameters."""
        return self.trade_parameters.copy()

    def backtest(self, df: pd.DataFrame, symbol: str, start_time=None, end_time=None) -> Tuple[float, int, int, int]:
        """
        Perform backtesting simulation.
        Note: Backtesting this strategy is limited since we need tick data
        """
        print("Note: Backtesting this strategy requires tick data for accurate results")
        return 0.0, 0, 0, 0  # Return dummy values since we can't properly backtest without tick data

class PriceActionStreamManager:
    """Manages multiple price action streams"""
    
    def __init__(self, ssid: str, api_v1, strategy_class):
        self.ssid = ssid
        self.api_v1 = api_v1
        self.strategy_class = strategy_class
        self.streams = {}  # {asset: {'api': api, 'stream': stream, 'strategy': strategy}}
        self.reconnect_delays = {}  # {asset: delay}
        self.last_data_times = {}  # {asset: last_time}
        
    async def setup_stream(self, asset: str):
        """Setup stream for a single asset"""
        try:
            # Create new API instance
            api = PocketOptionAsync(self.ssid)
            await asyncio.sleep(5)
            
            # Create new strategy instance
            strategy = self.strategy_class
            strategy.reset_monitoring()
            
            # Setup stream
            stream = await api.subscribe_symbol_timed(asset, timedelta(seconds=1))
            
            self.streams[asset] = {
                'api': api,
                'stream': stream,
                'strategy': strategy
            }
            self.reconnect_delays[asset] = 5  # Initial delay
            self.last_data_times[asset] = datetime.now()
            
            print(f"Stream established for {asset}")
            return True
        except Exception as e:
            print(f"Error setting up stream for {asset}: {e}")
            return False
            
    async def process_streams(self):
        """Process all active streams"""
        while True:
            # Create tasks for all streams
            tasks = []
            for asset in list(self.streams.keys()):
                task = asyncio.create_task(self.process_single_stream(asset))
                tasks.append(task)
                
            if not tasks:
                print("No active streams. Waiting...")
                await asyncio.sleep(5)
                continue
                
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Brief pause before next iteration
            await asyncio.sleep(1)
                
    async def process_single_stream(self, asset: str):
        """Process a single asset stream"""
        try:
            stream_data = self.streams.get(asset)
            if not stream_data:
                return
                
            api = stream_data['api']
            stream = stream_data['stream']
            strategy = stream_data['strategy']
            
            async for candle in stream:
                try:
                    current_time = datetime.now()
                    
                    # Check for timeout
                    last_time = self.last_data_times.get(asset)
                    if last_time and (current_time - last_time).total_seconds() > 10:
                        print(f"Data timeout detected for {asset}")
                        await self.reconnect_asset(asset)
                        break
                        
                    self.last_data_times[asset] = current_time
                    
                    # Process candle
                    call_signal, put_signal, signal_time = strategy.process_realtime_candle(candle)
                    
                    print(f"Received candle for {asset} at {current_time}: {candle}")
                    
                    # Handle signals
                    if call_signal:
                        print(f"\nCALL signal for {asset} at {signal_time}")
                        try:
                            await trade(api, self.api_v1, asset, 0)
                            await self.reconnect_asset(asset)
                            break
                        except Exception as e:
                            print(f"Error executing CALL trade for {asset}: {e}")
                            
                    elif put_signal:
                        print(f"\nPUT signal for {asset} at {signal_time}")
                        try:
                            await trade(api, self.api_v1, asset, 1)
                            await self.reconnect_asset(asset)
                            break
                        except Exception as e:
                            print(f"Error executing PUT trade for {asset}: {e}")
                            
                except Exception as e:
                    print(f"Error processing candle for {asset}: {e}")
                    
        except Exception as e:
            print(f"Stream error for {asset}: {e}")
            await self.reconnect_asset(asset)
            
    async def reconnect_asset(self, asset: str):
        """Handle reconnection for a single asset"""
        try:
            # Clean up existing stream
            if asset in self.streams:
                del self.streams[asset]
            
            delay = self.reconnect_delays.get(asset, 5)
            print(f"Reconnecting {asset} in {delay} seconds...")
            await asyncio.sleep(delay)
            
            # Attempt reconnection
            if await self.setup_stream(asset):
                self.reconnect_delays[asset] = 5  # Reset delay
            else:
                # Increase delay for next attempt
                self.reconnect_delays[asset] = min(delay * 2, 60)
                
        except Exception as e:
            print(f"Error reconnecting {asset}: {e}")
            
    async def add_asset(self, asset: str):
        """Add new asset to monitor"""
        await self.setup_stream(asset)
        
    async def remove_asset(self, asset: str):
        """Remove asset from monitoring"""
        if asset in self.streams:
            try:
                await self.streams[asset]['stream'].aclose()
            except:
                pass
            del self.streams[asset]
            
    async def cleanup(self):
        """Cleanup all streams"""
        for asset in list(self.streams.keys()):
            await self.remove_asset(asset)