"""
MACD Strategy Trading Bot for PocketOption

This script implements an automated trading strategy using the Moving Average Convergence Divergence (MACD)
indicator for binary options trading on PocketOption platform. The bot can operate in four modes:
1. Single asset trading
2. Multi-asset trading on best-performing pairs
3. Backtesting mode for strategy evaluation
4. CSV file analysis for signal validation

Features:
- Real-time MACD crossover signal detection
- Automated trade execution
- Multi-asset support
- Backtesting capabilities
- Payout optimization
- Automatic script restart mechanism
- CSV file analysis for strategy validation

Dependencies:
- pandas: Data manipulation and analysis
- ta: Technical analysis indicators
- asyncio: Asynchronous I/O operations
- BinaryOptionsToolsV2: Custom API wrapper for PocketOption
"""

import sys
import os
import json
import pandas as pd
import asyncio
import ssl
import websockets
import random
import nest_asyncio
import signal
import traceback
import atexit
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, List, Optional, Any
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from BinaryOptionsTools import pocketoption
from src.utils import determine_trade_result, analyze_market_structure, plot_market_structure, check_trade_conditions
from ssid_manager import SSIDManager

# Import strategies
from strategies.strategy_factory import StrategyFactory
from strategies.base_strategy import BaseStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from strategies.macd_sma_strategy import MACDSMAStrategy
from strategies.fractal_ema_strategy import FractalEMAStrategy
from strategies.williams_r_strategy import WilliamsRStrategy
from strategies.price_action_strategy import FastPriceActionStrategy, PriceActionStreamManager

nest_asyncio.apply()

# Global variables
selected_strategy = None  # Will be set in show_menu()
api = None  # Global PocketOptionAsync instance
api_v1 = None  # Global pocketoption instance
api_v1_prices = None # Global pocketoption instance for prices data using v1 API
trade_ssid = None
candle_ssid = None

# Restricted asset list - Temporary
RESTRICTED_ASSETS = {
    'KESUSD_otc', 'USDEGP_otc', 'EURCHF_otc', 'AUDUSD_otc', 'USDSGD_otc',
    'CADCHF_otc', 'USDCAD_otc', 'USDIDR_otc', 'GBPAUD_otc', 'AUDCAD_otc',
    'USDBDT_otc', 'USDMXN_otc', 'TNDUSD_otc', 'USDBRL_otc', 'EURJPY_otc',
    'SARCNY_otc', 'EURTRY_otc', 'IRRUSD_otc', 'QARCNY_otc', 'EURGBP_otc',
    'USDCHF_otc', 'GBPJPY_otc', 'USDINR_otc', 'AEDCNY_otc', 'USDCLP_otc',
    'USDCNH_otc', 'BHDCNY_otc', 'SYPUSD_otc', 'CHFNOK_otc', 'USDPKR_otc',
    'AUDNZD_otc', 'EURNZD_otc', 'NGNUSD_otc', 'USDCOP_otc', 'EURUSD_otc',
    'LBPUSD_otc', 'OMRCNY_otc', 'CADJPY_otc', 'EURRUB_otc', 'USDPHP_otc',
    'NZDUSD_otc', 'ZARUSD_otc', 'EURHUF_otc', 'JODCNY_otc', 'USDVND_otc',
    'USDRUB_otc', 'USDTHB_otc', 'MADUSD_otc', 'GBPUSD_otc', 'USDJPY_otc',
    'AUDCHF_otc', 'AUDJPY_otc', 'NZDJPY_otc', 'YERUSD_otc', 'CHFJPY_otc',
    'UAHUSD_otc', 'USDDZD_otc', 'USDARS_otc', 'USDMYR_otc',
    # 'AUDJPY', 'AUDCAD', 'GBPCAD', 'CHFJPY', 'EURCAD', 'EURJPY', 'GBPJPY',
    # 'GBPCHF', 'CADJPY', 'AUDCHF', 'AUDUSD', 'EURCHF', 'CADCHF', 'USDCAD',
    # 'GBPUSD', 'EURAUD', 'EURGBP', 'USDCHF', 'USDJPY', 'EURUSD', 'GBPAUD'
    # Commodities
    'UKBrent_otc', 'USCrude_otc', 'XAGUSD_otc', 'XAUUSD_otc', 'XNGUSD_otc',
    'XPDUSD_otc', 'XPTUSD_otc',
}
# RESTRICTED_ASSETS = {
#     'AUDJPY', 'AUDCAD', 'GBPCAD', 'CHFJPY', 'EURCAD', 'EURJPY', 'GBPJPY',
#     'GBPCHF', 'CADJPY', 'AUDCHF', 'AUDUSD', 'EURCHF', 'CADCHF', 'USDCAD',
#     'GBPUSD', 'EURAUD', 'EURGBP', 'USDCHF', 'USDJPY', 'EURUSD', 'GBPAUD'
# }

# Trading configuration
trading_config = {
    'trading_type': 'fixed',  # 'fixed', 'compound', or 'progressive_compound'
    'initial_amount': 1.0,
    'current_amount': 1.0,
    'max_compound_level': 5,
    'current_compound_level': 0,
    'consecutive_wins': 0,
    'save_csv': True,  # Default to saving CSV files
    # Progressive compound specific settings
    'max_stages': 5,    # Default value, will be configurable
    'current_stage': 1,
    'current_step': 1,
    'stage_profits': {},  # Will be initialized based on max_stages
    'current_profit': 0.0
}

def validate_trading_config(config):
    """
    Validate trading configuration values.
    Returns tuple (is_valid, error_message)
    """
    # Validate trading type
    if config['trading_type'] not in ['fixed', 'compound', 'progressive_compound']:
        return False, "Invalid trading type"

    if config['trading_type'] == 'progressive_compound':
        # Validate minimum initial amount
        if not isinstance(config['initial_amount'], (int, float)) or config['initial_amount'] < 1.10:
            return False, "Initial amount must be at least $1.10 for progressive compound"

        # Validate max stages
        if not isinstance(config['max_stages'], int) or not 1 <= config['max_stages'] <= 10:
            return False, "Max stages must be between 1 and 10"

        # Validate stage and step
        if not 1 <= config['current_stage'] <= config['max_stages']:
            return False, f"Current stage must be between 1 and {config['max_stages']}"
        if not 1 <= config['current_step'] <= 2:
            return False, "Current step must be 1 or 2"
            
        # Validate stage profits structure
        if not isinstance(config['stage_profits'], dict):
            return False, "Stage profits must be a dictionary"
        for stage in range(1, config['max_stages'] + 1):
            if str(stage) not in config['stage_profits']:
                return False, f"Missing stage {stage} in stage_profits"
                
        # Validate current profit
        if not isinstance(config['current_profit'], (int, float)):
            return False, "Current profit must be a number"
    else:
        # Validate numerical values
        if not isinstance(config['initial_amount'], (int, float)) or config['initial_amount'] <= 0:
            return False, "Initial amount must be a positive number"
        
        if not isinstance(config['max_compound_level'], int) or config['max_compound_level'] <= 0:
            return False, "Max compound level must be a positive integer"

        # Current amount can't be negative    
        if config['current_amount'] < 0:
            return False, "Current amount cannot be negative"

        # Current compound level can't exceed max
        if config['current_compound_level'] > config['max_compound_level']:
            return False, "Current compound level cannot exceed maximum"

    # Validate save_csv setting
    if not isinstance(config.get('save_csv', True), bool):
        return False, "save_csv setting must be a boolean"

    return True, ""

def load_trading_config():
    """Load trading configuration with validation"""
    try:
        with open('trading_config.json', 'r') as f:
            loaded_config = json.load(f)
            
        merged_config = trading_config.copy()
        merged_config.update(loaded_config)
        
        is_valid, error = validate_trading_config(merged_config)
        if not is_valid:
            print(f"Invalid config loaded: {error}. Using defaults.")
            return trading_config

        return merged_config
            
    except FileNotFoundError:
        # Create initial config file
        save_trading_config(trading_config)
        return trading_config
    except json.JSONDecodeError:
        print("Malformed trading_config.json. Using defaults.")
        return trading_config

def save_trading_config(config):
    """Save trading configuration with validation"""
    is_valid, error = validate_trading_config(config)
    if not is_valid:
        print(f"Cannot save invalid config: {error}")
        return False
        
    with open('trading_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    return True

# Add trading settings functions
async def change_trading_settings():
    """Handle changes to trading settings"""
    global trading_config
    
    while True:
        print("\n=== Trading Settings ===")
        print("1. Change trading type (Fixed/Compound/Progressive)")
        print("2. Change initial amount")
        if trading_config['trading_type'] == 'compound':
            print("3. Change max compound level")
        elif trading_config['trading_type'] == 'progressive_compound':
            print("3. View/Reset stage progress")
        print("4. Toggle CSV saving (Currently: {})".format("On" if trading_config.get('save_csv', True) else "Off"))
        print("\n0. Back to main menu")
        
        choice = input("\nChoose setting to change: ").strip()
        
        if choice == "0":
            return
            
        if choice == "1":
            print("\nSelect trading type:")
            print("1. Fixed")
            print("2. Compound")
            print("3. Progressive Compound")
            type_choice = input("Choice (1/2/3): ").strip()
            
            if type_choice == "1":
                trading_config['trading_type'] = 'fixed'
            elif type_choice == "2":
                trading_config['trading_type'] = 'compound'
                # Ask for max compound level
                while True:
                    try:
                        level = int(input("\nEnter max compound level (1-10): "))
                        if 1 <= level <= 10:
                            trading_config['max_compound_level'] = level
                            break
                        print("Please enter a number between 1 and 10")
                    except ValueError:
                        print("Please enter a valid number")
            elif type_choice == "3":
                trading_config['trading_type'] = 'progressive_compound'
                # Ask for number of stages
                while True:
                    try:
                        stages = int(input("\nEnter number of stages (1-10): "))
                        if 1 <= stages <= 10:
                            trading_config['max_stages'] = stages
                            # Initialize stage profits with the chosen number of stages
                            trading_config.update({
                                'current_stage': 1,
                                'current_step': 1,
                                'stage_profits': {str(i): 0.0 for i in range(1, stages + 1)},
                                'current_profit': 0.0
                            })
                            break
                        print("Please enter a number between 1 and 10")
                    except ValueError:
                        print("Please enter a valid number")
            
        elif choice == "2":
            while True:
                try:
                    amount = float(input("\nEnter initial amount (must be positive): $"))
                    if trading_config['trading_type'] == 'progressive_compound':
                        if amount < 1.10:
                            print("Progressive compound trading requires minimum $1.10 initial amount")
                            continue
                    elif amount <= 0:
                        print("Amount must be positive")
                        continue
                        
                    trading_config['initial_amount'] = amount
                    trading_config['current_amount'] = amount  # Reset current amount
                    
                    if trading_config['trading_type'] == 'compound':
                        trading_config['consecutive_wins'] = 0  # Reset wins
                        trading_config['current_compound_level'] = 0  # Reset compound level
                    elif trading_config['trading_type'] == 'progressive_compound':
                        trading_config['current_stage'] = 1  # Reset stage
                        trading_config['current_step'] = 1  # Reset step
                        trading_config['current_profit'] = 0.0  # Reset current profit
                        trading_config['stage_profits'] = {str(i): 0.0 for i in range(1, trading_config['max_stages'] + 1)}  # Reset stage profits
                    break
                    
                except ValueError:
                    print("Please enter a valid number")
                    
        elif choice == "3":
            if trading_config['trading_type'] == 'compound':
                while True:
                    try:
                        level = int(input("\nEnter max compound level (1-10): "))
                        if 1 <= level <= 10:
                            trading_config['max_compound_level'] = level
                            break
                        print("Please enter a number between 1 and 10")
                    except ValueError:
                        print("Please enter a valid number")
            elif trading_config['trading_type'] == 'progressive_compound':
                print("\n=== Current Progress ===")
                print(f"Stage: {trading_config['current_stage']}/{trading_config['max_stages']}")
                print(f"Step: {trading_config['current_step']}")
                print("\nStage Profits:")
                for stage in range(1, trading_config['max_stages'] + 1):
                    profit = trading_config['stage_profits'].get(str(stage), 0.0)
                    print(f"Stage {stage}: ${profit:.2f}")
                print(f"\nCurrent step profit: ${trading_config['current_profit']:.2f}")
                
                reset = input("\nWould you like to reset progress? (y/n): ").strip().lower()
                if reset == 'y':
                    trading_config['current_stage'] = 1
                    trading_config['current_step'] = 1
                    trading_config['current_profit'] = 0.0
                    trading_config['stage_profits'] = {str(i): 0.0 for i in range(1, trading_config['max_stages'] + 1)}
                    print("Progress reset to stage 1")

        elif choice == "4":
            trading_config['save_csv'] = not trading_config.get('save_csv', True)
            print(f"\nCSV saving is now {'On' if trading_config['save_csv'] else 'Off'}")
        
        # Validate and save new config
        is_valid, error = validate_trading_config(trading_config)
        if is_valid:
            if save_trading_config(trading_config):
                print("\nSettings updated successfully!")
            else:
                print("\nFailed to save settings!")
        else:
            print(f"\nInvalid configuration: {error}")
            # Reload last valid config
            trading_config = load_trading_config()

class AssetDataManager:
    def __init__(self):
        self.asset_data = {}  # Stores DataFrame for each asset
        self.last_update = {}  # Tracks last update time for each asset
        self._lock = asyncio.Lock()  # For thread-safe data updates
        self.processed_assets = set()  # Add tracking of processed assets
        self.state_file = "trading_state.json"  # Add state file path
        # Add status tracking
        self.fetch_count = 0
        self.last_status_print = datetime.now()
        self.skipped_trades = set()  # Add set to track skipped trade times

        # Create directory for debug CSV files
        # self.debug_dir = 'debug_data'
        # os.makedirs(self.debug_dir, exist_ok=True)

    async def update_asset_data(self, asset: str, new_data: pd.DataFrame):
        async with self._lock:
            if asset not in self.asset_data:
                self.asset_data[asset] = new_data
            else:
                # Merge existing and new data
                combined = pd.concat([self.asset_data[asset], new_data])
                # Remove duplicates and sort by time
                combined = combined.drop_duplicates(subset=['time'], keep='last').sort_values('time')
                self.asset_data[asset] = combined
            self.last_update[asset] = datetime.now()

            # Save to CSV for debugging
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"{self.debug_dir}/{asset}_{timestamp}.csv"
            # self.asset_data[asset].to_csv(filename, index=False)
            # print(f"Saved debug data for {asset} to {filename}")

    async def get_latest_data(self, asset: str) -> pd.DataFrame:
        async with self._lock:            
            df = self.asset_data.get(asset, pd.DataFrame())
            
            # Save to CSV when data is retrieved
            # if not df.empty:
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     filename = f"{self.debug_dir}/{asset}_retrieved_{timestamp}.csv"
            #     df.to_csv(filename, index=False)
            #     print(f"Saved retrieved data for {asset} to {filename}")
            
            return df

    async def save_state(self):
        """Save current state to file"""
        state = {
            'processed_assets': list(self.processed_assets),
            'last_update': {k: v.isoformat() for k, v in self.last_update.items()},
            'method': self.payout_method,
            'min_payout': self.min_payout
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    async def load_state(self):
        """Load state from file"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.processed_assets = set(state.get('processed_assets', []))
                self.payout_method = state.get('method')
                self.min_payout = state.get('min_payout', 70)
        except FileNotFoundError:
            pass

    async def get_status_summary(self):
        """Get concise status summary"""
        return {
            'total_assets': len(self.asset_data),
            'processed': len(self.processed_assets),
            'last_update': min(self.last_update.values()) if self.last_update else None,
            'fetch_count': self.fetch_count
        }

    async def is_trade_skipped(self, asset: str, trade_time: str) -> bool:
        """Check if this trade was already skipped"""
        trade_key = f"{asset}_{trade_time}"
        if trade_key in self.skipped_trades:
            return True
        return False

    async def mark_trade_skipped(self, asset: str, trade_time: str):
        """Mark this trade as skipped"""
        trade_key = f"{asset}_{trade_time}"
        self.skipped_trades.add(trade_key)

async def background_candle_fetcher(data_manager: AssetDataManager):
    """Continuously fetches candles for all assets"""
    status_interval = 60  # Print status every 30 seconds
    last_status_time = datetime.now()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Background fetcher started")  # Confirm start

    while True:
        try:
            current_time = datetime.now()
            
            # Print periodic status
            if (current_time - last_status_time).total_seconds() >= status_interval:
                status = await data_manager.get_status_summary()
                print(f"\n[{current_time.strftime('%H:%M:%S')}] Fetcher Status:")
                print(f"Assets: {status['total_assets']}/{len(data_manager.assets)} | Processed: {status['processed']}")
                print(f"Fetch cycles: {status['fetch_count']}")
                last_status_time = current_time

            # Get current payouts
            current_raw_payouts = await api.payout()
            active_assets = len(data_manager.assets)

            # Process each asset
            for asset in data_manager.assets[:]:  # Create copy to allow modification during iteration
                # Skip if payout no longer meets criteria
                payout_value = current_raw_payouts.get(asset, 0)

                if data_manager.payout_method == '1':
                    if payout_value < max(current_raw_payouts.values()):
                        print(f"↓ {asset} dropped (payout: {payout_value}%)")
                        data_manager.assets.remove(asset)
                        continue
                else:
                    if payout_value < data_manager.min_payout:
                        print(f"↓ {asset} dropped (payout: {payout_value}%)")
                        data_manager.assets.remove(asset)
                        continue

                # Fetch new candles
                new_candles = await get_candles(candle_ssid, asset, action, need_restart=False)
                await data_manager.update_asset_data(asset, new_candles)

                if asset in data_manager.processed_assets:
                    data_manager.processed_assets.remove(asset)

            data_manager.fetch_count += 1

            # If assets changed, log it
            if len(data_manager.assets) != active_assets:
                print(f"← Active assets: {len(data_manager.assets)}")

            # If all assets processed, start new cycle
            data_manager.processed_assets.clear()  # Clear processed assets
            
            # Get fresh list of assets based on current payouts
            if data_manager.payout_method == '1':
                new_assets = await get_best_payouts(api, True, log=False)
            else:
                new_assets = await get_best_payouts(api, False, data_manager.min_payout, log=False)
            
            data_manager.assets = new_assets  # Update shared assets list
            await data_manager.save_state()  # Save state after cycle reset

            # Ensure we have a consistent delay between cycles
            await asyncio.sleep(1)
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Fetcher error: {e}")
            traceback_str = traceback.format_exc()
            print(f"Full traceback:\n{traceback_str}")
            await asyncio.sleep(5)

async def signal_processor(data_manager: AssetDataManager):
    """Processes signals using latest available data"""
    global selected_strategy

    signals_checked = 0
    last_status_time = datetime.now()
    status_interval = 60  # Print status every 30 seconds
    MAX_PROCESSING_DELAY = 3  # Maximum allowed delay in seconds

    while True:
        try:
            current_time = datetime.now()
            
            # Print periodic status
            if (current_time - last_status_time).total_seconds() >= status_interval:
                print(f"\n[{current_time.strftime('%H:%M:%S')}] Signal Processor:")
                print(f"Signals checked: {signals_checked}")
                print(f"Processed assets: {len(data_manager.processed_assets)}")
                last_status_time = current_time

            for asset in data_manager.assets[:]:  # Create copy to allow modification
                if asset in data_manager.processed_assets:
                    continue

                df = await data_manager.get_latest_data(asset)
                if df.empty:
                    continue

                signals_checked += 1
                call_signal, put_signal, signal_time, trade_time = selected_strategy.check_trade_entry(df)
                
                if call_signal or put_signal:
                    # Check if this trade was already skipped
                    if await data_manager.is_trade_skipped(asset, trade_time):
                        continue

                    # TODO: Saving data when trading
                    # Create output directory
                    output_dir = 'candles_data'
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    allow_buy = False
                    allow_sell = False

                    if not df.empty:
                        structure = analyze_market_structure(df)
                        current_candle = df.iloc[-1]
                        current_price = current_candle['close']
                        current_idx = current_candle.name
                        signals = check_trade_conditions(current_price, current_idx, structure, proximity_pct=0.01) # 1% proximity
                        # The real decision your strategy will use:
                        current_decision = signals['decision_current']
                        print("\n[1] CURRENT DECISION (Based on Horizontal Levels Only)")
                        print(f"  Current price: {current_decision['current_price']}")
                        print(f"  Allow Buy?  -> {current_decision['allow_buy']}")
                        print(f"  Reason:        {current_decision['buy_reason']}")
                        print(f"  Allow Sell? -> {current_decision['allow_sell']}")
                        print(f"  Reason:        {current_decision['sell_reason']}")
                        allow_buy = current_decision['allow_buy']
                        allow_sell = current_decision['allow_sell']

                        # The hypothetical decision for your information:
                        trend_decision = signals['decision_with_trends']
                        print("\n[2] HYPOTHETICAL DECISION (If Trend Lines Were Rules)")
                        print(f"  Current price: {trend_decision['current_price']}")
                        print(f"  Allow Buy?  -> {trend_decision['allow_buy']}")
                        print(f"  Reason:        {trend_decision['buy_reason']}")
                        print(f"  Allow Sell? -> {trend_decision['allow_sell']}")
                        print(f"  Reason:        {trend_decision['sell_reason']}")

                        # filename = f"{output_dir}/{selected_strategy.__class__.__name__}_{asset}_{timestamp}.csv"
                        # await asyncio.sleep(0)
                        # df.to_csv(filename, index=False)
                        # await plot_market_structure(df, structure, asset, timestamp, selected_strategy.__class__.__name__)
                        # print(f"Saved candle data to {filename} at {datetime.now()}")
                    # END TODO
                    
                    signal_type = "CALL" if call_signal else "PUT"

                    # Convert trade_time to datetime and add 2 hours to current_time
                    trade_time_dt = pd.to_datetime(trade_time)
                    current_time = datetime.now(timezone.utc) + timedelta(hours=2)

                    if not trade_time_dt.tzinfo:
                        trade_time_dt = trade_time_dt.tz_localize('UTC')

                    time_diff = current_time - trade_time_dt
                    time_diff_seconds = abs(time_diff.total_seconds())
                    
                    print(f"\n[{current_time.strftime('%H:%M:%S')}] ↑ {signal_type} signal for {asset}")
                    print(f"Current time: {current_time}")
                    print(f"Signal time: {signal_time}")
                    print(f"Trade time: {trade_time}")
                    print(f"Processing delay: {time_diff_seconds:.2f} seconds")

                    # Check processing delay
                    if time_diff_seconds <= MAX_PROCESSING_DELAY:
                        if call_signal and allow_buy:
                            print(f"✓ Processing delay acceptable, executing trade")
                            await trade(api, api_v1, asset, 0)
                        elif put_signal and allow_sell:
                            print(f"✓ Processing delay acceptable, executing trade")
                            await trade(api, api_v1, asset, 1)
                        else:
                            print(f"⚠️ Skipped trade due to signal type mismatch")
                            
                            # Mark this trade as skipped
                            await data_manager.mark_trade_skipped(asset, trade_time)
                    else:
                        print(f"⚠️ Skipped trade due to high processing delay ({time_diff_seconds:.2f}s > {MAX_PROCESSING_DELAY}s)")

                        # Mark this trade as skipped
                        await data_manager.mark_trade_skipped(asset, trade_time)
                    
                # Mark asset as processed
                data_manager.processed_assets.add(asset)
                await data_manager.save_state()  # Save state after processing

            await asyncio.sleep(0.1)  # Wait before next check
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Processor error: {e}")
            traceback_str = traceback.format_exc()
            print(f"Full traceback:\n{traceback_str}")
            await asyncio.sleep(1)

async def run_continuous_trading(api, initial_assets: List[str], payout_method: str, min_payout: int = 70):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting continuous trading system")
    print(f"Initial assets: {len(initial_assets)}")
    print(f"Mode: {'Maximum payout' if payout_method == '1' else f'Minimum {min_payout}% payout'}")

    data_manager = AssetDataManager()
    data_manager.payout_method = payout_method
    data_manager.min_payout = min_payout
    data_manager.assets = initial_assets.copy()  # Initialize shared assets list

    # Load previous state
    await data_manager.load_state()
    
    # Create tasks for both processes
    print("\nInitializing background processes...")
    fetcher = asyncio.create_task(background_candle_fetcher(data_manager))
    processor = asyncio.create_task(signal_processor(data_manager))
    
    try:
        # Wait for both tasks (they'll run until cancelled)
        print("System running...")
        await asyncio.gather(fetcher, processor)
    except asyncio.CancelledError:
        print("\nShutting down...")
        fetcher.cancel()
        processor.cancel()
        await asyncio.gather(fetcher, processor, return_exceptions=True)
        await data_manager.save_state()  # Save state on exit

# Initialize trading config
trading_config = load_trading_config()

# Mute pd warnings
pd.set_option('future.no_silent_downcasting', True)

# Main part of the code
async def main(ssid: str, asset: str|list|None = None, action: str|None = None, is_demo: bool = True):
    """
    Main entry point for the trading bot. Handles different operation modes based on the action parameter.
    
    Args:
        ssid (str): Session ID for PocketOption authentication
        asset (str|list|None): Trading asset symbol(s) or None for auto-selection
        action (str|None): Operation mode ('1'=single asset, '2'=multi-asset, '3'=backtest)
    """
    global selected_strategy, api, api_v1, api_v1_prices, trading_config, trade_ssid, candle_ssid
    
    # Initialize strategy factory
    StrategyFactory.register_strategy(MACDStrategy)
    StrategyFactory.register_strategy(BollingerBandsStrategy)
    StrategyFactory.register_strategy(MACDSMAStrategy)
    StrategyFactory.register_strategy(FractalEMAStrategy)
    StrategyFactory.register_strategy(WilliamsRStrategy)
    StrategyFactory.register_strategy(FastPriceActionStrategy)
    
    # Initialize strategy if not yet set
    if not selected_strategy:
        if strategy_name:  # Use command line argument if available
            selected_strategy = StrategyFactory.get_strategy(strategy_name)
        else:
            selected_strategy = await select_strategy()

    # Initialize API instances if not already initialized
    candle_ssid = ssid
    if not is_demo:
        demo_SSID = [
            '''42["auth",{"session":"vtftn12e6f5f5008moitsd6skl","isDemo":1,"uid":27658142,"platform":2}]''',
            '''42["auth",{"session":"j079fsgog45pjnbsj9a2hvpnnb","isDemo":1,"uid":102766033,"platform":3,"isFastHistory":true}]''',
            '''42["auth",{"session":"upen8g2mcd3cvu5ai5i4jjl6si","isDemo":1,"uid":102365452,"platform":3,"isFastHistory":false}]''',
        ]
        candle_ssid = SSIDManager.acquire_ssid(demo_SSID)
        if candle_ssid is None:
            print("No demo SSIDs available for candle data. Please try again later.")
            return

        # Register cleanup
        atexit.register(SSIDManager.release_ssid, candle_ssid)
        
        # Start periodic update task
        update_task = asyncio.create_task(periodic_ssid_update(candle_ssid))
    
    trade_ssid = ssid
    if api is None or api_v1 is None or api_v1_prices is None:
        print("Initializing API connections...")
        api = PocketOptionAsync(candle_ssid)
        api_v1 = pocketoption(ssid, is_demo)
        api_v1_prices = pocketoption(candle_ssid)
        await asyncio.sleep(5)

    try:
        # If asset is a JSON string (from restart), try to parse it
        if isinstance(asset, str) and (asset.startswith('[') or asset.startswith('{')):
            try:
                data = json.loads(asset)
                if isinstance(data, dict) and data.get('type') == 'simple':
                    # Extract just the asset string for action '1'
                    asset = str(data['asset'])
                    if 'trading_config' in data:
                        trading_config.update(data['trading_config'])
                        save_trading_config(trading_config)
            except json.JSONDecodeError:
                pass

        if action == '1':
            if selected_strategy.__class__.__name__ == 'FastPriceActionStrategy':
                await run_price_action_strategy(api, api_v1, candle_ssid, asset, selected_strategy)
            else:
                count_loop = 0
                while True:
                    try:
                        count_loop += 1
                        # if count_loop > 1:
                        #     print("Entering 2nd loop, Restarting script...")
                        #     restart_script(ssid, asset, '1')

                        # Wait for next candle
                        await wait_for_next_candle()
                        
                        # Fetch and process candles
                        df_clean = await get_candles(candle_ssid, asset, action)
                        
                        # Check trade entry conditions using current strategy
                        call_signal, put_signal, signal_time, trade_time = selected_strategy.check_trade_entry(df_clean)
                        
                        if call_signal:
                            print(f"BUY CALL at {signal_time} at {datetime.now()} UTC")
                            await trade(api, api_v1, asset, 0)
                            # Reconnecting after each trade if in compound trading type
                            if trading_config['trading_type'] == 'compound':
                                api = PocketOptionAsync(candle_ssid)
                                api_v1_prices = pocketoption(candle_ssid)
                                await asyncio.sleep(5)
                        elif put_signal:
                            print(f"BUY PUT at {signal_time} at {datetime.now()} UTC")
                            await trade(api, api_v1, asset, 1)
                            # Reconnecting after each trade if in compound trading type
                            if trading_config['trading_type'] == 'compound':
                                api = PocketOptionAsync(candle_ssid)
                                api_v1_prices = pocketoption(candle_ssid)
                                await asyncio.sleep(5)
                    except KeyboardInterrupt:
                        # Re-raise to be caught by outer try block
                        raise
                    except Exception as e:
                        print(f"Error in trading loop: {e}")
                        print("Restarting script...")
                        restart_script(ssid, asset, action)

        elif action == '2':
            payouts = []
            if type(asset) == list:
                # For restarted runs, show remaining assets
                payouts = asset
                print("\nContinuing run with remaining assets:")
                current_raw_payouts = await api.payout()
                for asset_name in payouts:
                    payout_value = current_raw_payouts.get(asset_name, 0)
                    print(f"{asset_name}: {payout_value}%")
            else:
                # Get initial payouts
                raw_payouts = await api.payout()
                print("\n=== Available Assets and Payouts ===")
                for asset_name, payout in sorted(raw_payouts.items(), key=lambda x: x[1], reverse=True):
                    print(f"{asset_name}: {payout}%")
                
                # Ask user for payout selection method
                while True:
                    method = input("\nSelect trading method:\n1. Trade only assets with maximum payout\n2. Trade assets above minimum payout (70%)\nChoice (1/2): ").strip()
                    if method in ['1', '2']:
                        break
                    print("Invalid choice. Please enter 1 or 2.")
                
                if method == '1':
                    payouts = await get_best_payouts(api, True)  # get_max=True for maximum payout only
                    print(f"\nSelected {len(payouts)} assets with maximum payout for trading")
                else:
                    min_payout = input("\nEnter minimum payout percentage (default 70): ").strip()
                    try:
                        min_payout = int(min_payout) if min_payout else 70
                    except ValueError:
                        min_payout = 70
                        print("Invalid input. Using default minimum payout: 70%")
                    
                    payouts = await get_best_payouts(api, False, min_payout)  # get_max=False for minimum payout threshold
                    print(f"\nSelected {len(payouts)} assets with payout >= {min_payout}% for trading")

            if selected_strategy.__class__.__name__ == 'FastPriceActionStrategy':
                await run_price_action_strategy_multi(api, api_v1, candle_ssid, payouts, selected_strategy)
            else:
                try:
                    count_loop = 0
                    while payouts:  # Continue until all assets are processed
                        current_asset = payouts[0]  # Get the current asset
                        
                        # Get current payout for this asset
                        current_raw_payouts = await api.payout()
                        payout_value = current_raw_payouts.get(current_asset, 0)
                        print(f"\nProcessing asset: {current_asset} (Current Payout: {payout_value}%)")

                        # Wait for next candle only on first run, after restart, or after a trade
                        count_loop += 1
                        if count_loop == 1:
                            await wait_for_next_candle()
                        
                        # Fetch and process candles
                        df_clean = await get_candles(candle_ssid, current_asset, action, need_restart=False)

                        # Check trade entry conditions using current strategy
                        call_signal, put_signal, signal_time, trade_time = selected_strategy.check_trade_entry(df_clean)
                        
                        if call_signal:
                            print(f"BUY CALL at {signal_time} at {datetime.now()} UTC")
                            await trade(api, api_v1, current_asset, 0)
                            count_loop = 0  # Reset counter after trade
                        elif put_signal:
                            print(f"BUY PUT at {signal_time} at {datetime.now()} UTC")
                            await trade(api, api_v1, current_asset, 1)
                            count_loop = 0  # Reset counter after trade
                        else:
                            count_loop = -1  # No signal found, keep checking without waiting

                        # Remove the processed asset only if we got a signal or need to move to next asset
                        if count_loop == -1:
                            payouts.pop(0)
                            print(f"No signals found. Remaining assets: {len(payouts)}")
                        elif count_loop == 0:
                            payouts.pop(0)
                            print(f"Trade executed. Remaining assets: {len(payouts)}")

                    # When all assets are processed, return to menu
                    if not payouts:
                        print("\nAll assets processed. Returning to main menu...")
                        await show_menu(ssid)
                        return

                except KeyboardInterrupt:
                    # Save state before exiting
                    if payouts:
                        print("\nTrading interrupted. Saving state...")
                        restart_script(ssid, payouts, action)
                    raise
                except Exception as e:
                    print(f"Error in multi-asset mode: {e}")
                    if payouts:  # Only restart if there are remaining assets
                        print("\nError encountered. Saving state and restarting...")
                        restart_script(ssid, payouts, action)
                    raise

        elif action == '3':  # Continuous multi-pair trading
            continuous_mode = True
            payouts = []
            processed_assets = set()
            state_file = "trading_state.json"
            payout_method = None
            min_payout = 70  # Default minimum payout

            # If assets provided from backtesting, use them directly
            if isinstance(asset, list) and asset:
                payouts = asset.copy()  # Use the provided assets
                payout_method = 2
                print(f"\nStarting continuous trading with {len(payouts)} selected assets from backtesting")
                print("Assets:", ", ".join(payouts))
                
                # Save initial state
                with open(state_file, 'w') as f:
                    json.dump({
                        'assets': payouts,
                        'processed_assets': list(processed_assets)
                    }, f)
            else:
                # Load saved preferences
                if os.path.exists(state_file):
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                            payout_method = state.get('method')
                            min_payout = state.get('min_payout', 70)
                            processed_assets = set(state.get('processed_assets', []))
                            print(f"Restored trading preferences - Method: {'Maximum Payout' if payout_method == '1' else f'Minimum {min_payout}%'}")
                            print(f"Loaded {len(processed_assets)} previously processed assets")
                    except Exception as e:
                        print(f"Error loading state file: {e}")
                        payout_method = None

                # If no saved preferences, get initial preferences
                if payout_method is None:
                    print("\nFirst time setup - Please select your trading preferences")
                    while True:
                        method = input("\nSelect trading method:\n1. Trade only assets with maximum payout\n2. Trade assets above minimum payout (70%)\nChoice (1/2): ").strip()
                        if method in ['1', '2']:
                            payout_method = method
                            if method == '2':
                                min_input = input("\nEnter minimum payout percentage (default 70): ").strip()
                                try:
                                    min_payout = int(min_input) if min_input else 70
                                except ValueError:
                                    min_payout = 70
                                    print("Invalid input. Using default minimum payout: 70%")

                            # Save initial preferences
                            with open(state_file, 'w') as f:
                                json.dump({
                                    'method': payout_method,
                                    'min_payout': min_payout,
                                    'processed_assets': list(processed_assets)
                                }, f)
                                
                            # Save trading config with CSV preference
                            save_trading_config(trading_config)
                            break
                        print("Invalid choice. Please enter 1 or 2.")
            
            while continuous_mode:
                try:
                    if not payouts:  # Get new batch of assets when list is empty
                        if isinstance(asset, list) and asset:
                            # If trading backtested assets, reset the list instead of getting new assets
                            payouts = asset.copy()
                            processed_assets.clear()
                            print("\nResetting to initial backtested assets for new cycle")
                        else:
                            # Update payouts while preserving preferences
                            raw_payouts = await api.payout()
                            print("\n=== Current Available Assets and Payouts ===")
                            for asset_name, payout in sorted(raw_payouts.items(), key=lambda x: x[1], reverse=True):
                                print(f"{asset_name}: {payout}%")
                            
                            # Use saved preferences to get assets
                            all_payouts = []
                            if payout_method == '1':
                                all_payouts = await get_best_payouts(api, True)
                                method_desc = "maximum payout"
                            else:
                                all_payouts = await get_best_payouts(api, False, min_payout)
                                method_desc = f"payout >= {min_payout}%"

                            # Filter out processed assets
                            payouts = [p for p in all_payouts if p not in processed_assets]
                            
                            # If no unprocessed assets left, reset and start new cycle
                            if not payouts:
                                print("\nAll assets have been processed. Starting new trading cycle...")
                                processed_assets.clear()
                                payouts = all_payouts
                                # Save state to reflect reset
                                with open(state_file, 'w') as f:
                                    json.dump({
                                        'method': payout_method,
                                        'min_payout': min_payout,
                                        'processed_assets': []
                                    }, f)
                                
                            print(f"\nSelected {len(payouts)} assets with {method_desc} for trading")

                    count_loop = 0
                    while payouts:  # Continue until all assets are processed
                        current_asset = payouts[0]  # Get the current asset
                        
                        # Get current payout and validate
                        current_raw_payouts = await api.payout()
                        payout_value = current_raw_payouts.get(current_asset, 0)
                        
                        # Skip if payout no longer meets criteria
                        if (payout_method == '1' and payout_value < max(current_raw_payouts.values())) or \
                           (payout_method == '2' and payout_value < min_payout):
                            payouts.pop(0)
                            print(f"Skipping {current_asset} - Payout {payout_value}% no longer meets criteria")
                            continue
                        
                        print(f"\nProcessing asset: {current_asset} (Current Payout: {payout_value}%)")

                        # Wait for next candle with error handling
                        count_loop += 1
                        if count_loop == 1:
                            try:
                                await wait_for_next_candle()
                            except Exception as e:
                                print(f"Error waiting for next candle: {e}")
                                await asyncio.sleep(1)
                        
                        # Fetch and process candles with retry
                        retry_count = 0
                        while retry_count < 3:
                            try:
                                df_clean = await get_candles(candle_ssid, current_asset, action, need_restart=False)
                                break
                            except Exception as e:
                                retry_count += 1
                                if retry_count == 3:
                                    print(f"Failed to get candles after 3 retries: {e}")
                                    payouts.pop(0)  # Skip this asset
                                    break
                                print(f"Error getting candles (attempt {retry_count}/3): {e}")
                                await asyncio.sleep(1)

                        if retry_count == 3:
                            continue

                        # Check trade entry conditions
                        call_signal, put_signal, signal_time, trade_time = selected_strategy.check_trade_entry(df_clean)
                        
                        if call_signal:
                            print(f"BUY CALL at {signal_time} at {datetime.now()} UTC")
                            await trade(api,api_v1, current_asset, 0)
                            count_loop = 0
                            payouts.pop(0)  # Move to next asset after trade
                            # Add to processed assets and update state after successful trade
                            processed_assets.add(current_asset)
                            # Save state to file
                            with open(state_file, 'w') as f:
                                json.dump({
                                    'method': payout_method,
                                    'min_payout': min_payout,
                                    'processed_assets': list(processed_assets)
                                }, f)
                        elif put_signal:
                            print(f"BUY PUT at {signal_time} at {datetime.now()} UTC")
                            await trade(api,api_v1, current_asset, 1)
                            count_loop = 0
                            payouts.pop(0)  # Move to next asset after trade
                            # Add to processed assets and update state after successful trade
                            processed_assets.add(current_asset)
                            # Save state to file
                            with open(state_file, 'w') as f:
                                json.dump({
                                    'method': payout_method,
                                    'min_payout': min_payout,
                                    'processed_assets': list(processed_assets)
                                }, f)
                        else:
                            # No signal found
                            payouts.pop(0)
                            # Add to processed assets and update state even when no trade was made
                            processed_assets.add(current_asset)
                            print(f"No signals found. Remaining assets: {len(payouts)}")

                except KeyboardInterrupt:
                    print("\nTrading interrupted. Saving state...")
                    # Save state before restart
                    with open(state_file, 'w') as f:
                        json.dump({
                            'method': payout_method,
                            'min_payout': min_payout,
                            'processed_assets': list(processed_assets)
                        }, f)
                    restart_script(ssid, {'method': payout_method, 'min_payout': min_payout}, action)  # Save preferences for next run
                    continuous_mode = False
                except Exception as e:
                    print(f"\nError in continuous mode: {e}")
                    traceback_str = traceback.format_exc()
                    print(f"Full traceback:\n{traceback_str}")
                    # Save state before brief pause
                    with open(state_file, 'w') as f:
                        json.dump({
                            'method': payout_method,
                            'min_payout': min_payout,
                            'processed_assets': list(processed_assets)
                        }, f)
                    restart_script(ssid, {'method': payout_method, 'min_payout': min_payout}, action)
                    await asyncio.sleep(5)  # Brief pause before continuing

        elif action == '4':  # Continuous multi-pair trading with simultaneous
            continuous_mode = True
            payouts = []
            processed_assets = set()
            state_file = "trading_state.json"
            payout_method = None
            min_payout = 70  # Default minimum payout

            # If assets provided from backtesting, use them directly
            if isinstance(asset, list) and asset:
                payouts = asset.copy()  # Use the provided assets
                print(f"\nStarting simultaneous trading with {len(payouts)} selected assets from backtesting")
                print("Assets:", ", ".join(payouts))
                
                # Save initial state
                with open(state_file, 'w') as f:
                    json.dump({
                        'assets': payouts,
                        'processed_assets': list(processed_assets)
                    }, f)
            else:
                # Load saved preferences
                if os.path.exists(state_file):
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                            payout_method = state.get('method')
                            min_payout = state.get('min_payout', 70)
                            processed_assets = set(state.get('processed_assets', []))
                            print(f"Restored trading preferences - Method: {'Maximum Payout' if payout_method == '1' else f'Minimum {min_payout}%'}")
                            print(f"Loaded {len(processed_assets)} previously processed assets")
                    except Exception as e:
                        print(f"Error loading state file: {e}")
                        payout_method = None

                # If no saved preferences, get initial preferences
                if payout_method is None:
                    print("\nFirst time setup - Please select your trading preferences")
                    while True:
                        method = input("\nSelect trading method:\n1. Trade only assets with maximum payout\n2. Trade assets above minimum payout (70%)\nChoice (1/2): ").strip()
                        if method in ['1', '2']:
                            payout_method = method
                            if method == '2':
                                min_input = input("\nEnter minimum payout percentage (default 70): ").strip()
                                try:
                                    min_payout = int(min_input) if min_input else 70
                                except ValueError:
                                    min_payout = 70
                                    print("Invalid input. Using default minimum payout: 70%")

                            # Save initial preferences
                            with open(state_file, 'w') as f:
                                json.dump({
                                    'method': payout_method,
                                    'min_payout': min_payout,
                                    'processed_assets': list(processed_assets)
                                }, f)
                                
                            # Save trading config with CSV preference
                            save_trading_config(trading_config)
                            break
                        print("Invalid choice. Please enter 1 or 2.")
            
            while continuous_mode:
                try:
                    if not payouts:  # Get new batch of assets when list is empty
                        if isinstance(asset, list) and asset:
                            # If trading backtested assets, reset the list instead of getting new assets
                            payouts = asset.copy()
                            processed_assets.clear()
                            print("\nResetting to initial backtested assets for new cycle")
                        else:
                            # Update payouts while preserving preferences
                            raw_payouts = await api.payout()
                            print("\n=== Current Available Assets and Payouts ===")
                            for asset_name, payout in sorted(raw_payouts.items(), key=lambda x: x[1], reverse=True):
                                print(f"{asset_name}: {payout}%")
                            
                            # Use saved preferences to get assets
                            all_payouts = []
                            if payout_method == '1':
                                all_payouts = await get_best_payouts(api, True)
                                method_desc = "maximum payout"
                            else:
                                all_payouts = await get_best_payouts(api, False, min_payout)
                                method_desc = f"payout >= {min_payout}%"

                            # Filter out processed assets
                            payouts = [p for p in all_payouts if p not in processed_assets]
                            
                            # If no unprocessed assets left, reset and start new cycle
                            if not payouts:
                                print("\nAll assets have been processed. Starting new trading cycle...")
                                processed_assets.clear()
                                payouts = all_payouts
                                # Save state to reflect reset
                                with open(state_file, 'w') as f:
                                    json.dump({
                                        'method': payout_method,
                                        'min_payout': min_payout,
                                        'processed_assets': []
                                    }, f)
                                
                            print(f"\nSelected {len(payouts)} assets with {method_desc} for trading")

                    # Get current payouts and validate
                    current_raw_payouts = await api.payout()
                    valid_assets = []
                    for asset in payouts[:]:
                        payout_value = current_raw_payouts.get(asset, 0)
                        if (payout_method == '1' and payout_value < max(current_raw_payouts.values())) or \
                        (payout_method == '2' and payout_value < min_payout):
                            payouts.remove(asset)
                            print(f"Skipping {asset} - Payout {payout_value}% no longer meets criteria")
                        else:
                            valid_assets.append(asset)
                            print(f"Valid asset: {asset} (Payout: {payout_value}%)")

                    if not valid_assets:
                        print("No valid assets remaining. Starting new cycle...")
                        processed_assets.clear()
                        payouts = []  # Will trigger new asset fetch on next iteration
                        continue

                    print(f"\nProcessing {len(valid_assets)} assets simultaneously...")

                    # Wait for next candle
                    try:
                        await wait_for_next_candle()
                    except Exception as e:
                        print(f"Error waiting for next candle: {e}")
                        await asyncio.sleep(1)
                        continue

                    # Fetch data for all valid assets simultaneously
                    print("\nFetching candle data...")
                    assets_data = await get_candles_simultaneously(ssid, valid_assets, action, need_restart=False)
                    
                    if not assets_data:
                        print("Failed to retrieve data for any assets. Retrying...")
                        api = PocketOptionAsync(candle_ssid)
                        api_v1_prices = pocketoption(candle_ssid)
                        await asyncio.sleep(5)
                        continue

                    print(f"\nSuccessfully retrieved data for {len(assets_data)}/{len(valid_assets)} assets")
                    
                    # Process signals and execute trades
                    signal_results = []
                    for asset_name, df in assets_data.items():
                        try:
                            call_signal, put_signal, signal_time, trade_time = selected_strategy.check_trade_entry(df)
                            
                            if call_signal or put_signal:
                                signal_type = "CALL" if call_signal else "PUT"
                                payout = current_raw_payouts.get(asset_name, "Unknown")
                                signal_results.append({
                                    'asset': asset_name,
                                    'signal': signal_type,
                                    'time': signal_time,
                                    'payout': payout
                                })
                        except Exception as e:
                            print(f"Error checking signals for {asset_name}: {e}")
                            continue

                    # Sort results by payout
                    signal_results.sort(key=lambda x: x['payout'] if isinstance(x['payout'], (int, float)) else 0, reverse=True)

                    if signal_results:
                        print(f"\n=== Trading Signals Detected ({len(signal_results)}) ===")
                        print(f"{'Asset':<10} | {'Signal':<5} | {'Time':<20} | {'Payout':<7}")
                        print("-" * 50)
                        
                        for result in signal_results:
                            print(f"{result['asset']:<10} | {result['signal']:<5} | {result['time']:<20} | {result['payout']}%")
                        
                        # Execute trades based on trading type
                        trade_tasks = []
                        for result in signal_results:
                            asset_name = result['asset']
                            signal_type = result['signal']
                            command = 0 if signal_type == "CALL" else 1
                            
                            print(f"Setting up trade for {asset_name} - {signal_type}")
                            trade_tasks.append(trade(api, api_v1, asset_name, command))

                        # Execute all trades simultaneously
                        try:
                            if trading_config['trading_type'] == 'compound' or trading_config['trading_type'] == 'progressive_compound':
                                trade_results = await asyncio.gather(*trade_tasks)
                                if all(trade_results):
                                    print("\nAll trades successful! Advancing compound/step/stage level.")
                                else:
                                    print("\nSome trades failed. Resetting compound/step/stage level.")
                            else:
                                await asyncio.gather(*trade_tasks)
                        except Exception as e:
                            print(f"\nError executing trades: {e}")

                    # Update processed assets
                    for asset_name in assets_data.keys():
                        processed_assets.add(asset_name)
                        if asset_name in payouts:
                            payouts.remove(asset_name)

                    # Save state
                    with open(state_file, 'w') as f:
                        json.dump({
                            'method': payout_method,
                            'min_payout': min_payout,
                            'processed_assets': list(processed_assets)
                        }, f)

                    # Brief pause before next iteration
                    await asyncio.sleep(1)

                except KeyboardInterrupt:
                    print("\nTrading interrupted. Saving state...")
                    with open(state_file, 'w') as f:
                        json.dump({
                            'method': payout_method,
                            'min_payout': min_payout,
                            'processed_assets': list(processed_assets)
                        }, f)
                    restart_script(ssid, {'method': payout_method, 'min_payout': min_payout}, action)
                    continuous_mode = False
                except Exception as e:
                    print(f"\nError in continuous mode: {e}")
                    traceback_str = traceback.format_exc()
                    print(f"Full traceback:\n{traceback_str}")
                    with open(state_file, 'w') as f:
                        json.dump({
                            'method': payout_method,
                            'min_payout': min_payout,
                            'processed_assets': list(processed_assets)
                        }, f)
                    restart_script(ssid, {'method': payout_method, 'min_payout': min_payout}, action)
                    await asyncio.sleep(5)

        elif action == '5':  # Multi-asset Analysis
            # Get current payouts
            raw_payouts = await api.payout()
            print("\n=== Available Assets and Payouts ===")
            for asset_name, payout in sorted(raw_payouts.items(), key=lambda x: x[1], reverse=True):
                print(f"{asset_name}: {payout}%")
            
            # Let user select assets or choose by payout
            selection_method = input("\nSelect assets by:\n1. Manual selection\n2. Top payout\n3. Minimum payout threshold\nChoice (1-3): ").strip()
            
            selected_assets = []
            
            if selection_method == '1':
                # Manual selection
                print("\nEnter asset symbols one by one. Type 'done' when finished.")
                while True:
                    asset_input = input("Asset symbol (or 'done'): ").strip()
                    if asset_input.lower() == 'done':
                        break
                    if asset_input in raw_payouts:
                        selected_assets.append(asset_input)
                        print(f"Added {asset_input} (Payout: {raw_payouts[asset_input]}%)")
                    else:
                        print(f"Asset {asset_input} not found. Please try again.")
                        
            elif selection_method == '2':
                # Top N assets by payout
                try:
                    top_n = int(input("\nEnter number of top assets to analyze: ").strip())
                    sorted_assets = sorted(raw_payouts.items(), key=lambda x: x[1], reverse=True)
                    selected_assets = [asset for asset, _ in sorted_assets[:top_n]]
                    print(f"\nSelected top {len(selected_assets)} assets")
                except ValueError:
                    print("Invalid input. Defaulting to top 5 assets.")
                    sorted_assets = sorted(raw_payouts.items(), key=lambda x: x[1], reverse=True)
                    selected_assets = [asset for asset, _ in sorted_assets[:5]]

            elif selection_method == '3':
                # By minimum payout threshold
                try:
                    min_payout = int(input("\nEnter minimum payout percentage: ").strip())
                    selected_assets = [asset for asset, payout in raw_payouts.items() if payout >= min_payout]
                    print(f"\nSelected {len(selected_assets)} assets with payout >= {min_payout}%")
                except ValueError:
                    print("Invalid input. Defaulting to minimum 70% payout.")
                    selected_assets = [asset for asset, payout in raw_payouts.items() if payout >= 70]
            
            if not selected_assets:
                print("No assets selected. Returning to main menu.")
                return

            # Ask for number of trades based on trading style
            trades_to_execute = None
            if trading_config['trading_type'] == 'fixed':
                user_input = input("\nEnter number of trades to execute (press Enter for all signals): ").strip()
                if user_input:
                    try:
                        trades_to_execute = int(user_input)
                        if trades_to_execute <= 0:
                            print("Invalid number. Returning to main menu.")
                            return
                    except ValueError:
                        print("Invalid input. Returning to main menu.")
                        return
            else:  # Compound trading
                user_input = input("\nEnter number of trades to execute (required for compound trading): ").strip()
                try:
                    trades_to_execute = int(user_input)
                    if trades_to_execute <= 0:
                        print("Invalid number. Returning to main menu.")
                        return
                except ValueError:
                    print("Invalid input. Returning to main menu.")
                    return
                    
            print(f"\nAnalyzing {len(selected_assets)} assets: {', '.join(selected_assets[:5])}" + 
                (f" and {len(selected_assets)-5} more..." if len(selected_assets) > 5 else ""))
            
            # Wait for next candle to ensure we have fresh data
            await wait_for_next_candle()
                
            # Fetch data for all selected assets simultaneously
            print("\nFetching candle data for all selected assets...")
            assets_data = await get_candles_simultaneously(ssid, selected_assets, action)
            
            if not assets_data:
                print("Failed to retrieve data for any assets. Returning to menu.")
                api = PocketOptionAsync(candle_ssid)
                api_v1_prices = pocketoption(candle_ssid)
                await asyncio.sleep(5)
                return
                
            print(f"\nSuccessfully retrieved data for {len(assets_data)}/{len(selected_assets)} assets")
            
            # Check for signals across all assets
            print("\nAnalyzing for trading signals...")
            signal_results = []
            
            for asset_name, df in assets_data.items():
                call_signal, put_signal, signal_time, trade_time = selected_strategy.check_trade_entry(df)
                
                if call_signal or put_signal:
                    signal_type = "CALL" if call_signal else "PUT"
                    payout = raw_payouts.get(asset_name, "Unknown")
                    signal_results.append({
                        'asset': asset_name,
                        'signal': signal_type,
                        'time': signal_time,
                        'payout': payout
                    })
            
            # Sort results by payout
            signal_results.sort(key=lambda x: x['payout'] if isinstance(x['payout'], (int, float)) else 0, reverse=True)
            
            # Display results and execute trades
            if signal_results:
                print(f"\n=== Trading Signals Detected ({len(signal_results)}) ===")
                print(f"{'Asset':<10} | {'Signal':<5} | {'Time':<20} | {'Payout':<7}")
                print("-" * 50)
                
                for result in signal_results:
                    print(f"{result['asset']:<10} | {result['signal']:<5} | {result['time']:<20} | {result['payout']}%")
                
                if trading_config['trading_type'] == 'fixed':
                    # Execute trades for fixed trading
                    trades_to_process = signal_results[:trades_to_execute] if trades_to_execute else signal_results
                    print(f"\nExecuting {len(trades_to_process)} trades simultaneously...")
                    
                    # Create tasks for all trades
                    trade_tasks = []
                    for result in trades_to_process:
                        asset_name = result['asset']
                        signal_type = result['signal']
                        command = 0 if signal_type == "CALL" else 1
                        
                        print(f"Setting up trade for {asset_name} - {signal_type}")
                        trade_tasks.append(trade(api, api_v1, asset_name, command))
                    
                    # Execute all trades simultaneously
                    try:
                        await asyncio.gather(*trade_tasks)
                        print(f"\nCompleted {len(trades_to_process)} trades. Returning to main menu.")
                    except Exception as e:
                        print(f"\nError executing trades: {e}")
                    
                else:
                    # Execute trades for compound trading
                    trades_to_process = signal_results[:trades_to_execute]
                    if len(trades_to_process) < trades_to_execute:
                        print(f"\nWarning: Only {len(trades_to_process)} signals available.")
                    
                    print(f"\nExecuting {len(trades_to_process)} compound trades simultaneously...")
                    
                    # Create tasks for all trades
                    trade_tasks = []
                    for result in trades_to_process:
                        asset_name = result['asset']
                        signal_type = result['signal']
                        command = 0 if signal_type == "CALL" else 1
                        
                        print(f"Setting up trade for {asset_name} - {signal_type}")
                        trade_tasks.append(trade(api, api_v1, asset_name, command))
                    
                    # Execute all trades simultaneously
                    try:
                        trade_results = await asyncio.gather(*trade_tasks)
                        
                        # Check if all trades were successful
                        if all(trade_results):
                            print("\nAll trades successful! Advancing compound level.")
                            # Compound level advancement is handled in the trade function
                        else:
                            print("\nSome trades failed. Resetting compound level.")
                            # Reset is handled in the trade function
                        
                        print(f"\nCompleted {len(trades_to_process)} compound trades. Returning to main menu.")
                        
                    except Exception as e:
                        print(f"\nError executing compound trades: {e}")
                        print("Resetting compound level due to error.")
                        # Reset compound level on error
                        trading_config['current_amount'] = trading_config['initial_amount']
                        trading_config['consecutive_wins'] = 0
                        trading_config['current_compound_level'] = 0
                        save_trading_config(trading_config)
            else:
                print("\nNo trading signals detected across selected assets.")
                
            return

        elif action == '6':  # Backtesting
            # Handle the initial asset parameter which could be:
            # 1. None (fresh start)
            # 2. A list of payouts to test
            # 3. A tuple of (payouts, previous_results) if we're continuing after a restart
            payouts = []
            results = []
            
            if asset is None:
                # Get initial payouts
                raw_payouts = await api.payout()
                print("\n=== Available Assets and Payouts ===")
                for asset_name, payout in sorted(raw_payouts.items(), key=lambda x: x[1], reverse=True):
                    print(f"{asset_name}: {payout}%")
                payouts = await get_best_payouts(api)
                print(f"\nSelected {len(payouts)} assets for testing with payout >= 70%")
            elif isinstance(asset, tuple) and len(asset) == 2:
                payouts, results = asset
            elif isinstance(asset, list):
                # For restarted runs, show remaining assets
                payouts = asset
                print(f"\nContinuing test with {len(payouts)} remaining assets")

            # Ask for time range preference
            print("\nSelect time range for backtesting:")
            print("1. Full history")
            print("2. Recent time period")
            time_choice = input("Choose option (1 or 2): ").strip()

            time_filter = None
            period_choice = None
            custom_hours = None
            time_period = 'Full history'  # Default time period description

            # Add reference time to pass to backtest function
            reference_time = None

            if time_choice == "2":
                print("\nSelect time period:")
                print("1. Last 5 minutes")
                print("2. Last 15 minutes")
                print("3. Last 30 minutes")
                print("4. Last hour")
                print("5. Last 4 hours")
                print("6. Last 8 hours")
                print("7. Last 12 hours")
                print("8. Last 24 hours")
                print("9. Custom hours")
                
                period_choice = input("Choose period (1-9): ").strip()
                
                time_map = {
                    "1": timedelta(minutes=5),
                    "2": timedelta(minutes=15),
                    "3": timedelta(minutes=30),
                    "4": timedelta(hours=1),
                    "5": timedelta(hours=4),
                    "6": timedelta(hours=8),
                    "7": timedelta(hours=12),
                    "8": timedelta(hours=24)
                }

                # Set time period description
                time_period_map = {
                    "1": "Last 5 minutes",
                    "2": "Last 15 minutes",
                    "3": "Last 30 minutes",
                    "4": "Last hour",
                    "5": "Last 4 hours",
                    "6": "Last 8 hours",
                    "7": "Last 12 hours",
                    "8": "Last 24 hours"
                }

            for current_asset in payouts[:]:
                try:
                    df_clean = await get_candles(candle_ssid, current_asset, action, need_restart=False)
                    
                    payouts.remove(current_asset)
                    
                    if not df_clean.empty:
                        # Convert time column to UTC if it's not already
                        if 'time' in df_clean.columns:
                            # Convert time to datetime if it's not already
                            if not pd.api.types.is_datetime64_any_dtype(df_clean['time']):
                                df_clean['time'] = pd.to_datetime(df_clean['time'])
                                
                            # Add timezone information if not present
                            if not df_clean['time'].dt.tz:
                                df_clean['time'] = df_clean['time'].dt.tz_localize('UTC')
                            elif df_clean['time'].dt.tz != timezone.utc:
                                df_clean['time'] = df_clean['time'].dt.tz_convert('UTC')

                            # Calculate time filter based on the latest data point
                            if time_choice == "2" and period_choice:
                                reference_time = df_clean['time'].max()
                                if period_choice in time_map:
                                    time_filter = reference_time - time_map[period_choice]
                                    time_period = time_period_map[period_choice]
                                elif period_choice == "9":
                                    custom_hours = float(input("\nEnter number of hours to look back: "))
                                    if custom_hours > 0:
                                        time_filter = reference_time - timedelta(hours=custom_hours)
                                        time_period = f"Last {custom_hours} hours"
                                    else:
                                        print("Invalid hours input")
                                        continue

                                # Filter the data based on time_filter
                                df_clean = df_clean[
                                    (df_clean['time'] >= time_filter) & 
                                    (df_clean['time'] <= reference_time)
                                ]
                                
                                if not df_clean.empty:
                                    print(f"\nAnalyzing {current_asset} from {df_clean['time'].min()} to {df_clean['time'].max()}")
                                    print(f"Total candles in period: {len(df_clean)}")
                                else:
                                    print(f"No data for {current_asset} in specified time period")
                                    continue

                        win_rate, total_trades, wins, losses = backtest(df_clean, current_asset, time_filter, reference_time)
                        results.append((current_asset, win_rate, total_trades, wins, losses))

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Error in backtest mode: {e}")
                    traceback_str = traceback.format_exc()
                    print(f"Full traceback:\n{traceback_str}")
                    # Package both payouts and current results before restart
                    restart_script(ssid, (payouts, results), action)

            # Show final results only when all assets are processed
            if not payouts and results:  # Only show when we've processed all assets
                print('\n===Final Backtest Results (Sorted by Win Rate)===')
                print(f"Time period: {time_period}")
                results.sort(key=lambda x: x[1], reverse=True)  # Sort by win rate
                for asset_name, win_rate, total_trades, wins, losses in results:
                    print(f"{asset_name}: Win Rate = {win_rate:.2f}%, T = {total_trades}, W = {wins}, L = {losses}")
                
                # Ask if user wants to trade with profitable assets
                while True:
                    trade_choice = input("\nWould you like to trade with profitable assets? (y/n): ").strip().lower()
                    if trade_choice in ['y', 'n']:
                        break
                    print("Please enter 'y' for yes or 'n' for no.")
                    
                if trade_choice == 'y':
                    # Ask for minimum win rate
                    while True:
                        try:
                            min_win_rate = float(input("\nEnter minimum win rate percentage (0-100): "))
                            if 0 <= min_win_rate <= 100:
                                break
                            print("Please enter a number between 0 and 100")
                        except ValueError:
                            print("Please enter a valid number")
                    
                    # Filter assets based on win rate
                    selected_assets = [asset for asset, win_rate, _, _, _ in results if win_rate >= min_win_rate]
                    
                    if not selected_assets:
                        print(f"\nNo assets found with win rate >= {min_win_rate}%")
                        print("Returning to main menu...")
                        await show_menu(ssid)
                        return
                        
                    print(f"\nSelected {len(selected_assets)} assets with win rate >= {min_win_rate}%:")
                    for asset in selected_assets:
                        result = next(r for r in results if r[0] == asset)
                        print(f"{asset}: Win Rate = {result[1]:.2f}%, T = {result[2]}, W = {result[3]}, L = {result[4]}")
                        
                    # Ask for trading mode
                    print("\nSelect trading mode:")
                    print("1. Regular multi-asset trading")
                    print("2. Simultaneous multi-asset trading")
                    
                    while True:
                        mode_choice = input("Choose mode (1 or 2): ").strip()
                        if mode_choice in ['1', '2']:
                            break
                        print("Please enter 1 or 2")
                    
                    # Start trading with selected assets
                    try:
                        if mode_choice == '1':
                            print("\nStarting regular multi-asset trading...")
                            await main(ssid, selected_assets, '3', is_demo)  # Use action '3' for continuous trading
                        else:
                            print("\nStarting simultaneous multi-asset trading...")
                            await main(ssid, selected_assets, '4', is_demo)  # Use action '4' for simultaneous trading
                            
                    except KeyboardInterrupt:
                        print("\nTrading stopped by user")
                        print("Returning to main menu...")
                        await show_menu(ssid)
                        return
                    except Exception as e:
                        print(f"\nError starting trading: {e}")
                        traceback_str = traceback.format_exc()
                        print(f"Full traceback:\n{traceback_str}")
                        print("Returning to main menu...")
                        await show_menu(ssid)
                        return
                else:
                    print("\nReturning to main menu...")
                    await show_menu(ssid)
                    return
            elif not results:
                print("No results to display.")
                print("\nReturning to main menu...")
                await show_menu(ssid)
                return

        elif action == '7':
            # List available CSV files
            print("\nAvailable CSV files in candles_data folder:")
            csv_files = [f for f in os.listdir('candles_data') if f.endswith('.csv')]
            for i, file in enumerate(csv_files, 1):
                print(f"{i}. {file}")
                
            filename = input('\nEnter the name of the CSV file to analyze: ').strip()
            await analyze_csv_file(filename)
            
            # Return to menu after analysis
            print("\nAnalysis completed. Returning to main menu...")
            await show_menu(ssid)
            return

        elif action == '8':  # New background fetching mode
            continuous_mode = True
            payouts = []
            processed_assets = set()
            state_file = "trading_state.json"
            payout_method = None
            min_payout = 70

            # If assets provided from backtesting, use them directly
            if isinstance(asset, list) and asset:
                payouts = asset.copy()  # Use the provided assets
                payout_method = 2
                print(f"\nStarting continuous trading with {len(payouts)} selected assets from backtesting")
                print("Assets:", ", ".join(payouts))
                
                # Save initial state
                with open(state_file, 'w') as f:
                    json.dump({
                        'assets': payouts,
                        'processed_assets': list(processed_assets)
                    }, f)
            else:
                # Load or get initial preferences for asset selection
                if os.path.exists(state_file):
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                            payout_method = state.get('method')
                            min_payout = state.get('min_payout', 70)
                            processed_assets = set(state.get('processed_assets', []))
                            print(f"Restored trading preferences - Method: {'Maximum Payout' if payout_method == '1' else f'Minimum {min_payout}%'}")
                            print(f"Loaded {len(processed_assets)} previously processed assets")
                    except Exception as e:
                        print(f"Error loading state file: {e}")
                        payout_method = None

                # If no saved preferences, get initial preferences
                if payout_method is None:
                    print("\nFirst time setup - Please select your trading preferences")
                    while True:
                        method = input("\nSelect trading method:\n1. Trade only assets with maximum payout\n2. Trade assets above minimum payout (70%)\nChoice (1/2): ").strip()
                        if method in ['1', '2']:
                            payout_method = method
                            if method == '2':
                                min_input = input("\nEnter minimum payout percentage (default 70): ").strip()
                                try:
                                    min_payout = int(min_input) if min_input else 70
                                except ValueError:
                                    min_payout = 70
                                    print("Invalid input. Using default minimum payout: 70%")

                            # Save initial preferences
                            with open(state_file, 'w') as f:
                                json.dump({
                                    'method': payout_method,
                                    'min_payout': min_payout,
                                    'processed_assets': list(processed_assets)
                                }, f)
                                
                            # Save trading config with CSV preference
                            save_trading_config(trading_config)
                            break
                        print("Invalid choice. Please enter 1 or 2.")

                # Get initial assets based on preferences
                raw_payouts = await api.payout()
                if payout_method == '1':
                    payouts = await get_best_payouts(api, True)
                    print(f"\nSelected assets with maximum payout")
                else:
                    payouts = await get_best_payouts(api, False, min_payout)
                    print(f"\nSelected assets with payout >= {min_payout}%")

            if not payouts:
                print("No valid assets to trade. Returning to main menu...")
                return

            print(f"\nStarting background fetching mode with {len(payouts)} assets")
            print("Assets:", ", ".join(payouts))

            try:
                data_manager = AssetDataManager()
                
                # Create and run the continuous trading tasks
                await run_continuous_trading(api, payouts, payout_method, min_payout)
                
            except KeyboardInterrupt:
                print("\nTrading interrupted by user")
            except Exception as e:
                print(f"Error in background fetching mode: {e}")
                traceback_str = traceback.format_exc()
                print(f"Full traceback:\n{traceback_str}")
        else:
            print("Invalid action. Please choose 1, 2, 3, 4, 5, 6, 7, or 8.")

    except KeyboardInterrupt:
        print("\nReturning to main menu...")

        if not is_demo:
            update_task.cancel()  # Cancel the update task
            SSIDManager.release_ssid(candle_ssid)  # Release the demo SSID
    except Exception as e:
        print(f"\nError in main: {e}")
        traceback_str = traceback.format_exc()
        print(f"Full traceback:\n{traceback_str}")

        if not is_demo:
            update_task.cancel()  # Cancel the update task
            SSIDManager.release_ssid(candle_ssid)  # Release the demo SSID
    finally:
        # Cleanup
        if not is_demo:
            try:
                update_task.cancel()  # Ensure update task is cancelled
            except:
                pass
            SSIDManager.release_ssid(candle_ssid)  # Release the demo SSID

async def get_candles(ssid, asset, action, need_restart = True):
    """
    Fetches and processes candlestick data for the specified asset.
    
    Args:
        api: PocketOptionAsync instance
        ssid (str): Session ID
        asset (str|list|tuple): Trading asset symbol or data structure
        action (str): Operation mode
        need_restart (bool): Whether to restart script on error
    
    Returns:
        pd.DataFrame: Processed candlestick data with MACD signals
    """
    global trading_config, selected_strategy, api, trade_ssid
    current_asset = None
    remaining_payouts = None
    results = None

    # Handle various input types
    if isinstance(asset, tuple) and len(asset) == 2:
        payouts, results = asset
        if not isinstance(payouts, list):
            print(f"Invalid payouts type: {type(payouts)}")
            return pd.DataFrame()
        if not payouts:  # No more assets to process
            print("No more assets to process, showing final results...")
            return pd.DataFrame()  # Return empty DataFrame to trigger results display
        current_asset = payouts[0]  # Get the next asset to process
        remaining_payouts = payouts[1:]  # Store remaining assets
        print(f"Processing asset {current_asset} ({len(payouts)} assets remaining)")
    
    elif isinstance(asset, list):
        if not asset:  # No more assets to process
            print("No assets to process")
            return pd.DataFrame()
        if len(asset) > 0 and isinstance(asset[0], str) and is_serialized(asset[0]):
            # Handle JSON-serialized data
            data = json.loads(asset[0])
            if isinstance(data, dict):
                if 'trading_config' in data:
                    # Restore trading configuration
                    trading_config.update(data['trading_config'])
                    save_trading_config(trading_config)
                
                if data['type'] == 'tuple':
                    payouts = data['payouts']
                    results = data['results']
                elif data['type'] == 'list':
                    payouts = data['assets']
                    results = []
                elif data['type'] == 'simple':
                    # For action '1', directly use the asset string
                    current_asset = str(data['asset'])
                    print(f"Using asset from data: {current_asset}")
                else:
                    current_asset = asset[0]
                    remaining_payouts = asset[1:]
                    print(f"Processing asset from list: {current_asset}")
        else:
            current_asset = asset[0]
            remaining_payouts = asset[1:]
            print(f"Processing asset from list: {current_asset}")
    else:
        current_asset = asset

    strategy_timeframe = selected_strategy.strategy_timeframe
    time = strategy_timeframe['candles_history']  # Time window for historical data
    force_candle = strategy_timeframe['force_candle'] if 'force_candle' in strategy_timeframe else 30
    max_retries = 2 if action != '4' else 5  # More retries for backtest mode

    # Main retry loop for API calls
    for retry in range(max_retries):
        try:
            # Check if API instances need to be reinitialized
            if api is None or api_v1 is None:
                print("Reinitializing API connections...")
                api = PocketOptionAsync(ssid)
                await asyncio.sleep(5)  # Wait for connection

            # Create task for candles request
            candles_task = asyncio.create_task(api.get_candles(current_asset, force_candle, time))
            full_candles_task = None
            use_full_candles_history = strategy_timeframe['candles'] >= 60

            if use_full_candles_history:
                full_candles_history = strategy_timeframe['full_candles_history'] if 'full_candles_history' else time
                full_candles_task = asyncio.create_task(api.get_candles(current_asset, strategy_timeframe['candles'], full_candles_history))

            try:
                # Wait for request with timeout
                candles = None
                full_candles = None
                if use_full_candles_history:
                    candles, full_candles = await asyncio.wait_for(
                        asyncio.gather(candles_task, full_candles_task),
                        timeout = 5
                    )

                    # Verify data
                    full_candles_df = pd.DataFrame(full_candles)
                    if len(full_candles_df) > 150:
                        # Need to exchange values of candles and full_candles
                        temp_candles = candles
                        candles = full_candles
                        full_candles = temp_candles

                else:
                    candles = await asyncio.wait_for(
                        asyncio.gather(candles_task),
                        timeout=5
                    )
                    candles = candles[0]  # Extract result from gather
                    if not candles:
                        raise ValueError("Empty candles data received")
                break  # Success - exit retry loop
                
            except asyncio.TimeoutError:
                print(f"Request timeout (attempt {retry + 1}/{max_retries})")
                if use_full_candles_history:
                    for task in [candles_task, full_candles_task]:
                        if not task.done():
                            task.cancel()
                else:
                    if not candles_task.done():
                        candles_task.cancel()
                raise
            except asyncio.CancelledError:
                if use_full_candles_history:
                    for task in [candles_task, full_candles_task]:
                        if not task.done():
                            task.cancel()
                else:
                    if not candles_task.done():
                        candles_task.cancel()
                raise
            
        except (asyncio.TimeoutError, ConnectionError) as e:
            print(f"Connection error fetching candles (attempt {retry + 1}/{max_retries}): {str(e)}")
            
            # Cancel any pending tasks
            if 'candles_task' in locals() and not candles_task.done():
                candles_task.cancel()
            
            if use_full_candles_history:
                if 'full_candles_task' in locals() and not full_candles_task.done():
                    full_candles_task.cancel()
            
            if retry < max_retries - 1:
                # Attempt connection refresh
                print("Reconnecting APIs...")
                api = PocketOptionAsync(ssid)
                await asyncio.sleep(5)
                continue
            else:
                print("Max retries reached")
                if need_restart:
                    restart_script(trade_ssid, asset, action)
                raise
                
        except Exception as e:
            print(f"Error fetching candles: {e}")
            if need_restart:
                restart_state = remaining_payouts if results is None else (remaining_payouts, results)
                restart_script(trade_ssid, restart_state, action)
            raise

    # Process candle data
    try:
        # Use the async processing function
        return await process_candle_data_async(candles, current_asset, full_candles)

    except Exception as e:
        print(f"Error processing candle data: {e}")
        traceback_str = traceback.format_exc()
        print(f"Full traceback:\n{traceback_str}")
        if need_restart:
            restart_state = remaining_payouts if results is None else (remaining_payouts, results)
            restart_script(ssid, restart_state, action)
        return pd.DataFrame()

async def get_candles_simultaneously(ssid, assets, action, need_restart=True):
    """
    Fetches and processes candlestick data for a list of assets simultaneously.
    """
    global trading_config, selected_strategy, api, api_v1_prices, trade_ssid
    
    if not assets or not isinstance(assets, list):
        print("Invalid assets parameter, must be a non-empty list")
        return {}

    # Store current prices for all assets before starting
    asset_prices = {}
    print("\nGetting current prices for all assets...")
    for asset in assets:
        try:
            prices_df = api_v1_prices.GetPrices(asset, 1, count=2)
            if prices_df is not None and not prices_df.empty:
                asset_prices[asset] = prices_df['price'].iloc[-1]
            await asyncio.sleep(0.5)  # Small delay between price requests
        except Exception as e:
            print(f"Error getting price for {asset}: {e}")
    
    if not asset_prices:
        print("Failed to get any prices. Aborting.")
        return {}
        
    strategy_timeframe = selected_strategy.strategy_timeframe
    time = strategy_timeframe['candles_history']
    
    results = {}
    total_assets = len(assets)
    completed = 0
    failed = 0
    start_time = datetime.now()
    
    progress_width = 50
    
    async def update_progress():
        nonlocal completed, failed
        progress = (completed + failed) / total_assets
        filled = int(progress_width * progress)
        bar = '█' * filled + '-' * (progress_width - filled)
        percent = progress * 100
        print(f'\rProgress: |{bar}| {percent:.1f}% Complete ({completed} success, {failed} failed)', end='')
    
    if api is None:
        print("Initializing API connection...")
        api = PocketOptionAsync(ssid)
        api_v1_prices = pocketoption(candle_ssid)
        await asyncio.sleep(5)
    
    print(f"\nFetching data for {len(assets)} assets concurrently...")
    
    batch_size = 100
    asset_batches = [assets[i:i + batch_size] for i in range(0, len(assets), batch_size)]
    
    try:
        for batch in asset_batches:
            print(f"\nProcessing batch of {len(batch)} assets...")
            
            # Create tasks for the batch
            tasks = []
            for asset in batch:
                task = asyncio.create_task(process_asset(asset, time, asset_prices))
                tasks.append(task)
            
            # Process batch with timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30
                )
                
                # Process batch results
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"\nTask error: {result}")
                        failed += 1
                    elif isinstance(result, tuple) and len(result) == 2:
                        asset, df = result
                        if df is not None and not df.empty:
                            results[asset] = df
                            completed += 1
                        else:
                            failed += 1
                    
                    await update_progress()
                    
            except asyncio.TimeoutError:
                print(f"\nBatch processing timeout - some assets may have failed")
                failed += len(batch)
            
            await asyncio.sleep(1)
        
        await update_progress()
        print("\n")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nProcessing completed in {duration:.2f} seconds")
        print(f"Success rate: {(completed/total_assets)*100:.1f}%")
        print(f"Successful: {completed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed assets:")
            failed_assets = set(assets) - set(results.keys())
            for asset in failed_assets:
                print(f"- {asset}")
        
    except Exception as e:
        print(f"\nError in concurrent operations: {e}")
        traceback_str = traceback.format_exc()
        print(f"Full traceback:\n{traceback_str}")
        
        if need_restart:
            restart_script(trade_ssid, assets, action)
            
    return results

async def process_asset(asset, time, all_prices):
    """
    Process a single asset's fetch and data processing completely independently.
    """
    global selected_strategy, api, api_v1_prices, candle_ssid

    max_retries = 3
    retry_count = 0
    timeframe = selected_strategy.strategy_timeframe
    force_candle = timeframe['force_candle'] if 'force_candle' in timeframe else 30
    known_price = all_prices[asset]
    
    while retry_count < max_retries:
        try:
            # Create a new coroutine for each attempt
            candles = await asyncio.wait_for(api.get_candles(asset, force_candle, time), timeout=10)
            
            if not candles:
                print(f"No data returned for {asset}")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue
                return asset, None

            # Verify data corresponds to correct asset using known price
            if candles and len(candles) > 0:
                latest_candle = candles[-1]
                latest_price = latest_candle['close']
                
                price_diff_percent = abs(latest_price - known_price) / known_price * 100
                
                if price_diff_percent > 1:  # 1% threshold
                    # Try to find matching asset
                    potential_matches = []
                    for other_asset, other_price in all_prices.items():
                        if other_asset != asset:
                            other_diff_percent = abs(latest_price - other_price) / other_price * 100
                            if other_diff_percent < 1:  # Within 1% threshold
                                potential_matches.append((other_asset, other_diff_percent))
                    
                    if potential_matches:
                        # Sort by closest price match
                        potential_matches.sort(key=lambda x: x[1])
                        matched_asset, match_diff = potential_matches[0]
                        
                        # Process data for the matched asset instead
                        print(f"Processing data for {matched_asset} instead of {asset}")
                        df_clean = await process_candle_data_async(candles, matched_asset)
                        if df_clean.empty:
                            print(f"Processing yielded empty dataframe for {matched_asset}")
                            return matched_asset, None
                        return matched_asset, df_clean
                    else:
                        print("No matching asset found for this price - data appears invalid")
                        return asset, None
                else:
                    print(f"Price verification passed for {asset} (diff: {price_diff_percent:.2f}%)")
            
            # The data is already in the correct dictionary format
            # No need for additional formatting, just pass it directly to process_candle_data_async
            df_clean = await process_candle_data_async(candles, asset)
            
            if df_clean.empty:
                print(f"Processing yielded empty dataframe for {asset}")
                return asset, None
            
            return asset, df_clean
            
        except asyncio.TimeoutError:
            print(f"Timeout fetching data for {asset}")
            return asset, None
    
    print(f"Failed to process {asset}")
    return asset, None

def generate_signals(df):
    """Generate trading signals using selected strategy."""
    global selected_strategy
    return selected_strategy.generate_signals(df)

async def trade(api, api_v1, asset: str, command: int):
    """
    Executes trades on PocketOption platform.
    
    Args:
        api_v1: PocketOption API instance
        asset (str): Trading asset symbol
        command (int): Trade direction (0=buy/call, 1=sell/put)
        
    Uses strategy's trade parameters for expiry time and amount.
    Handles compound trading state management.
    
    Returns:
        bool: True if trade was successful, False otherwise
        
    Raises:
        ValueError: If trading_config validation fails
    """
    global selected_strategy, trading_config

    # Validate configuration before trading    
    is_valid, error = validate_trading_config(trading_config)
    if not is_valid:
        print(f"Invalid trading configuration: {error}")
        raise ValueError(f"Cannot trade with invalid configuration: {error}")
        
    # Validate inputs
    if not asset:
        raise ValueError("Asset cannot be empty")
    if command not in [0, 1]:
        raise ValueError("Command must be 0 (buy/call) or 1 (sell/put)")

    try:
        # Get strategy parameters
        try:
            params = selected_strategy.trade_parameters
            if not params:
                raise ValueError("No trade parameters found in selected strategy")
            time = params.get('expiration', 120)  # Default 2-minute expiry
        except AttributeError as e:
            print(f"Error accessing strategy parameters: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error getting strategy parameters: {e}")
            return False

        amount = 0

        if trading_config['trading_type'] == 'progressive_compound':
            current_stage = trading_config['current_stage']
            current_step = trading_config['current_step']
            
            # Determine trade amount based on stage and step
            if current_step == 1:
                if current_stage == 1:
                    amount = trading_config['initial_amount']
                else:
                    # Use only the profit from previous stage, not cumulative
                    amount = float(trading_config['stage_profits'][str(current_stage - 1)])
            else:
                # Second step - use profit from first step
                amount = trading_config['current_profit']
                
            print(f"\nProgressive Compound - Stage {current_stage}/{trading_config['max_stages']}, Step {current_step}")
            print(f"Trading amount: ${amount:.2f}")
        else:
            # Get current trading state
            try:
                amount = trading_config['current_amount']
                if not isinstance(amount, (int, float)):
                    raise ValueError(f"Invalid amount type: {type(amount)}")
            except KeyError:
                print("Trading amount not found in configuration")
                return False
            except Exception as e:
                print(f"Error accessing trading state: {e}")
                return False
                
            print(f"Starting trade - Asset: {asset}, Amount: {amount}, Type: {trading_config['trading_type']}")        # Get payout percentage for the asset with retries
                
        if amount <= 0:
            raise ValueError(f"Invalid trade amount: {amount}")        # Execute trade based on command
        trade_id = None
        trade_type = "BUY" if command == 0 else "SELL"
        print(f"{trade_type}ING {asset} for ${amount} at {time} seconds")
        
        try:
            if command == 0:
                result, trade_id = api_v1.Call(amount=amount, active=asset, expiration=time, add_check_win=False)
            else:
                result, trade_id = api_v1.Put(amount=amount, active=asset, expiration=time, add_check_win=False)
                
            if not trade_id:
                print("Trade placement failed - no trade ID returned")
                return False
                
            print(f"Trade placed successfully - ID: {trade_id}")
            
        except asyncio.TimeoutError as e:
            print(f"Timeout placing {trade_type} order: {e}")
            return False
        except ConnectionError as e:
            print(f"Connection error placing {trade_type} order: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error placing {trade_type} order: {e}")
            return False
        
        if trading_config['trading_type'] == 'compound' or trading_config['trading_type'] == 'progressive_compound':
            # Wait for trade to complete when in compound
            await asyncio.sleep(time)
            
            # Check trade result with retry
            max_retries = 3
            trade_data = None
            profit = 0
            for retry in range(max_retries):
                try:
                    profit, trade_data = await get_trade_result(api_v1, trade_id)
                    if trade_data:
                        break
                    print(f"Retry {retry + 1}/{max_retries} getting trade result")
                    await asyncio.sleep(1)
                except asyncio.TimeoutError as e:
                    print(f"Timeout getting trade result (attempt {retry + 1}): {e}")
                    if retry == max_retries - 1:
                        return False
                    continue
                except ConnectionError as e:
                    print(f"Connection error getting trade result (attempt {retry + 1}): {e}")
                    if retry == max_retries - 1:
                        return False
                    continue
                except Exception as e:
                    print(f"Error getting trade result (attempt {retry + 1}): {e}")
                    if retry == max_retries - 1:
                        return False
                    continue
            
            if not trade_data:
                print("Failed to get trade result after retries")
                return False

            trade_result = determine_trade_result(trade_data)
            print(f"Trade result: {trade_result}")
            print(f"Profit: ${profit}")
            # Handle compound trading state updates based on result
            try:
                if trade_result == 'win':
                    try:
                        if trading_config['trading_type'] == 'compound':
                            # Calculate compound amount based on payout
                            if not isinstance(profit, (int, float)) or profit < 0:
                                raise ValueError(f"Invalid profit calculation: {profit}")
                                
                            trading_config['current_amount'] += profit
                            trading_config['consecutive_wins'] += 1
                            
                            # Handle max compound level
                            if trading_config['current_compound_level'] >= trading_config['max_compound_level']:
                                print("Reached max compound level - resetting to initial amount")
                                trading_config['current_amount'] = trading_config['initial_amount']
                                trading_config['current_compound_level'] = 0
                                trading_config['consecutive_wins'] = 0
                            else:
                                trading_config['current_compound_level'] = min(
                                    trading_config['current_compound_level'] + 1,
                                    trading_config['max_compound_level']
                                )
                                print(f"New compound level: {trading_config['current_compound_level']}")

                        elif trading_config['trading_type'] == 'progressive_compound':
                            if current_step == 1:
                                # Store profit and move to step 2
                                trading_config['current_profit'] = profit
                                trading_config['current_step'] = 2
                                print(f"Step 1 complete. Profit: ${profit:.2f}")
                            else:  # step 2
                                # Complete the stage
                                total_stage_profit = trading_config['current_profit'] + profit
                                trading_config['stage_profits'][str(current_stage)] = total_stage_profit
                                print(f"Stage {current_stage} complete! Total profit: ${total_stage_profit:.2f}")
                                
                                # Move to next stage
                                if current_stage < trading_config['max_stages']:
                                    trading_config['current_stage'] += 1
                                    trading_config['current_step'] = 1
                                    trading_config['current_profit'] = 0.0
                                else:
                                    print("All stages complete! Starting over...")
                                    trading_config['current_stage'] = 1
                                    trading_config['current_step'] = 1
                                    trading_config['current_profit'] = 0.0
                                    trading_config['stage_profits'] = {str(i): 0.0 for i in range(1, trading_config['max_stages'] + 1)}
                            
                    except ValueError as e:
                        print(f"Error calculating profit: {e}")
                        return False
                    except Exception as e:
                        print(f"Error updating win state: {e}")
                        return False
                    
                else:
                    try:
                        if trading_config['trading_type'] == 'compound':
                            # Reset on loss
                            print("Trade lost - resetting compound state")
                            trading_config['current_amount'] = trading_config['initial_amount']
                            trading_config['consecutive_wins'] = 0
                            trading_config['current_compound_level'] = 0
                        elif trading_config['trading_type'] == 'progressive_compound':
                            if current_step == 1:
                                if current_stage == 1:
                                    print("Lost initial investment. Need new deposit.")
                                    return False
                                else:
                                    # Go back one stage instead of resetting to stage 1
                                    previous_stage = current_stage - 1
                                    print(f"Lost stage {current_stage} investment. Reverting to stage {previous_stage}.")
                                    trading_config['current_stage'] = previous_stage
                                    trading_config['current_step'] = 1
                                    trading_config['current_profit'] = 0.0
                                    # Clear profits only for current and later stages
                                    for stage in range(current_stage, trading_config['max_stages'] + 1):
                                        trading_config['stage_profits'][str(stage)] = 0.0
                            else:  # step 2
                                print("Lost step 2. Reverting to step 1 of current stage.")
                                trading_config['current_step'] = 1
                                trading_config['current_profit'] = 0.0
                    except Exception as e:
                        print(f"Error resetting state after loss: {e}")
                        return False

                # Validate updated state
                is_valid, error = validate_trading_config(trading_config)
                if not is_valid:
                    print(f"Invalid trading configuration after update: {error}")
                    return False

                # Save updated state
                if not save_trading_config(trading_config):
                    print("Failed to save updated trading config")
                    return False
                    
                return trade_result == 'win'
                
            except Exception as e:
                print(f"Error updating compound state: {e}")
                traceback_str = traceback.format_exc()
                print(f"Full traceback:\n{traceback_str}")
                return False
                    
            return trade_result == 'win'
    except Exception as e:
        print(f"Error in trade function : {e}")

def restart_script(ssid: str, asset: str|list|tuple|None = None, command: str|None = None):
    """
    Restarts the script with the given parameters to maintain trading state.
    
    Args:
        ssid (str): Session ID for authentication
        asset (str|list|tuple|None): Asset(s) to trade. Can be:
            - str: Single asset symbol
            - list: Multiple asset symbols
            - tuple: (payouts, results) for backtesting continuity
            - None: No asset specified
        command (str|None): Action to perform after restart
    """
    # Save current trading config before restart
    global trading_config
    save_trading_config(trading_config)
    try:
        # Get the Python executable and script paths
        python_exec = sys.executable
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        # Change to script directory to ensure relative paths work
        os.chdir(script_dir)

        # Build argument list with absolute paths
        args = [python_exec]
        
        if not getattr(sys, 'frozen', False):
            args.append(script_path)

        # Add ssid
        args.append(str(ssid))                

        # Handle asset argument based on type
        if asset is not None:
            if isinstance(asset, tuple) and len(asset) == 2:
                # Handle tuple of (payouts, results)
                payouts, results = asset
                if not isinstance(payouts, list):
                    payouts = list(payouts)
                if not isinstance(results, list):
                    results = list(results)
                # Create a flat serialized structure to avoid nesting issues
                json_str = json.dumps({
                    "type": "tuple",
                    "payouts": payouts,
                    "results": results,
                    "trading_config": trading_config  # Include trading configuration
                })
                args.append(json_str)
            elif isinstance(asset, list):
                # Regular list serialization with trading config
                json_str = json.dumps({
                    "type": "list",
                    "assets": asset,
                    "trading_config": trading_config
                })
                args.append(json_str)
            else:
                # String or other simple type with trading config
                json_str = json.dumps({
                    "type": "simple",
                    "asset": asset if isinstance(asset, str) else str(asset),
                    "trading_config": trading_config
                })
                args.append(json_str)

        # Add command if present
        if command is not None:
            args.append(str(command))

        # Add current strategy if one is selected
        global selected_strategy
        if selected_strategy:
            # Use the class name instead of display name
            args.append(selected_strategy.__class__.__name__)

        print(f"Restarting script with ssid: {ssid}, asset: {asset}, action: {command}, and strategy: {selected_strategy.__class__.__name__ if selected_strategy else 'None'}")
        print(f"Command: {' '.join(args)}")
        
        try:
            os.execv(python_exec, args)
        except OSError as e:
            print(f"Failed to restart script: {e}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during restart: {e}")
        traceback_str = traceback.format_exc()
        print(f"Full traceback:\n{traceback_str}")
        sys.exit(1)

async def get_best_payouts(api, get_max: bool = False, min_payout: int = 70, log: bool = True):
    """
    Retrieves assets with the highest payout rates from PocketOption.
    
    Args:
        api: PocketOption API instance
        get_max (bool): If True, returns only assets with maximum payout
        min_payout (int): Minimum acceptable payout percentage
    
    Returns:
        list: List of asset symbols meeting the payout criteria
    """
    global RESTRICTED_ASSETS
    raw_payouts = await api.payout()

    # Filter to only include restricted assets
    payouts = {k: v for k, v in raw_payouts.items() if k in RESTRICTED_ASSETS}
    if not payouts:
        print("Warning: None of the restricted assets found in available payouts")
        return []

    if log:
        sorted_payouts = sorted(payouts.items(), key=lambda x: x[1], reverse=True)
        print("\nAvailable restricted assets and payouts:")
        for asset, payout in sorted_payouts:
            print(f"{asset}: {payout}%")

    # Handle maximum payout filtering
    if get_max:
        max_payout = max(payouts.values())
        return [k for k, v in payouts.items() if v >= max_payout]

    # Handle minimum payout filtering
    return [k for k, v in payouts.items() if v >= min_payout]

def get_keys_with_max_value(data_dict):
    """Get keys with maximum value from a dictionary"""
    if not data_dict:
        return []
    max_value = max(data_dict.values())
    return [k for k, v in data_dict.items() if v == max_value]

def get_keys_with_defined_value_min(data_dict, min_value):
    """Get keys from dictionary where value is >= min_value"""
    if not data_dict:
        return []
    return [k for k, v in data_dict.items() if v >= min_value]

def is_serialized(s):
    try:
        data = json.loads(s)  # Attempt JSON deserialization [[8]][[9]]
        # Check if the result is a list or object (dict)
        return isinstance(data, (list, dict))  # [[3]][[10]]
    except (json.JSONDecodeError, TypeError):
        return False  # Not a serialized object/array

def backtest(df, symbol, start_time=None, end_time=None):
    """Run backtesting using selected strategy."""
    global selected_strategy
    return selected_strategy.backtest(df, symbol, start_time, end_time)

async def analyze_csv_file(filename: str) -> pd.DataFrame:
    """
    Analyzes a CSV file using the specified strategy or MACD strategy by default.
    
    Args:
        filename (str): Name of the CSV file in the candles_data folder
        strategy (BaseStrategy): Trading strategy to use for analysis (optional)
        
    Returns:
        pd.DataFrame: Processed candlestick data with signals
    """
    global selected_strategy

    # Construct the full path
    file_path = os.path.join('candles_data', filename)
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None
        
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert time column to datetime if it exists
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            
        # Process only necessary columns
        required_columns = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV file must contain columns: {required_columns}")
            return None
            
        print("\nProcessing CSV file for signals...")
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"Total candles: {len(df)}")

        strategy = selected_strategy

        # If no strategy provided, use default MACD strategy
        if strategy is None:
            from strategies.macd_strategy import MACDStrategy
            strategy = MACDStrategy()
            
        # Generate signals using the strategy
        df_with_signals = strategy.generate_signals(df)
        df_with_signals = df_with_signals.dropna()
        
        # Display summary of signals
        signal_count = df_with_signals['call_signal'].sum() + df_with_signals['put_signal'].sum()
        print(f"\nFound {signal_count} trading signals in the data")
        print(f"Call signals: {df_with_signals['call_signal'].sum()}")
        print(f"Put signals: {df_with_signals['put_signal'].sum()}\n")
        
        # Perform backtesting if there are any signals
        if signal_count > 0:
            win_rate, total_trades, wins, losses = strategy.backtest(df_with_signals, filename)
            print(f"\nBacktest Summary:")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total Trades: {total_trades}")
            print(f"Wins: {wins}")
            print(f"Losses: {losses}")
        
        # Display the last few signals
        signals = df_with_signals[df_with_signals['call_signal'] | df_with_signals['put_signal']].tail()
        if not signals.empty:
            print("\nLast few signals:")
            for _, row in signals.iterrows():
                signal_type = "CALL" if row['call_signal'] else "PUT"
                print(f"Time: {row['time']}, Signal: {signal_type}, Close: {row['close']}")
        
        # Save analyzed data
        output_file = os.path.join('candles_data', f'analyzed_{strategy.__class__.__name__}_{filename}')
        df_with_signals.to_csv(output_file, index=False)
        print(f"\nAnalyzed data saved to: {output_file}")
                
        return df_with_signals
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def save_trading_state(state):
    """
    Save trading state to a file, including processed assets list.
    
    Args:
        state (dict): Trading state dictionary containing payouts and processed assets
    """
    try:
        with open("trading_state.json", 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Error saving trading state: {e}")

async def select_strategy() -> BaseStrategy:
    """Show menu to select trading strategy."""
    while True:
        print("\n=== Available Trading Strategies ===")
        strategies = StrategyFactory.get_available_strategies()
        for i, strategy in enumerate(strategies, 1):
            strategy_instance = StrategyFactory.get_strategy(strategy)
            print(f"{i}. {strategy_instance.name}")
            print(f"   {strategy_instance.description}")
        
        try:
            choice = int(input(f"\nSelect strategy (1-{len(strategies)}): "))
            if 1 <= choice <= len(strategies):
                strategy_name = strategies[choice - 1]
                return StrategyFactory.get_strategy(strategy_name)
        except ValueError:
            pass
        
        print("Invalid choice. Please try again.")

async def show_menu(ssid: str, is_demo: bool = True):
    """Show main menu and handle user selection."""
    global selected_strategy, trading_config
    
    # Always reload trading config when showing menu to ensure latest state
    trading_config = load_trading_config()
    
    # Select strategy if not already selected
    if not selected_strategy:
        selected_strategy = await select_strategy()
        
    # On first run, ask for initial amount
    if not trading_config.get('trading_type'):
        print("\nFirst time setup:")
        print("1. Fixed amount trading")
        print("2. Compound trading")
        type_choice = input("Select trading type (1/2): ").strip()
        
        if type_choice == "1":
            trading_config['trading_type'] = 'fixed'
        elif type_choice == "2":
            trading_config['trading_type'] = 'compound'
            while True:
                try:
                    level = int(input("\nEnter max compound level (1-10): "))
                    if 1 <= level <= 10:
                        trading_config['max_compound_level'] = level
                        break
                    print("Please enter a number between 1 and 10")
                except ValueError:
                    print("Please enter a valid number")
                    
        while True:
            try:
                amount = float(input("\nEnter initial trading amount: $"))
                if amount > 0:
                    trading_config['initial_amount'] = amount
                    trading_config['current_amount'] = amount
                    break
                print("Amount must be positive")
            except ValueError:
                print("Please enter a valid number")
                
        # Save initial configuration
        save_trading_config(trading_config)
    
    while True:
        print(f"\n=== Trading Bot - {selected_strategy.name} ===")
        # Show current trading settings
        print(f"\nTrading Settings:")
        print(f"Type: {trading_config['trading_type'].replace('_', ' ').title()}")
        print(f"Initial Amount: ${trading_config['initial_amount']:.2f}")
        
        if trading_config['trading_type'] == 'compound':
            print(f"Current Amount: ${trading_config['current_amount']:.2f}")
            print(f"Max Compound Level: {trading_config['max_compound_level']}")
            print(f"Current Level: {trading_config['current_compound_level']}")
            print(f"Consecutive Wins: {trading_config['consecutive_wins']}\n")
        elif trading_config['trading_type'] == 'progressive_compound':
            print(f"Current Stage: {trading_config['current_stage']}/{trading_config['max_stages']}")
            print(f"Current Step: {trading_config['current_step']}/2")
            print(f"Current Step Profit: ${trading_config['current_profit']:.2f}")
            print("\nStage Profits:")
            total_profit = 0
            for stage in range(1, trading_config['max_stages'] + 1):
                profit = float(trading_config['stage_profits'].get(str(stage), 0.0))
                total_profit += profit
                print(f"Stage {stage}: ${profit:.2f}")
            print(f"\nTotal Accumulated Profit: ${total_profit:.2f}\n")
        
        print("\nAvailable Actions:")
        print("1. Run on one pair")
        print("2. Run on best pairs")
        print("3. Run continuously on best pairs (The old way)")
        print("4. Run continuously on best pairs")
        print("5. Multi-asset Analysis")
        print("6. Backtest")
        print("7. Analyze CSV file")
        print("8. Run continuously with background fetching (New)")
        print("9. Change strategy")
        print("10. Change trading settings")
        print("0. Exit")
        action = input("\nChoose action (0-10): ").strip()
        
        if action == "0":
            print("Exiting program...")
            return True  # Signal clean exit
            
        if action == "9":
            selected_strategy = await select_strategy()
            continue
            
        if action == "10":
            await change_trading_settings()
            continue
            
        if action in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            asset = None
            if action == "1":
                asset = input("Enter asset symbol: ")
            try:
                await main(ssid, asset, action, is_demo)
            except KeyboardInterrupt:
                print("\nOperation cancelled - returning to main menu...")
                continue
            except Exception as e:
                print(f"\nError: {e}")
                print("Returning to main menu...")
                continue
        else:
            print("Invalid action. Please choose 1-8.")

def exit_program():
    """Exit the program cleanly"""
    print("\nExiting program...")
    sys.exit(0)

async def wait_for_next_candle():
    """
    Wait until the next candle based on strategy's timeframe.
    Returns the target time that was waited for.
    """
    global selected_strategy
    timeframe = selected_strategy.strategy_timeframe
    candle_seconds = timeframe['candles']  # Get candle interval in seconds
    
    # Calculate next candle start time
    now = datetime.now()
    if candle_seconds >= 60:
        # For minute-based strategies, align to full minutes
        next_candle_start = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        target_time = next_candle_start - timedelta(seconds=1)
    else:
        # For second-based strategies, align to next interval
        current_second = now.second
        next_interval = ((current_second // candle_seconds) + 1) * candle_seconds
        next_candle_start = now.replace(microsecond=0) + timedelta(
            seconds=(next_interval - current_second)
        )
        target_time = next_candle_start - timedelta(milliseconds=100)  # Small buffer
    
    # Calculate time until target
    time_until_target = (target_time - now).total_seconds()
    
    # Handle if target_time has passed
    if time_until_target < 0:
        if candle_seconds >= 60:
            next_candle_start += timedelta(minutes=1)
        else:
            next_candle_start += timedelta(seconds=candle_seconds)
        target_time = next_candle_start - timedelta(
            seconds=1 if candle_seconds >= 60 else 0,
            milliseconds=0 if candle_seconds >= 60 else 100
        )
        time_until_target = (target_time - now).total_seconds()

    # Start countdown
    try:
        while True:
            remaining = target_time - datetime.now()
            seconds_remaining = remaining.total_seconds()
            
            if seconds_remaining <= 0:
                break
                
            print(f"Next fetch in {seconds_remaining:.1f} seconds...", end='\r')
            # Use a shorter sleep interval to be more responsive to interrupts
            for _ in range(10):  # Split 1 second into 10 parts
                await asyncio.sleep(0.1)  # 100ms intervals

    except KeyboardInterrupt:
        raise  # Re-raise to be caught by outer try block
    except Exception as e:
        print(f"\nError in countdown: {e}")
        raise

    # Clear the line after countdown finishes
    print(" " * 40, end='\r')
    return target_time

async def get_trade_result(api_v1, trade_id: str):
    """
    Get data for a specific trade.
    If newly created trade

    Args:
        trade_id (str): ID of the trade

    Returns:
        Any: Trade data
    """
    try:
        profit, result = api_v1.CheckWin(trade_id)
        return profit, result
    except Exception as e:
        print(f'Error check win using API V1 : {str(e)}')

def convert_to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert regular candlestick data to Heikin Ashi candlesticks.
    
    Args:
        df (pd.DataFrame): DataFrame containing OHLC (Open, High, Low, Close) price data
                          Required columns: ['open', 'high', 'low', 'close']
    
    Returns:
        pd.DataFrame: DataFrame with Heikin Ashi OHLC values
                     Contains columns: ['ha_open', 'ha_high', 'ha_low', 'ha_close']
    """
    # Create a new DataFrame for Heikin Ashi values
    ha = pd.DataFrame(index=df.index)
    
    # Calculate Heikin Ashi Close values
    ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Calculate Heikin Ashi Open values
    # First open value is the same as regular open
    ha['ha_open'] = df['open'].copy()
    for i in range(1, len(df)):
        ha.iloc[i, ha.columns.get_loc('ha_open')] = (ha.iloc[i-1, ha.columns.get_loc('ha_open')] + 
                                                    ha.iloc[i-1, ha.columns.get_loc('ha_close')]) / 2
    
    # Calculate Heikin Ashi High and Low values
    ha['ha_high'] = ha[['ha_open', 'ha_close']].join(df['high']).max(axis=1)
    ha['ha_low'] = ha[['ha_open', 'ha_close']].join(df['low']).min(axis=1)
    
    return ha

async def save_dataframes_batch(dataframes_to_save):
    """Save multiple dataframes to CSV in a batch"""
    tasks = []
    for filename, df in dataframes_to_save.items():
        tasks.append(asyncio.create_task(save_df_to_csv(filename, df)))
    await asyncio.gather(*tasks)

async def save_df_to_csv(filename, df):
    """Save a single dataframe to CSV asynchronously"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, df.to_csv, filename, False)

async def process_candle_data_async(candles: list, current_asset: str, full_candles : list | None = None):
    """
    Asynchronous version of process_candle_data to allow concurrent processing
    
    Args:
        candles (list): Candle data from API
        current_asset (str): Asset symbol
        full_candles (list|None): Optional full candle data from API
        
    Returns:
        pd.DataFrame: Processed dataframe with signals
    """
    global selected_strategy, trading_config
    
    start_time = datetime.now()

    strategy_timeframe = selected_strategy.strategy_timeframe
    time = strategy_timeframe['candles_history']

    try:
        # Convert to DataFrame directly from list of dictionaries
        candles_df = pd.DataFrame(candles)

        if candles_df.empty:
            print(f"No valid data received for {current_asset}")
            return pd.DataFrame()

        # Create output directory
        output_dir = 'candles_data'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert time to datetime with flexible ISO format
        try:
            # First try with format='ISO8601'
            candles_df['time'] = pd.to_datetime(candles_df['time'], format='ISO8601')
            # Add timezone information if not present
            if not candles_df['time'].dt.tz:
                candles_df['time'] = candles_df['time'].dt.tz_localize('UTC')
        except ValueError:
            try:
                # If that fails, try with mixed format
                candles_df['time'] = pd.to_datetime(candles_df['time'], format='mixed')
                if not candles_df['time'].dt.tz:
                    candles_df['time'] = candles_df['time'].dt.tz_localize('UTC')
            except ValueError as e:
                print(f"Error converting time for {current_asset}: {e}")
                traceback_str = traceback.format_exc()
                print(f"Full traceback:\n{traceback_str}")
                # Try one more time with a specific format
                try:
                    candles_df['time'] = pd.to_datetime(candles_df['time'], format="%Y-%m-%dT%H:%M:%SZ")
                    if not candles_df['time'].dt.tz:
                        candles_df['time'] = candles_df['time'].dt.tz_localize('UTC')
                except Exception as e:
                    print(f"Final attempt to convert time failed: {e}")
                    traceback_str = traceback.format_exc()
                    print(f"Full traceback:\n{traceback_str}")
                    return pd.DataFrame()

        final_df = None
        
        # Check if data needs resampling
        needs_resampling = strategy_timeframe.get('resample') is not None

        if needs_resampling:
            candles_df.set_index('time', inplace=True)

            # First pass resampling - get valid intervals
            interval_counts = candles_df.resample(strategy_timeframe['resample']).size()
            
            # Handle last interval
            last_interval_idx = interval_counts.index[-1]
            if interval_counts[last_interval_idx] > 0:
                interval_counts[last_interval_idx] = strategy_timeframe['min_points']
            valid_intervals = interval_counts[interval_counts >= strategy_timeframe['min_points']].index

            # Resample historical data
            resampled_current = candles_df.resample(strategy_timeframe['resample']).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
            })

            # Yield control to allow other tasks to run
            await asyncio.sleep(0)

            # Special handling for last interval
            last_timestamp = candles_df.index[-1]
            last_interval = resampled_current.index[-1]
            if last_timestamp >= last_interval:
                last_interval_data = candles_df[candles_df.index >= last_interval]
                if not last_interval_data.empty:
                    resampled_current.loc[last_interval, 'open'] = last_interval_data['open'].iloc[0]
                    resampled_current.loc[last_interval, 'high'] = last_interval_data['high'].max()
                    resampled_current.loc[last_interval, 'low'] = last_interval_data['low'].min()
                    resampled_current.loc[last_interval, 'close'] = last_interval_data['close'].iloc[-1]

            # Clean up data
            resampled_current = resampled_current.dropna(subset=['open', 'high', 'low', 'close'])
            resampled_current = resampled_current[resampled_current.index.isin(valid_intervals)]
            
            # Combine and sort data
            final_df = resampled_current[~resampled_current.index.duplicated(keep='first')].sort_index()
            final_df.reset_index(inplace=True)
        else:
            # No resampling needed, use data as is
            final_df = candles_df.copy()
            final_df = final_df.sort_values('time').reset_index(drop=True)

        # Handle full candles if provided
        if full_candles:
            full_candles_df = pd.DataFrame(full_candles)

            if not full_candles_df.empty:
                # Convert time to datetime with flexible ISO format
                try:
                    full_candles_df['time'] = pd.to_datetime(full_candles_df['time'], format='ISO8601')
                    if not full_candles_df['time'].dt.tz:
                        full_candles_df['time'] = full_candles_df['time'].dt.tz_localize('UTC')
                except ValueError:
                    try:
                        full_candles_df['time'] = pd.to_datetime(full_candles_df['time'], format='mixed')
                        if not full_candles_df['time'].dt.tz:
                            full_candles_df['time'] = full_candles_df['time'].dt.tz_localize('UTC')
                    except ValueError:
                        try:
                            full_candles_df['time'] = pd.to_datetime(full_candles_df['time'], format="%Y-%m-%dT%H:%M:%SZ")
                            if not full_candles_df['time'].dt.tz:
                                full_candles_df['time'] = full_candles_df['time'].dt.tz_localize('UTC')
                        except Exception as e:
                            print(f"Error converting time for full candles: {e}")
                            return final_df  # Continue with just the regular candles

                if needs_resampling:
                    # Resample full candles data
                    full_candles_df.set_index('time', inplace=True)
                    resampled_full = full_candles_df.resample(strategy_timeframe['resample']).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                    })
                    resampled_full = resampled_full.dropna(subset=['open', 'high', 'low', 'close'])
                    resampled_full.reset_index(inplace=True)
                    full_candles_df = resampled_full
                else:
                    # Sort and reset index without resampling
                    full_candles_df = full_candles_df.sort_values('time').reset_index(drop=True)

                # Ensure both dataframes have the same columns
                required_columns = ['time', 'open', 'high', 'low', 'close']

                if not all(col in full_candles_df.columns for col in required_columns):
                    print("Missing required columns in full_candles_df")
                    print(f"Available columns: {full_candles_df.columns.tolist()}")
                    return final_df

                if not all(col in final_df.columns for col in required_columns):
                    print("Missing required columns in final_df")
                    print(f"Available columns: {final_df.columns.tolist()}")
                    return final_df

                # Combine the dataframes
                combined_df = pd.concat([full_candles_df, final_df], axis=0, ignore_index=True)

                # Sort and remove duplicates
                combined_df = combined_df.sort_values('time')
                original_len = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=['time'], keep='first')

                final_df = combined_df.reset_index(drop=True)

        # Add Heikin Ashi calculations
        ha_candles = convert_to_heikin_ashi(final_df)
        candles_pd = pd.concat([final_df, ha_candles], axis=1)
        
        # Yield control again before generating signals
        await asyncio.sleep(0)
        
        # Generate signals
        structure = analyze_market_structure(candles_pd)
        current_candle = candles_pd.iloc[-1]
        current_price = current_candle['close']
        current_idx = current_candle.name
        signals = check_trade_conditions(current_price, current_idx, structure, proximity_pct=0.01) # 1% proximity
        # The real decision your strategy will use:
        current_decision = signals['decision_current']
        # print("\n[1] CURRENT DECISION (Based on Horizontal Levels Only)")
        # print(f"  Current price: {current_decision['current_price']}")
        # print(f"  Allow Buy?  -> {current_decision['allow_buy']}")
        # print(f"  Reason:        {current_decision['buy_reason']}")
        # print(f"  Allow Sell? -> {current_decision['allow_sell']}")
        # print(f"  Reason:        {current_decision['sell_reason']}")

        # The hypothetical decision for your information:
        trend_decision = signals['decision_with_trends']
        # print("\n[2] HYPOTHETICAL DECISION (If Trend Lines Were Rules)")
        # print(f"  Current price: {trend_decision['current_price']}")
        # print(f"  Allow Buy?  -> {trend_decision['allow_buy']}")
        # print(f"  Reason:        {trend_decision['buy_reason']}")
        # print(f"  Allow Sell? -> {trend_decision['allow_sell']}")
        # print(f"  Reason:        {trend_decision['sell_reason']}")
        df = generate_signals(candles_pd)
        dropna_strategy = strategy_timeframe.get('dropna', 'any')
        df_clean = df.dropna(how=dropna_strategy)

        # Save to CSV if enabled
        if trading_config.get('save_csv', True):
            filename = f"{output_dir}/{selected_strategy.__class__.__name__}_{current_asset}_time{time}_{timestamp}.csv"
            filename_final = f"{output_dir}/{selected_strategy.__class__.__name__}_{current_asset}_time{time}_{timestamp}_final.csv"
            await asyncio.sleep(0)
            candles_pd.to_csv(filename, index=False)
            df_clean.to_csv(filename_final, index=False)
            await plot_market_structure(df_clean, structure, current_asset, timestamp, selected_strategy.__class__.__name__)
            print(f"Saved candle data to {filename} at {datetime.now()}")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return df_clean

    except Exception as e:
        print(f"Error processing candle data for {current_asset}: {e}")
        traceback_str = traceback.format_exc()
        print(f"Full traceback:\n{traceback_str}")
        return pd.DataFrame()


def get_ssid_choice():
    """Show SSID selection menu and return chosen SSID."""
    print("\n=== Available SSIDs ===")
    predefined_ssids = {
        "1": '''42["auth",{"session":"j079fsgog45pjnbsj9a2hvpnnb","isDemo":1,"uid":102766033,"platform":3,"isFastHistory":true}]''',
        "2": '''42["auth",{"session":"upen8g2mcd3cvu5ai5i4jjl6si","isDemo":1,"uid":102365452,"platform":3,"isFastHistory":false}]''',
        "3": r'''42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"e880474778a3286b35b5a23504b76206\";s:10:\"ip_address\";s:13:\"102.18.26.137\";s:10:\"user_agent\";s:70:\"Mozilla/5.0 (X11; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0\";s:13:\"last_activity\";i:1748984628;}f1f2b4e8c412d3cef2198dc3d0ffc662","isDemo":0,"uid":102766033,"platform":3,"isFastHistory":true}]'''
    }
    
    print("1. SSID 1 (rindra36@hotmail.com)")
    print("2. SSID 2 (aroniainamanohisoa06@gmail.com)")
    print("3. SSID 3 (rindra36@hotmail.com) - REAL")
    print("4. Enter custom SSID")
    
    while True:
        choice = input("\nChoose SSID (1-4): ").strip()
        if choice in ["1", "2", "3"]:
            return predefined_ssids[choice]
        elif choice == "4":
            return input("Enter your custom session ID: ").strip()
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

async def run_price_action_strategy(api: PocketOptionAsync, api_v1, ssid, asset: str, strategy: FastPriceActionStrategy):
    """
    Run the real-time price action strategy using subscribe_symbol_timed.
    
    Args:
        api: PocketOptionAsync instance
        api_v1: PocketOption V1 API instance
        ssid: Candle SSID
        asset: Trading asset symbol
        strategy: FastPriceActionStrategy instance
    """
    reconnect_delay = 5  # Initial reconnect delay
    max_reconnect_delay = 60  # Maximum reconnect delay
    stream = None
    stream_task = None
    last_candle_time = None
    max_silence_duration = 20  # Maximum seconds without data before considering stream stuck

    async def setup_stream():
        """Setup new stream connection"""
        nonlocal api, stream, last_candle_time
        try:
            # Create new API instance for fresh connection
            api = PocketOptionAsync(ssid)
            await asyncio.sleep(5)  # Wait for connection to establish

            # Subscribe to real-time data stream
            stream = await api.subscribe_symbol_timed(asset, timedelta(seconds=1))
            last_candle_time = datetime.now()  # Reset timer on new stream
            print("Stream connection established")
            return True
        except Exception as e:
            print(f"Error setting up stream: {e}")
            return False

    async def process_stream(stream):
        """Process stream in a separate task that can be cancelled"""
        nonlocal last_candle_time
        try:
            async for candle in stream:
                # Update last candle time
                last_candle_time = datetime.now()

                # Process candle
                call_signal, put_signal, signal_time = strategy.process_realtime_candle(candle)
                
                print(f"Received candle at {datetime.now()}: {candle}")
                
                # Handle trading signals
                if call_signal:
                    print(f"\nCALL signal at {signal_time}")
                    await trade(api, api_v1, asset, 0)
                    
                elif put_signal:
                    print(f"\nPUT signal at {signal_time}")
                    await trade(api, api_v1, asset, 1)

                # Allow other tasks to run
                await asyncio.sleep(0)
                    
        except asyncio.CancelledError:
            print("Stream processing cancelled")
            raise
        except Exception as e:
            print(f"Error in stream processing: {e}")
            return False

    async def check_stream_health():
        """Check if stream is healthy based on last candle time"""
        while True:
            try:
                if last_candle_time is not None:
                    time_since_last = (datetime.now() - last_candle_time).total_seconds()
                    if time_since_last > max_silence_duration:
                        print(f"No data received for {time_since_last:.1f} seconds")
                        return False
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error checking stream health: {e}")
                await asyncio.sleep(1)

    while True:  # Main loop to keep strategy running
        try:
            # Initialize or reinitialize stream if needed
            if stream is None:
                print(f"\nStarting/Restarting Price Action Strategy on {asset}")
                print("Initializing data stream...")
                strategy.reset_monitoring()
                
                if not await setup_stream():
                    print(f"Failed to setup stream. Retrying in {reconnect_delay} seconds...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue
                
                reconnect_delay = 5  # Reset delay on successful connection

            # Create task for stream processing
            stream_task = asyncio.create_task(process_stream(stream))

            # Create task for health checking
            health_task = asyncio.create_task(check_stream_health())
            
            try:
                # Wait for either task to complete
                done, pending = await asyncio.wait(
                    [stream_task, health_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check results
                for task in done:
                    if task == stream_task:
                        result = await task
                        if result:  # True means reconnect after trade
                            print("Reconnecting after trade...")
                            stream = None
                            await asyncio.sleep(1)
                    elif task == health_task:
                        if not await task:  # False means stream is unhealthy
                            print("Stream appears to be stuck, reconnecting...")
                            stream = None
                            await asyncio.sleep(5)

                if stream is not None:
                    continue  # Stream still good, keep processing

            except Exception as e:
                print(f"Error in stream processing: {e}")
                if not stream_task.done():
                    stream_task.cancel()
                if not health_task.done():
                    health_task.cancel()
                stream = None
                await asyncio.sleep(5)
                continue
                
        except KeyboardInterrupt:
            print("\nStrategy stopped by user")
            break
            
        except Exception as e:
            print(f"Major error in price action strategy: {e}")
            traceback.print_exc()
            stream = None
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
        
        finally:
            if stream is not None:
                try:
                    await stream.aclose()
                except:
                    pass

async def run_price_action_strategy_multi(api: PocketOptionAsync, api_v1, ssid: str, assets: List[str], strategy_class):
    """
    Run price action strategy on multiple pairs simultaneously.
    
    Args:
        api: PocketOptionAsync instance
        api_v1: PocketOption V1 API instance
        ssid: Session ID
        assets: List of asset symbols to monitor
        strategy_class: Strategy class to use
    """
    # Initialize stream manager
    manager = PriceActionStreamManager(ssid, api_v1, strategy_class)
    
    try:
        # Setup initial streams
        for asset in assets:
            await manager.add_asset(asset)
            
        # Process all streams
        await manager.process_streams()
        
    except KeyboardInterrupt:
        print("\nStrategy stopped by user")
    except Exception as e:
        print(f"Error in multi-pair strategy: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        await manager.cleanup()

if __name__ == '__main__':
    ssid = None
    asset = None
    action = None
    strategy_name = None
    is_demo = True
    demo_SSID = [
        '''42["auth",{"session":"vtftn12e6f5f5008moitsd6skl","isDemo":1,"uid":27658142,"platform":2}]''',
        '''42["auth",{"session":"j079fsgog45pjnbsj9a2hvpnnb","isDemo":1,"uid":102766033,"platform":3,"isFastHistory":true}]''',
        '''42["auth",{"session":"upen8g2mcd3cvu5ai5i4jjl6si","isDemo":1,"uid":102365452,"platform":3,"isFastHistory":false}]''',
    ]
    
    if len(sys.argv) > 1:
        ssid = sys.argv[1]

        if ssid not in demo_SSID:
            is_demo = False
        if len(sys.argv) > 2:
            asset = sys.argv[2]
        if len(sys.argv) > 3:
            action = sys.argv[3]
        if len(sys.argv) > 4:
            strategy_name = sys.argv[4]
    
    if ssid is None:
        ssid = get_ssid_choice()

        # if not demo_SSID or (demo_SSID and ssid not in demo_SSID):
        if ssid not in demo_SSID:
            is_demo = False

    async def run():
        global selected_strategy
        try:
            # Initialize strategy factory first
            StrategyFactory.initialize()
                
            # Initialize strategy if not yet set
            if not selected_strategy:
                if strategy_name:  # Use command line argument if available
                    selected_strategy = StrategyFactory.get_strategy(strategy_name)
                else:
                    selected_strategy = await select_strategy()
                    
            # If action and asset provided, resume the previous state
            if action and asset:
                await main(ssid, asset, action, is_demo)
            else:
                # Show menu for fresh start
                await show_menu(ssid, is_demo)
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"\nFatal error: {e}")
            traceback_str = traceback.format_exc()
            print(f"Full traceback:\n{traceback_str}")
        finally:
            # Allow for clean exit without exceptions
            if 'show_menu' in locals() and isinstance(show_menu, bool) and show_menu:
                sys.exit(0)
            else:
                return

    # Add periodic SSID lock update
    async def periodic_ssid_update(ssid):
        """Periodically update the SSID lock to prevent it from becoming stale"""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                SSIDManager.update_lock(ssid)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error updating SSID lock: {e}")
                break

    # Run the async main function
    asyncio.run(run())