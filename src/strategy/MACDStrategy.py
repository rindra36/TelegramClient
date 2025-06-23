import pandas as pd
import asyncio
import os
import sys
import json

from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from datetime import datetime, timedelta
from ta.trend import MACD

# Mute pd warnings
pd.set_option('future.no_silent_downcasting', True)

# Main part of the code
async def main(ssid: str, asset: str|list|None = None, action: str|None = None):
    # The api automatically detects if the 'ssid' is for real or demo account
    api = PocketOptionAsync(ssid)
    # await asyncio.sleep(6)
    await asyncio.sleep(5)

    if action == '1':
        count_loop = 0
        while True:
            count_loop += 1

            if count_loop > 1:
                print("Entering 2nd loop, Restarting script...")
                restart_script(ssid, asset, '1')

            # Calculate next candle start time (next full minute)
            now = datetime.now()
            next_candle_start = now.replace(second=0, microsecond=0) + timedelta(minutes=1)

            # Target execution time: **EXACTLY the next minute's start**
            target_time = next_candle_start
            
            # Calculate time until target
            time_until_target = (target_time - now).total_seconds()
            
            # Handle if target_time has passed (e.g., due to late execution)
            if time_until_target < 0:
                # Schedule for the following candle
                next_candle_start += timedelta(minutes=1)
                target_time = next_candle_start - timedelta(seconds=1)
                time_until_target = (target_time - now).total_seconds()

            # Start countdown
            while True:
                remaining = target_time - datetime.now()
                seconds_remaining = remaining.total_seconds()
                
                if seconds_remaining <= 0:
                    break
                    
                print(f"Next fetch in {seconds_remaining:.1f} seconds...", end='\r')
                await asyncio.sleep(1)

            # Clear the line after countdown finishes
            print(" " * 40, end='\r')
            
            # Fetch and process candles
            df_clean = await get_candles(api, ssid, asset, action)
            
            if not df_clean.empty:
                last_row = df_clean.iloc[-2]
                if last_row['call_signal']:
                    print(f"BUY CALL at {last_row['time']} at {datetime.now()} UTC")
                    await trade(api, asset, 0)
                elif last_row['put_signal']:
                    print(f"BUY PUT at {last_row['time']} at {datetime.now()} UTC")
                    await trade(api, asset, 1)
    elif action == '2':
        if type(asset) == list:
            payouts = asset
        else:
            payouts = await get_best_payouts(api, True)

        count_loop = 0
        for asset in payouts:
            count_loop += 1

            if count_loop > 1:
                print("Entering 2nd loop, Restarting script...")
                restart_script(ssid, payouts, '2')

            # Calculate next candle start time (next full minute)
            now = datetime.now()
            next_candle_start = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            
            # Target execution time: 1 seconds before next candle starts
            target_time = next_candle_start - timedelta(seconds=1)
            
            # Calculate time until target
            time_until_target = (target_time - now).total_seconds()
            
            # Handle if target_time has passed (e.g., due to late execution)
            if time_until_target < 0:
                # Schedule for the following candle
                next_candle_start += timedelta(minutes=1)
                target_time = next_candle_start - timedelta(seconds=1)
                time_until_target = (target_time - now).total_seconds()

            # Start countdown
            while True:
                remaining = target_time - datetime.now()
                seconds_remaining = remaining.total_seconds()
                
                if seconds_remaining <= 0:
                    break
                    
                print(f"Next fetch in {seconds_remaining:.1f} seconds...", end='\r')
                await asyncio.sleep(1)

            # Clear the line after countdown finishes
            print(" " * 40, end='\r')
            
            # Fetch and process candles
            try:
                df_clean = await get_candles(api, ssid, asset, action, need_restart=False)
            except Exception as e:
                print("Restarting script...")
                restart_script(ssid, payouts, '2')

            # Remove current asset from payouts
            payouts.remove(asset)
            
            if not df_clean.empty:
                last_row = df_clean.iloc[-1]
                if last_row['call_signal']:
                    print(f"BUY CALL at {last_row['time']}")
                    await trade(api, asset, 0)
                elif last_row['put_signal']:
                    print(f"BUY PUT at {last_row['time']}")
                    await trade(api, asset, 1)

    elif action == '3':
        if type(asset) == list:
            payouts = asset
        else:
            payouts = await get_best_payouts(api)

        results = []
        for asset in payouts:
            try:
                df_clean = await get_candles(api, ssid, asset, action, need_restart=False)
            except Exception as e:
                print("Restarting script...")
                restart_script(ssid, payouts, '3')
                return  # Exit the function to prevent further execution after restarting the script

            # Remove current asset from payouts
            payouts.remove(asset)

            if not df_clean.empty:
                win_rate = backtest(df_clean, asset)

                results.append((asset, win_rate))

        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            print('===Backtest Results (Sorted by Win Rate)===')
            for asset, win_rate in results:
                print(f"{asset}: Win Rate = {win_rate:.2f}%")
        else:
            print("No results to display.")
    else:
        print("Invalid action. Please choose 1, 2, or 3.")

async def get_candles(api, ssid, asset, action, need_restart = True):
    time = 39600
    frame = 60

    # Get candles with time = 10800 and frame = 60
    try:            
        candles = await asyncio.wait_for(api.get_candles(asset, frame, time), timeout=2)
        candles_history = await asyncio.wait_for(api.history(asset, time), timeout=2)
    except Exception as e:
        print(f"Error fetching candles: {e}")
        if need_restart:
            restart_script(ssid, asset, action)
        raise e
        return

    # Convert to DataFrame
    candles_only = pd.DataFrame.from_dict(candles)
    candles_history = pd.DataFrame.from_dict(candles_history)

    # Create output directory if it doesn't exist
    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_candle = f"{output_dir}/candles_{asset}_time{time}_{timestamp}.csv"
    candles_pd.to_csv(filename_candle, index=False)

    # Convert time to datetime and resample to 1-minute bars
    candles_only['time'] = pd.to_datetime(candles_only['time'], format='ISO8601')
    candles_only.set_index('time', inplace=True)
    candles_history['time'] = pd.to_datetime(candles_history['time'], format='ISO8601')
    candles_history.set_index('time', inplace=True)

    # Calculate the number of data points per minute
    minute_counts = candles_history.resample('min').size()

    # Filter to keep only minutes with ≥60 data points (1-second intervals)
    valid_minutes = minute_counts[minute_counts >= 60].index  # Adjust threshold as needed

    # Resample to 1-minute OHLC with dropna() for incomplete minutes
    candles_history_resampled = candles_history.resample('min').agg({
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
    }).dropna()

    # Keep only valid minutes
    candles_history_resampled = candles_history_resampled[candles_history_resampled.index.isin(valid_minutes)]

    # Reset the index to make 'time' a column again [[1]][[7]]:
    candles_only.reset_index(inplace=True)
    candles_history_resampled.reset_index(inplace=True)

    # Merge while prioritizing candles_only:
    merged_df = pd.concat([candles_only, candles_history_resampled])
    merged_df.drop_duplicates(subset=['time'], keep='first', inplace=True)  # [[7]][[9]]

    candles_pd = merged_df
    
    df = generate_signals(candles_pd)
    df_clean = df.dropna()

    # Create filename with timestamp and parameters
    filename = f"{output_dir}/MACDStrategy_{asset}_{timestamp}.csv"
    filename_candle_resampled = f"{output_dir}/candles_resampled_{asset}_time{time}_{timestamp}.csv"

    # Save to CSV (optional)
    # df.to_csv(filename, index=False)
    # candles_pd.to_csv(filename_candle_resampled, index=False)

    return df_clean

def generate_signals(df):
    # Calculate MACD
    close_prices = df['close'].astype(float)
    macd = MACD(close_prices, window_slow=20, window_fast=9, window_sign=3)
    df['macd_line'] = macd.macd()
    df['signal_line'] = macd.macd_signal()
    
    # Detect crossovers
    df['macd_above'] = df['macd_line'] > df['signal_line']
    df['prev_macd_above'] = df['macd_above'].shift(1)
    df['cross_up'] = (df['macd_above'] & ~df['prev_macd_above']
        .fillna(False)  # Fill missing values
        .infer_objects()  # Explicitly infer the optimal dtype
        .astype(bool))
    df['cross_down'] = (~df['macd_above'] & df['prev_macd_above'])
    
    # Check next candle's direction
    df['next_close'] = df['close'].shift(-1)
    df['next_open'] = df['open'].shift(-1)
    df['next_bullish'] = df['next_close'] > df['next_open']
    df['next_bearish'] = df['next_close'] < df['next_open']
    
    # Final signals : Only consider signals where the next candle has closed
    df['call_signal'] = (df['cross_up'] & df['next_bullish']) & (df['time'] < df['time'].iloc[-1] - timedelta(minutes=1))
    df['put_signal'] = (df['cross_down'] & df['next_bearish']) & (df['time'] < df['time'].iloc[-1] - timedelta(minutes=1))

    return df

async def trade(api, asset: str, command: int):
    time = 120
    amount = 1

    # Create the trade depending of the command
    if command == 0:
        # Buy order
        try:
            print(f"Buying {asset} for {amount}$ at {time} seconds")
            trade_id = await api.buy(asset, amount, time, check_win=False)
            print(f"Trade placed ID: {trade_id}")
        except Exception as e:
            print(f"Buy order failed: {e}")

    elif command == 1:
        # Sell order
        try:
            print(f"Selling {asset} for {amount}$ at {time} seconds")
            trade_id = await api.sell(asset, amount, time, check_win=False)
            print(f"Trade placed ID: {trade_id}")
        except Exception as e:
            print(f"Sell order failed: {e}")
    else:
        print("Unknown command")

def restart_script(ssid: str, asset: str|list|None = None, command: str|None = None):
    # Use the current Python interpreter (from the venv)
    python_exec = sys.executable  # [[3]][[5]]
    # Use the script’s own path (from sys.argv[0])
    script_path = sys.argv[0]  # [[2]][[7]]

    launcher = [python_exec, script_path]

    if type(asset) == list:
        asset = json.dumps(asset)

    arguments = [ssid, asset, command]
    
    os.execv(python_exec, launcher + arguments)

async def get_best_payouts(api, get_max: bool = False, min_payout: int = 70):
    payouts = await api.payout()

    if get_max:
        return get_keys_with_max_value(payouts)

    return get_keys_with_defined_value_min(payouts, min_payout)

def get_keys_with_max_value(data_dict):
    # Find the maximum value in the dictionary
    max_value = max(data_dict.values())
    
    # Get all keys that have this maximum value
    keys_with_max_value = [key for key, value in data_dict.items() if value == max_value]
    
    return keys_with_max_value

def get_keys_with_defined_value_min(data_dict, min_value):
    # Get all keys that have this minimum value
    keys_with_min_value = [key for key, value in data_dict.items() if value >= min_value]

    if not keys_with_min_value:
        print('No assets found with payout >= 70%')
        return []

    # Sort by value in descending order
    keys_with_min_value.sort(key=lambda x: data_dict[x], reverse=True)
    
    print(f"Found {len(keys_with_min_value)} assets with payout >= 70%")
    for key in keys_with_min_value:
        print(f"{key}: {data_dict[key]}%")
    
    return keys_with_min_value

def is_serialized(s):
    try:
        data = json.loads(s)  # Attempt JSON deserialization [[8]][[9]]
        # Check if the result is a list or object (dict)
        return isinstance(data, (list, dict))  # [[3]][[10]]
    except (json.JSONDecodeError, TypeError):
        return False  # Not a serialized object/array

def backtest(df, payout):
    print(f'Backtesting with : {payout}')
    wins = 0
    losses = 0
    total_trades = 0

    close_prices = df['close']

    # Get the last valid index we can check (length - 4 to ensure we have 3 candles ahead)
    max_index = len(df) - 4

    for i in range(max_index + 1):
        if not df['call_signal'].iloc[i] and not df['put_signal'].iloc[i]:
            continue

        total_trades += 1
        opening_price = close_prices.iloc[i + 1]
        
        # Since we limited the range with max_index, this access will always be safe
        closing_price = close_prices.iloc[i + 3]

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
            total_trades -= 1  # If no closing price, do not count this trade
            continue

    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    print(f'Backtest results for {payout}: Total Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%')

    return win_rate


if __name__ == '__main__':
    ssid = asset = action = None

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        ssid = args[0]
        asset = args[1]

        if is_serialized(asset):
            asset = json.loads(asset)

        action = args[2]
        print(f'Restarting script with ssid: {ssid} and asset: {asset} and action: {action}')

    if ssid is None:
        ssid = input('Please enter your ssid: ')

    if len(sys.argv) == 1:
        print('1. Run on one pair')
        print('2. Run on best pairs')
        print('3. Backtest')

        action = input('Please choose the action: ').strip()
        
        
    if asset is None and action == '1':
        asset = input('Please choose the asset: ')

    asyncio.run(main(ssid, asset, action))