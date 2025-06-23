import pandas as pd
import asyncio
import os

from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from PocketOptionMethod import PocketOptionMethod
from datetime import datetime, timedelta
from ta.trend import MACD
from typing import Tuple, Any, Optional, Callable, TypeVar, Coroutine
from tenacity import retry, stop_after_attempt, wait_fixed

T = TypeVar('T')  # Type variable for generic return types

# Main part of the code
async def main(account_id, wallet_type, asset):
    apiMethod = PocketOptionMethod(int(account_id), wallet_type)
    await asyncio.sleep(5)  # Wait for connection to establish

    api = apiMethod.api_async

    while True:
        # Calculate next candle start time (next full minute)
        now = datetime.now()
        next_candle_start = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Target execution time: 7 seconds before next candle starts
        target_time = next_candle_start - timedelta(seconds=7)
        
        # Calculate time until target
        time_until_target = (target_time - now).total_seconds()
        
        # Handle if target_time has passed (e.g., due to late execution)
        if time_until_target < 0:
            # Schedule for the following candle
            next_candle_start += timedelta(minutes=1)
            target_time = next_candle_start - timedelta(seconds=7)
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
        df_clean = await get_candles(api, asset)
        
        if not df_clean.empty:
            last_row = df_clean.iloc[-1]
            if last_row['call_signal']:
                await trade(apiMethod, asset, 0)
                print(f"BUY CALL at {last_row['time']}")
            elif last_row['put_signal']:
                await trade(apiMethod, asset, 1)
                print(f"BUY PUT at {last_row['time']}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0),
    retry_error_cls=asyncio.TimeoutError
)
async def get_candles(api, asset: str):
    time = 39600
    frame = 60

    # Get candles with time = 10800 and frame = 60
    while True:
        try:            
            # candles = await api.get_candles(asset, frame, time)
            # candles = await execute_with_retry(api.get_candles, asset, frame, time)
            candles = await asyncio.wait_for(api.get_candles(asset, frame, time), timeout=15)
            # candles_history = await api.history(asset, time)
            # candles_history = await execute_with_retry(api.history, asset, time)
            candles_history = await asyncio.wait_for(api.history(asset, time), timeout=15)

            if candles and candles_history:
                break
        except asyncio.TimeoutError as e:
            print(f"TimeoutError: {e}")
            raise e

    print('Breaking the loop')

    # Convert to DataFrame
    candles_only = pd.DataFrame.from_dict(candles)
    candles_history = pd.DataFrame.from_dict(candles_history)

    # Create output directory if it doesn't exist
    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename_candle = f"{output_dir}/candles_{asset}_time{time}_{timestamp}.csv"
    # candles_pd.to_csv(filename_candle, index=False)

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
    df.to_csv(filename, index=False)
    candles_pd.to_csv(filename_candle_resampled, index=False)

    return df_clean

async def execute_with_retry(func: Callable[..., T], *args, **kwargs) -> Optional[T]:
    """
    Execute a function with retry mechanism.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Optional[T]: Function result or None if all retries fail
    """
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    use_delay_retry = True

    for attempt in range(MAX_RETRIES):
        try:
            result = await func(*args, **kwargs)
            if not result and attempt < MAX_RETRIES - 1:
                print(f"Empty result on attempt {attempt + 1}, retrying...")
                if use_delay_retry:
                    await asyncio.sleep(RETRY_DELAY)
                continue
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                if use_delay_retry:
                    await asyncio.sleep(RETRY_DELAY)
            else:
                print(f"All retry attempts failed for {func.__name__}")
                raise


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
    
    # Final signals
    df['call_signal'] = df['cross_up'] & df['next_bullish']
    df['put_signal'] = df['cross_down'] & df['next_bearish']

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


if __name__ == '__main__':
    # Initialize the application API client
    # ssid = input('Please enter your ssid: ')
    account_id = input('Choose the account ID [1, 2]: ')
    wallet_type = input('Choose the wallet type [demo, real]: ')
    # asset = input('Choose the asset: ')
    asset = 'CHFJPY_otc'
    asyncio.run(main(account_id, wallet_type, asset))