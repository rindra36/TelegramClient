"""
Utility functions for binary options trading operations.
Handles time conversions, configuration loading, and safe trading operations.
"""

import os
import json
import asyncio
import logging
from logging import RootLogger
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytz
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
from zoneinfo import ZoneInfo
from pathlib import Path
from telethon import TelegramClient
from BinaryOptionsToolsV2.tracing import start_logs

# Type aliases
TimeOffset = Union[int, float]
JsonDict = Dict[str, Any]

# Constants
VALID_TIMEZONES = {
    -5: 'America/Chicago',  # UTC-5
    -4: 'America/New_York',  # UTC-4
    -3: 'America/Sao_Paulo',  # UTC-3
    +7: 'Asia/Bangkok', # UTC+7
    +3: 'Africa/Nairobi' # UTC+3
}

TRADE_EXECUTION_BUFFER: float = -0.  # Seconds to subtract from wait time for API preparation
ROOT_PATH = Path(__file__).parents[1]
CONFIG_PATH = f'{ROOT_PATH}/assets/env/telegramCredentials.json'


class TimeZoneError(ValueError):
    """Custom exception for timezone-related errors."""
    pass


def get_wait_time(entry_time_str: str, timezone_offset: TimeOffset = -4) -> Optional[float]:
    """
    Calculate wait time until specified entry time in given timezone.

    Args:
        entry_time_str: Time string in "HH:MM" format
        timezone_offset: Timezone offset (-3 or -4)

    Returns:
        Seconds to wait, or None if target time has already passed

    Raises:
        TimeZoneError: If timezone offset is invalid
        ValueError: If time string format is invalid
    """
    if timezone_offset not in VALID_TIMEZONES:
        raise TimeZoneError(f"Timezone offset must be one of: {list(VALID_TIMEZONES.keys())}")

    try:
        hour, minute = map(int, entry_time_str.split(':'))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except ValueError:
        raise ValueError("Time must be in HH:MM format (24-hour)")

    # Get current time in target timezone
    target_tz = ZoneInfo(VALID_TIMEZONES[timezone_offset])
    now = datetime.now(target_tz)

    # Create target time for today
    target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # If target time has passed, return None
    if target_time <= now:
        return None

    return (target_time - now).total_seconds()


async def waiting_time(entry_time: str, timezone_offset: TimeOffset = -4) -> bool:
    """
    Wait until the specified entry time accounting for timezone offset.

    Args:
        entry_time: Time string in "HH:MM" format
        timezone_offset: Timezone offset (-3 or -4)

    Returns:
        bool: True if wait completed successfully, False if time already passed or error occurred
    """
    try:
        wait_seconds = get_wait_time(entry_time, timezone_offset)
        if wait_seconds is None:
            logging.warning(f"Entry time has already passed : {entry_time}")
            return False

        adjusted_wait = max(0, wait_seconds + TRADE_EXECUTION_BUFFER) if wait_seconds > TRADE_EXECUTION_BUFFER else 0
        logging.info(f"Waiting {adjusted_wait:.2f} seconds...")
        await asyncio.sleep(adjusted_wait)
        return True

    except TimeZoneError as e:
        logging.error(f"Invalid timezone: {e}")
        return False
    except ValueError as e:
        logging.error(f"Invalid time format: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during wait: {e}")
        return False


def get_telegram_credentials() -> JsonDict:
    """
    Load Telegram API credentials from configuration file.

    Returns:
        dict: Telegram credentials containing 'id' and 'hash'

    Raises:
        FileNotFoundError: If credentials file doesn't exist
        json.JSONDecodeError: If credentials file is invalid JSON
        KeyError: If required credentials are missing
    """
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            credentials = json.load(f)

        # Validate required fields
        if not all(key in credentials for key in ['id', 'hash']):
            raise KeyError("Missing required credentials (id or hash)")

        return credentials
    except FileNotFoundError:
        logging.error(f"Credentials file not found: {CONFIG_PATH}")
        raise
    except json.JSONDecodeError:
        logging.error("Invalid JSON in credentials file")
        raise
    except Exception as e:
        logging.error(f"Error loading credentials: {e}")
        raise


async def safe_trade(pocket_option: Any, channel: str, check_win: bool = False) -> Optional[str]:
    """
    Execute a trade operation with error handling.

    Args:
        pocket_option: PocketOption instance
        channel: Trading channel identifier
        check_win: Whether to check for win status

    Returns:
        str: Trade ID if successful, None if failed
    """
    try:
        return await pocket_option.trade(channel, check_win)
    except Exception as e:
        logging.error(f"Trade execution failed: {e}")
        return None


def setup_logging(session_name: str) -> None:
    """Configure logging settings."""
    logging.basicConfig(
        format='[%(levelname)s %(asctime)s] %(name)s: %(message)s',
        level=logging.INFO
    )
    start_logs(
        path=f"{ROOT_PATH}/logs/{session_name}/",
        level="INFO",
        terminal=False
    )


def load_chats() -> Dict[str, Any]:
    """Load monitored chat configurations."""
    with open(f'{ROOT_PATH}/assets/env/chats.json', 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def load_credentials() -> Dict[str, str]:
    """Load Telegram API credentials."""
    credentials = get_telegram_credentials()
    if not credentials.get('id') or not credentials.get('hash'):
        raise ValueError("Empty Telegram credentials")
    return credentials


def setup_client(session_name, id, hash) -> TelegramClient:
    """Initialize and configure Telegram client."""
    return TelegramClient(
        f'{session_name}-codeanywhere',
        id,
        hash
    )


async def find_trade_in_opened_deals(pocket_option: any, channel: str, need_data: bool = False) -> Optional[str]:
    """
    Find the trade ID in the list of opened deals.

    Args:
        channel: Channel identifier

    Returns:
        Optional[str]: Trade ID if found, None otherwise
    """
    channel_data = pocket_option.get_channel_data(channel)
    opened_deals = await pocket_option.get_opened_deals()

    if not opened_deals:
        logging.error('No opened deals found')
        return None

    for deal in opened_deals:
        if deal.get('asset') == channel_data.get('asset') and deal.get('amount') == channel_data.get('amount'):
            return deal.get('id') if not need_data else deal
    return None


async def find_trade_in_closed_deals(pocket_option: any, trade_id: str) -> Optional[Dict[str, Any]]:
    """
    Find the trade result in the list of closed deals.

    Args:
        trade_id: Trade identifier

    Returns:
        Optional[Dict[str, Any]]: Trade data if found, None otherwise
    """
    closed_deals = await pocket_option.get_closed_deals()

    if not closed_deals:
        logging.error('No closed deals found')
        return None

    for deal in closed_deals:
        if deal.get('id') == trade_id:
            return deal
    return None


async def get_trade_result(pocket_option: any, trade_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the result of a specific trade.

    Args:
        trade_id: Trade identifier
        need_timeout:
            Set to True to wait for timeout for newly placed trades
            Set to False if the trades has already placed for some times

    Returns:
        Optional[Dict[str, Any]]: Trade result data if found, None otherwise
    """
    try:
        return await pocket_option.get_trade_data(trade_id)
    except Exception as e:
        logging.warning(f'Trade data not found with check_win: {str(e)}')


def determine_trade_result(trade_data: Dict[str, Any]) -> str:
    """
    Determine the result of a trade.

    Args:
        trade_data: Trade data dictionary

    Returns:
        str: Trade result ('win', 'lose', 'draw')
    """
    if 'result' in trade_data:
        return trade_data.get('result', 'lose')
    if 'profit' in trade_data:
        return 'win' if trade_data.get('profit') > 0 else 'lose'
    else:
        return trade_data


async def wait_until_close_timestamp(close_timestamp: int, waiting_second: int = 10, timezone_offset: int = 3, additional_offset: int = 2):
    """
    Wait until {waiting_second} seconds before the close timestamp, adjusted for the given time zone and additional offset.

    Args:
        close_timestamp (int): The close timestamp in seconds since the epoch.
        timezone_offset (int): The time zone offset in hours.
        additional_offset (int): Additional offset in hours to adjust the timestamp.
    """
    # Convert the timestamp to a datetime object
    target_time = datetime.fromtimestamp(close_timestamp, pytz.utc)

    # Adjust the time zone
    target_time = target_time.astimezone(pytz.FixedOffset(timezone_offset * 60))

    # Subtract the additional offset
    target_time -= timedelta(hours=additional_offset)

    # Subtract 10 seconds
    target_time -= timedelta(seconds=waiting_second)

    logging.info(f'TargetTime : {target_time}')

    # Calculate the wait time
    now = datetime.now(pytz.utc).astimezone(pytz.FixedOffset(timezone_offset * 60))
    wait_time = (target_time - now).total_seconds()

    logging.info(f'WaitTime : {wait_time}')

    if wait_time > 0:
        await asyncio.sleep(wait_time)

def filter_keys_by_allowed_list(all_keys, allowed_keys):
    # Return only keys that are in the allowed list
    return [key for key in all_keys if key in allowed_keys]

def check_other_asset(message: str):
    ASSETS_DICT = {
        'ADA-USD_otc': r'Cardano.*(\(?OTC\)?)?',
        'AVAX_otc': r'Avalanche.*(\(?OTC\)?)?',
        'AMZN_otc': r'Amazon.*(\(?OTC\)?)?',
        'AUS200_otc': r'AUS 200.*(\(?OTC\)?)?',
        'BABA_otc': r'Alibaba.*(\(?OTC\)?)?',
        'BITB_otc': r'Bitcoin ETF.*(\(?OTC\)?)?',
        'BNB-USD_otc': r'BNB.*(\(?OTC\)?)?',
        'BTCUSD_otc': r'Bitcoin.*(\(?OTC\)?)?',
        'CITI_otc': r'Citigroup Inc.*(\(?OTC\)?)?',
        'CITI_otc': r'Citigroup.*(\(?OTC\)?)?',
        'D30EUR_otc': r'D30EUR.*(\(?OTC\)?)?',
        'DJI30_otc': r'DJI30.*(\(?OTC\)?)?',
        'DOGE_otc': r'Dogecoin.*(\(?OTC\)?)?',
        'DOTUSD_otc': r'Polkadot.*(\(?OTC\)?)?',
        'E35EUR_otc': r'E35EUR.*(\(?OTC\)?)?',
        'E50EUR_otc': r'E50EUR.*(\(?OTC\)?)?',
        'ETHUSD_otc': r'Ethereum.*(\(?OTC\)?)?',
        'F40EUR_otc': r'F40EUR.*(\(?OTC\)?)?',
        'FDX_otc': r'FedEx.*(\(?OTC\)?)?',
        'JNJ_otc': r'Johnson & Johnson.*(\(?OTC\)?)?',
        'JPN225_otc': r'JPN225.*(\(?OTC\)?)?',
        'LINK_otc': r'Chainlink.*(\(?OTC\)?)?',
        'LTCUSD_otc': r'Litecoin.*(\(?OTC\)?)?',
        'MATIC_otc': r'Polygon.*(\(?OTC\)?)?',
        'MSFT_otc': r'Microsoft.*(\(?OTC\)?)?',
        'NFLX_otc': r'Netflix.*(\(?OTC\)?)?',
        'NASUSD_otc': r'US100.*(\(?OTC\)?)?',
        'SOL-USD_otc': r'Solana.*(\(?OTC\)?)?',
        'SP500_otc': r'SP500.*(\(?OTC\)?)?',
        'TON-USD_otc': r'Toncoin.*(\(?OTC\)?)?',
        'TRX-USD_otc': r'TRON.*(\(?OTC\)?)?',
        'TWITTER_otc': r'Twitter.*(\(?OTC\)?)?',
        'UKBrent_otc': r'Brent Oil.*(\(?OTC\)?)?',
        'USCrude_otc': r'WTI Crude Oil.*(\(?OTC\)?)?',
        'VISA_otc': r'VISA.*(\(?OTC\)?)?',
        'XAGUSD_otc': r'Silver.*(\(?OTC\)?)?',
        'XAUUSD_otc': r'Gold.*(\(?OTC\)?)?',
        'XNGUSD_otc': r'Natural Gas.*(\(?OTC\)?)?',
        'XPDUSD_otc': r'Palladium spot.*(\(?OTC\)?)?',
        'XPTUSD_otc': r'Platinum spot.*(\(?OTC\)?)?',
        'XPRUSD_otc': r'American Express.*(\(?OTC\)?)?',
        '#AAPL_otc': r'Apple.*(\(?OTC\)?)?',
        '#AXP_otc': r'American Express.*(\(?OTC\)?)?',
        '#CSCO_otc': r'Cisco.*(\(?OTC\)?)?',
        '#FB_otc': r'FACEBOOK(?: INC)?.*(\(?OTC\)?)?',
        '#INTC_otc': r'Intel.*(\(?OTC\)?)?',
        '#JNJ_otc': r'Johnson & Johnson.*(\(?OTC\)?)?',
        '#MCD_otc': r"McDonald's.*(\(?OTC\)?)?",
        '#MCD_otc': r"MCDONALD.*(\(?OTC\)?)?",
        '#MSFT_otc': r'Microsoft.*(\(?OTC\)?)?',
        '#PFE_otc': r'Pfizer Inc.*(\(?OTC\)?)?',
        '#TSLA_otc': r'Tesla.*(\(?OTC\)?)?',
        '#XOM_otc': r'ExxonMobil.*(\(?OTC\)?)?',
        '#XOM_otc': r'Exxon Mobile.*(\(?OTC\)?)?'
    }

    for value, regex in ASSETS_DICT.items():
        asset_match = re.search(regex, message, re.IGNORECASE)
        if asset_match:
            if not 'OTC' in asset_match.group():
                value = value.replace('_otc', '')
            
            return value
    return None

def find_levels(df, is_support: bool, window_size: int = 5, tolerance: float = 0.005):
    """
    Finds horizontal support or resistance levels by identifying and clustering pivot points.
    
    Args:
        df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
        is_support (bool): True to find support levels, False for resistance.
        window_size (int): The number of candles on each side of a pivot to check.
        tolerance (float): The percentage tolerance to group nearby levels.

    Returns:
        list: A list of identified level prices.
    """
    if is_support:
        price_col = 'low'
        # A pivot low is a low with 'window_size' lower lows on both sides
        pivots = df[price_col] == df[price_col].rolling(window_size * 2 + 1, center=True).min()
    else:
        price_col = 'high'
        # A pivot high is a high with 'window_size' higher highs on both sides
        pivots = df[price_col] == df[price_col].rolling(window_size * 2 + 1, center=True).max()
        
    pivot_prices = df[pivots][price_col]
    if pivot_prices.empty:
        return []

    # Cluster the pivot prices
    levels = []
    for price in pivot_prices:
        is_new_level = True
        for i, level in enumerate(levels):
            if abs(level - price) / price < tolerance:
                # Price is close to an existing level, update the level to the average
                levels[i] = (level * (df[price_col] == level).sum() + price) / ((df[price_col] == level).sum() + 1)
                is_new_level = False
                break
        if is_new_level:
            levels.append(price)
            
    return sorted(levels)

def find_trend_lines(df, is_support: bool, window_size: int = 5, tolerance: float = 0.01):
    """
    Finds trend lines by connecting pivot points.
    
    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'open', 'close'.
        is_support (bool): True to find support trend lines (connecting lows), False for resistance (connecting highs).
        window_size (int): The window size for pivot detection.
        tolerance (float): The percentage tolerance for a point to be considered on the line.

    Returns:
        list: A list of dictionaries, where each dict represents a trend line 
              with 'points' (indices and prices) and 'slope'.
    """
    if is_support:
        price_col = 'low'
        pivots_idx = df.index[df[price_col] == df[price_col].rolling(window_size * 2 + 1, center=True).min()]
    else:
        price_col = 'high'
        pivots_idx = df.index[df[price_col] == df[price_col].rolling(window_size * 2 + 1, center=True).max()]

    if len(pivots_idx) < 2:
        return []

    trend_lines = []
    # Check all combinations of 2 pivots
    for i in range(len(pivots_idx)):
        for j in range(i + 1, len(pivots_idx)):
            p1_idx, p2_idx = pivots_idx[i], pivots_idx[j]
            p1_price, p2_price = df[price_col].loc[p1_idx], df[price_col].loc[p2_idx]
            
            # Calculate slope
            slope = (p2_price - p1_price) / (p2_idx - p1_idx)
            
            # Check for a third pivot point that confirms the trend line
            touches = [{'idx': p1_idx, 'price': p1_price}, {'idx': p2_idx, 'price': p2_price}]
            
            # Check if price violates the line between the two points
            is_violated = False
            for k in range(p1_idx + 1, p2_idx):
                line_price_at_k = p1_price + slope * (k - p1_idx)
                if is_support and df['low'].loc[k] < line_price_at_k:
                    is_violated = True
                    break
                elif not is_support and df['high'].loc[k] > line_price_at_k:
                    is_violated = True
                    break
            if is_violated:
                continue

            for k_idx in pivots_idx:
                if k_idx not in [p1_idx, p2_idx]:
                    line_price_at_k = p1_price + slope * (k_idx - p1_idx)
                    # Check if the pivot is close to the line
                    if abs(df[price_col].loc[k_idx] - line_price_at_k) / line_price_at_k < tolerance:
                         touches.append({'idx': k_idx, 'price': df[price_col].loc[k_idx]})
            
            # A valid trend line needs at least 3 touches
            if len(touches) >= 3:
                # Sort touches by index
                touches = sorted(touches, key=lambda x: x['idx'])
                trend_lines.append({'points': touches, 'slope': slope})
    
    # Optional: Filter out redundant lines (lines that are too similar)
    # This part can be complex, for now we return all valid 3+ touch lines.
    
    return trend_lines

def analyze_market_structure(df: pd.DataFrame, window_size: int = 5, tolerance_horiz: float = 0.005, tolerance_trend: float = 0.01):
    """
    Analyzes the dataframe to find all S/R levels, flips, and trend lines and returns them in a structured dictionary.
    
    Args:
        df (pd.DataFrame): Input market data.
        window_size (int): Window for pivot detection.
        tolerance_horiz (float): Tolerance for clustering horizontal levels.
        tolerance_trend (float): Tolerance for points on a trend line.

    Returns:
        dict: A dictionary containing all identified market structures.
              Keys: 'horizontal_support', 'horizontal_resistance', 
                    'trendline_support', 'trendline_resistance',
                    'flipped_to_support', 'flipped_to_resistance'.
    """
    # 1. Find standard levels first
    support_levels = find_levels(df, is_support=True, window_size=window_size, tolerance=tolerance_horiz)
    resistance_levels = find_levels(df, is_support=False, window_size=window_size, tolerance=tolerance_horiz)

    # 2. Find flipped levels based on the standard levels
    flipped_to_support = find_flipped_levels(df, resistance_levels, is_support_flip=True, window_size=window_size, tolerance=tolerance_horiz)
    flipped_to_resistance = find_flipped_levels(df, support_levels, is_support_flip=False, window_size=window_size, tolerance=tolerance_horiz)
    
    # 3. Find trend lines
    support_trends = find_trend_lines(df, is_support=True, window_size=window_size, tolerance=tolerance_trend)
    resistance_trends = find_trend_lines(df, is_support=False, window_size=window_size, tolerance=tolerance_trend)
    
    market_structure = {
        "horizontal_support": sorted([s for s in support_levels if s not in flipped_to_resistance]),
        "horizontal_resistance": sorted([r for r in resistance_levels if r not in flipped_to_support]),
        "flipped_support": sorted(flipped_to_support),
        "flipped_resistance": sorted(flipped_to_resistance),
        "trendline_support": support_trends,
        "trendline_resistance": resistance_trends
    }
    
    return market_structure

async def plot_market_structure(df: pd.DataFrame, market_structure: dict, asset: str, timestamp: str, selected_strategy: str):
    """
    Plots the candlestick chart along with the provided market structure.
    
    Args:
        df (pd.DataFrame): The market data.
        market_structure (dict): The dictionary of levels and lines from analyze_market_structure.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot candlestick
    ax.plot(df.index, df['close'], color='gray', alpha=0.7, label='Close Price')
    for idx, row in df.iterrows():
        color = 'cyan' if row['close'] >= row['open'] else 'magenta'
        ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1, zorder=1)
        ax.plot([idx, idx], [row['open'], row['close']], color=color, linewidth=4, zorder=1)

    plot_end_idx = df.index[-1]

    # --- Plotting with Labels ---
    def plot_line_with_label(level, color, linestyle, label_prefix):
        ax.axhline(y=level, color=color, linestyle=linestyle, linewidth=1.2, zorder=2)
        ax.text(plot_end_idx + 1, level, f" {label_prefix} {level:.4f}", va='center', color=color, fontsize=9)

    # support_levels = market_structure['horizontal_support']
    # resistance_levels = market_structure['horizontal_resistance']
        
    # # Plot horizontal levels
    # for level in support_levels:
    #     ax.axhline(y=level, color='lime', linestyle='--', linewidth=1.2, label=f'Support {level:.4f}')
    # for level in resistance_levels:
    #     ax.axhline(y=level, color='red', linestyle='--', linewidth=1.2, label=f'Resistance {level:.4f}')

    for level in market_structure["horizontal_support"]:
        plot_line_with_label(level, 'lime', '--', 'Sup:')
    for level in market_structure["horizontal_resistance"]:
        plot_line_with_label(level, 'red', '--', 'Res:')
    for level in market_structure["flipped_support"]:
        plot_line_with_label(level, 'cyan', ':', 'Flip Sup:') # Dotted Cyan for Flipped Support
    for level in market_structure["flipped_resistance"]:
        plot_line_with_label(level, 'orange', ':', 'Flip Res:') # Dotted Orange for Flipped Resistance

    # support_trends = market_structure['trendline_support']
    # resistance_trends = market_structure['trendline_resistance']
        
    # # Plot trend lines
    # for trend in support_trends:
    #     points = trend['points']
    #     x = [p['idx'] for p in points]
    #     y = [p['price'] for p in points]
    #     # Extend the line
    #     line_x = np.arange(x[0], df.index[-1])
    #     line_y = y[0] + trend['slope'] * (line_x - x[0])
    #     ax.plot(line_x, line_y, color='lime', linestyle='-', linewidth=1.5)

    # for trend in resistance_trends:
    #     points = trend['points']
    #     x = [p['idx'] for p in points]
    #     y = [p['price'] for p in points]
    #     # Extend the line
    #     line_x = np.arange(x[0], df.index[-1])
    #     line_y = y[0] + trend['slope'] * (line_x - x[0])
    #     ax.plot(line_x, line_y, color='red', linestyle='-', linewidth=1.5)

    # Plot trend lines
    # for trend in market_structure["trendline_support"]:
    #     points = trend['points']
    #     line_x = np.arange(points[0]['idx'], plot_end_idx + 1)
    #     line_y = points[0]['price'] + trend['slope'] * (line_x - points[0]['idx'])
    #     ax.plot(line_x, line_y, color='lime', linestyle='-', linewidth=1.5, zorder=2)

    # for trend in market_structure["trendline_resistance"]:
    #     points = trend['points']
    #     line_x = np.arange(points[0]['idx'], plot_end_idx + 1)
    #     line_y = points[0]['price'] + trend['slope'] * (line_x - points[0]['idx'])
    #     ax.plot(line_x, line_y, color='red', linestyle='-', linewidth=1.5, zorder=2)
        
    ax.set_title("Market Structure Analysis (with Flips and Labels)")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date/Time Index")
    ax.grid(True, linestyle='--', alpha=0.2)
    plt.margins(x=0.08) # Add space on the right for labels
    
    # Create output directory
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{selected_strategy}_{asset}_time_{timestamp}.png"
    await asyncio.sleep(0)
    plt.savefig(filename)

def check_trade_conditions(current_price: float, current_index: int, market_structure: dict, proximity_pct: float = 0.005):
    """
    Analyzes trade conditions against S/R levels and provides two decision sets:
    1. The primary decision based only on horizontal and flipped levels.
    2. A hypothetical decision that includes trend lines as hard rules.

    Args:
        current_price (float): The current market price.
        current_index (int): The index of the current candle.
        market_structure (dict): The dictionary of levels from analyze_market_structure.
        proximity_pct (float): The percentage to define "nearness" to a level.

    Returns:
        dict: A nested dictionary containing 'decision_current' and 'decision_with_trends'.
    """
    # --- Step 1: Find all nearby zones of each type ---
    
    nearby_hr = [ # Horizontal Resistance
        {'type': 'Resistance', 'name': 'Horizontal Resistance', 'value': l, 'distance': abs(l - current_price) / current_price}
        for l in market_structure['horizontal_resistance']
        if current_price <= l and (l - current_price) / current_price < proximity_pct
    ]
    nearby_fr = [ # Flipped Resistance
        {'type': 'Resistance', 'name': 'Flipped Resistance', 'value': l, 'distance': abs(l - current_price) / current_price}
        for l in market_structure['flipped_resistance']
        if current_price <= l and (l - current_price) / current_price < proximity_pct
    ]
    nearby_hs = [ # Horizontal Support
        {'type': 'Support', 'name': 'Horizontal Support', 'value': l, 'distance': abs(l - current_price) / current_price}
        for l in market_structure['horizontal_support']
        if current_price >= l and (current_price - l) / current_price < proximity_pct
    ]
    nearby_fs = [ # Flipped Support
        {'type': 'Support', 'name': 'Flipped Support', 'value': l, 'distance': abs(l - current_price) / current_price}
        for l in market_structure['flipped_support']
        if current_price >= l and (current_price - l) / current_price < proximity_pct
    ]

    nearby_tr = [] # Trendline Resistance
    for trend in market_structure['trendline_resistance']:
        p1 = trend['points'][0]
        line_val = p1['price'] + trend['slope'] * (current_index - p1['idx'])
        if current_price <= line_val and (line_val - current_price) / current_price < proximity_pct:
            nearby_tr.append({'type': 'Resistance', 'name': 'Trend Line Resistance', 'value': line_val, 'distance': abs(line_val - current_price) / current_price})

    nearby_ts = [] # Trendline Support
    for trend in market_structure['trendline_support']:
        p1 = trend['points'][0]
        line_val = p1['price'] + trend['slope'] * (current_index - p1['idx'])
        if current_price >= line_val and (current_price - line_val) / current_price < proximity_pct:
            nearby_ts.append({'type': 'Support', 'name': 'Trend Line Support', 'value': line_val, 'distance': abs(current_price - line_val) / current_price})

    # --- Step 2: Define a helper function to generate decisions from zones ---
    
    def _determine_decision(resistances, supports):
        all_zones = resistances + supports
        
        # If no zones are nearby, all trades are allowed.
        if not all_zones:
            return {
                'current_price': current_price,
                'allow_buy': True, 'buy_reason': "Clear",
                'allow_sell': True, 'sell_reason': "Clear"
            }
        
        # Find the single closest zone to the price.
        closest_zone = min(all_zones, key=lambda x: x['distance'])
        
        # If the closest zone is a SUPPORT, block sells and allow buys.
        if closest_zone['type'] == 'Support':
            return {
                'current_price': current_price,
                'allow_buy': True,
                'buy_reason': "Clear",
                'allow_sell': False,
                'sell_reason': f"Blocked by {closest_zone['name']} at ~{closest_zone['value']:.4f}"
            }
        
        # If the closest zone is a RESISTANCE, block buys and allow sells.
        elif closest_zone['type'] == 'Resistance':
            return {
                'current_price': current_price,
                'allow_buy': False,
                'buy_reason': f"Blocked by {closest_zone['name']} at ~{closest_zone['value']:.4f}",
                'allow_sell': True,
                'sell_reason': "Clear"
            }

    # --- Step 3: Calculate the two different decision sets ---
    
    # Decision 1: The "real" decision using only horizontal and flipped levels
    current_resistances = nearby_hr + nearby_fr
    current_supports = nearby_hs + nearby_fs
    decision_current = _determine_decision(current_resistances, current_supports)

    # Decision 2: The hypothetical decision including trend lines
    all_resistances = current_resistances + nearby_tr
    all_supports = current_supports + nearby_ts
    decision_with_trends = _determine_decision(all_resistances, all_supports)

    return {
        'decision_current': decision_current,
        'decision_with_trends': decision_with_trends
    }

def find_flipped_levels(df: pd.DataFrame, original_levels: list, is_support_flip: bool, window_size: int = 5, tolerance: float = 0.005):
    """
    Identifies levels that have "flipped" from support to resistance or vice-versa.

    Args:
        df (pd.DataFrame): The market data.
        original_levels (list): The list of pre-identified support or resistance levels.
        is_support_flip (bool): True to check for resistance flipping to support, 
                                False for support flipping to resistance.
        window_size (int): The window for pivot detection.
        tolerance (float): The tolerance for a price to be considered "at" a level.

    Returns:
        list: A list of prices for levels that have flipped.
    """
    flipped_levels = []
    
    for level in original_levels:
        # Find the index of the last time this level acted as its original role
        if is_support_flip: # Original role was RESISTANCE
            price_col_original = 'high'
            pivots_original = df[price_col_original] == df[price_col_original].rolling(window_size*2+1, center=True).max()
            relevant_pivots = df[(pivots_original) & (abs(df[price_col_original] - level) / level < tolerance)]
        else: # Original role was SUPPORT
            price_col_original = 'low'
            pivots_original = df[price_col_original] == df[price_col_original].rolling(window_size*2+1, center=True).min()
            relevant_pivots = df[(pivots_original) & (abs(df[price_col_original] - level) / level < tolerance)]

        if relevant_pivots.empty:
            continue
            
        last_original_pivot_idx = relevant_pivots.index[-1]
        
        # Data after the last original touch
        df_after = df.loc[last_original_pivot_idx + 1:]
        
        # Check for a break of the level
        if is_support_flip: # Break ABOVE resistance
            break_occurred = (df_after['close'] > level).any()
        else: # Break BELOW support
            break_occurred = (df_after['close'] < level).any()
            
        if not break_occurred:
            continue
            
        # Check for a retest in the new role
        if is_support_flip: # Retest as SUPPORT (pivot low)
            price_col_new = 'low'
            pivots_new = df_after[price_col_new] == df_after[price_col_new].rolling(window_size*2+1, center=True).min()
            retest_pivots = df_after[(pivots_new) & (abs(df_after[price_col_new] - level) / level < tolerance)]
        else: # Retest as RESISTANCE (pivot high)
            price_col_new = 'high'
            pivots_new = df_after[price_col_new] == df_after[price_col_new].rolling(window_size*2+1, center=True).max()
            retest_pivots = df_after[(pivots_new) & (abs(df_after[price_col_new] - level) / level < tolerance)]
            
        if not retest_pivots.empty:
            flipped_levels.append(level)
            
    return list(set(flipped_levels)) # Return unique levels