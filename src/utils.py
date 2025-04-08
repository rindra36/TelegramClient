"""
Utility functions for binary options trading operations.
Handles time conversions, configuration loading, and safe trading operations.
"""

import json
import asyncio
import logging
from logging import RootLogger

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
    -4: 'America/New_York',  # UTC-4
    -3: 'America/Sao_Paulo'  # UTC-3
}

TRADE_EXECUTION_BUFFER: float = 0.5  # Seconds to subtract from wait time for API preparation
ROOT_PATH = Path(__file__).parents[1]
CONFIG_PATH = f'{ROOT_PATH}/assets/telegramCredentials.json'


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
    with open(f'{ROOT_PATH}/assets/chats.json', 'r', encoding='utf-8') as f:
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
        f'{session_name}-home',
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