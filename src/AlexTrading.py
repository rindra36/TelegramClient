"""
Alex Trading Bot Module

This module implements a Telegram bot that monitors specific channels for trading signals and
executes trades on PocketOption platform.
"""

import asyncio
import logging
import re

from typing import Dict, Optional
from telethon import events
from PocketOptionAPI import PocketOptionAPI
from src.utils import setup_logging, load_credentials, load_chats, safe_trade, find_trade_in_opened_deals, \
    get_trade_result, determine_trade_result, setup_client, wait_until_close_timestamp

# Constants
SESSION_NAME = 'Alex'
TRADE_PATTERNS = {
    'ASSET': r'([A-Z]{3})/([A-Z]{3})\s+(\(OTC\))?',
    'EXPIRATION': r'Duration\s+(\d)\s+minutes?$',
    'ACTION': r'Deal\s+for\s+(BUY|SELL)'
}

class AlexTradingBot:
    """
    Trading bot that monitors Telegram channels for signals and executes trades on PocketOption.

    This bot processes messages from configured Telegram channels, extracts trading signals,
    and executes corresponding trades on the PocketOption platform.
    """
    def __init__(self):
        """Initialize the trading bot with configuration and API connections."""
        setup_logging(SESSION_NAME)
        self.credentials = load_credentials()
        self.chats = load_chats()
        self.client = setup_client(SESSION_NAME, self.credentials['id'], self.credentials['hash'])
        self.pocket_option = PocketOptionAPI(2, 'real')

    async def handle_trade_execution(self, message: str) -> bool:
        """
        Process trading signals and execute trades based on message content.

        Args:
            message: Raw message text from Telegram

        Returns:
            bool: True if trade execution was successful, False otherwise
        """
        channel = SESSION_NAME

        # Extract trading parameters
        action, asset, expiration = self.parse_trading_signal(message)
        if not all([action, asset, expiration]):
            return False
        
        # Initialize trade data
        self.pocket_option.set_channel_data(channel, {
            'action': action,
            'asset': asset,
            'expiration': int(expiration) * 60,
            'amount': 1.0,
        })

        return await self._execute_new_trade(channel)    

    async def _execute_new_trade(self, channel: str) -> bool:
        """Set up and execute a new trade."""
        logging.info(f'Trade parameters: {self.pocket_option.get_channel_data(channel)}')

        try:
            return await self.execute_trade_cycle(channel)
        except Exception as e:
            logging.error(f'Trade cycle failed: {e}')
            return False

    @staticmethod
    def parse_trading_signal(message: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse trading signal from message text.

        Returns:
            tuple: (action, asset, expiration_time)
        """
        action = expiration = asset = None

        # Verify message format
        if not re.search(r'Deal\s+for\s+', message, re.IGNORECASE):
            return action, asset, expiration

        # Check for action pattern
        action_match = re.search(TRADE_PATTERNS['ACTION'], message, re.IGNORECASE)
        if action_match:
            action = action_match.group(1)

        # Extract asset if this is the message
        asset_match = re.search(TRADE_PATTERNS['ASSET'], message, re.IGNORECASE)
        if asset_match:
            has_otc = ''
            if asset_match.group(3) and asset_match.group(3).lower() == '(otc)':
                has_otc = '_otc'
            print(asset_match)
            asset = f'{asset_match.group(1)}{asset_match.group(2)}{has_otc}'

        # Extract expiration if this is the message
        expiration_match = re.search(TRADE_PATTERNS['EXPIRATION'], message, re.IGNORECASE)
        if expiration_match:
            expiration = expiration_match.group(1)

        return action, asset, expiration

    async def execute_trade_cycle(self, channel: str) -> bool:
        """
        Execute a complete trade cycle including verification and result checking.

        Args:
            channel: Channel identifier for the trade

        Returns:
            bool: True if trade was successful, False otherwise
        """
        logging.info('INITIATE TRADE')
        trade_id = await safe_trade(self.pocket_option, channel)

        if not trade_id:
            trade_id = await find_trade_in_opened_deals(self.pocket_option, channel)
            if not trade_id:
                return False

        # Remove the channel data right after the trade is placed
        self.pocket_option.remove_channel_data(channel)
        logging.info(f'TRADE PLACED SUCCESSFULLY: "{trade_id}" AND WAITING FOR RESPONSE')

        return await self._monitor_trade_result(trade_id, channel)

    async def _monitor_trade_result(self, trade_id: str, channel: str) -> bool:
        """Monitor and process trade results."""
        trade_data = await get_trade_result(self.pocket_option, trade_id)
        if not trade_data:
            return False

        trade_result = determine_trade_result(trade_data)
        logging.info(f'Trade result: {trade_result}')

        if trade_result in ['win', 'draw']:
            logging.info(f'Trade successful: {trade_result}')
            return True

        logging.warning(f'Trade {trade_id} failed ({trade_result})')
        return False

    async def start(self) -> None:
        """Start the trading bot"""

        # @self.client.on(events.NewMessage(chats=self.chats[SESSION_NAME]))
        @self.client.on(events.NewMessage)
        async def message_handler(event):
            await self.handle_trade_execution(event.raw_text)

        await self.client.start()
        await self.client.run_until_disconnected()


def main():
    """Main entry point for the trading bot."""
    try:
        bot = AlexTradingBot()
        asyncio.run(bot.start())
    except Exception as e:
        logging.error(f'Bot initialization failed : {e}')


if __name__ == "__main__":
    main()