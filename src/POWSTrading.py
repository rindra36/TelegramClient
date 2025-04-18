"""
POWS Trading Bot Module

This module implements a Telegram bot that monitors specific channels for trading signals and
executes trades on PocketOption platform.
"""

import asyncio
import logging
import re

from typing import Dict
from telethon import events
from PocketOptionAPI import PocketOptionAPI
from src.utils import setup_logging, load_credentials, load_chats, safe_trade, find_trade_in_opened_deals, \
    get_trade_result, determine_trade_result, setup_client, wait_until_close_timestamp

# Constants
SESSION_NAME = 'POWS'
TRADE_PATTERNS = {
    'ASSET': r'([A-Z]{3})/.*([A-Z]{3}).*OTC',
    'EXPIRATION': r'(\d)\s+minutes?$',
    'ACTION': r'(UP|DOWN)$'
}

class POWSTradingBot:
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
        self.pocket_option = PocketOptionAPI(2)

    async def handle_trade_execution(self, message: str) -> bool:
        """
        Process trading signals and execute trades based on message content.

        Args:
            message: Raw message text from Telegram

        Returns:
            bool: True if trade execution was successful, False otherwise
        """
        channel = SESSION_NAME

        # Normally, one must be empty at this point. Only when we need to check one value that all is full
        # Remove the channel data because normally, it is a new data
        if self._should_reset_channel_data(channel):
            self.pocket_option.remove_channel_data(channel)

        if not self._has_complete_trade_data(channel):
            extracted_data = self.parse_trading_signal(message)
            if not extracted_data:
                return False

            logging.info(extracted_data)
            self.pocket_option.set_value(channel, extracted_data['key'], extracted_data['value'])

            if self._has_complete_trade_data(channel):
                return await self._execute_new_trade(channel)

        return False

    def _should_reset_channel_data(self, channel: str) -> bool:
        """Check if channel data should be reset."""
        return all([
            self.pocket_option.get_value(channel, 'action'),
            self.pocket_option.get_value(channel, 'asset'),
            self.pocket_option.get_value(channel, 'expiration')
        ])

    def _has_complete_trade_data(self, channel: str) -> bool:
        """Check if all required trade data is present."""
        return all([
            self.pocket_option.get_value(channel, key)
            for key in ['action', 'asset', 'expiration']
        ])

    async def _execute_new_trade(self, channel: str) -> bool:
        """Set up and execute a new trade."""
        self.pocket_option.set_value(channel, 'amount', 1.0)
        logging.info(f'Trade parameters: {self.pocket_option.get_channel_data(channel)}')

        try:
            return await self.execute_trade_cycle(channel)
        except Exception as e:
            logging.error(f'Trade cycle failed: {e}')
            return False

    @staticmethod
    def parse_trading_signal(message: str) -> Dict[str, str] | bool:
        """
        Parse trading signal from message text.

        Args:
            message: Raw message text to parse

        Returns:
            Dict containing extracted key-value pair or False if no match found
        """
        # Check for action pattern
        action_match = re.search(TRADE_PATTERNS['ACTION'], message, re.IGNORECASE)
        if action_match:
            if action_match.group(1).upper() == 'UP':
                return {'key': 'action', 'value': 'BUY'}
            elif action_match.group(1).upper() == 'DOWN':
                return {'key': 'action', 'value': 'SELL'}

        # Extract asset if this is the message
        asset_match = re.search(TRADE_PATTERNS['ASSET'], message, re.IGNORECASE)
        if asset_match:
            return {'key': 'asset', 'value': f'{asset_match.group(1)}{asset_match.group(2)}_otc'}

        # Extract expiration if this is the message
        expiration_match = re.search(TRADE_PATTERNS['EXPIRATION'], message, re.IGNORECASE)
        if expiration_match:
            return {'key': 'expiration', 'value': int(expiration_match.group(1)) * 60}

        return False

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
        bot = POWSTradingBot()
        asyncio.run(bot.start())
    except Exception as e:
        logging.error(f'Bot initialization failed : {e}')


if __name__ == "__main__":
    main()