"""
AureaBot Trading Bot Module

This module implements a Telegram bot that monitors specific channels for trading signals and
executes trades on PocketOption platform.
"""

import asyncio
import logging
import re


from datetime import datetime, timedelta, timezone
from typing import Dict
from telethon import events
from PocketOptionAPI import PocketOptionAPI
from BinaryOptionsToolsV2.validator import Validator
from src.utils import setup_logging, load_credentials, load_chats, safe_trade, find_trade_in_opened_deals, \
    get_trade_result, determine_trade_result, setup_client

class AureaTradingBot:
    """
    Trading bot that monitors Telegram channels for signals and executes trades on PocketOption.

    This bot processes messages from configured Telegram channels, extracts trading signals,
    and executes corresponding trades on the PocketOption platform.
    """
    def __init__(self):
        """Initialize the trading bot with configuration and API connections."""
        self.SESSION_NAME = 'Aurea'
        self.TRADE_PATTERNS = {
            'ACTION': r'(CALL|PUT)',
            'ASSETS': r'Analyzing signals for (.*)',
            'OPEN_PRICE': r'Entry Point:\s+(\d+\.\d+)',
            'RESISTANCE': r'Resistance Level:\s+(\d+\.\d+)',
            'SUPPORT': r'Caution: Avoid if price.*(\d+\.\d+)'
        }
        setup_logging(self.SESSION_NAME)
        self.credentials = load_credentials()
        self.chats = load_chats()
        self.client = setup_client(self.SESSION_NAME, self.credentials['id'], self.credentials['hash'])
        self.pocket_option = PocketOptionAPI(2, 'real')
        self.asset = None

    async def handle_trade_execution(self, message: str) -> bool:
        """
        Process trading signals and execute trades based on message content.

        Args:
            message: Raw message text from Telegram

        Returns:
            bool: True if trade execution was successful, False otherwise
        """
        channel = self.SESSION_NAME

        # Normally, one must be empty at this point. Only when we need to check one value that all is full
        # Remove the channel data because normally, it is a new data
        if self._should_reset_channel_data(channel):
            self.pocket_option.remove_channel_data(channel)

        if not self._has_complete_trade_data(channel):
            extracted_data = self.parse_trading_signal(message)
            if not extracted_data:
                return False

            logging.info(f'AureaBot.py {extracted_data}')
            if type(extracted_data) == list:
                for data in extracted_data:
                    self.pocket_option.set_value(channel, data['key'], data['value'])
            else:
                self.pocket_option.set_value(channel, extracted_data['key'], extracted_data['value'])

            if self._has_complete_trade_data(channel):
                return await self._execute_new_trade(channel)

        return False

    def _should_reset_channel_data(self, channel: str) -> bool:
        """Check if channel data should be reset."""
        return all([
            self.pocket_option.get_value(channel, 'action'),
            self.pocket_option.get_value(channel, 'asset'),
            self.pocket_option.get_value(channel, 'open_price'),
            self.pocket_option.get_value(channel, 'resistance'),
            self.pocket_option.get_value(channel, 'support'),
        ])

    def _has_complete_trade_data(self, channel: str) -> bool:
        """Check if all required trade data is present."""
        return all([
            self.pocket_option.get_value(channel, key)
            for key in ['action', 'asset', 'resistance', 'support', 'open_price']
        ])

    async def _execute_new_trade(self, channel: str) -> bool:
        """Set up and execute a new trade."""
        self.pocket_option.set_value(channel, 'amount', 1.0)
        logging.info(f'AureaBot.py Trade parameters: {self.pocket_option.get_channel_data(channel)}')

        try:
            return await self.execute_trade_cycle(channel)
        except Exception as e:
            logging.error(f'AureaBot.py Trade cycle failed: {e}')
            return False

    def parse_trading_signal(self, message: str) -> Dict[str, str] | bool:
        """
        Parse trading signal from message text.

        Args:
            message: Raw message text to parse

        Returns:
            Dict containing extracted key-value pair or False if no match found
        """
        """
        Check for :
            - action pattern
            - open price pattern
            - resistance value pattern
            - support value pattern
            - asset pattern
        """
        action_match = re.search(self.TRADE_PATTERNS['ACTION'], message, re.IGNORECASE)
        open_price_match = re.search(self.TRADE_PATTERNS['OPEN_PRICE'], message, re.IGNORECASE)
        resistance_match = re.search(self.TRADE_PATTERNS['RESISTANCE'], message, re.IGNORECASE)
        support_match = re.search(self.TRADE_PATTERNS['SUPPORT'], message, re.IGNORECASE)
        if action_match and open_price_match and resistance_match and support_match:
            return [
                {'key': 'action', 'value': action_match.group(1)},
                {'key': 'open_price', 'value': open_price_match.group(1)},
                {'key': 'resistance', 'value': resistance_match.group(1)},
                {'key': 'support', 'value': support_match.group(1)}
            ]

        # Extract asset if this is the message
        asset_match = re.search(self.TRADE_PATTERNS['ASSETS'], message, re.IGNORECASE)
        if asset_match:
            return {'key': 'asset', 'value': asset_match.group(1)}

        return False
    
    async def execute_trade_cycle(self, channel: str) -> bool:
        """
        Execute a complete trade cycle including verification and result checking.

        Args:
            channel: Channel identifier for the trade

        Returns:
            bool: True if trade was successful, False otherwise
        """
        logging.info('AureaBot.py INITIATE PENDING TRADE')
        raw_order = r'42["openPendingOrder",{"openType":1,"amount":%f,"asset":"%s","openTime":0,"openPrice":%f,"timeframe":300,"minPayout":10,"command":%d}]'
        command = 1 if self.pocket_option.get_value(channel, 'action') == 'PUT' else 0
        formatted_order = raw_order % (float(self.pocket_option.get_value(channel, 'amount')), self.pocket_option.get_value(channel, 'asset'), float(self.pocket_option.get_value(channel, 'open_price')), command)
        validator = Validator.contains('{"data":{"ticket":"')
        response = await self.pocket_option.create_raw_order(formatted_order, validator)

        # Remove the channel data right after the trade is placed
        self.pocket_option.remove_channel_data(channel)
        logging.info(f'AureaBot.py PENDING TRADE PLACED SUCCESSFULLY: "{response}"')

    async def start(self) -> None:
        """Start the trading bot"""

        @self.client.on(events.NewMessage(chats=self.chats[self.SESSION_NAME]))
        async def message_handler(event):
            await self.handle_trade_execution(event.raw_text)

        await self.client.start()
        await self.client.run_until_disconnected()


def main():
    """Main entry point for the trading bot."""
    try:
        bot = AureaTradingBot()
        asyncio.run(bot.start())
    except Exception as e:
        logging.error(f'Bot initialization failed : {e}')


if __name__ == "__main__":
    main()