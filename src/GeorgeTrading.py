"""
George Trading Bot Module

This module implements a Telegram bot that monitors specific channels for trading signals and
executes trades on PocketOption platform.
"""

import asyncio
import logging
import re

from typing import Dict
from src.utils import safe_trade, find_trade_in_opened_deals, get_trade_result, determine_trade_result

class GeorgeTradingBot:
    """
    Trading bot that monitors Telegram channels for signals and executes trades on PocketOption.

    This bot processes messages from configured Telegram channels, extracts trading signals,
    and executes corresponding trades on the PocketOption platform.
    """
    def __init__(self, pocket_option):
        """Initialize the trading bot with configuration and API connections."""
        self.SESSION_NAME = 'George'
        self.TRADE_PATTERNS = {
            'ASSET_EXPIRATION_TIME_ACTION': r'(\w{3})(?:\s+)?(\w{3})(?:\s+)?(?:OTC\s+)?(?:REAL(?:\s+)?MARKET\s+)?.*?(\d+)(?:-)(\d+)(?:\s+)MIN.*?(?:\s+)?.*?(BUY|SELL)',
            'GO_MESSAGE': r'ALL GO NOW'
        }
        self.pocket_option = pocket_option

    async def handle_trade_execution(self, message: str) -> bool:
        """
        Process trading signals and execute trades based on message content.

        Args:
            message: Raw message text from Telegram

        Returns:
            bool: True if trade execution was successful, False otherwise
        """
        channel = self.SESSION_NAME

        go_message = re.search(self.TRADE_PATTERNS['GO_MESSAGE'], message, re.IGNORECASE)

        # Normally, one must be empty at this point. Only when we need to check one value that all is full
        # Remove the channel data because normally, it is a new data
        if self._should_reset_channel_data(channel) and not go_message:
            self.pocket_option.remove_channel_data(channel)

        if not self._has_complete_trade_data(channel) or go_message:
            if not go_message:
                extracted_data = self.parse_trading_signal(message)
                if not extracted_data:
                    return False

                logging.info(f'GeorgeTrading.py {extracted_data}')
                if type(extracted_data) == list:
                    for data in extracted_data:
                        self.pocket_option.set_value(channel, data['key'], data['value'])
                else:
                    self.pocket_option.set_value(channel, extracted_data['key'], extracted_data['value'])

                if not go_message:
                    logging.info(f'GeorgeTrading.py Waiting for the go message')
                    return False

            logging.info(f'Go message received, proceeding with trade execution {self.pocket_option.get_channel_data(channel)}')
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
        logging.info(f'GeorgeTrading.py Trade parameters: {self.pocket_option.get_channel_data(channel)}')

        try:
            return await self.execute_trade_cycle(channel)
        except Exception as e:
            logging.error(f'GeorgeTrading.py Trade cycle failed: {e}')
            return False

    def parse_trading_signal(self, message: str) -> Dict[str, str] | bool:
        """
        Parse trading signal from message text.

        Args:
            message: Raw message text to parse

        Returns:
            Dict containing extracted key-value pair or False if no match found
        """
        # Check for asset, expiration, and action in the message
        asset_expiration_action_match = re.search(self.TRADE_PATTERNS['ASSET_EXPIRATION_TIME_ACTION'], message, re.IGNORECASE)
        if asset_expiration_action_match:
            has_otc = ''
            if 'otc' in message.lower():
                has_otc = '_otc'

            return [
                {'key': 'asset', 'value': f'{asset_expiration_action_match.group(1)}{asset_expiration_action_match.group(2)}{has_otc}'},
                {'key': 'expiration', 'value': int(asset_expiration_action_match.group(4)) * 60},
                {'key': 'action', 'value': asset_expiration_action_match.group(5).upper()}
            ]

        return False

    async def execute_trade_cycle(self, channel: str) -> bool:
        """
        Execute a complete trade cycle including verification and result checking.

        Args:
            channel: Channel identifier for the trade

        Returns:
            bool: True if trade was successful, False otherwise
        """
        # Execute the trade 5 times
        for _ in range(5):
            logging.info(f'GeorgeTrading.py INITIATE TRADE {_ + 1}')
            trade_id = await safe_trade(self.pocket_option, channel)

            if not trade_id:
                logging.error(f'GeorgeTrading.py Failed to place trade {_ + 1}')
                if _ == 4:
                    return False
                
                continue

            logging.info(f'GeorgeTrading.py TRADE PLACED SUCCESSFULLY: "{trade_id}"')

        # Remove the channel data right after the trade is placed
        self.pocket_option.remove_channel_data(channel)

        return True

        # return await self._monitor_trade_result(trade_id, channel)

    async def _monitor_trade_result(self, trade_id: str, channel: str) -> bool:
        """Monitor and process trade results."""
        trade_data = await get_trade_result(self.pocket_option, trade_id)
        if not trade_data:
            return False

        trade_result = determine_trade_result(trade_data)
        logging.info(f'GeorgeTrading.py Trade result: {trade_result}')

        if trade_result in ['win', 'draw']:
            logging.info(f'GeorgeTrading.py Trade successful: {trade_result}')
            return True

        logging.warning(f'GeorgeTrading.py Trade {trade_id} failed ({trade_result})')
        return False

    async def start(self, message) -> None:
        """Start the trading bot"""
        await self.handle_trade_execution(message)
