"""
Prince Trading Bot Module

This module implements a Telegram bot that monitors specific channels for trading signals and
executes trades on PocketOption platform.
"""

import asyncio
import logging
import re

from typing import Dict
from src.utils import safe_trade, find_trade_in_opened_deals, get_trade_result, determine_trade_result

class PrinceTradingBot:
    """
    Trading bot that monitors Telegram channels for signals and executes trades on PocketOption.

    This bot processes messages from configured Telegram channels, extracts trading signals,
    and executes corresponding trades on the PocketOption platform.
    """
    def __init__(self, pocket_option):
        """Initialize the trading bot with configuration and API connections."""
        self.SESSION_NAME = 'Prince'
        self.TRADE_PATTERNS = {
            'ASSET': r'([A-Z]{3})([A-Z]{3})(?:\s+OTC)?$',
            'ACTION_EXPIRATION': r'(CALL|PUT).*for\s+(\d+)\s+minute'
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

        # Normally, one must be empty at this point. Only when we need to check one value that all is full
        # Remove the channel data because normally, it is a new data
        # if self._should_reset_channel_data(channel):
        #     self.pocket_option.remove_channel_data(channel)

        if not self._has_complete_trade_data(channel):
            extracted_data = self.parse_trading_signal(message)
            if not extracted_data:
                return False

            logging.info(f'PrinceTrading.py {extracted_data}')
            if type(extracted_data) == list:
                for data in extracted_data:
                    self.pocket_option.set_value(channel, data['key'], data['value'])
            else:
                # Normally, one must be empty at this point. Only when we need to check one value that all is full
                # Remove the channel data because normally, it is a new data
                self.pocket_option.remove_channel_data(channel)
                self.pocket_option.set_value(channel, extracted_data['key'], extracted_data['value'])

            if self._has_complete_trade_data(channel):
                return await self._execute_new_trade(channel)

        return False

    def _should_reset_channel_data(self, channel: str) -> bool:
        """Check if channel data should be reset."""
        return all([
            self.pocket_option.get_value(channel, 'action'),
            self.pocket_option.get_value(channel, 'asset'),
            self.pocket_option.get_value(channel, 'expiration'),
            self.pocket_option.get_value(channel, 'amount')
        ])

    def _has_complete_trade_data(self, channel: str) -> bool:
        """Check if all required trade data is present."""
        return all([
            self.pocket_option.get_value(channel, key)
            for key in ['action', 'asset', 'expiration', 'amount']
        ])

    async def _execute_new_trade(self, channel: str) -> bool:
        """Set up and execute a new trade."""
        logging.info(f'PrinceTrading.py Trade parameters: {self.pocket_option.get_channel_data(channel)}')

        try:
            return await self.execute_trade_cycle(channel)
        except Exception as e:
            logging.error(f'PrinceTrading.py Trade cycle failed: {e}')
            return False

    def parse_trading_signal(self, message: str) -> Dict[str, str] | bool:
        """
        Parse trading signal from message text.

        Args:
            message: Raw message text to parse

        Returns:
            Dict containing extracted key-value pair or False if no match found
        """
        # Check for action pattern
        action_expiration_match = re.search(self.TRADE_PATTERNS['ACTION_EXPIRATION'], message, re.IGNORECASE)
        if action_expiration_match:
            amount = 1.0

            if 'safe' in message.lower():
                amount *= 2

            return [
                {'key': 'action', 'value': action_expiration_match.group(1)},
                {'key': 'expiration', 'value': int(action_expiration_match.group(2)) * 60},
                {'key': 'amount', 'value': amount}
            ]

        # Extract asset if this is the message
        asset_expiration_match = re.search(self.TRADE_PATTERNS['ASSET'], message, re.IGNORECASE)
        if asset_expiration_match:
            has_otc = ''
            if 'otc' in message.lower():
                has_otc = '_otc'

            return {'key': 'asset', 'value': f'{asset_expiration_match.group(1)}{asset_expiration_match.group(2)}{has_otc}'}

        return False

    async def execute_trade_cycle(self, channel: str) -> bool:
        """
        Execute a complete trade cycle including verification and result checking.

        Args:
            channel: Channel identifier for the trade

        Returns:
            bool: True if trade was successful, False otherwise
        """
        logging.info('PrinceTrading.py INITIATE TRADE')
        trade_id = await safe_trade(self.pocket_option, channel)

        if not trade_id:
            trade_id = await find_trade_in_opened_deals(self.pocket_option, channel)
            if not trade_id:
                return False

        # Remove the amount data right after the trade is placed
        self.pocket_option.set_value(channel, 'amount', None)
        logging.info(f'PrinceTrading.py TRADE PLACED SUCCESSFULLY: "{trade_id}"')

        return True

        # return await self._monitor_trade_result(trade_id, channel)

    async def _monitor_trade_result(self, trade_id: str, channel: str) -> bool:
        """Monitor and process trade results."""
        trade_data = await get_trade_result(self.pocket_option, trade_id)
        if not trade_data:
            return False

        trade_result = determine_trade_result(trade_data)
        logging.info(f'PrinceTrading.py Trade result: {trade_result}')

        if trade_result in ['win', 'draw']:
            logging.info(f'PrinceTrading.py Trade successful: {trade_result}')
            return True

        logging.warning(f'PrinceTrading.py Trade {trade_id} failed ({trade_result})')
        return False

    async def start(self, message) -> None:
        """Start the trading bot"""
        await self.handle_trade_execution(message)
