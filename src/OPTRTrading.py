"""
OPTR Trading Bot Module

This module implements a trading bot for OTC Pro Trading Robot using the PocketOption API.
It monitors Telegram messages and executes trades based on signals.

Key Features:
- Monitors specific Telegram channels for trading signals
- Executes trades through PocketOption API
- Implements retry mechanism for failed trades
- Provides comprehensive logging
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from telethon import TelegramClient, events

from PocketOptionAPI import PocketOptionAPI
from src.utils import get_telegram_credentials, safe_trade, setup_logging, load_credentials, setup_client, load_chats, \
    find_trade_in_opened_deals, get_trade_result, determine_trade_result

# Constants
SESSION_NAME = 'OPTR'
DO_MARTINGALE = True # To control if we will going to Martingale or not because it is not working properly actually, set to True if want to go to Martingale

# Message pattern constants
TRADE_PATTERNS = {
    'PREPARE': r'Preparing trading asset',
    'SUMMARY': r'Summary:',
    'REPORT': r'Closing price',
    'ASSET': r'([A-Z]{3})/.*([A-Z]{3})\s+OTC',
    'EXPIRATION': r'Expiration time:\s+(\d)',
    'ACTION': r'(BUY|SELL)'
}


@dataclass
class TradeSignal:
    """Represents a trading signal with all necessary parameters"""
    asset: Optional[str] = None
    expiration: Optional[int] = None
    action: Optional[str] = None
    amount: float = 1.0
    retry_count: int = 0
    trade_id: Optional[str] = None


class OPTRTradingBot:
    """
    Main trading bot class that handles Telegram messages and executes trades
    """

    def __init__(self):
        setup_logging(SESSION_NAME)
        self.credentials = load_credentials()
        self.trades: Dict[str, TradeSignal] = {}
        self.pocket_option = PocketOptionAPI('real')
        self.client = setup_client(SESSION_NAME, self.credentials['id'], self.credentials['hash'])
        self.chats = load_chats()
        self.pocket_option.MAX_RETRY = 2
        self.pocket_option.MAX_BID = 4

    async def handle_trade_execution(self, message: str) -> bool:
        """
        Process trading signals and execute trades based on message content.

        Args:
            message: Raw message text from Telegram

        Returns:
            bool: True if trade execution was successful, False otherwise
        """
        channel = SESSION_NAME

        # Check if message matches any trading pattern
        pattern = r'(Preparing trading asset|Closing price|Summary:|Result:)'
        match = re.search(pattern, message, re.IGNORECASE)

        if not match:
            return False

        match_type = match.group(1)

        # Extract trading parameters
        try:
            if 'Preparing trading asset' in match_type:
                await self.handle_prepare_signal(message, channel)
            elif message.find('Closing price') > -1:
                await self.handle_trade_report(channel)
            elif 'Summary:' in match_type:
                await self.handle_summary_signal(message, channel)
            else:
                await self.handle_trade_result(message, channel)
        except Exception as e:
            logging.error(f'Error processing message: {e}')
            self.pocket_option.remove_channel_data(channel)
            self.cleanup_trade(channel)

    async def handle_prepare_signal(self, message: str, channel: str) -> None:
        """Handle preparation signal for new trade"""
        if self.pocket_option.has_channel(channel):
            self.pocket_option.remove_channel_data(channel)
            self.cleanup_trade(channel)

        asset_match = re.search(TRADE_PATTERNS['ASSET'], message)
        if not asset_match:
            logging.error('No assets found in preparation message')
            return

        asset = f'{asset_match.group(1)}{asset_match.group(2)}_otc'
        self.trades[channel] = TradeSignal(asset=asset)
        self.pocket_option.set_value(channel, 'asset', asset)
        logging.info(f'Initialized trade preparation for asset: {asset}')

    async def handle_summary_signal(self, message: str, channel: str) -> None:
        """Handle trade summary signal and execute initial trade"""
        if channel not in self.trades:
            logging.warning(f'No active trade found for channel {channel}')
            return

        retry = 0
        amount = 1.0

        expiration_match = re.search(TRADE_PATTERNS['EXPIRATION'], message, re.IGNORECASE)
        if not expiration_match:
            logging.error(f'No expiration found in preparation message : {message}')
            return
        expiration = int(expiration_match.group(1)) * 60

        action_match = re.search(TRADE_PATTERNS['ACTION'], message, re.IGNORECASE)
        if not action_match:
            logging.error('No action found in preparation message')
            return
        action = f'{action_match.group(1)}'

        self.trades[channel] = TradeSignal(action=action, expiration=expiration, amount=amount, retry_count=retry)
        self.pocket_option.set_value(channel, 'action', action)
        self.pocket_option.set_value(channel, 'expiration', expiration)
        self.pocket_option.set_value(channel, 'amount', amount)

        logging.info(self.pocket_option.get_channel_data(channel))

        trade_id = await self.execute_trade(channel, retry)
        if trade_id:
            self.trades[channel].trade_id = trade_id

    async def handle_trade_result(self, message: str, channel: str) -> None:
        """Handle trade result and manage retries if needed"""
        if channel not in self.trades:
            logging.warning(f'No active trade found for channel {channel}')
            return

        trade_id = self.trades[channel].trade_id
        trade_result = await self.handle_trade_checking(channel, trade_id)
        if not trade_result:
            expiration_match = re.search(TRADE_PATTERNS['EXPIRATION'], message, re.IGNORECASE)
            if not expiration_match:
                logging.error('No expiration found in preparation message')
                return
            expiration = int(expiration_match.group(1)) * 60
            amount = self.trades[channel].amount
            amount *= 2
            current_retry = self.trades[channel].retry_count
            retry = current_retry + 1
            self.trades[channel] = TradeSignal(expiration=expiration, amount=amount, retry_count=retry)
            self.pocket_option.set_value(channel, 'expiration', expiration)
            self.pocket_option.set_value(channel, 'amount', amount)
            martingale_message = '' if current_retry == 0 else f'MARTINGALE {current_retry} '
            logging.warning(f'{martingale_message}Trade "{trade_id}" failed ({trade_result})')
            await self.handle_trade_retry(channel)

    async def execute_trade(self, channel, retry) -> Optional[str]:
        if self.pocket_option.get_value(channel, 'amount') > self.pocket_option.MAX_BID:
            self.pocket_option.remove_channel_data(channel)
            self.cleanup_trade(channel)
            return False

        martingale_message = '' if retry == 0 else f'MARTINGALE {retry} '

        logging.info(f'{martingale_message}INITIATE TRADE')

        trade_id = await safe_trade(self.pocket_option, channel)

        if not trade_id:
            trade_id = await find_trade_in_opened_deals(self.pocket_option, channel)
            if not trade_id:
                return None

        logging.info(f'{martingale_message}TRADE PLACED SUCCESSFULLY: "{trade_id}" AND WAITING FOR RESPONSE')

        return trade_id

    async def handle_trade_retry(self, channel) -> Optional[str]:
        retry = self.trades[channel].retry_count

        logging.info(self.pocket_option.get_channel_data(channel))

        trade_id = await self.execute_trade(channel, retry)
        if trade_id:
            self.trades[channel].trade_id = trade_id
            return trade_id

        return None

    async def handle_trade_report(self, channel):
        if channel not in self.trades:
            logging.warning(f'No active trade found for channel {channel}')
            return

        trade_id = self.trades[channel].trade_id
        trade_result = await self.handle_trade_checking(channel, trade_id)
        if not trade_result:
            retry = self.trades[channel].retry_count
            if retry <= self.pocket_option.MAX_RETRY:
                while retry < self.pocket_option.MAX_RETRY:
                    current_retry = self.trades[channel].retry_count
                    next_retry = current_retry + 1
                    amount = self.trades[channel].amount
                    amount *= 2
                    self.trades[channel] = TradeSignal(amount=amount, retry_count=next_retry)
                    self.pocket_option.set_value(channel, 'amount', amount)
                    martingale_message = '' if current_retry == 0 else f'MARTINGALE {current_retry} '
                    logging.warning(f'{martingale_message}Trade "{trade_id}" failed ({trade_result})')
                    trade_id = await self.handle_trade_retry(channel)
                    if not trade_id:
                        return None
                    trade_result = await self.handle_trade_checking(channel, trade_id)
                    if trade_result:
                        break

                    retry = next_retry
                    self.trades[channel].retry_count = retry
            else:
                logging.info(f'Trade failed: {trade_result}')
                self.pocket_option.remove_channel_data(channel)
                self.cleanup_trade(channel)

            if self.pocket_option.has_channel(channel):
                self.pocket_option.remove_channel_data(channel)
                self.cleanup_trade(channel)

    async def handle_trade_checking(self, channel, trade_id):
        """Check the result of a trade"""
        trade_data = await get_trade_result(self.pocket_option, trade_id)
        if not trade_data:
            return None

        trade_result = determine_trade_result(trade_data)

        if trade_result in ['win', 'draw']:
            logging.info(f'Trade successful: {trade_result}')
            self.pocket_option.remove_channel_data(channel)
            self.cleanup_trade(channel)
            return True
        else:
            logging.warning(f'Trade "{trade_id}" failed ({trade_result})')
            return False

    def cleanup_trade(self, channel: str) -> None:
        """Remove trade data for a specific channel"""
        if channel in self.trades:
            del self.trades[channel]

    async def start(self) -> None:
        """Start the trading bot"""

        @self.client.on(events.NewMessage(chats=self.chats[SESSION_NAME]))
        async def message_handler(event):
            await self.handle_trade_execution(event.raw_text)

        await self.client.start()
        await self.client.run_until_disconnected()


def main():
    """Main entry point for the trading bot."""
    try:
        bot = OPTRTradingBot()
        asyncio.run(bot.start())
    except Exception as e:
        logging.error(f'Bot initialization failed: {e}')

# Main execution
if __name__ == '__main__':
    main()