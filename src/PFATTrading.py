"""
PFAT Trading Bot Module

This module implements a Telegram bot that monitors specific channels for trading signals
and executes trades on PocketOption platform using the Martingale strategy.
"""

import logging
import re
import asyncio
import traceback
from typing import Optional

from src.utils import waiting_time, safe_trade, \
    find_trade_in_opened_deals, get_trade_result, determine_trade_result


class PFATTradingBot:
    """
    Trading bot that monitors Telegram channels for signals and executes trades on PocketOption.
    """

    def __init__(self, pocket_option):
        """Initialize the trading bot with configuration and credentials."""
        self.SESSION_NAME = 'PFAT'
        self.TIMEZONE_OFFSET = -3
        self.DO_MARTINGALE = True # To control if we will going to Martingale or not because it is not working properly actually, set to True if want to go to Martingale
        self.MAX_RETRY = 1
        self.INIT_AMOUNT = 1
        self.pocket_option = pocket_option

    async def handle_trade_execution(self, message: str) -> bool:
        """
        Process trading signals and execute trades based on message content.

        Args:
            message: Raw message text from Telegram

        Returns:
            bool: True if trade execution was successful, False otherwise
        """
        # Extract trading parameters
        action, asset, entry = self.parse_trading_signal(message)
        if not all([action, asset, entry]):
            return False

        channel = self.SESSION_NAME
        retry = 0
        amount = self.INIT_AMOUNT
        expiration = 1 * 60  # 5 minutes

        temp_data = None
        if (self.pocket_option.has_channel(channel)):
            temp_data = {
                'action': action,
                'asset': asset,
                'expiration': expiration,
                'amount': amount,
                'entry': entry
            }

            logging.info(f"PFATTrading.py Trade parameters put in TEMP : {temp_data}")
        else:
            # Initialize trade data
            self.pocket_option.set_channel_data(channel, {
                'action': action,
                'asset': asset,
                'expiration': expiration,
                'amount': amount,
            })

            # Adding the entry point
            self.pocket_option.set_value(channel, 'entry', entry)
            logging.info(f"PFATTrading.py Trade parameters: {self.pocket_option.get_channel_data(channel)}")

        # Execute trade with Martingale strategy
        while retry <= self.MAX_RETRY:
            if retry > 0 or await waiting_time(entry, timezone_offset=self.TIMEZONE_OFFSET):
                try:
                    # Putting the temp_data into the channel
                    if temp_data:
                        self.pocket_option.set_channel_data(channel, {
                            'action': temp_data['action'],
                            'asset': temp_data['asset'],
                            'expiration': temp_data['expiration'],
                            'amount': temp_data['amount'],
                        })
                        self.pocket_option.set_value(channel, 'entry', temp_data['entry'])
                        temp_data = None

                        logging.info(f"PFATTrading.py Loading Temp trade parameters: {self.pocket_option.get_channel_data(channel)}")

                    logging.info(f"PFATTrading.py Current trade parameters: {self.pocket_option.get_channel_data(channel)}")

                    if retry == 0:
                        amount = self.INIT_AMOUNT
                        self.pocket_option.set_value(channel, 'amount', amount)
                        
                    success = await self.execute_trade_cycle(channel, retry)
                    if success:
                        return True
                    elif success is None:
                        return False
                    if not self.DO_MARTINGALE:
                        logging.warning(f'PFATTrading.py Martingale needed here but not activated. Stop on retry {retry}')
                        return False
                    retry += 1

                    if retry >= self.MAX_RETRY:
                        amount = self.INIT_AMOUNT * 2
                        self.pocket_option.set_value(channel, 'amount', amount)

                    self.pocket_option.set_value(channel, 'retry', retry)
                except Exception as e:
                    traceback_exception = traceback.format_exc()
                    logging.error(f'PFATTrading.py Trade cycle failed: {e}', traceback_exception)
                    break
            else:
                logging.warning(f'PFATTrading.py Entry time has passed : {message}')
                self.pocket_option.remove_channel_data(channel)
                break

        # self.cleanup_channel_data(channel, retry)
        return False

    @staticmethod
    def parse_trading_signal(message: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse trading signal from message text.

        Returns:
            tuple: (action, asset, entry_time)
        """
        action = entry = asset = None

        # Verify message format
        if not re.search(r'Si pierdes, hasta 1 Martin Gale', message, re.IGNORECASE):
            return action, asset, entry

        # Extract action
        action_match = re.search(r'(CALL|PUT)', message, re.IGNORECASE)
        if action_match:
            action = action_match.group(1)

        # Extract asset
        asset_match = re.search(r'([A-Z]{3})([A-Z]{3}).*OTC', message)
        if asset_match:
            asset = f'{asset_match.group(1)}{asset_match.group(2)}_otc'

        # Extract entry time
        entry_match = re.search(r'(\d{1,2}:\d{2})', message)
        if entry_match:
            entry = entry_match.group(1)

        return action, asset, entry

    async def execute_trade_cycle(self, channel: str, retry: int) -> Optional[bool]:
        """
        Execute a complete trade cycle including verification and result checking.

        Returns:
            bool: True if trade was successful, False otherwise
        """
        martingale_message = ''
        if retry > 0:
            martingale_message = f'MARTINGALE {retry} '
        logging.info(f'PFATTrading.py {martingale_message}INITIATE TRADE')

        trade_id = await safe_trade(self.pocket_option, channel)

        if not trade_id:
            trade_id = await find_trade_in_opened_deals(self.pocket_option, channel)
            if not trade_id:
                return None

        logging.info(f'PFATTrading.py {martingale_message}TRADE PLACED SUCCESSFULLY: "{trade_id}" AND WAITING FOR RESPONSE')

        expiration = int(self.pocket_option.get_value(channel, 'expiration')) - 5
        logging.info(f'PFATTrading.py Waiting for the expiration time minus 5 seconds : {expiration}')
        await asyncio.sleep(expiration)

        trade_data = await get_trade_result(self.pocket_option, trade_id)
        if not trade_data:
            return None

        trade_result = determine_trade_result(trade_data)
        logging.info(f'PFATTrading.py Trade result: {trade_result}')

        if trade_result in ['win', 'draw']:
            logging.info(f'PFATTrading.py Trade successful: {trade_result}')
            # self.pocket_option.remove_channel_data(channel)
            return True
        elif trade_result != 'lose' and trade_result != 'loss':
            return None

        message_retrying = ', retrying...' if retry < self.MAX_RETRY else ''
        logging.warning(f'PFATTrading.py {martingale_message}Trade {trade_id} failed ({trade_result}){message_retrying}')
        return False

    def cleanup_channel_data(self, channel: str, retry: int) -> None:
        """Clean up channel data after trade completion or failure."""
        if self.pocket_option.has_channel(channel):
            current_retry = self.pocket_option.get_value(channel, 'retry')
            if current_retry is not None and current_retry > self.MAX_RETRY:
                self.pocket_option.remove_channel_data(channel)

    async def start(self, message) -> None:
        """Start the trading bot."""
        await self.handle_trade_execution(message)
