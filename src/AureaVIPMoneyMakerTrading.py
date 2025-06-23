"""
AureaVIPMoneyMaker Trading Bot Module

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


class AureaVIPMoneyMakerTradingBot:
    """
    Trading bot that monitors Telegram channels for signals and executes trades on PocketOption.
    """

    def __init__(self, pocket_option):
        """Initialize the trading bot with configuration and credentials."""
        self.SESSION_NAME = 'AureaVIPMoneyMaker'
        self.TIMEZONE_OFFSET = -5
        self.DO_MARTINGALE = True # To control if we will going to Martingale or not because it is not working properly actually, set to True if want to go to Martingale
        self.MAX_RETRY = 1
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
        amount = 1.0
        expiration = 5 * 60  # 5 minutes

        # Initialize trade data
        self.pocket_option.set_channel_data(channel, {
            'action': action,
            'asset': asset,
            'expiration': expiration,
            'amount': amount,
        })

        logging.info(f"AureaVIPMoneyMakerTrading.py Trade parameters: {self.pocket_option.get_channel_data(channel)}")

        # Execute trade with Martingale strategy
        while retry <= self.MAX_RETRY:
            if retry > 0 or await waiting_time(entry, timezone_offset=self.TIMEZONE_OFFSET):
                try:
                    success = await self.execute_trade_cycle(channel, retry)
                    if success:
                        return True
                    elif success is None:
                        return False
                    if not self.DO_MARTINGALE:
                        logging.warning(f'AureaVIPMoneyMakerTrading.py Martingale needed here but not activated. Stop on retry {retry}')
                        return False
                    retry += 1
                    amount *= 2
                    self.pocket_option.set_value(channel, 'retry', retry)
                    self.pocket_option.set_value(channel, 'amount', amount)
                except Exception as e:
                    traceback_exception = traceback.format_exc()
                    logging.error(f'AureaVIPMoneyMakerTrading.py Trade cycle failed: {e}', traceback_exception)
                    break
            else:
                logging.warning(f'AureaVIPMoneyMakerTrading.py Entry time has passed : {message}')
                self.pocket_option.remove_channel_data(channel)
                break

        self.cleanup_channel_data(channel, retry)
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
        if not re.search(r'Aurea Pro', message, re.IGNORECASE):
            return action, asset, entry

        # Extract action
        action_match = re.search(r'(CALL|PUT)', message, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).upper()

        # Extract asset
        PAIRS = ['ADA-USD_otc', 'AEDCNY_otc', 'AMZN_otc', 'AUDCAD_otc', 'AUDCHF_otc', 'AUDJPY_otc', 'AUDNZD_otc', 'AUDUSD_otc', 'AUS200_otc', 'BABA_otc', 'BHDCNY_otc', 'BITB_otc', 'BNB-USD_otc', 'BTCUSD_otc', 'CADCHF_otc', 'CADJPY_otc', 'CHFJPY_otc', 'CHFNOK_otc', 'CITI_otc', 'D30EUR_otc', 'DJI30_otc', 'DOGE_otc', 'DOTUSD_otc', 'E35EUR_otc', 'E50EUR_otc', 'ETHUSD_otc', 'EURCHF_otc', 'EURGBP_otc', 'EURHUF_otc', 'EURJPY_otc', 'EURNZD_otc', 'EURRUB_otc', 'EURTRY_otc', 'EURUSD_otc', 'F40EUR_otc', 'FDX_otc', 'GBPJPY_otc', 'GBPAUD_otc', 'GBPUSD_otc', 'IRRUSD_otc', 'JODCNY_otc', 'JNJ_otc', 'JPN225_otc', 'LBPUSD_otc', 'LINK_otc', 'LTCUSD_otc', 'MADUSD_otc', 'MATIC_otc', 'MSFT_otc', 'NFLX_otc', 'NASUSD_otc', 'NZDJPY_otc', 'NZDUSD_otc', 'OMRCNY_otc', 'QARCNY_otc', 'SARCNY_otc', 'SOL-USD_otc', 'SP500_otc', 'SYPUSD_otc', 'TON-USD_otc', 'TRX-USD_otc', 'TWITTER_otc', 'UKBrent_otc', 'USDARS_otc', 'USDBDT_otc', 'USDBRL_otc', 'USDCAD_otc', 'USDCHF_otc', 'USDCLP_otc', 'USDCNH_otc', 'USDCOP_otc', 'USDDZD_otc', 'USDEGP_otc', 'USDIDR_otc', 'USDINR_otc', 'USDJPY_otc', 'USDMXN_otc', 'USDMYR_otc', 'USDPHP_otc', 'USDPKR_otc', 'USDRUB_otc', 'USDSGD_otc', 'USDTHB_otc', 'USDVND_otc', 'USCrude_otc', 'VISA_otc', 'XAGUSD_otc', 'XAUUSD_otc', 'XNGUSD_otc', 'XPDUSD_otc', 'XPTUSD_otc', 'XPRUSD_otc', 'YERUSD_otc', '#AALP_otc', '#AXP_otc', '#BA_otc', '#CSCO_otc', '#FB_otc', '#INTC_otc', '#JNJ_otc', '#MCD_otc', '#MSFT_otc', '#PFE_otc', '#TSLA_otc', '#XOM_otc', 'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURHUF', 'EURJPY', 'EURUSD', 'F40EUR', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
        asset_match = re.search(rf'({"|".join(PAIRS)})', message, re.IGNORECASE)
        if asset_match:
            asset = asset_match.group(1)

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
        logging.info(f'AureaVIPMoneyMakerTrading.py {martingale_message}INITIATE TRADE')

        trade_id = await safe_trade(self.pocket_option, channel)

        if not trade_id:
            trade_id = await find_trade_in_opened_deals(self.pocket_option, channel)
            if not trade_id:
                return None

        logging.info(f'AureaVIPMoneyMakerTrading.py {martingale_message}TRADE PLACED SUCCESSFULLY: "{trade_id}" AND WAITING FOR RESPONSE')

        expiration = int(self.pocket_option.get_value(channel, 'expiration')) - 5
        logging.info(f'AureaVIPMoneyMakerTrading.py Waiting for the expiration time minus 5 seconds : {expiration}')
        await asyncio.sleep(expiration)

        trade_data = await get_trade_result(self.pocket_option, trade_id)
        if not trade_data:
            return None

        trade_result = determine_trade_result(trade_data)
        logging.info(f'AureaVIPMoneyMakerTrading.py Trade result: {trade_result}')

        if trade_result in ['win', 'draw']:
            logging.info(f'AureaVIPMoneyMakerTrading.py Trade successful: {trade_result}')
            self.pocket_option.remove_channel_data(channel)
            return True
        elif trade_result != 'lose' and trade_result != 'loss':
            return None

        message_retrying = ', retrying...' if retry < self.MAX_RETRY else ''
        logging.warning(f'AureaVIPMoneyMakerTrading.py {martingale_message}Trade {trade_id} failed ({trade_result}){message_retrying}')
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
