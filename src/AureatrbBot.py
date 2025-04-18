"""
AureaBot Trading Bot Module

This module implements a Telegram bot that monitors specific channels for trading signals and
executes trades on PocketOption platform.
"""

import asyncio
import logging
import re
import time
import random

from typing import Dict, Optional
from datetime import datetime
from telethon import events
from PocketOptionAPI import PocketOptionAPI
from src.utils import setup_logging, load_credentials, load_chats, safe_trade, find_trade_in_opened_deals, \
    get_trade_result, determine_trade_result, setup_client, wait_until_close_timestamp, filter_keys_by_allowed_list

# Constants
SESSION_NAME = 'AureaBot'
TRADE_PATTERNS = {
    'ACTION': r'(BUY|SELL)'
}

ALWAYS_AVAILABLE_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'AUDJPY', 'AUDNZD', 'NZDJPY',
    'EURUSD_otc', 'GBPUSD_otc', 'USDJPY_otc', 'AUDUSD_otc', 'AUDJPY_otc', 'AUDNZD_otc', 'NZDJPY_otc',
    'USDCAD', 'USDCAD_otc', 'USDCHF', 'USDCHF_otc', 'NZDUSD', 'NZDUSD_otc', 'CADJPY', 'CADJPY_otc',
    'EURGBP', 'EURGBP_otc', 'GBPJPY', 'GBPJPY_otc', 'CHFJPY', 'CHFJPY_otc'  # Added LESS_IDEAL pairs
]

IDEAL_1230PM_CST_PAIRS = [
    'EURUSD', 'EURUSD_otc',
    'GBPUSD', 'GBPUSD_otc',
    'USDCAD', 'USDCAD_otc',
    'USDJPY', 'USDJPY_otc',
    'USDCHF', 'USDCHF_otc',
    'NZDUSD', 'NZDUSD_otc',
    'AUDUSD', 'AUDUSD_otc',
    'CADJPY', 'CADJPY_otc'
]

LESS_IDEAL_1230PM_CST_PAIRS = [
    'EURGBP', 'EURGBP_otc',
    'GBPJPY', 'GBPJPY_otc',
    'CHFJPY', 'CHFJPY_otc'
]

IDEAL_1230PM_CST_PRIORITY = {
    'EXCELLENT': ['USDCAD_otc'],
    'GOOD': ['EURUSD_otc', 'GBPUSD_otc', 'USDJPY_otc'],
    'MODERATE': ['USDCHF_otc', 'NZDUSD_otc', 'AUDUSD_otc', 'CADJPY_otc']
}

class AureaBotTradingBot:
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
        self.pocket_option = PocketOptionAPI(1)
        self.asset = None

    async def handle_trade_execution(self, message: str) -> bool:
        """
        Process trading signals and execute trades based on message content.

        Args:
            message: Raw message text from Telegram

        Returns:
            bool: True if trade execution was successful, False otherwise
        """
        channel = SESSION_NAME
        expiration = 3 * 60 # 3 minutes
        amount = 1.0
        asset = self.asset # 

        # Extract trading parameters
        action = self.parse_trading_signal(message)
        if not all([action, asset]):
            return False
        
        # Initialize trade data
        self.pocket_option.set_channel_data(channel, {
            'action': action,
            'asset': asset,
            'expiration': expiration,
            'amount': amount,
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
    def parse_trading_signal(message: str) -> Optional[str]:
        """
        Parse trading signal from message text.

        Returns:
            tuple: (action, asset, expiration_time)
        """
        action = None

        # Check for action pattern
        action_match = re.search(TRADE_PATTERNS['ACTION'], message, re.IGNORECASE)
        if action_match:
            action = action_match.group(1)

        return action.upper()

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

        logging.info(f'TRADE PLACED SUCCESSFULLY: "{trade_id}" AND WAITING FOR RESPONSE')

        expiration = int(self.pocket_option.get_value(channel, 'expiration')) - 5
        logging.info(f'Waiting for the expiration time minus 5 seconds : {expiration}')
        await asyncio.sleep(expiration)

        monitoring_result = await self._monitor_trade_result(trade_id, channel)

        # if not monitoring_result:
            # current_amount = self.pocket_option.get_value(channel, 'amount')
            # self.pocket_option.set_value(channel, 'amount', current_amount * 2)

            # logging.info('RETRY INITIATE TRADE')
            # trade_id = await safe_trade(self.pocket_option, channel)

            # if not trade_id:
            #     trade_id = await find_trade_in_opened_deals(self.pocket_option, channel)
            #     if not trade_id:
            #         return False

            # logging.info(f'RETRY TRADE PLACED SUCCESSFULLY: "{trade_id}" AND WAITING FOR RESPONSE')

            # expiration = int(self.pocket_option.get_value(channel, 'expiration')) - 5
            # logging.info(f'Waiting for the expiration time minus 5 seconds : {expiration}')
            # await asyncio.sleep(expiration)

            # monitoring_result = await self._monitor_trade_result(trade_id, channel)

        # Remove the channel data right after the trade is placed
        self.pocket_option.remove_channel_data(channel)

        return monitoring_result

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

        @self.client.on(events.NewMessage(chats=self.chats[SESSION_NAME], from_users='me'))
        async def message_handler(event):
            message = event.raw_text

            if '1M' not in message:
                self.asset = message

        @self.client.on(events.NewMessage(chats=self.chats[SESSION_NAME]))
        async def message_handler(event):
            message = event.raw_text

            # if 'Choose a new trading pair' in message:
            #     # Wait for the channel to close before proceeding
            #     logging.info("Waiting for current trade to finish")
            #     i = 1
            #     while True:
            #         if not self.pocket_option.has_channel(SESSION_NAME):
            #             break

            #         logging.info(f"Waiting 30s - {i} time(s)")
            #         await asyncio.sleep(30)
            #         i += 1

            #     logging.info("Current trade finished, proceeding to choose a new trading pair")

            #     # Wait for a random time between 1 and 10 minutes
            #     range_waiting_time = range(1, 3) # Put this when going to sleep : range(3, 8)
            #     waiting_time = random.choice(range_waiting_time) * 60
            #     logging.info(f'Waiting for {waiting_time} seconds')
            #     # time.sleep(waiting_time)

            #     # Choose a random account
            #     accounts = [1, 2]
            #     account = random.choice(accounts)
            #     account = 2
            #     logging.info(f'Using account {account}')

            #     # self.pocket_option = PocketOptionAPI(account, 'real')

            #     # Get the best payout asset between the given list of assets
            #     current_time = datetime.now()
            #     current_hour = current_time.hour
            #     current_minute = current_time.minute
                
            #     def get_priority_filtered_payouts(payouts, priority_level):
            #         priority_pairs = IDEAL_1230PM_CST_PRIORITY[priority_level]
            #         return [pair for pair in payouts if pair in priority_pairs]

            #     payouts = await self.pocket_option.get_best_payout()
            #     # allowed_keys = ['ADA-USD_otc','AEDCNY_otc','AMZN_otc','AUDCAD_otc','AUDCHF_otc','AUDJPY_otc','AUDNZD_otc','AUDUSD_otc','AUS200_otc','BABA_otc','BHDCNY_otc','BITB_otc','BNB-USD_otc','BTCUSD_otc','CADCHF_otc','CADJPY_otc','CHFJPY_otc','CHFNOK_otc','CITI_otc','D30EUR_otc','DJI30_otc','DOGE_otc','DOTUSD_otc','E35EUR_otc','E50EUR_otc','ETHUSD_otc','EURCHF_otc','EURGBP_otc','EURHUF_otc','EURJPY_otc','EURNZD_otc','EURRUB_otc','EURTRY_otc','EURUSD_otc','F40EUR_otc','FDX_otc','GBPJPY_otc','GBPAUD_otc','GBPUSD_otc','IRRUSD_otc','JODCNY_otc','JNJ_otc','JPN225_otc','LBPUSD_otc','LINK_otc','LTCUSD_otc','MADUSD_otc','MATIC_otc','MSFT_otc','NFLX_otc','NASUSD_otc','NZDJPY_otc','NZDUSD_otc','OMRCNY_otc','QARCNY_otc','SARCNY_otc','SOL-USD_otc','SP500_otc','SYPUSD_otc','TON-USD_otc','TRX-USD_otc','TWITTER_otc','UKBrent_otc','USDARS_otc','USDBDT_otc','USDBRL_otc','USDCAD_otc','USDCHF_otc','USDCLP_otc','USDCNH_otc','USDCOP_otc','USDDZD_otc','USDEGP_otc','USDIDR_otc','USDINR_otc','USDJPY_otc','USDMXN_otc','USDMYR_otc','USDPHP_otc','USDPKR_otc','USDRUB_otc','USDSGD_otc','USDTHB_otc','USDVND_otc','USCrude_otc','VISA_otc','XAGUSD_otc','XAUUSD_otc','XNGUSD_otc','XPDUSD_otc','XPTUSD_otc','XPRUSD_otc','YERUSD_otc','#AALP_otc','#AXP_otc','#BA_otc','#CSCO_otc','#FB_otc','#INTC_otc','#JNJ_otc','#MCD_otc','#MSFT_otc','#PFE_otc','#TSLA_otc','#XOM_otc','AUDCAD','AUDCHF','AUDJPY','AUDNZD','AUDUSD','CADCHF','CADJPY','CHFJPY','EURAUD','EURCAD','EURCHF','EURGBP','EURHUF','EURJPY','EURUSD','F40EUR','GBPAUD','GBPCAD','GBPCHF','GBPJPY','GBPUSD','NZDJPY','NZDUSD','USDCAD','USDCHF','USDJPY']
                
            #     # At 12:30 PM CST (18:30 UTC), use priority-based selection
            #     if current_hour == 18 and current_minute == 30:
            #         # Try EXCELLENT pairs first
            #         filtered_payouts = get_priority_filtered_payouts(payouts, 'EXCELLENT')
            #         if not filtered_payouts:
            #             # Try GOOD pairs
            #             filtered_payouts = get_priority_filtered_payouts(payouts, 'GOOD')
            #             if not filtered_payouts:
            #                 # Try MODERATE pairs
            #                 filtered_payouts = get_priority_filtered_payouts(payouts, 'MODERATE')
            #                 if not filtered_payouts:
            #                     # If no ideal pairs available, fall back to ALWAYS_AVAILABLE_PAIRS
            #                     filtered_payouts = filter_keys_by_allowed_list(payouts, ALWAYS_AVAILABLE_PAIRS)
            #     else:
            #         # Outside of specific time, use ALWAYS_AVAILABLE_PAIRS
            #         filtered_payouts = filter_keys_by_allowed_list(payouts, ALWAYS_AVAILABLE_PAIRS)
                
            #     payout = random.choice(filtered_payouts) if filtered_payouts else None
            #     if not payout:
            #         logging.error("No suitable trading pairs found")
            #         return
                
            #     self.asset = payout
            #     await event.respond(self.asset)
            #     logging.info(f'Using asset {self.asset}')

            # if 'Choose an expiry time' in message:
            #     while True:
            #         # Get the current time
            #         current_time = datetime.now()
            #         seconds = current_time.second

            #         if seconds == 0:
            #             await event.respond('1M')
            #             break
            #         elif seconds >= 55:
            #             # Alert that we're approaching potential signal time
            #             print(f"Approaching candle close... {60-seconds} seconds remaining", end='\r')
            #             time.sleep(0.2)  # More frequent updates when close to signal time
            #         else:
            #             # Regular monitoring message
            #             print(f"Waiting... Response will be sent in {60-seconds} seconds", end='\r')
            #             time.sleep(1)

            action_match = re.search(TRADE_PATTERNS['ACTION'], message, re.IGNORECASE)
            if action_match:
                await self.handle_trade_execution(event.raw_text)

        await self.client.start()
        await self.client.run_until_disconnected()


def main():
    """Main entry point for the trading bot."""
    try:
        bot = AureaBotTradingBot()
        asyncio.run(bot.start())
    except Exception as e:
        logging.error(f'Bot initialization failed : {e}')


if __name__ == "__main__":
    main()
