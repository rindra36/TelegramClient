from typing import Tuple, Any, Optional, Callable, TypeVar, Coroutine
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync, PocketOption
from BinaryOptionsToolsV2.validator import Validator
from BinaryOptionsTools import pocketoption
from datetime import timedelta
import asyncio
import logging
import nest_asyncio

T = TypeVar('T')  # Type variable for generic return types

class PocketOptionMethod:
    """
    A wrapper class for PocketOptionAsync that provides trading functionality
    with retry mechanisms and error handling.
    """

    # Class constants
    SSID = {
        # rindra36@gmail.com
        # 1: {
        #     'DEMO': '''42["auth",{"session":"24k4ea9r0a1qck71rfojrgnl8o","isDemo":1,"uid":96282099,"platform":3}]''',
        #     'REAL': r'''42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"b2254c2e6b4ac99a32b83c1cf3bc1c5a\";s:10:\"ip_address\";s:11:\"102.18.37.8\";s:10:\"user_agent\";s:70:\"Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0\";s:13:\"last_activity\";i:1744740582;}05c047b229940bbd6729c6f02e0e33b9","isDemo":0,"uid":96282099,"platform":3}]'''
        # },
        1: {
            'DEMO': '''42["auth",{"session":"j079fsgog45pjnbsj9a2hvpnnb","isDemo":1,"uid":102766033,"platform":3,"isFastHistory":true}]''',
            'REAL': r'''42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"e880474778a3286b35b5a23504b76206\";s:10:\"ip_address\";s:13:\"102.18.26.137\";s:10:\"user_agent\";s:70:\"Mozilla/5.0 (X11; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0\";s:13:\"last_activity\";i:1748984628;}f1f2b4e8c412d3cef2198dc3d0ffc662","isDemo":0,"uid":102766033,"platform":3,"isFastHistory":true}]'''
        },
        #  aronihmax@gmail.com
        # 2: {
        #     'DEMO': '''42["auth",{"session":"u1n8euostr7agl269u7bo55f6l","isDemo":1,"uid":98422233,"platform":3}]''',
        #     'REAL': r'''42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"5ccd052cd6fa15869a64b979fc15191c\";s:10:\"ip_address\";s:13:\"102.18.45.228\";s:10:\"user_agent\";s:70:\"Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0\";s:13:\"last_activity\";i:1744140658;}f118368f92ce971e6c44cd3a09aa0e78","isDemo":0,"uid":98422233,"platform":3}]'''
        # }
        2: {
            'DEMO': '''42["auth",{"session":"upen8g2mcd3cvu5ai5i4jjl6si","isDemo":1,"uid":102365452,"platform":3,"isFastHistory":false}]''',
            'REAL': r'''42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"63469838d39604af877c0f5759e0fde5\";s:10:\"ip_address\";s:12:\"102.18.38.77\";s:10:\"user_agent\";s:70:\"Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0\";s:13:\"last_activity\";i:1748113468;}40e4371abb280075c4d016787b9fc52a","isDemo":0,"uid":102365452,"platform":3,"isFastHistory":false}]'''
        }
    }
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    def __init__(self, account: int, wallet_type: str = 'demo'):
        """
        Initialize PocketOptionMethod with specified wallet type.

        Args:
            wallet_type (str): Type of wallet ('demo' or 'real')
        """
        self.wallet_type = wallet_type.lower()
        self.ssid = self.SSID[account]['REAL'] if 'real' in self.wallet_type else self.SSID[account]['DEMO']
        self.api_async = PocketOptionAsync(self.ssid)
        self._logger = logging.getLogger(__name__)
        self.isDemo = False if 'real' in self.wallet_type else True
        self.api_v1 = pocketoption(self.ssid, self.isDemo)
        self.use_delay_retry = True
        nest_asyncio.apply()

    async def get_balance(self) -> float | None:
        """
        Get current account balance.

        Returns:
            float: Current balance
        """
        return await self.execute_with_retry(self.api_async.balance)

    async def get_history(self, asset: str, time: int) -> Any:
        """
        Get trading history for specific asset and timeframe.

        Args:
            asset (str): Asset symbol
            time (int): Timeframe in seconds

        Returns:
            Any: Trading history data
        """
        return await self.execute_with_retry(self.api_async.history, asset, time)

    async def buy(self, asset: str, amount: float, time: int, use_v1:bool = True, check_win: bool = True) -> Tuple[str, Any]:
        """
        Execute a buy order.

        Args:
            asset (str): Asset symbol
            amount (float): Trade amount
            time (int): Expiration time in seconds
            check_win (bool): Whether to check if trade was successful

        Returns:
            Tuple[str, Any]: Trade ID and result data

        Raises:
            ValueError: If buy operation fails
        """
        if use_v1:
            try:
                return self.api_v1.Call(amount=amount, active=asset, expiration=time, add_check_win=check_win)
            except Exception as e:
                self._logger.error(f"Buy operation failed: {str(e)}")
                raise ValueError(f"Failed to buy: {str(e)}") from e
        else:
            try:
                return await self.api_async.buy(asset=asset, amount=amount, time=time, check_win=check_win)
            except Exception as e:
                self._logger.error(f"Buy operation failed: {str(e)}")
                raise ValueError(f"Failed to buy: {str(e)}") from e

    async def sell(self, asset: str, amount: float, time: int, use_v1: bool = True, check_win: bool = True) -> Tuple[str, Any]:
        """
        Execute a sell order.

        Args:
            asset (str): Asset symbol
            amount (float): Trade amount
            time (int): Expiration time in seconds
            check_win (bool): Whether to check if trade was successful

        Returns:
            Tuple[str, Any]: Trade ID and result data

        Raises:
            ValueError: If sell operation fails
        """
        if use_v1:
            try:
                return self.api_v1.Put(amount=amount, active=asset, expiration=time, add_check_win=check_win)
            except Exception as e:
                self._logger.error(f"Buy operation failed: {str(e)}")
                raise ValueError(f"Failed to buy: {str(e)}") from e
        else:
            try:
                return await self.api_async.sell(asset=asset, amount=amount, time=time, check_win=check_win)
            except Exception as e:
                self._logger.error(f"Sell operation failed: {str(e)}")
                raise ValueError(f"Failed to sell: {str(e)}") from e

    async def execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> Optional[T]:
        """
        Execute a function with retry mechanism.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Optional[T]: Function result or None if all retries fail
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                result = await func(*args, **kwargs)
                if not result and attempt < self.MAX_RETRIES - 1:
                    self._logger.warning(f"Empty result on attempt {attempt + 1}, retrying...")
                    if self.use_delay_retry:
                        await asyncio.sleep(self.RETRY_DELAY)
                    continue
                return result
            except Exception as e:
                self._logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    if self.use_delay_retry:
                        await asyncio.sleep(self.RETRY_DELAY)
                else:
                    self._logger.error(f"All retry attempts failed for {func.__name__}")
                    raise

    async def get_opened_deals(self) -> list | None:
        """
        Get list of currently opened deals.

        Returns:
            list: List of opened deals
        """
        return await self.execute_with_retry(self.api_async.opened_deals)

    async def get_closed_deals(self) -> list | None:
        """
        Get list of closed deals.

        Returns:
            list: List of closed deals
        """
        return await self.execute_with_retry(self.api_async.closed_deals)

    async def get_trade_data(self, trade_id: str) -> Any:
        """
        Get data for a specific trade.
        If newly created trade

        Args:
            trade_id (str): ID of the trade

        Returns:
            Any: Trade data
        """
        try:
            profit, result = self.api_v1.CheckWin(trade_id)
            return result
        except Exception as e:
            logging.warning(f'Error check win using API V1 : {str(e)}')

        try:
            return await self.api_async.check_win(trade_id)
        except Exception as e:
            logging.warning(f'Error check win using API V2 : {str(e)}')

        logging.info('Try using execute_with_retry')
        return await self.execute_with_retry(self.api_async.check_win, trade_id)

    async def get_best_payout(self):
        """
        Get the best payout.
        """
        await asyncio.sleep(5)
        payouts = await self.execute_with_retry(self.api_async.payout)
        
        return self.get_keys_with_max_value(payouts)

    def get_keys_with_max_value(self, data_dict):
        # Find the maximum value in the dictionary
        max_value = max(data_dict.values())
        
        # Get all keys that have this maximum value
        keys_with_max_value = [key for key, value in data_dict.items() if value == max_value]
        
        return keys_with_max_value
    
    async def create_raw_order(self, message: str, validator: Validator):
        # Disable delay on retry
        self.use_delay_retry = False
        raw_order = await self.execute_with_retry(self.api_async.create_raw_order_with_timout, message, validator, timedelta(seconds=5))
        self.use_delay_retry = True
        return raw_order
    
    async def get_candles(self, asset: str, period: int, offset: int) -> list[dict]:
        return await self.execute_with_retry(self.api_async.get_candles, asset, period, offset)