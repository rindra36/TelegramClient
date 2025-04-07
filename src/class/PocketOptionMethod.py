from typing import Tuple, Any, Optional, Callable, TypeVar, Coroutine
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync, PocketOption
from BinaryOptionsTools import pocketoption
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
    DEMO_SSID = '''42["auth",{"session":"24k4ea9r0a1qck71rfojrgnl8o","isDemo":1,"uid":96282099,"platform":3}]'''
    REAL_SSID = r'''42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"b14deaeb687db05a7a49318bdb0973b6\";s:10:\"ip_address\";s:12:\"102.18.29.29\";s:10:\"user_agent\";s:70:\"Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0\";s:13:\"last_activity\";i:1743529087;}cc8be88ba075e7d0598e1e887eea1e50","isDemo":0,"uid":96282099,"platform":3}]'''
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    def __init__(self, wallet_type: str = 'demo'):
        """
        Initialize PocketOptionMethod with specified wallet type.

        Args:
            wallet_type (str): Type of wallet ('demo' or 'real')
        """
        self.wallet_type = wallet_type.lower()
        self.ssid = self.REAL_SSID if 'real' in self.wallet_type else self.DEMO_SSID
        self.api_async = PocketOptionAsync(self.ssid)
        self._logger = logging.getLogger(__name__)
        self.isDemo = False if 'real' in self.wallet_type else True
        self.api_v1 = pocketoption(self.ssid, self.isDemo)
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
                    await asyncio.sleep(self.RETRY_DELAY)
                    continue
                return result
            except Exception as e:
                self._logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
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
