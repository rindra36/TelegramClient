from typing import Dict, Any, Optional, Union
import asyncio
import logging
from PocketOptionMethod import PocketOptionMethod


class PocketOptionAPI:
    """
    A high-level API wrapper for PocketOption trading operations.
    Handles trading operations, balance management, and channel-based trading data.
    """

    MAX_RETRY = 3
    LIMIT_BALANCE = 25
    MAX_BID = 4  # Maximum bid amount considering Martingale strategy with base amount of $1

    def __init__(self, wallet_type: str = 'demo'):
        """
        Initialize PocketOption API wrapper.

        Args:
            wallet_type (str): Type of wallet ('demo' or 'real')
        """
        self._variables: Dict[str, Dict[str, Any]] = {}
        self.wallet_type = wallet_type.lower()
        self._logger = logging.getLogger(__name__)
        self.pocket_option_method = PocketOptionMethod(self.wallet_type)

    @property
    def variables(self) -> Dict[str, Dict[str, Any]]:
        """Get the channel variables dictionary."""
        return self._variables

    @variables.setter
    def variables(self, value: Dict[str, Dict[str, Any]]) -> None:
        """
        Set the channel variables dictionary.

        Args:
            value: Dictionary containing channel data

        Raises:
            TypeError: If value is not a dictionary
        """
        if not isinstance(value, dict):
            raise TypeError("Value must be a dictionary")
        self._variables = value

    def get_channel_data(self, channel: str) -> Dict[str, Any]:
        """
        Get data for a specific channel.

        Args:
            channel: Channel identifier

        Returns:
            Dictionary containing channel data
        """
        return self._variables.get(channel, {})

    def set_channel_data(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Set data for a specific channel.

        Args:
            channel: Channel identifier
            data: Dictionary containing channel data

        Raises:
            TypeError: If data is not a dictionary
            KeyError: If required keys are missing
        """
        if not isinstance(data, dict):
            raise TypeError("Channel data must be a dictionary")

        required_keys = {"asset", "action", "expiration", "amount"}
        if not all(key in data for key in required_keys):
            raise KeyError(f"Data must contain keys: {required_keys}")

        self._variables[channel] = data

    def remove_channel_data(self, channel: str) -> bool:
        """
        Remove data for a specific channel.

        Args:
            channel: Channel identifier

        Returns:
            bool: True if channel was removed, False if it didn't exist
        """
        return bool(self._variables.pop(channel, None))

    def has_channel(self, channel: str) -> bool:
        """
        Check if a channel exists.

        Args:
            channel: Channel identifier

        Returns:
            bool: True if channel exists, False otherwise
        """
        return channel in self._variables

    def get_value(self, channel: str, key: str) -> Optional[Any]:
        """
        Get a specific value from channel data.

        Args:
            channel: Channel identifier
            key: Key to retrieve

        Returns:
            Value associated with the key or None if not found
        """
        channel_data = self.get_channel_data(channel)
        return channel_data.get(key)

    def set_value(self, channel: str, key: str, value: Any) -> None:
        """
        Set a specific value in channel data.

        Args:
            channel: Channel identifier
            key: Key to set
            value: Value to store
        """
        if channel not in self._variables:
            self._variables[channel] = {}
        if key != "channel":  # Prevent overwriting channel identifier
            self._variables[channel][key] = value

    async def trade(self, channel: str, check_win: bool = False, use_v1: bool = True) -> Union[str, bool, None]:
        """
        Execute a trade for a specific channel.

        Args:
            channel: Channel identifier
            check_win: Whether to check if trade was successful

        Returns:
            str: Trade ID if check_win is False
            bool: Trade result if check_win is True
            None: If trade fails

        Raises:
            ValueError: If required trading data is missing
        """
        up = ['BUY', 'CALL']
        down = ['SELL', 'PUT']

        channel_data = self.get_channel_data(channel)
        asset = channel_data.get("asset")
        action = channel_data.get("action")
        expiration = channel_data.get("expiration")
        amount = channel_data.get("amount")

        wins = ['win', 'draw']

        if not all([asset, action, expiration, amount]):
            raise ValueError("Missing required data for trading")

        try:
            if action in up:
                trade_id, result = await self.pocket_option_method.buy(
                    asset=asset,
                    amount=amount,
                    time=expiration,
                    check_win=check_win,
                    use_v1=use_v1
                )
            elif action in down:
                trade_id, result = await self.pocket_option_method.sell(
                    asset=asset,
                    amount=amount,
                    time=expiration,
                    check_win=check_win,
                    use_v1=use_v1
                )
            else:
                raise ValueError(f'Invalid action: {action}')

            if check_win:
                self._logger.info(f"Trade result: {result}")
                return result in wins if use_v1 else result.get('result') in wins
            return result if use_v1 else trade_id

        except Exception as e:
            self._logger.error(f"Trade error: {str(e)}")
            return None

    async def get_balance_amount(self) -> float:
        """
        Retrieve the balance amount from the PocketOption account.

        Returns:
            float: The balance amount.
        """
        await asyncio.sleep(5)
        balance = await self.pocket_option_method.get_balance()
        return balance

    async def is_breaking_balance_limit(self) -> bool:
        """
        Check if the balance is below the predefined limit.

        Returns:
            bool: True if the balance is below the limit, False otherwise.
        """
        balance = await self.get_balance_amount()
        if balance < self.LIMIT_BALANCE:
            self._logger.info(f"Balance {balance} is below the limit.")
            return True
        return False

    async def get_opened_deals(self) -> list:
        """
        Retrieve the list of opened deals.

        Returns:
            list: A list of opened deals.
        """
        return await self.pocket_option_method.get_opened_deals()

    async def get_closed_deals(self) -> list:
        """
        Retrieve the list of closed deals.

        Returns:
            list: A list of closed deals.
        """
        return await self.pocket_option_method.get_closed_deals()

    async def get_trade_data(self, trade_id: str) -> Any:
        """
        Get data for a specific trade.

        Args:
            trade_id (str): ID of the trade

        Returns:
            Any: Trade data
        """
        return await self.pocket_option_method.get_trade_data(trade_id)