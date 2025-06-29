"""
Base Strategy Module for Trading Bot

This module defines the base strategy class and common interfaces that all trading strategies
must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
import pandas as pd

class BaseStrategy(ABC):
    """Abstract base class defining the common interface for all trading strategies."""

    def __init__(self, name: str, description: str) -> None:
        """Initialize base strategy with name and description."""
        self.name = name
        self.description = description
        self.parameters: Dict = {}
        self._trade_parameters: Dict = {
            'expiration': 120,  # Default 2-minute expiry
            'amount': 1.0,      # Default $1 trade amount
        }
        self._strategy_timeframe = {
            'candles': 60, # For 1-minute in seconds
            'resample': '1min', # For resampling to 1-minute candles
            'min_points': 60  # Minimum points for resampling
        }  # Default strategy timeframe for signal generation

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from candlestick data.
        
        Args:
            df (pd.DataFrame): Candlestick data with OHLC prices
        
        Returns:
            pd.DataFrame: Enhanced dataframe with trading signals
        """
        pass

    @abstractmethod
    def backtest(self, df: pd.DataFrame, symbol: str, start_time=None, end_time=None) -> Tuple[float, int, int, int]:
        """
        Perform backtesting of the strategy.
        
        Args:
            df (pd.DataFrame): Historical price data with signals
            symbol (str): Trading symbol being tested
        
        Returns:
            tuple: (win_rate, total_trades, wins, losses)
                - win_rate: Percentage of winning trades
                - total_trades: Total number of trades executed
                - wins: Number of winning trades
                - losses: Number of losing trades
        """
        pass

    @abstractmethod
    def get_required_candle_history(self) -> int:
        """
        Get the amount of historical candle data required for the strategy.
        
        Returns:
            int: Number of historical candles required
        """
        pass

    @abstractmethod
    def get_trade_parameters(self, asset: str) -> Dict[str, Any]:
        """
        Get the trading parameters for this strategy.
        
        Args:
            asset: The trading asset symbol
            
        Returns:
            Dict[str, Any]: A dictionary containing the trade parameters.
                Must include at least:
                - expiration: Trade expiration time in seconds
                - amount: Base trade amount
        """
        pass

    @abstractmethod
    def check_trade_entry(self, df_clean: pd.DataFrame) -> Tuple[bool, bool, str, str]:
        """
        Check for trade entry signals based on strategy-specific conditions.
        
        Args:
            df_clean (pd.DataFrame): Clean dataframe with indicators and signals
            
        Returns:
            Tuple[bool, bool, str]: (call_signal, put_signal, signal_time)
                - call_signal: True if should enter a call trade
                - put_signal: True if should enter a put trade
                - signal_time: Timestamp of the signal
                - trade_time: Timestamp of the trade entry time
        """
        pass

    # @abstractmethod
    # async def get_candles(self, api, candles: dict, candles_history: dict, timeframe: str = '1m') -> pd.DataFrame:
    #     """
    #     Process and resample candlestick data according to strategy requirements.
        
    #     Args:
    #         api: API instance for data fetching
    #         candles: Current candles data
    #         candles_history: Historical candles data
    #         timeframe: Timeframe for resampling (e.g. '15s', '1m', '5m', '15m')
    #                   Format: <number><unit> where unit is 's' for seconds or 'm' for minutes
        
    #     Returns:
    #         pd.DataFrame: Processed and resampled candlestick data
    #     """
    #     pass

    @property
    def parameters(self) -> Dict:
        """Get strategy technical parameters."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: Dict) -> None:
        """Set strategy technical parameters."""
        self._parameters = value

    @property 
    def trade_parameters(self) -> Dict:
        """Get strategy trade parameters."""
        return self._trade_parameters

    @trade_parameters.setter
    def trade_parameters(self, value: Dict) -> None:
        """Set strategy trade parameters."""
        self._trade_parameters = value

    @property
    def strategy_timeframe(self) -> str:
        """Get the strategy's preferred timeframe for signal generation."""
        return self._strategy_timeframe

    @strategy_timeframe.setter 
    def strategy_timeframe(self, value: str) -> None:
        """Set the strategy's preferred timeframe with validation."""
        self._strategy_timeframe = value
