"""
Strategy Manager Module

This module provides centralized strategy management and parameter handling for trading bots.
"""

from typing import Dict, Any, Optional
from .strategies.base_strategy import BaseStrategy
from .strategies.strategy_factory import StrategyFactory

class StrategyManager:
    """Manages trading strategies and their parameters across the system."""
    
    _instance = None
    _current_strategy: Optional[BaseStrategy] = None
    _strategy_factory = StrategyFactory()

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(StrategyManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the strategy manager with default settings."""
        # Initialize with MACD strategy by default
        self._current_strategy = self._strategy_factory.get_strategy('MACDStrategy')

    def set_strategy(self, strategy_name: str) -> None:
        """
        Set the current trading strategy.
        
        Args:
            strategy_name: Name of the strategy to use
        """
        self._current_strategy = self._strategy_factory.get_strategy(strategy_name)

    def get_strategy(self) -> Optional[BaseStrategy]:
        """Get the current trading strategy."""
        return self._current_strategy

    def get_trade_parameters(self, asset: str, channel_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get trade parameters by combining strategy and channel parameters.
        
        Args:
            asset: Trading asset symbol
            channel_params: Channel-specific parameters
            
        Returns:
            Combined trade parameters with channel overrides
        """
        if not self._current_strategy:
            return channel_params

        # Get base parameters from strategy
        strategy_params = self._current_strategy.trade_parameters.copy()

        # Allow channel to override specific parameters
        for key in ['amount', 'expiration', 'action', 'asset']:
            if key in channel_params:
                strategy_params[key] = channel_params[key]

        return strategy_params

    def get_required_candle_history(self) -> int:
        """Get required historical candle data for current strategy."""
        if not self._current_strategy:
            return 39600  # Default to ~11 hours
        return self._current_strategy.get_required_candle_history()

    def generate_signals(self, df):
        """Generate trading signals using current strategy."""
        if not self._current_strategy:
            raise ValueError("No trading strategy selected")
        return self._current_strategy.generate_signals(df)
