"""
Strategy Factory Module

This module provides a factory class for creating and managing trading strategies.
"""

from typing import Dict, Type, List
from .base_strategy import BaseStrategy
from .macd_strategy import MACDStrategy
from .bollinger_bands_strategy import BollingerBandsStrategy
from .macd_sma_strategy import MACDSMAStrategy
from .fractal_ema_strategy import FractalEMAStrategy
from .williams_r_strategy import WilliamsRStrategy
from .price_action_strategy import FastPriceActionStrategy

class StrategyFactory:
    """Factory class for creating and managing trading strategies."""
    
    # Class-level registry of available strategies
    _strategies: Dict[str, Type[BaseStrategy]] = {}
    
    @classmethod
    def register_strategy(cls, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a new strategy class.
        
        Args:
            strategy_class: Class implementing BaseStrategy interface
        """
        cls._strategies[strategy_class.__name__] = strategy_class
    
    @classmethod
    def get_strategy(cls, strategy_name: str) -> BaseStrategy:
        """
        Create and return a strategy instance by name.
        
        Args:
            strategy_name: Name of the strategy class to instantiate
            
        Returns:
            BaseStrategy: Instance of the requested strategy
            
        Raises:
            ValueError: If strategy_name is not found in registry
        """
        if strategy_name not in cls._strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        return cls._strategies[strategy_name]()
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of registered strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def initialize(cls) -> None:
        """Register all built-in strategies."""
        cls.register_strategy(MACDStrategy)
        cls.register_strategy(BollingerBandsStrategy)
        cls.register_strategy(MACDSMAStrategy)
        cls.register_strategy(FractalEMAStrategy)
        cls.register_strategy(WilliamsRStrategy)
        cls.register_strategy(FastPriceActionStrategy)

# Initialize built-in strategies
StrategyFactory.initialize()
