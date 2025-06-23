import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
import asyncio

def find_levels(df, is_support: bool, window_size: int = 5, tolerance: float = 0.005):
    """
    Finds horizontal support or resistance levels by identifying and clustering pivot points.
    
    Args:
        df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
        is_support (bool): True to find support levels, False for resistance.
        window_size (int): The number of candles on each side of a pivot to check.
        tolerance (float): The percentage tolerance to group nearby levels.

    Returns:
        list: A list of identified level prices.
    """
    if is_support:
        price_col = 'low'
        # A pivot low is a low with 'window_size' lower lows on both sides
        pivots = df[price_col] == df[price_col].rolling(window_size * 2 + 1, center=True).min()
    else:
        price_col = 'high'
        # A pivot high is a high with 'window_size' higher highs on both sides
        pivots = df[price_col] == df[price_col].rolling(window_size * 2 + 1, center=True).max()
        
    pivot_prices = df[pivots][price_col]
    if pivot_prices.empty:
        return []

    # Cluster the pivot prices
    levels = []
    for price in pivot_prices:
        is_new_level = True
        for i, level in enumerate(levels):
            if abs(level - price) / price < tolerance:
                # Price is close to an existing level, update the level to the average
                levels[i] = (level * (df[price_col] == level).sum() + price) / ((df[price_col] == level).sum() + 1)
                is_new_level = False
                break
        if is_new_level:
            levels.append(price)
            
    return sorted(levels)

def find_trend_lines(df, is_support: bool, window_size: int = 5, tolerance: float = 0.01):
    """
    Finds trend lines by connecting pivot points.
    
    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'open', 'close'.
        is_support (bool): True to find support trend lines (connecting lows), False for resistance (connecting highs).
        window_size (int): The window size for pivot detection.
        tolerance (float): The percentage tolerance for a point to be considered on the line.

    Returns:
        list: A list of dictionaries, where each dict represents a trend line 
              with 'points' (indices and prices) and 'slope'.
    """
    if is_support:
        price_col = 'low'
        pivots_idx = df.index[df[price_col] == df[price_col].rolling(window_size * 2 + 1, center=True).min()]
    else:
        price_col = 'high'
        pivots_idx = df.index[df[price_col] == df[price_col].rolling(window_size * 2 + 1, center=True).max()]

    if len(pivots_idx) < 2:
        return []

    trend_lines = []
    # Check all combinations of 2 pivots
    for i in range(len(pivots_idx)):
        for j in range(i + 1, len(pivots_idx)):
            p1_idx, p2_idx = pivots_idx[i], pivots_idx[j]
            p1_price, p2_price = df[price_col].loc[p1_idx], df[price_col].loc[p2_idx]
            
            # Calculate slope
            slope = (p2_price - p1_price) / (p2_idx - p1_idx)
            
            # Check for a third pivot point that confirms the trend line
            touches = [{'idx': p1_idx, 'price': p1_price}, {'idx': p2_idx, 'price': p2_price}]
            
            # Check if price violates the line between the two points
            is_violated = False
            for k in range(p1_idx + 1, p2_idx):
                line_price_at_k = p1_price + slope * (k - p1_idx)
                if is_support and df['low'].loc[k] < line_price_at_k:
                    is_violated = True
                    break
                elif not is_support and df['high'].loc[k] > line_price_at_k:
                    is_violated = True
                    break
            if is_violated:
                continue

            for k_idx in pivots_idx:
                if k_idx not in [p1_idx, p2_idx]:
                    line_price_at_k = p1_price + slope * (k_idx - p1_idx)
                    # Check if the pivot is close to the line
                    if abs(df[price_col].loc[k_idx] - line_price_at_k) / line_price_at_k < tolerance:
                         touches.append({'idx': k_idx, 'price': df[price_col].loc[k_idx]})
            
            # A valid trend line needs at least 3 touches
            if len(touches) >= 3:
                # Sort touches by index
                touches = sorted(touches, key=lambda x: x['idx'])
                trend_lines.append({'points': touches, 'slope': slope})
    
    # Optional: Filter out redundant lines (lines that are too similar)
    # This part can be complex, for now we return all valid 3+ touch lines.
    
    return trend_lines

# --- NEW REFACTORED FUNCTIONS ---

def analyze_market_structure(df: pd.DataFrame, window_size: int = 5, tolerance_horiz: float = 0.005, tolerance_trend: float = 0.01):
    """
    Analyzes the dataframe to find all S/R levels and trend lines and returns them in a structured dictionary.
    
    Args:
        df (pd.DataFrame): Input market data.
        window_size (int): Window for pivot detection.
        tolerance_horiz (float): Tolerance for clustering horizontal levels.
        tolerance_trend (float): Tolerance for points on a trend line.

    Returns:
        dict: A dictionary containing all identified market structures.
              Keys: 'horizontal_support', 'horizontal_resistance', 
                    'trendline_support', 'trendline_resistance'.
    """
    support_levels = find_levels(df, is_support=True, window_size=window_size, tolerance=tolerance_horiz)
    resistance_levels = find_levels(df, is_support=False, window_size=window_size, tolerance=tolerance_horiz)
    
    support_trends = find_trend_lines(df, is_support=True, window_size=window_size, tolerance=tolerance_trend)
    resistance_trends = find_trend_lines(df, is_support=False, window_size=window_size, tolerance=tolerance_trend)
    
    market_structure = {
        "horizontal_support": support_levels,
        "horizontal_resistance": resistance_levels,
        "trendline_support": support_trends,
        "trendline_resistance": resistance_trends
    }
    
    return market_structure

def plot_market_structure(df: pd.DataFrame, market_structure: dict):
    """
    Plots the candlestick chart along with the provided market structure.
    
    Args:
        df (pd.DataFrame): The market data.
        market_structure (dict): The dictionary of levels and lines from analyze_market_structure.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot candlestick
    ax.plot(df.index, df['close'], color='gray', alpha=0.7, label='Close Price')
    for idx, row in df.iterrows():
        color = 'cyan' if row['close'] >= row['open'] else 'magenta'
        ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
        ax.plot([idx, idx], [row['open'], row['close']], color=color, linewidth=4)

    support_levels = market_structure['horizontal_support']
    resistance_levels = market_structure['horizontal_resistance']
        
    # Plot horizontal levels
    for level in support_levels:
        ax.axhline(y=level, color='lime', linestyle='--', linewidth=1.2, label=f'Support {level:.4f}')
    for level in resistance_levels:
        ax.axhline(y=level, color='red', linestyle='--', linewidth=1.2, label=f'Resistance {level:.4f}')

    support_trends = market_structure['trendline_support']
    resistance_trends = market_structure['trendline_resistance']
        
    # Plot trend lines
    for trend in support_trends:
        points = trend['points']
        x = [p['idx'] for p in points]
        y = [p['price'] for p in points]
        # Extend the line
        line_x = np.arange(x[0], df.index[-1])
        line_y = y[0] + trend['slope'] * (line_x - x[0])
        ax.plot(line_x, line_y, color='lime', linestyle='-', linewidth=1.5)

    for trend in resistance_trends:
        points = trend['points']
        x = [p['idx'] for p in points]
        y = [p['price'] for p in points]
        # Extend the line
        line_x = np.arange(x[0], df.index[-1])
        line_y = y[0] + trend['slope'] * (line_x - x[0])
        ax.plot(line_x, line_y, color='red', linestyle='-', linewidth=1.5)
        
    ax.set_title("Candlestick Chart with Support & Resistance")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date/Time Index")
    ax.grid(True, linestyle='--', alpha=0.2)
    
    plt.savefig('analyze_and_plot_2.png')

def check_trade_conditions(current_price: float, current_index: int, market_structure: dict, proximity_pct: float = 0.005):
    """
    Checks if a trade is allowed based on proximity to support and resistance.
    
    Args:
        current_price (float): The current market price.
        current_index (int): The index of the current candle.
        market_structure (dict): The dictionary of levels from analyze_market_structure.
        proximity_pct (float): The percentage to define "nearness" to a level.

    Returns:
        dict: A dictionary with 'allow_buy', 'allow_sell', and 'reason'.
    """
    # Default state: trades are allowed
    conditions = {'allow_buy': True, 'allow_sell': True, 'reason': 'Clear'}

    # 1. Check against Horizontal Resistance
    for res_level in market_structure['horizontal_resistance']:
        if abs(res_level - current_price) / current_price < proximity_pct:
            conditions['allow_buy'] = False
            conditions['reason'] = f'Price near horizontal resistance at {res_level:.4f}'
            return conditions # Exit early once a condition is met

    # 2. Check against Horizontal Support
    for sup_level in market_structure['horizontal_support']:
        if abs(sup_level - current_price) / current_price < proximity_pct:
            conditions['allow_sell'] = False
            conditions['reason'] = f'Price near horizontal support at {sup_level:.4f}'
            return conditions

    # 3. Check against Trend Line Resistance
    for trend in market_structure['trendline_resistance']:
        # Calculate the trend line's value at the current index
        p1 = trend['points'][0]
        slope = trend['slope']
        resistance_line_value_now = p1['price'] + slope * (current_index - p1['idx'])
        
        # Check if current price is above or near the trend line
        if current_price >= resistance_line_value_now or \
           abs(resistance_line_value_now - current_price) / current_price < proximity_pct:
            conditions['allow_buy'] = False
            conditions['reason'] = f'Price near resistance trend line'
            return conditions

    # 4. Check against Trend Line Support
    for trend in market_structure['trendline_support']:
        # Calculate the trend line's value at the current index
        p1 = trend['points'][0]
        slope = trend['slope']
        support_line_value_now = p1['price'] + slope * (current_index - p1['idx'])
        
        # Check if current price is below or near the trend line
        if current_price <= support_line_value_now or \
           abs(support_line_value_now - current_price) / current_price < proximity_pct:
            conditions['allow_sell'] = False
            conditions['reason'] = f'Price near support trend line'
            return conditions

    return conditions

async def main(ssid: str):
    api = PocketOptionAsync(ssid)    
    await asyncio.sleep(5)  # Wait for connection

    candles = await api.get_candles("TNDUSD_otc", 60, 10800)
    df = pd.DataFrame.from_dict(candles)

    # 1. Analyze the market to get the S/R structures
    structure = analyze_market_structure(df)

    # 2. Get the current price and index for decision making
    current_candle = df.iloc[-1]
    current_price = current_candle['close']
    current_idx = current_candle.name # .name gets the index label

    print(f"--- Strategy Check ---")
    print(f"Current Price: {current_price:.4f} at index {current_idx}")

    # 3. Check if a trade is permissible based on our rules
    trade_signals = check_trade_conditions(current_price, current_idx, structure, proximity_pct=0.01) # 1% proximity

    # 4. Make a decision based on the result
    print(f"\nDecision Engine Output:")
    print(f"  Allow Buy?  -> {trade_signals['allow_buy']}")
    print(f"  Allow Sell? -> {trade_signals['allow_sell']}")
    print(f"  Reason: {trade_signals['reason']}")

    # Example of integrating into a strategy:
    # if my_primary_buy_signal and trade_signals['allow_buy']:
    #     execute_buy_order()
    # elif my_primary_sell_signal and trade_signals['allow_sell']:
    #     execute_sell_order()

    # 5. (Optional) Visualize the analysis
    print("\nPlotting the analysis...")
    plot_market_structure(df, structure)


if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    asyncio.run(main(ssid))