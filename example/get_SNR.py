import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
import asyncio
from lib.PocketOptionAPI.pocketoptionapi_async import utils

def detect_support_resistance(df, window=20, cluster_eps=0.005, trend_r2_threshold=0.8):
    """
    Detect support/resistance levels (horizontal or trend lines) from candlestick data.
    """
    df = df.copy()
    df['swing_high'] = df['high'].rolling(window, center=True).apply(lambda x: x.argmax() == len(x)//2, raw=True)
    df['swing_low'] = df['low'].rolling(window, center=True).apply(lambda x: x.argmin() == len(x)//2, raw=True)
    
    swing_highs = df[df['swing_high'] == 1][['high']].values
    swing_lows = df[df['swing_low'] == 1][['low']].values
    
    # Horizontal levels via clustering
    def cluster_levels(prices, eps=cluster_eps):
        if len(prices) < 2: return []
        cluster = DBSCAN(eps=eps, min_samples=2).fit(prices)
        labels = cluster.labels_
        unique_prices = [np.mean(prices[labels == i]) for i in set(labels) if i != -1]
        return [{"type": "horizontal", "price": p} for p in unique_prices]
    
    horizontal_levels = cluster_levels(np.vstack([swing_highs, swing_lows]))
    
    # Trend lines (minimum 2 points)
    def detect_trend_lines(prices):
        trends = []
        n = len(prices)
        if n < 2: return []

        for i in range(n - 2):
            for j in range(i + 2, n):
                X = np.array(range(i, j)).reshape(-1, 1)
                y = prices[i:j].ravel()  # Flatten to 1D to ensure proper model output shape
                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)

                if r2 > trend_r2_threshold:
                    # Safe scalar extraction using .item()
                    slope = float(model.coef_[0].item())         # 1D array -> scalar
                    intercept = float(model.intercept_.item())   # Scalar -> float

                    trends.append({
                        "type": "trend",
                        "slope": slope,
                        "intercept": intercept
                    })
        return trends
    
    trend_highs = detect_trend_lines(swing_highs)
    trend_lows = detect_trend_lines(swing_lows)
    
    return horizontal_levels + trend_highs + trend_lows

def detect_support_resistance_2(df, window=20, min_touches=2, cluster_threshold=0.01, 
                            trend_threshold=0.0005, plot=True, figsize=(15, 8)):
    """
    Detect horizontal support/resistance levels and trend lines from candlestick data.
    
    This function works by:
    1. Finding significant peaks (resistance) and troughs (support) in price data
    2. Clustering nearby levels to identify major support/resistance zones
    3. Determining if levels are horizontal or trending based on slope analysis
    4. Validating levels by counting how many times price touches them
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Candlestick data with columns: 'open', 'high', 'low', 'close'
        Index should be datetime or sequential
    window : int, default=20
        Lookback window for peak/trough detection (larger = fewer, stronger levels)
    min_touches : int, default=2
        Minimum number of times price must touch a level for it to be valid
    cluster_threshold : float, default=0.01
        Price clustering threshold as percentage (1% = 0.01)
    trend_threshold : float, default=0.0005
        Slope threshold to distinguish horizontal vs trending lines
    plot : bool, default=True
        Whether to plot the results
    figsize : tuple, default=(15, 8)
        Figure size for plotting
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'horizontal_support': List of horizontal support levels
        - 'horizontal_resistance': List of horizontal resistance levels  
        - 'trend_support': List of trend support lines (slope, intercept, start_idx, end_idx)
        - 'trend_resistance': List of trend resistance lines (slope, intercept, start_idx, end_idx)
        - 'support_touches': Touch counts for each support level
        - 'resistance_touches': Touch counts for each resistance level
    """
    
    # Step 1: Find peaks and troughs using scipy's peak detection
    # We use both high/low prices and closing prices for more comprehensive detection
    
    # Find resistance levels (peaks in highs)
    high_peaks, _ = find_peaks(df['high'].values, distance=window//2)
    
    # Find support levels (troughs in lows) - we invert the data to find peaks
    low_peaks, _ = find_peaks(-df['low'].values, distance=window//2)
    
    # Also find peaks/troughs in closing prices for additional confirmation
    close_peaks, _ = find_peaks(df['close'].values, distance=window//2)
    close_troughs, _ = find_peaks(-df['close'].values, distance=window//2)
    
    # Combine all resistance and support candidates
    resistance_candidates = np.concatenate([
        df.iloc[high_peaks]['high'].values,
        df.iloc[close_peaks]['close'].values
    ])
    
    support_candidates = np.concatenate([
        df.iloc[low_peaks]['low'].values,
        df.iloc[close_troughs]['close'].values
    ])
    
    print(f"Found {len(resistance_candidates)} resistance candidates and {len(support_candidates)} support candidates")
    
    # Step 2: Cluster nearby levels to identify major zones
    # This helps us group levels that are very close together into single zones
    
    def cluster_levels(levels, threshold):
        """Group nearby price levels using DBSCAN clustering"""
        if len(levels) == 0:
            return []
        
        # Reshape for sklearn
        levels_reshaped = levels.reshape(-1, 1)
        
        # Use DBSCAN to cluster nearby levels
        # eps is our clustering threshold (percentage of average price)
        avg_price = np.mean(df['close'])
        eps = threshold * avg_price
        
        clustering = DBSCAN(eps=eps, min_samples=1).fit(levels_reshaped)
        
        # Get the average level for each cluster
        clustered_levels = []
        for cluster_id in np.unique(clustering.labels_):
            cluster_levels = levels[clustering.labels_ == cluster_id]
            clustered_levels.append(np.mean(cluster_levels))
        
        return sorted(clustered_levels)
    
    # Cluster the levels
    clustered_resistance = cluster_levels(resistance_candidates, cluster_threshold)
    clustered_support = cluster_levels(support_candidates, cluster_threshold)
    
    print(f"After clustering: {len(clustered_resistance)} resistance levels, {len(clustered_support)} support levels")
    
    # Step 3: Count touches for each level and validate
    def count_touches(level, is_resistance=True):
        """Count how many times price touches a level within a small tolerance"""
        tolerance = cluster_threshold * np.mean(df['close']) / 2
        
        if is_resistance:
            # For resistance, check if high prices touch the level
            touches = np.sum((df['high'] >= level - tolerance) & (df['high'] <= level + tolerance))
            # Also check for close approaches
            close_approaches = np.sum((df['close'] >= level - tolerance) & (df['close'] <= level + tolerance))
        else:
            # For support, check if low prices touch the level
            touches = np.sum((df['low'] >= level - tolerance) & (df['low'] <= level + tolerance))
            # Also check for close approaches
            close_approaches = np.sum((df['close'] >= level - tolerance) & (df['close'] <= level + tolerance))
        
        return max(touches, close_approaches)
    
    # Validate levels by touch count
    valid_resistance = []
    valid_support = []
    resistance_touches = []
    support_touches = []
    
    for level in clustered_resistance:
        touches = count_touches(level, is_resistance=True)
        if touches >= min_touches:
            valid_resistance.append(level)
            resistance_touches.append(touches)
    
    for level in clustered_support:
        touches = count_touches(level, is_resistance=False)
        if touches >= min_touches:
            valid_support.append(level)
            support_touches.append(touches)
    
    print(f"Valid levels: {len(valid_resistance)} resistance, {len(valid_support)} support")
    
    # Step 4: Detect trend lines by connecting significant points
    def detect_trend_lines(peaks_idx, price_values, is_resistance=True):
        """Detect trend lines by connecting peaks/troughs"""
        trend_lines = []
        
        if len(peaks_idx) < 2:
            return trend_lines
        
        # Try connecting different combinations of peaks
        for i in range(len(peaks_idx)):
            for j in range(i + 2, len(peaks_idx)):  # Skip adjacent peaks
                x1, x2 = peaks_idx[i], peaks_idx[j]
                y1, y2 = price_values[i], price_values[j]
                
                # Calculate line parameters
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Check if this is a significant trend line
                if abs(slope) > trend_threshold * np.mean(df['close']):
                    # Count how many other points are near this line
                    line_touches = 0
                    tolerance = cluster_threshold * np.mean(df['close'])
                    
                    for k in range(x1, x2 + 1):
                        expected_y = slope * k + intercept
                        actual_y = df['high'].iloc[k] if is_resistance else df['low'].iloc[k]
                        
                        if abs(actual_y - expected_y) <= tolerance:
                            line_touches += 1
                    
                    # If enough points touch this line, it's a valid trend line
                    if line_touches >= min_touches:
                        trend_lines.append({
                            'slope': slope,
                            'intercept': intercept,
                            'start_idx': x1,
                            'end_idx': x2,
                            'touches': line_touches
                        })
        
        return trend_lines
    
    # Find trend lines for resistance and support
    trend_resistance = detect_trend_lines(high_peaks, df.iloc[high_peaks]['high'].values, True)
    trend_support = detect_trend_lines(low_peaks, df.iloc[low_peaks]['low'].values, False)
    
    # Step 5: Classify levels as horizontal or trending
    horizontal_resistance = valid_resistance.copy()
    horizontal_support = valid_support.copy()
    
    print(f"Trend lines found: {len(trend_resistance)} resistance, {len(trend_support)} support")
    
    # Step 6: Plotting
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot candlesticks (simplified as line chart for clarity)
        ax.plot(df.index, df['close'], label='Close Price', linewidth=1, alpha=0.7)
        ax.fill_between(df.index, df['low'], df['high'], alpha=0.1, color='gray', label='High-Low Range')
        
        # Plot horizontal support levels
        for i, level in enumerate(horizontal_support):
            ax.axhline(y=level, color='green', linestyle='-', alpha=0.7, linewidth=2)
            ax.text(df.index[-1], level, f'S{i+1} ({support_touches[i]} touches)', 
                   verticalalignment='bottom', color='green', fontweight='bold')
        
        # Plot horizontal resistance levels  
        for i, level in enumerate(horizontal_resistance):
            ax.axhline(y=level, color='red', linestyle='-', alpha=0.7, linewidth=2)
            ax.text(df.index[-1], level, f'R{i+1} ({resistance_touches[i]} touches)', 
                   verticalalignment='bottom', color='red', fontweight='bold')
        
        # Plot trend support lines
        for i, trend in enumerate(trend_support):
            x_start, x_end = trend['start_idx'], trend['end_idx']
            y_start = trend['slope'] * x_start + trend['intercept']
            y_end = trend['slope'] * x_end + trend['intercept']
            
            ax.plot([df.index[x_start], df.index[x_end]], [y_start, y_end], 
                   color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(df.index[x_end], y_end, f'TS{i+1}', color='darkgreen', fontweight='bold')
        
        # Plot trend resistance lines
        for i, trend in enumerate(trend_resistance):
            x_start, x_end = trend['start_idx'], trend['end_idx']
            y_start = trend['slope'] * x_start + trend['intercept']
            y_end = trend['slope'] * x_end + trend['intercept']
            
            ax.plot([df.index[x_start], df.index[x_end]], [y_start, y_end], 
                   color='darkred', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(df.index[x_end], y_end, f'TR{i+1}', color='darkred', fontweight='bold')
        
        ax.set_title('Support and Resistance Levels Detection', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.savefig('detect_support_resistance_2.png')
    
    # Return comprehensive results
    return {
        'horizontal_support': horizontal_support,
        'horizontal_resistance': horizontal_resistance,
        'trend_support': trend_support,
        'trend_resistance': trend_resistance,
        'support_touches': support_touches,
        'resistance_touches': resistance_touches
    }

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

def analyze_and_plot(df: pd.DataFrame):
    """
    Main function to analyze the data, find all S/R levels and trend lines, and plot them.
    """
    print("Analyzing Support & Resistance...")
    
    # 1. Find Horizontal Levels
    support_levels = find_levels(df, is_support=True)
    resistance_levels = find_levels(df, is_support=False)
    
    # 2. Find Trend Lines
    support_trends = find_trend_lines(df, is_support=True)
    resistance_trends = find_trend_lines(df, is_support=False)

    # --- Reporting Results ---
    print("\n--- Identified Levels ---")
    print(f"Horizontal Support Levels: {[f'{level:.4f}' for level in support_levels]}")
    print(f"Horizontal Resistance Levels: {[f'{level:.4f}' for level in resistance_levels]}")
    
    print("\n--- Identified Trend Lines ---")
    if support_trends:
        print(f"Found {len(support_trends)} support trend line(s).")
        for i, trend in enumerate(support_trends):
             print(f"  - Support Trend {i+1}: Connects {len(trend['points'])} points with slope {trend['slope']:.4f}")
    else:
        print("No significant support trend lines found.")

    if resistance_trends:
        print(f"Found {len(resistance_trends)} resistance trend line(s).")
        for i, trend in enumerate(resistance_trends):
             print(f"  - Resistance Trend {i+1}: Connects {len(trend['points'])} points with slope {trend['slope']:.4f}")
    else:
        print("No significant resistance trend lines found.")
    
    # --- Plotting ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot candlestick
    ax.plot(df.index, df['close'], color='gray', alpha=0.7, label='Close Price')
    for idx, row in df.iterrows():
        color = 'cyan' if row['close'] >= row['open'] else 'magenta'
        ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
        ax.plot([idx, idx], [row['open'], row['close']], color=color, linewidth=4)
        
    # Plot horizontal levels
    for level in support_levels:
        ax.axhline(y=level, color='lime', linestyle='--', linewidth=1.2, label=f'Support {level:.4f}')
    for level in resistance_levels:
        ax.axhline(y=level, color='red', linestyle='--', linewidth=1.2, label=f'Resistance {level:.4f}')
        
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
    plt.show()
    plt.savefig('analyze_and_plot.png')

def detect_support_resistance_3(df, window=5, min_touches=2, cluster_threshold=0.01, 
                            trend_threshold=0.0005, plot=True, figsize=(15, 8)):
    """
    Detect horizontal support/resistance levels and trend lines from candlestick data.
    
    This function works by:
    1. Finding significant peaks (resistance) and troughs (support) in price data
    2. Clustering nearby levels to identify major support/resistance zones
    3. Determining if levels are horizontal or trending based on slope analysis
    4. Validating levels by counting how many times price touches them
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Candlestick data with columns: 'open', 'high', 'low', 'close'
        Index should be datetime or sequential
    window : int, default=20
        Lookback window for peak/trough detection (larger = fewer, stronger levels)
    min_touches : int, default=2
        Minimum number of times price must touch a level for it to be valid
    cluster_threshold : float, default=0.01
        Price clustering threshold as percentage (1% = 0.01)
    trend_threshold : float, default=0.0005
        Slope threshold to distinguish horizontal vs trending lines
    plot : bool, default=True
        Whether to plot the results
    figsize : tuple, default=(15, 8)
        Figure size for plotting
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'horizontal_support': List of horizontal support levels
        - 'horizontal_resistance': List of horizontal resistance levels  
        - 'trend_support': List of trend support lines (slope, intercept, start_idx, end_idx)
        - 'trend_resistance': List of trend resistance lines (slope, intercept, start_idx, end_idx)
        - 'support_touches': Touch counts for each support level
        - 'resistance_touches': Touch counts for each resistance level
    """
    
    # Step 1: Find peaks and troughs using scipy's peak detection
    # We use both high/low prices and closing prices for more comprehensive detection
    
    # Find resistance levels (peaks in highs)
    high_peaks, _ = find_peaks(df['high'].values, distance=window//2)
    
    # Find support levels (troughs in lows) - we invert the data to find peaks
    low_peaks, _ = find_peaks(-df['low'].values, distance=window//2)
    
    # Also find peaks/troughs in closing prices for additional confirmation
    close_peaks, _ = find_peaks(df['close'].values, distance=window//2)
    close_troughs, _ = find_peaks(-df['close'].values, distance=window//2)
    
    # Combine all resistance and support candidates
    resistance_candidates = np.concatenate([
        df.iloc[high_peaks]['high'].values,
        df.iloc[close_peaks]['close'].values
    ])
    
    support_candidates = np.concatenate([
        df.iloc[low_peaks]['low'].values,
        df.iloc[close_troughs]['close'].values
    ])
    
    print(f"Found {len(resistance_candidates)} resistance candidates and {len(support_candidates)} support candidates")
    
    # Step 2: Cluster nearby levels to identify major zones
    # This helps us group levels that are very close together into single zones
    
    def cluster_levels(levels, threshold):
        """Group nearby price levels using DBSCAN clustering"""
        if len(levels) == 0:
            return []
        
        # Reshape for sklearn
        levels_reshaped = levels.reshape(-1, 1)
        
        # Use DBSCAN to cluster nearby levels
        # eps is our clustering threshold (percentage of average price)
        avg_price = np.mean(df['close'])
        eps = threshold * avg_price
        
        clustering = DBSCAN(eps=eps, min_samples=1).fit(levels_reshaped)
        
        # Get the average level for each cluster
        clustered_levels = []
        for cluster_id in np.unique(clustering.labels_):
            cluster_levels = levels[clustering.labels_ == cluster_id]
            clustered_levels.append(np.mean(cluster_levels))
        
        return sorted(clustered_levels)
    
    # Cluster the levels
    clustered_resistance = cluster_levels(resistance_candidates, cluster_threshold)
    clustered_support = cluster_levels(support_candidates, cluster_threshold)
    
    print(f"After clustering: {len(clustered_resistance)} resistance levels, {len(clustered_support)} support levels")
    
    # Step 3: Count touches for each level and validate
    def count_touches(level, is_resistance=True):
        """Count how many times price touches a level within a small tolerance"""
        tolerance = cluster_threshold * np.mean(df['close']) / 2
        
        if is_resistance:
            # For resistance, check if high prices touch the level
            touches = np.sum((df['high'] >= level - tolerance) & (df['high'] <= level + tolerance))
            # Also check for close approaches
            close_approaches = np.sum((df['close'] >= level - tolerance) & (df['close'] <= level + tolerance))
        else:
            # For support, check if low prices touch the level
            touches = np.sum((df['low'] >= level - tolerance) & (df['low'] <= level + tolerance))
            # Also check for close approaches
            close_approaches = np.sum((df['close'] >= level - tolerance) & (df['close'] <= level + tolerance))
        
        return max(touches, close_approaches)
    
    # Validate levels by touch count
    valid_resistance = []
    valid_support = []
    resistance_touches = []
    support_touches = []
    
    for level in clustered_resistance:
        touches = count_touches(level, is_resistance=True)
        if touches >= min_touches:
            valid_resistance.append(level)
            resistance_touches.append(touches)
    
    for level in clustered_support:
        touches = count_touches(level, is_resistance=False)
        if touches >= min_touches:
            valid_support.append(level)
            support_touches.append(touches)
    
    print(f"Valid levels: {len(valid_resistance)} resistance, {len(valid_support)} support")
    
    # Step 4: Detect trend lines by connecting significant points
    def detect_trend_lines(peaks_idx, price_values, is_resistance=True):
        """Detect trend lines by connecting peaks/troughs"""
        trend_lines = []
        
        if len(peaks_idx) < 2:
            return trend_lines
        
        # Try connecting different combinations of peaks
        for i in range(len(peaks_idx)):
            for j in range(i + 2, len(peaks_idx)):  # Skip adjacent peaks
                x1, x2 = peaks_idx[i], peaks_idx[j]
                y1, y2 = price_values[i], price_values[j]
                
                # Calculate line parameters
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Check if this is a significant trend line
                if abs(slope) > trend_threshold * np.mean(df['close']):
                    # Count how many other points are near this line
                    line_touches = 0
                    tolerance = cluster_threshold * np.mean(df['close'])
                    
                    for k in range(x1, x2 + 1):
                        expected_y = slope * k + intercept
                        actual_y = df['high'].iloc[k] if is_resistance else df['low'].iloc[k]
                        
                        if abs(actual_y - expected_y) <= tolerance:
                            line_touches += 1
                    
                    # If enough points touch this line, it's a valid trend line
                    if line_touches >= min_touches:
                        trend_lines.append({
                            'slope': slope,
                            'intercept': intercept,
                            'start_idx': x1,
                            'end_idx': x2,
                            'touches': line_touches
                        })
        
        return trend_lines
    
    # Find trend lines for resistance and support
    trend_resistance = detect_trend_lines(high_peaks, df.iloc[high_peaks]['high'].values, True)
    trend_support = detect_trend_lines(low_peaks, df.iloc[low_peaks]['low'].values, False)
    
    # Step 5: Classify levels as horizontal or trending
    horizontal_resistance = valid_resistance.copy()
    horizontal_support = valid_support.copy()
    
    print(f"Trend lines found: {len(trend_resistance)} resistance, {len(trend_support)} support")
    
    # Step 6: Plotting with both candlesticks and line chart
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.3), 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # === TOP SUBPLOT: CANDLESTICK CHART ===
        def plot_candlesticks(ax, df):
            """
            Plot candlestick chart manually using matplotlib bars and lines.
            This approach gives us full control over the appearance and works
            well with our support/resistance overlays.
            """
            # Create x-axis positions (we'll use integer positions for simplicity)
            x_pos = np.arange(len(df))
            
            # Determine colors for each candle
            # Green/white for bullish (close > open), red/black for bearish
            colors = ['green' if close >= open_price else 'red' 
                     for close, open_price in zip(df['close'], df['open'])]
            
            # Plot the high-low lines (wicks)
            for i, (high, low) in enumerate(zip(df['high'], df['low'])):
                ax.plot([x_pos[i], x_pos[i]], [low, high], color='black', linewidth=1)
            
            # Plot the open-close rectangles (candle bodies)
            for i, (open_price, close, color) in enumerate(zip(df['open'], df['close'], colors)):
                height = abs(close - open_price)
                bottom = min(open_price, close)
                
                # Create rectangle for candle body
                # Use different styles for bullish vs bearish candles
                if color == 'green':  # Bullish candle
                    ax.bar(x_pos[i], height, bottom=bottom, width=0.6, 
                          color='lightgreen', edgecolor='green', linewidth=1)
                else:  # Bearish candle
                    ax.bar(x_pos[i], height, bottom=bottom, width=0.6, 
                          color='lightcoral', edgecolor='red', linewidth=1)
            
            # Set x-axis labels to show actual dates/times at reasonable intervals
            # Show every nth label to avoid overcrowding
            label_interval = max(1, len(df) // 10)  # Show roughly 10 labels
            tick_positions = x_pos[::label_interval]
            tick_labels = [str(df.index[i])[:10] for i in range(0, len(df), label_interval)]  # First 10 chars of date
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            return x_pos  # Return positions for overlaying support/resistance
        
        # Plot candlesticks on top subplot
        x_positions = plot_candlesticks(ax1, df)
        
        # Overlay support and resistance on candlestick chart
        # Plot horizontal support levels
        for i, level in enumerate(horizontal_support):
            ax1.axhline(y=level, color='green', linestyle='-', alpha=0.8, linewidth=2.5)
            ax1.text(len(df)-1, level, f'  S{i+1} ({support_touches[i]}T)', 
                    verticalalignment='center', color='green', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot horizontal resistance levels  
        for i, level in enumerate(horizontal_resistance):
            ax1.axhline(y=level, color='red', linestyle='-', alpha=0.8, linewidth=2.5)
            ax1.text(len(df)-1, level, f'  R{i+1} ({resistance_touches[i]}T)', 
                    verticalalignment='center', color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot trend support lines on candlestick chart
        for i, trend in enumerate(trend_support):
            x_start, x_end = trend['start_idx'], trend['end_idx']
            y_start = trend['slope'] * x_start + trend['intercept']
            y_end = trend['slope'] * x_end + trend['intercept']
            
            ax1.plot([x_start, x_end], [y_start, y_end], 
                    color='darkgreen', linestyle='--', linewidth=3, alpha=0.9)
            ax1.text(x_end, y_end, f'  TS{i+1}', color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot trend resistance lines on candlestick chart
        for i, trend in enumerate(trend_resistance):
            x_start, x_end = trend['start_idx'], trend['end_idx']
            y_start = trend['slope'] * x_start + trend['intercept']
            y_end = trend['slope'] * x_end + trend['intercept']
            
            ax1.plot([x_start, x_end], [y_start, y_end], 
                    color='darkred', linestyle='--', linewidth=3, alpha=0.9)
            ax1.text(x_end, y_end, f'  TR{i+1}', color='darkred', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax1.set_title('Candlestick Chart with Support & Resistance Levels', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # === BOTTOM SUBPLOT: LINE CHART (Original Style) ===
        # Plot the traditional line chart for comparison
        ax2.plot(df.index, df['close'], label='Close Price', linewidth=1.5, alpha=0.8, color='blue')
        ax2.fill_between(df.index, df['low'], df['high'], alpha=0.15, color='gray', label='High-Low Range')
        
        # Plot horizontal support levels on line chart
        for i, level in enumerate(horizontal_support):
            ax2.axhline(y=level, color='green', linestyle='-', alpha=0.7, linewidth=2)
            ax2.text(df.index[-1], level, f'S{i+1}', 
                   verticalalignment='bottom', color='green', fontweight='bold')
        
        # Plot horizontal resistance levels on line chart
        for i, level in enumerate(horizontal_resistance):
            ax2.axhline(y=level, color='red', linestyle='-', alpha=0.7, linewidth=2)
            ax2.text(df.index[-1], level, f'R{i+1}', 
                   verticalalignment='bottom', color='red', fontweight='bold')
        
        # Plot trend support lines on line chart
        for i, trend in enumerate(trend_support):
            x_start, x_end = trend['start_idx'], trend['end_idx']
            y_start = trend['slope'] * x_start + trend['intercept']
            y_end = trend['slope'] * x_end + trend['intercept']
            
            ax2.plot([df.index[x_start], df.index[x_end]], [y_start, y_end], 
                   color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
            ax2.text(df.index[x_end], y_end, f'TS{i+1}', color='darkgreen', fontweight='bold')
        
        # Plot trend resistance lines on line chart
        for i, trend in enumerate(trend_resistance):
            x_start, x_end = trend['start_idx'], trend['end_idx']
            y_start = trend['slope'] * x_start + trend['intercept']
            y_end = trend['slope'] * x_end + trend['intercept']
            
            ax2.plot([df.index[x_start], df.index[x_end]], [y_start, y_end], 
                   color='darkred', linestyle='--', linewidth=2, alpha=0.8)
            ax2.text(df.index[x_end], y_end, f'TR{i+1}', color='darkred', fontweight='bold')
        
        ax2.set_title('Line Chart View (Traditional)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.savefig('detect_support_resistance_3.png')
    
    # Return comprehensive results
    return {
        'horizontal_support': horizontal_support,
        'horizontal_resistance': horizontal_resistance,
        'trend_support': trend_support,
        'trend_resistance': trend_resistance,
        'support_touches': support_touches,
        'resistance_touches': resistance_touches
    }

async def main(ssid: str):
    api = PocketOptionAsync(ssid)    
    await asyncio.sleep(5)  # Wait for connection

    candles = await api.get_candles("NZDUSD_otc", 60, 10800)
    df = pd.DataFrame.from_dict(candles)

    # Detect levels
    # levels = detect_support_resistance(df)
    # levels_2 = detect_support_resistance_2(df)
    levels_3 = analyze_and_plot(df)
    levels_4 = detect_support_resistance_3(df)

    # print("------------------------------------------------------------")
    # print(levels)
    # print("------------------------------------------------------------")
    # print(levels_2)
    # print("------------------------------------------------------------")
    # levels_3
    print("------------------------------------------------------------")
    print(levels_4)


if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    asyncio.run(main(ssid))

# # Example usage function
# def example_usage():
#     """
#     Example of how to use the support/resistance detection function.
#     This creates sample data to demonstrate the functionality.
#     """
    
#     # Create sample candlestick data
#     np.random.seed(42)
#     dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
#     # Generate realistic price data with trends and levels
#     price = 100
#     prices = [price]
    
#     for i in range(199):
#         # Add some trending behavior and noise
#         change = np.random.normal(0, 1) + 0.02 * np.sin(i/20)  # Trending component
#         price += change
#         prices.append(price)
    
#     prices = np.array(prices)
    
#     # Create OHLC data
#     sample_data = pd.DataFrame({
#         'open': prices + np.random.normal(0, 0.5, len(prices)),
#         'high': prices + np.abs(np.random.normal(0, 1, len(prices))),
#         'low': prices - np.abs(np.random.normal(0, 1, len(prices))),
#         'close': prices + np.random.normal(0, 0.3, len(prices))
#     }, index=dates)
    
#     # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
#     sample_data['high'] = np.maximum(sample_data['high'], 
#                                    np.maximum(sample_data['open'], sample_data['close']))
#     sample_data['low'] = np.minimum(sample_data['low'], 
#                                   np.minimum(sample_data['open'], sample_data['close']))
    
#     print("Sample candlestick data created. Running support/resistance detection...")
    
#     # Run the detection
#     results = detect_support_resistance(sample_data, window=15, min_touches=3)
    
#     return results

# # Uncomment the line below to run the example
# example_results = example_usage()
# print(example_results)