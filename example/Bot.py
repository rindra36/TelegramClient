import os
import talib  
import pandas as pd  
import numpy as np  
from datetime import datetime  
import requests  # For API integration
from bs4 import BeautifulSoup

# --- Step 1: Fetch Historical Data (Example: Alpha Vantage) [[7]]  
def fetch_data(symbol, interval='5min'):  
    # API_KEY = 'EIHE1J45K5KJYOMI'  
    # url = f'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={symbol[:3]}&to_symbol={symbol[3:]}&interval={interval}&apikey={API_KEY}'
    # url = f'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={symbol[:3]}&to_symbol={symbol[3:]}&interval={interval}&apikey=demo'
    
    # data = requests.get(url).json()
    # # df = pd.DataFrame(data[f'Time Series FX ({interval})']).T  
    # df = pd.DataFrame(data[f'quotes']).T  
    # df.columns = ['open', 'high', 'low', 'close']  

    # url = 'https://marketdata.tradermade.com/api/v1/timeseries?api_key=3uCIT5poAkGHv0-NkCS1&currency=GBPJPY&format=index&start_date=2025-04-29-00:00&end_date=2025-05-01-19:35&interval=minute&period=5'
    # data = requests.get(url).json()
    # quotes = data['quotes'].values()
    
    # # Convert to DataFrame
    # df = pd.DataFrame(quotes)
    
    # # Convert 'date' to datetime and set as index
    # df['date'] = pd.to_datetime(df['date'])
    # df.set_index('date', inplace=True)
    # df.sort_index(inplace=True)

    # Load CSV data
    df = pd.read_csv('candles_data/GBPJPY_historical_data.csv', 
                    parse_dates=['Date'], 
                    date_parser=lambda x: pd.date_(x, format='%m/%d/%Y %H:%M'))

    # Clean data
    df = df.replace(',', '', regex=True)  # Remove commas from numbers
    df = df.astype({
        'Open': float,
        'High': float,
        'Low': float,
        'Close': float,
        'Change(Pips)': float,
        'Change(%)': float
    })

    # Format for trading bot
    df = df.sort_values('Date').reset_index(drop=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]  # Keep essential columns

    # Set datetime index
    df.set_index('Date', inplace=True)

    print(df.head())  # Debugging: Check the first few rows of the DataFrame

    return df  

# --- Step 2: Strategy Logic [[7]]  
def generate_signals(df):  
    # Calculate indicators  
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)  
    df['macd'], df['macd_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)  
    df['upper'], df['mid'], df['lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)  
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)  

    # Entry Rules  
    df['signal'] = 0  
    df.loc[(df['close'] > df['upper']) & (df['rsi'] > 70) & (df['macd'] > df['macd_signal']), 'signal'] = 1  # Call  
    df.loc[(df['close'] < df['lower']) & (df['rsi'] < 30) & (df['macd'] < df['macd_signal']), 'signal'] = -1  # Put  

    # Volatility Filter [[7]]  
    df['vol_filter'] = np.where(df['atr'] > 0.0015, 1, 0)  # Skip low volatility  
    df['signal'] = df['signal'] * df['vol_filter']  
    return df  

# --- Step 3: Risk Management [[7]]  
def calculate_position_size(account_balance, atr, risk_percent=0.01):  
    risk_per_trade = account_balance * risk_percent  
    pip_value = 0.0001  # For EUR/USD  
    position_size = risk_per_trade / (atr * pip_value)  
    return position_size  

# --- Step 4: Backtesting Example [[7]]  
def backtest(df):  
    # df['returns'] = df['signal'].shift(1) * (df['close'].pct_change())  
    # df['cumulative_returns'] = (1 + df['returns']).cumprod()  
    # win_rate = len(df[df['returns'] > 0]) / len(df[df['signal'] != 0])  
    # print(f"Win Rate: {win_rate:.2%}, Total Return: {df['cumulative_returns'].iloc[-1]:.2f}")

    df['returns'] = df['signal'].shift(1) * df['close'].pct_change()

    signals = df[df['signal'] != 0]
    if len(signals) == 0:
        print("No valid trades found. Check strategy parameters.")
        return

    # Track individual trade outcomes
    trade_log = []
    
    for i in range(len(signals)):
        entry_time = signals.index[i]
        direction = "call" if signals['signal'].iloc[i] == 1 else "put"
        return_pct = signals['returns'].iloc[i]  # From previous code
        
        # Classify as win/loss
        outcome = "WIN" if return_pct > 0 else "LOSS"
        
        trade_log.append({
            "entry_time": entry_time,
            "direction": direction,
            "return_pct": return_pct,
            "outcome": outcome
        })
    
    # Save trade log to CSV
    log_df = pd.DataFrame(trade_log)
    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_candle = f"{output_dir}/trade_log_{timestamp}.csv"
    log_df.to_csv(filename_candle, index=False)
    
    # Calculate aggregate metrics
    win_rate = len(log_df[log_df['outcome'] == "WIN"]) / len(log_df)
    total_return = (1 + log_df['return_pct']).prod()
    
    print(f"Win Rate: {win_rate:.2%}, Total Return: {total_return:.2f}")
    print("Trade log saved to trade_log.csv")

# --- Step 5: API Integration (Example: IQ Option) [[10]]  
class BrokerAPI:  
    def __init__(self, api_key):  
        self.api_key = api_key  

    def place_order(self, symbol, direction, amount, timestamp):  
        # Simplified API call (replace with broker-specific endpoints)  
        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{formatted_time}] Order placed: {direction} {amount:.2f} on {symbol}")

# --- Main Execution ---  
if __name__ == "__main__":  
    # Load data
    asset = 'GBPJPY'
    df = fetch_data(asset)  
    df = generate_signals(df)  
    
    # Create output directory if it doesn't exist
    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_candle = f"{output_dir}/bot_{timestamp}.csv"
    df.to_csv(filename_candle, index=True)

    backtest(df)  

    # Live trading loop (simplified)  
    account_balance = 10000  # $10,000  
    for i in range(len(df)-1, 0, -1):  
        if df['signal'].iloc[i] != 0:  
            atr = df['atr'].iloc[i]
            order_time = df.index[i]
            position_size = calculate_position_size(account_balance, atr, risk_percent=0.02)  
            api = BrokerAPI('YOUR_API_KEY')  
            api.place_order(asset, 'call' if df['signal'].iloc[i] == 1 else 'put', position_size, order_time)