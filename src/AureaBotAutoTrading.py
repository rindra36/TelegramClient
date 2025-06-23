global selected_pair  # inserted
global mtg_value  # inserted
global initial_amount  # inserted
global bot_running  # inserted
global trade_history  # inserted
global mtg_status  # inserted
import time
import pandas as pd
import numpy as np
import argparse
import json
import logging
# from BinaryOptionsTools.platforms.pocketoption.stable_api import PocketOption
# from BinaryOptionsToolsV2.pocketoption import PocketOption as PocketOptionV2
# from pocketoptionapi.stable_api import PocketOption
# from PocketOptionMethod import PocketOptionMethod
from BinaryOptionsTools.platforms.pocketoption.stable_api import PocketOption
from BinaryOptionsToolsV2.pocketoption import PocketOption as PocketOptionV2
from datetime import datetime, timedelta
from pathlib import Path
import pytz
import keyboard
import traceback
import nest_asyncio
import asyncio
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('bot.log')
logger.addHandler(console_handler)
logger.addHandler(file_handler)
BOT_NAME = 'Aurea TRB 1'
DISCORD_INVITE = 'Join our Discord: https://discord.gg/wv7FRSHuYW'
keys_data = {'541211': datetime.now(pytz.timezone('America/New_York')) + timedelta(days=7), '121542': datetime.now(pytz.timezone('America/New_York')) + timedelta(days=7), '215813': datetime.now(pytz.timezone('America/New_York')) + timedelta(days=7), '658794': datetime.now(pytz.timezone('America/New_York')) + timedelta(days=7), '131515': datetime.now(pytz.timezone('America/New_York')) + timedelta(days=7), '151566': datetime.now(pytz.timezone('America/New_York')) + timedelta(days=7), '213257': datetime.now(pytz.timezone('America/New_York')) + timedelta(days=7), '245878': datetime.now(pytz.timezone('America/New_York')) + timedelta(days=7), '154679': datetime.now(pytz.timezone('America/New_York')) + timedelta(days=7)}
PAIRS = ['ADA-USD_otc', 'AEDCNY_otc', 'AMZN_otc', 'AUDCAD_otc', 'AUDCHF_otc', 'AUDJPY_otc', 'AUDNZD_otc', 'AUDUSD_otc', 'AUS200_otc', 'BABA_otc', 'BHDCNY_otc', 'BITB_otc', 'BNB-USD_otc', 'BTCUSD_otc', 'CADCHF_otc', 'CADJPY_otc', 'CHFJPY_otc', 'CHFNOK_otc', 'CITI_otc', 'D30EUR_otc', 'DJI30_otc', 'DOGE_otc', 'DOTUSD_otc', 'E35EUR_otc', 'E50EUR_otc', 'ETHUSD_otc', 'EURCHF_otc', 'EURGBP_otc', 'EURHUF_otc', 'EURJPY_otc', 'EURNZD_otc', 'EURRUB_otc', 'EURTRY_otc', 'EURUSD_otc', 'F40EUR_otc', 'FDX_otc', 'GBPJPY_otc', 'GBPAUD_otc', 'GBPUSD_otc', 'IRRUSD_otc', 'JODCNY_otc', 'JNJ_otc', 'JPN225_otc', 'LBPUSD_otc', 'LINK_otc', 'LTCUSD_otc', 'MADUSD_otc', 'MATIC_otc', 'MSFT_otc', 'NFLX_otc', 'NASUSD_otc', 'NZDJPY_otc', 'NZDUSD_otc', 'OMRCNY_otc', 'QARCNY_otc', 'SARCNY_otc', 'SOL-USD_otc', 'SP500_otc', 'SYPUSD_otc', 'TON-USD_otc', 'TRX-USD_otc', 'TWITTER_otc', 'UKBrent_otc', 'USDARS_otc', 'USDBDT_otc', 'USDBRL_otc', 'USDCAD_otc', 'USDCHF_otc', 'USDCLP_otc', 'USDCNH_otc', 'USDCOP_otc', 'USDDZD_otc', 'USDEGP_otc', 'USDIDR_otc', 'USDINR_otc', 'USDJPY_otc', 'USDMXN_otc', 'USDMYR_otc', 'USDPHP_otc', 'USDPKR_otc', 'USDRUB_otc', 'USDSGD_otc', 'USDTHB_otc', 'USDVND_otc', 'USCrude_otc', 'VISA_otc', 'XAGUSD_otc', 'XAUUSD_otc', 'XNGUSD_otc', 'XPDUSD_otc', 'XPTUSD_otc', 'XPRUSD_otc', 'YERUSD_otc', '#AALP_otc', '#AXP_otc', '#BA_otc', '#CSCO_otc', '#FB_otc', '#INTC_otc', '#JNJ_otc', '#MCD_otc', '#MSFT_otc', '#PFE_otc', '#TSLA_otc', '#XOM_otc', 'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURHUF', 'EURJPY', 'EURUSD', 'F40EUR', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
mtg_status = False
mtg_value = 1.0
bot_running = True
selected_pair = None
initial_amount = 0.0
tz = pytz.timezone('America/New_York')
trade_history = []
ROOT_PATH = Path(__file__).parents[1]
CONFIG_PATH = f'{ROOT_PATH}/assets/env/config.json'
TRADE_HISTORY_PATH = f'{ROOT_PATH}/logs/trade_history/trade_history.json'

def check_key(key):
    """Check if the provided key is valid and within the expiry period."""  # inserted
    if key in keys_data:
        if datetime.now(tz) < keys_data[key]:
            return True
        print('Your key has expired. Please contact us on Discord.')
        return False
    return False

def get_key_input():
    """Prompt the user to enter a key."""  # inserted
    key = input('Please enter your access key to continue: ').strip()
    if check_key(key):
        print('Key validated successfully. You can now use the bot.')
    else:  # inserted
        print('Invalid or expired key. Exiting the bot.')
        exit()

def load_trade_history():
    """Load the trade history from a JSON file."""  # inserted
    try:
        with open(TRADE_HISTORY_PATH, 'r') as file:
            pass  # postinserted
    except FileNotFoundError:
            return json.load(file)
            return []
    except json.JSONDecodeError as e:
        logger.error(f'Error loading trade history: {e}')
        return []
    else:  # inserted
        pass

def save_trade_history():
    """Save the trade history to a JSON file."""  # inserted
    with open(TRADE_HISTORY_PATH, 'w') as file:
        json.dump(trade_history, file)

def calculate_bollinger_bands(data, period=14, std_dev=2):
    """Calculate Bollinger Bands."""  # inserted
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper_band = sma + std * std_dev
    lower_band = sma - std * std_dev
    return (upper_band, sma, lower_band)

def calculate_stochastic(data, window=7):
    """Calculate Stochastic Oscillator."""  # inserted
    lowest_low = data['low'].rolling(window=window).min()
    highest_high = data['high'].rolling(window=window).max()
    stochastic = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    return stochastic

def select_pair_with_payouts(api):
    """Displays pairs and their payouts, allows user to select one."""  # inserted
    print('\nAvailable Pairs with Payouts:')
    pair_payouts = GetMaximumProfitPairs(api, PAIRS)
    if not pair_payouts:
        print('No pairs with a payout above 70% available.')
        return
    for i, (pair, payout) in enumerate(pair_payouts, 1):
        print(f'({i}) {pair} - Payout: {payout}%')
    while True:
        try:
            choice = int(input('\nEnter the number of the pair you want to analyze: '))
            if 1 <= choice <= len(pair_payouts):
                print(f'You selected: {pair_payouts[choice - 1][0]} with a payout of {pair_payouts[choice - 1][1]}%')
                return pair_payouts[choice - 1][0]
            print('Invalid choice. Please select a valid number.')
        except ValueError:
                print('Invalid input. Please enter a number.')
    pass

def mtg_on_off():
    """Turn MTG On/Off."""  # inserted
    global mtg_status  # inserted
    print(f"\nMTG is currently {('ON' if mtg_status else 'OFF')}.")
    choice = input('Would you like to turn it ON or OFF? (on/off): ').strip().lower()
    if choice == 'on':
        mtg_status = True
        print('MTG is now ON.')
    else:  # inserted
        if choice == 'off':
            mtg_status = False
            print('MTG is now OFF.')
        else:  # inserted
            print('Invalid input. Please enter \'on\' or \'off\'.')

def mod_mtg():
    """Modify the MTG value."""  # inserted
    global mtg_value  # inserted
    try:
        new_value = float(input('\nEnter new MTG value (e.g., 1.5, 2): ').strip())
        if new_value > 0:
            mtg_value = new_value
            print(f'MTG value set to {mtg_value}.')
        else:  # inserted
            print('MTG value must be greater than 0.')
    except ValueError:
        print('Invalid input. Please enter a valid number.')
    else:  # inserted
        pass

def mod_amount():
    """Modify the trade amount."""  # inserted
    global initial_amount  # inserted
    try:
        new_amount = float(input('\nEnter new trade amount (e.g., 10.0): ').strip())
        if new_amount > 0:
            initial_amount = new_amount
            print(f'Trade amount set to {initial_amount}.')
        else:  # inserted
            print('Trade amount must be greater than 0.')
    except ValueError:
        print('Invalid input. Please enter a valid number.')
    else:  # inserted
        pass

def restart_connection():
    """Simulate restarting connection."""  # inserted
    print('\nRestarting connection...')
    time.sleep(2)
    print('Connection restarted successfully.')

def stop_current_session():
    """Stop the current session without turning off the bot."""  # inserted
    print('\nStopping current session...')
    time.sleep(1)
    print('Session stopped. You can start a new session anytime.')

def turn_off_bot():
    """Turn off the bot."""  # inserted
    global bot_running  # inserted
    print('\nTurning off bot...')
    bot_running = False

def show_status():
    """Display bot status."""  # inserted
    print('\n====== BOT STATUS ======')
    print(f"MTG Status: {('ON' if mtg_status else 'OFF')}")
    print(f'MTG Value: {mtg_value}')
    print(f'Trade Amount: {initial_amount}')
    print('========================')

def view_trade_history():
    """Display the history of executed trades."""  # inserted
    if not trade_history:
        print('\nNo trades have been executed yet.')
        return
    print('\n=== Trade History ===')
    for trade in trade_history:
        print(f"{trade['time']} - {trade['pair']} - {trade['decision']} - {trade['result']} - Amount: {trade['amount']} - Payout: {trade.get('payout', 'N/A')}")
        if 'indicators' in trade:
            indicators = trade['indicators']
            price_value = indicators.get('price', 'N/A')
            stochastic_value = indicators.get('stochastic', 'N/A')
            if isinstance(price_value, (int, float)):
                price_value = f'{price_value:.2f}'
            if isinstance(stochastic_value, (int, float)):
                stochastic_value = f'{stochastic_value:.2f}'
            print(f'  Indicators: Price={price_value}, Stochastic={stochastic_value}')

def view_trade_stats():
    """Display overall trade statistics."""  # inserted
    if not trade_history:
        print('\nNo trades executed to show statistics.')
        return
    wins = sum((1 for trade in trade_history if trade['result'] == 'Win'))
    losses = sum((1 for trade in trade_history if trade['result'] == 'Lose'))
    total = len(trade_history)
    win_rate = wins / total * 100 if total > 0 else 0
    total_profit = 0
    for trade in trade_history:
        if 'amount' in trade and trade['result'] == 'Win':
            total_profit += trade['amount'] * 0.7
        else:  # inserted
            if 'amount' in trade and trade['result'] == 'Lose':
                total_profit -= trade['amount']
    print('\n=== Trade Statistics ===')
    print(f'Total Trades: {total}')
    print(f'Wins: {wins}')
    print(f'Losses: {losses}')
    print(f'Win Rate: {win_rate:.2f}%')
    if total > 0:
        print(f'Estimated Profit: {total_profit:.2f}')

def reset_trade_statistics():
    """Reset trade statistics by clearing the trade history."""  # inserted
    global trade_history  # inserted
    trade_history = []
    save_trade_history()
    print('Trade history and statistics have been reset.')

def test_api_connection(api):
    """Test API connection and display status."""  # inserted
    print('\nTesting API connection...')
    if api.check_connect():
        print('API connection is healthy.')
    else:  # inserted
        print('API connection failed.')

def GetMaximumProfitPairs(api, pairs):
    data = []
    for pair in pairs:
        try:
            profit = api.GetPayout(pair)
            if profit is not None and profit >= 70:
                data.append((pair, profit))
            else:  # inserted
                logger.info(f'Skipping pair {pair} due to low or invalid payout')
        except Exception as e:
            traceback_exception = traceback.format_exc()
            print(f'Error: {e}', traceback_exception)
    return data

def get_parser():
    parser = argparse.ArgumentParser(prog=BOT_NAME, description='A trading bot using multiple strategies.')
    parser.add_argument('-r', '--real', action='store_true', help='Use real or demo accounts, defaults to demo')
    parser.add_argument('-c', '--config', default=CONFIG_PATH, help='Path to configuration file')
    return parser

def load_config(path):
    with open(path) as file:
        return json.load(file)

def get_config():
    print(f'{BOT_NAME}\n{DISCORD_INVITE}\n')
    args = get_parser().parse_args()
    config = load_config(args.config)
    ssid = config.get('ssid') or '42["auth",{"session":"24k4ea9r0a1qck71rfojrgnl8o","isDemo":1,"uid":96282099,"platform":3}]' or input('Please paste your SSID: ')
    pairs = PAIRS
    return (ssid, config.get('demo', True), pairs)

def fetch_candles(api, apiV2, active, period, num_candles):
    """Fetch candles for a given active pair."""  # inserted
    try:
        while not api.check_connect():
            time.sleep(1)
        print('Fetching candles')

        candles = apiV2.get_candles(active, period, 7200)
        # print(f'Fetched candles V2')

        # candles_v1 = api.get_candles(active, period)
        # print(f'Fetched candles V1')

        # candles_df_v1 = pd.DataFrame(candles_v1)
        candles_df = pd.DataFrame(candles)
        # print(f'Candles V1 : {candles_df_v1}')
        # print(f'Candles V2 : {candles_df}')

        # Custom parsing function to handle both timestamp formats
        def parse_timestamp(ts):
            try:
                # First try parsing with milliseconds
                return pd.to_datetime(ts, format='%Y-%m-%dT%H:%M:%S.%fZ', utc=True)
            except ValueError:
                try:
                    # If that fails, try parsing without milliseconds
                    return pd.to_datetime(ts, format='%Y-%m-%dT%H:%M:%SZ', utc=True)
                except ValueError:
                    # If both fail, try the generic ISO8601 parser
                    return pd.to_datetime(ts, format='ISO8601', utc=True)

        # Apply the custom parsing function
        candles_df['time'] = candles_df['time'].apply(parse_timestamp)
        candles_df['time'] = candles_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        print('Fetched candles')
        if len(candles_df) > num_candles:
            candles_df = candles_df.iloc[-num_candles:]
        if candles_df.isnull().values.any():
            candles_df.fillna(0, inplace=True)
        required_columns = ['time', 'open', 'high', 'low', 'close']
        candles_df.columns = required_columns
        candles_df['time'] = pd.to_datetime(candles_df['time'])
        return candles_df
    except Exception as e:
        traceback_exception = traceback.format_exc()
        print(f'Error: {e}', traceback_exception)
        return None

def check_win(api, trade_id):
    """Check the result of a trade given its trade ID."""  # inserted
    try:
        return api.check_win(trade_id)
    except Exception as e:
        logger.error(f'Error checking trade result for ID {trade_id}: {e}')
        return 'Error'

def backtest(api, apiV2, active: str, initial_amount: float) -> float:
    """Run a backtest on the selected pair."""  # inserted
    candles_df = fetch_candles(api, apiV2, active, 30, 180)
    if candles_df is None or candles_df.empty:
        logger.error(f'No data available for backtesting {active}.')
        return 0.0
    close_prices = candles_df['close']
    if len(close_prices) < 2:
        logger.error('Not enough candle data to perform backtest.')
        return 0.0
    upper_band, _, lower_band = calculate_bollinger_bands(candles_df)
    stochastic = calculate_stochastic(candles_df)
    wins = 0
    losses = 0
    total_trades = 0
    for i in range(len(candles_df) - 1):
        if i >= len(stochastic) or i >= len(lower_band):
            break
        latest_price = close_prices.iloc[i]
        latest_stochastic = stochastic.iloc[i]
        buy_signal = latest_price < lower_band.iloc[i] and latest_stochastic < 20
        sell_signal = latest_price > upper_band.iloc[i] and latest_stochastic > 80
        if buy_signal or sell_signal:
            total_trades += 1
            next_price = close_prices.iloc[i + 1]
            if buy_signal and next_price > latest_price:
                wins += 1
            else:  # inserted
                if sell_signal and next_price < latest_price:
                    wins += 1
                else:  # inserted
                    losses += 1
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    logger.info(f'Backtest results for {active}: Total Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%')
    return win_rate

def run_backtests_on_all_pairs(api, apiV2):
    """Run backtests on all pairs and return results sorted by win rate."""  # inserted
    results = []
    pairs_with_profit = GetMaximumProfitPairs(api, PAIRS)
    if not pairs_with_profit:
        logger.warning('No pairs available for testing with a payout above 70%.')
        return []
    logger.info('Starting backtest for all available pairs...')
    for pair, payout in pairs_with_profit:
        win_rate = backtest(api, apiV2, pair, initial_amount)
        results.append((pair, win_rate))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results

def run(api, apiV2, pairs, initial_amount):
    """Start the trading process for all selected pairs or a single selected pair."""  # inserted
    if selected_pair == 'all':
        pairs = GetMaximumProfitPairs(api, pairs)
        if not pairs:
            logger.warning('No available pairs with payout above 70%.')
            return
        print('Starting trading on all selected pairs. Please wait...\n')
        time.sleep(5)
        for pair, payout in pairs:
            logger.info(f'Analyzing {pair} for trading opportunities...')
            candles_df = fetch_candles(api, apiV2, pair, 5, 200)
            if candles_df is None or candles_df.empty:
                logger.warning(f'No data available for {pair}, skipping...')
                continue
            close_prices = candles_df['close']
            upper_band, _, lower_band = calculate_bollinger_bands(candles_df)
            stochastic = calculate_stochastic(candles_df)
            latest_upper = upper_band.iloc[(-1)]
            latest_lower = lower_band.iloc[(-1)]
            latest_price = close_prices.iloc[(-1)]
            latest_stochastic = stochastic.iloc[(-1)]
            buy_signal = latest_price < latest_lower and latest_stochastic < 20
            sell_signal = latest_price > latest_upper and latest_stochastic > 80
            if not buy_signal and (not sell_signal):
                continue
            action = 'call' if buy_signal else 'put'
            amount_to_trade = initial_amount * mtg_value if mtg_status else initial_amount
            trade_response = api.buy(amount_to_trade, pair, action, 50)
            if trade_response[0]:
                trade_id = trade_response[1]
                logger.info(f'Trade placed at new candle: {action.upper()} on {pair} for Amount: {amount_to_trade}, Trade ID: {trade_id}')
                time.sleep(5)
                result = check_win(api, trade_id)
                # trade_history.append({'time': datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S'), 'pair': pair, 'decision': action, 'result': result, 'amount': amount_to_trade, 'indicators': {'price': float(latest_price), 'bb_upper': float(latest_upper), 'bb_lower': float(latest_lower), 'stochastic': float(latest_stochastic)}})
                # save_trade_history()
            else:  # inserted
                logger.warning(f'Failed to place trade on {pair}. Response: {trade_response}')
    else:  # inserted
        analyze_pair(api, apiV2, selected_pair, initial_amount)

def analyze_pair(api, apiV2, selected_pair, initial_amount):
    """Analyze the selected pair for trading opportunities using Bollinger Bands and Stochastic."""  # inserted
    logger.info(f'Analyzing {selected_pair} for trading opportunities...')
    pending_trade = False
    pending_action = None
    while bot_running:
        if keyboard.is_pressed('esc'):
            print('\nStopped analyzing. Returning to menu...\n')
            return
        current_time = datetime.now(tz)
        seconds_to_next_minute = 60 - current_time.second
        candles_df = fetch_candles(api, apiV2, selected_pair, 30, 200)
        if candles_df is not None and candles_df.empty:
            time.sleep(2)
            continue
        close_prices = candles_df['close']
        upper_band, _, lower_band = calculate_bollinger_bands(candles_df)
        stochastic = calculate_stochastic(candles_df)
        latest_upper = upper_band.iloc[(-1)]
        latest_lower = lower_band.iloc[(-1)]
        latest_price = close_prices.iloc[(-1)]
        latest_stochastic = stochastic.iloc[(-1)]
        buy_signal = latest_price < latest_lower and latest_stochastic < 20
        sell_signal = latest_price > latest_upper and latest_stochastic > 80
        if (buy_signal or sell_signal) and (not pending_trade):
            pending_trade = True
            pending_action = 'call' if buy_signal else 'put'
            logger.info(f'Signal detected: {pending_action.upper()} on {selected_pair}. Will execute at next candle open.')
            amount_to_trade = initial_amount * mtg_value if mtg_status else initial_amount
            trade_response = api.buy(amount_to_trade, selected_pair, pending_action, 50)
            if trade_response[0]:
                trade_id = trade_response[1]
                logger.info(f'Trade placed at new candle: {pending_action.upper()} on {selected_pair} for Amount: {amount_to_trade}, Trade ID: {trade_id}')
                new_trade = {'time': datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S'), 'pair': selected_pair, 'decision': pending_action, 'result': 'Pending', 'amount': amount_to_trade, 'indicators': {'price': float(latest_price), 'bb_upper': float(latest_upper), 'bb_lower': float(latest_lower), 'stochastic': float(latest_stochastic)}}
                trade_history.append(new_trade)
                save_trade_history()
                time.sleep(5)
                result = check_win(api, trade_id)
                new_trade['result'] = result
                save_trade_history()
                logger.info(f'Trade result: {result}')
                pending_trade = False
            else:  # inserted
                logger.warning(f'Failed to place trade on {selected_pair}. Response: {trade_response}')
        time.sleep(1)

if __name__ == '__main__':
    # get_key_input()
    ssid, demo, pairs = get_config()
    logger.info(f'Using real account: {not demo}')
    api = PocketOption(ssid, demo)
    apiV2 = PocketOptionV2(ssid)
    # pocketOptionMethod = PocketOptionMethod(1, 'demo')
    # api = pocketOptionMethod.api_v1.api
    # apiV2 = pocketOptionMethod.api_sync
    # apiV2= pocketOptionMethod.api_async
    api.connect()
    time.sleep(10)
    trade_history = load_trade_history()
    while bot_running:
        print('\n=== Aurea Bot Menu ===')
        print('1. Trade Amount')
        print('2. Modify MTG Value')
        print('3. Turn MTG On/Off')
        print('4. Select Trading Pair')
        print('5. Select All Pairs')
        print('6. Restart Connection')
        print('7. Stop Current Session')
        print('8. Turn Off Bot')
        print('9. Instructions')
        print('10. View Trade History')
        print('11. View Trade Statistics')
        print('12. Test API Connection')
        print('13. Reset Trade Statistics')
        print('14. Backtest All Pairs')
        print('\n=== Join our Discord: https://discord.gg/wv7FRSHuYW ===')
        choice = input('Select an option: ').strip()
        if choice == '1':
            mod_amount()
        else:  # inserted
            if choice == '2':
                mod_mtg()
            else:  # inserted
                if choice == '3':
                    mtg_on_off()
                else:  # inserted
                    if choice == '4':
                        selected_pair = select_pair_with_payouts(api)
                        if selected_pair:
                            logger.info(f'Selected trading pair: {selected_pair}. Starting trading...')
                            run(api, apiV2, [selected_pair], initial_amount)
                    else:  # inserted
                        if choice == '5':
                            selected_pair = 'all'
                            if mtg_status:
                                initial_amount *= 2
                            logger.info(f"Starting trading on all pairs with {('doubled' if mtg_status else 'same')} amount...")
                            run(api, apiV2, pairs, initial_amount)
                        else:  # inserted
                            if choice == '6':
                                restart_connection()
                            else:  # inserted
                                if choice == '7':
                                    stop_current_session()
                                else:  # inserted
                                    if choice == '8':
                                        turn_off_bot()
                                    else:  # inserted
                                        if choice == '9':
                                            print('\nInstructions:\n1. Use option 4 to select a specific trading pair, or option 5 to trade all pairs.\n2. The bot will automatically start trading with the selected pair.\n3. Press the ESC key at any time to stop trading for the current pair and return to the menu.\n4. You can modify MTG settings or turn off the bot from the menu.\n5. Option 10 shows a log of executed trades, and option 11 shows overall statistics.\n6. Option 12 tests the API connection.\n7. Ensure to use this bot in a controlled environment and be aware of financial risks.\n')
                                        else:  # inserted
                                            if choice == '10':
                                                view_trade_history()
                                            else:  # inserted
                                                if choice == '11':
                                                    view_trade_stats()
                                                else:  # inserted
                                                    if choice == '12':
                                                        test_api_connection(api)
                                                    else:  # inserted
                                                        if choice == '13':
                                                            reset_trade_statistics()
                                                        else:  # inserted
                                                            if choice == '14':
                                                                backtest_results = run_backtests_on_all_pairs(api, apiV2)
                                                                if backtest_results:
                                                                    print('\n=== Backtest Results (Sorted by Win Rate) ===')
                                                                    for pair, win_rate in backtest_results:
                                                                        print(f'{pair}: Win Rate = {win_rate:.2f}%')
                                                                else:  # inserted
                                                                    print('No results to display.')
                                                            else:  # inserted
                                                                print('Invalid option. Please select a number between 1 and 14.')
        show_status()
    save_trade_history()
    nest_asyncio.apply()