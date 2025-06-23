import os

from datetime import datetime
from src.utils import load_credentials, setup_client, load_chats

creadentials = load_credentials()
client = setup_client('anon', creadentials['id'], creadentials['hash'])

async def main():
    # Get all dialogs
    # async for dialog in client.iter_dialogs():
    #     print(f'{dialog.id}:"{dialog.name}"')

    chats = load_chats()
    chat_to_load = 'POFSOM'

    # Get messages from a specific chat
    # id = -1001770790377  # Replace with the chat ID you want to check

    id = chats[chat_to_load]
    limit = 100  # Number of messages to retrieve

    # You can print the message history of any chat:
    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_candle = f"{output_dir}/{chat_to_load}_{timestamp}.txt"
    open(filename_candle, 'w').close()
        
    async for message in client.iter_messages(id, limit=limit):
        print("\n\n=========================================\n\n")
        print(f'ID: {message.id}\n')
        print(f'Date: {message.date}\n')
        print(f'Text: {message.text}\n')
    # async for message in client.iter_messages(id):
        with open(filename_candle, 'a') as file:
            file.write("\n\n=========================================\n\n")
            file.write(f'ID: {message.id}\n')
            file.write(f'Date: {message.date}\n')
            file.write(f'Text: {message.text}\n')

    # GavinBotM1
    # id = -1002126431207  # M1
    # messages = []
    # async for message in client.iter_messages(id, limit=200):
    #     messages.append({
    #         'id': message.id,
    #         'date': message.date,
    #         'text': message.text
    #     })
    # analyze_messages_gavin_bot(messages)

    # GavinBotM5
    # id = -1001989007993  # M5
    # messages = []
    # async for message in client.iter_messages(id, limit=1000):
    #     messages.append({
    #         'id': message.id,
    #         'date': message.date,
    #         'text': message.text
    #     })
    # analyze_messages_gavin_bot(messages)

    # Youssef
    # id = -1002074799242  # Youssef
    # messages = []
    # async for message in client.iter_messages(id, limit=1000):
    #     messages.append({
    #         'id': message.id,
    #         'date': message.date,
    #         'text': message.text
    #     })
    # analyze_messages_youssef(messages)

    # Matthew
    # id = -1002101961419  # Matthew
    # messages = []
    # async for message in client.iter_messages(id):
    #     messages.append({
    #         'id': message.id,
    #         'date': message.date,
    #         'text': message.text
    #     })
    # analyze_messages_matthew(messages)

def analyze_messages_gavin_bot(messages):
    from collections import defaultdict

    results = {
        'nonOTC': {
            'count': 0,
            'messages': []
        },
        'entry_time_nonzero': {
            'count': 0,
            'messages': []
        },
        'result_counts': {
            'win': 0,
            'win1': 0,
            'win2': 0,
            'loss': 0,
            'total_win': 0
        },
        'daily_stats': defaultdict(lambda: {
            'total_win': 0,
            'win1': 0,
            'win2': 0,
            'loss': 0,
            'messages': []
        })
    }

    for msg in messages:
        if not msg['text']:
            print(f'Skipping empty message: {msg}')
            continue

        text = msg['text'].strip()
        lines = text.split('\n')

        # Process signals
        if len(lines) > 1 and lines[0].startswith('🚀 POCKET OPTION'):
            for line in lines:
                # Check for stock name without -OTC
                if line.startswith('啐 '):
                    stock = line.split(' ', 1)[1]
                    if '-OTC' not in stock:
                        results['nonOTC']['count'] += 1
                        results['nonOTC']['messages'].append(msg['id'])
                
                # Check entry time seconds
                if line.startswith('⌚️ '):
                    time_str = line.split(' ', 1)[1]
                    seconds = time_str.split(':', 2)[-1]
                    if seconds != '00':
                        results['entry_time_nonzero']['count'] += 1
                        results['entry_time_nonzero']['messages'].append(msg['id'])

        # Process results
        else:
            result = text.strip()
            date_key = msg['date'].date().isoformat()

            if result == 'WIN ✅':
                results['result_counts']['win'] += 1
            elif result == 'WIN ✅¹':
                results['result_counts']['win1'] += 1
            elif result == 'WIN ✅²':
                results['result_counts']['win2'] += 1
            elif result == '✖️ Loss':
                results['result_counts']['loss'] += 1

            # Update daily stats
            if result in ['WIN ✅', 'WIN ✅¹', 'WIN ✅²']:
                results['daily_stats'][date_key]['total_win'] += 1

                if result == 'WIN ✅':
                    results['daily_stats'][date_key]['messages'].append(result)
                elif result == 'WIN ✅¹':
                    results['daily_stats'][date_key]['win1'] += 1
                    results['daily_stats'][date_key]['messages'].append(result)
                elif result == 'WIN ✅²':
                    results['daily_stats'][date_key]['win2'] += 1
                    results['daily_stats'][date_key]['messages'].append(result)
            elif result == '✖️ Loss':
                results['daily_stats'][date_key]['loss'] += 1
                results['daily_stats'][date_key]['messages'].append(result)

    # Calculate total wins
    results['result_counts']['total_win'] = (results['result_counts']['win'] + 
                                            results['result_counts']['win1'] + 
                                            results['result_counts']['win2'])

    # Convert defaultdict to normal dict for better usability
    results['daily_stats'] = dict(results['daily_stats'])

    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_candle = f"{output_dir}/gavinbot_{timestamp}.txt"
    open(filename_candle, 'w').close()
    with open(filename_candle, 'a') as file:
        file.write(f"Total WIN: {results['result_counts']['total_win']}\n")
        file.write(f"Total WIN1: {results['result_counts']['win1']}\n")
        file.write(f"Total WIN2: {results['result_counts']['win2']}\n")
        file.write(f"Total Loss: {results['result_counts']['loss']}\n")
        # Write daily stats
        file.write("Daily Stats:\n")
        for date, stats in results['daily_stats'].items():
            file.write(f"{date}: TotalWin={stats['total_win']}, WIN1={stats['win1']}, WIN2={stats['win2']}, Loss={stats['loss']}\n")
            # Reverse the messages for each day and make it more readable in the file
            file.write("Messages:\n")
            for message in reversed(stats['messages']):
                file.write(f"{message}\n")

    # Return the results for further use if needed
    print(results)
    
    return results

def analyze_messages_youssef(messages):
    from collections import defaultdict

    results = {
        'result_counts': {
            'win': 0,
            'win1': 0,
            'win2': 0,
            'win3': 0,
            'loss': 0,
            'total_win': 0
        },
        'daily_stats': defaultdict(lambda: {
            'total_win': 0,
            'win1': 0,
            'win2': 0,
            'win3': 0,
            'loss': 0,
            'messages': []
        })
    }

    for msg in messages:
        if not msg['text']:
            print(f'Skipping empty message: {msg}')
            continue

        text = msg['text'].strip()
        result = text.strip()
        date_key = msg['date'].date().isoformat()

        if result == '✅ WIN ✅ - Direct victory.':
            results['result_counts']['win'] += 1
        elif result == '✅ WIN¹ ✅ - Victory in Martingale 1.':
            results['result_counts']['win1'] += 1
        elif result == '✅ WIN² ✅ - Victory in Martingale 2.':
            results['result_counts']['win2'] += 1
        elif result == '✅ WIN³ ✅ - Victory in Martingale 3.':
            results['result_counts']['win3'] += 1
        elif result == '❌':
            results['result_counts']['loss'] += 1

        # Update daily stats
        if result in ['✅ WIN ✅ - Direct victory.', '✅ WIN¹ ✅ - Victory in Martingale 1.', '✅ WIN² ✅ - Victory in Martingale 2.', '✅ WIN³ ✅ - Victory in Martingale 3.']:
            results['daily_stats'][date_key]['total_win'] += 1

            if result == '✅ WIN ✅ - Direct victory.':
                results['daily_stats'][date_key]['messages'].append(result)
            elif result == '✅ WIN¹ ✅ - Victory in Martingale 1.':
                results['daily_stats'][date_key]['win1'] += 1
                results['daily_stats'][date_key]['messages'].append(result)
            elif result == '✅ WIN² ✅ - Victory in Martingale 2.':
                results['daily_stats'][date_key]['win2'] += 1
                results['daily_stats'][date_key]['messages'].append(result)
            elif result == '✅ WIN³ ✅ - Victory in Martingale 3.':
                results['daily_stats'][date_key]['win3'] += 1
                results['daily_stats'][date_key]['messages'].append(result)
        elif result == '❌':
            results['daily_stats'][date_key]['loss'] += 1
            results['daily_stats'][date_key]['messages'].append(result)

    # Calculate total wins
    results['result_counts']['total_win'] = (results['result_counts']['win'] + 
                                            results['result_counts']['win1'] + 
                                            results['result_counts']['win2'] +
                                            results['result_counts']['win3'])

    # Convert defaultdict to normal dict for better usability
    results['daily_stats'] = dict(results['daily_stats'])

    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_candle = f"{output_dir}/youssef_{timestamp}.txt"
    open(filename_candle, 'w').close()
    with open(filename_candle, 'a') as file:
        file.write(f"Total WIN: {results['result_counts']['total_win']}\n")
        file.write(f"Total WIN1: {results['result_counts']['win1']}\n")
        file.write(f"Total WIN2: {results['result_counts']['win2']}\n")
        file.write(f"Total WIN3: {results['result_counts']['win3']}\n")
        file.write(f"Total Loss: {results['result_counts']['loss']}\n")
        # Write daily stats
        file.write("Daily Stats:\n")
        for date, stats in results['daily_stats'].items():
            file.write(f"{date}: TotalWin={stats['total_win']}, WIN1={stats['win1']}, WIN2={stats['win2']}, WIN3={stats['win3']}, Loss={stats['loss']}\n")
            # Reverse the messages for each day and make it more readable in the file
            file.write("Messages:\n")
            for message in reversed(stats['messages']):
                file.write(f"{message}\n")

    # Return the results for further use if needed
    print(results)
    
    return results

def analyze_messages_matthew(messages):
    from collections import defaultdict

    results = {
        'result_counts': {
            'win': 0,
            'win1': 0,
            'win2': 0,
            'win3': 0,
            'loss': 0,
            'total_win': 0
        },
        'daily_stats': defaultdict(lambda: {
            'total_win': 0,
            'win1': 0,
            'win2': 0,
            'win3': 0,
            'loss': 0
        })
    }

    for msg in messages:
        if not msg['text']:
            print(f'Skipping empty message: {msg}')
            continue

        text = msg['text'].strip()
        result = text.strip()
        date_key = msg['date'].date().isoformat()

        if result == '✅ WIN⁰ ✅ - Direct WIN 🏆👏':
            results['result_counts']['win'] += 1
        elif result == '✅ WIN¹ ✅ - Victory in Martingale 1 \U0001faf5':
            results['result_counts']['win1'] += 1
        elif result == '✅ WIN² ✅ - Victory in Martingale 2 \U0001faf5':
            results['result_counts']['win2'] += 1
        elif result == '✅ WIN³ ✅ - Victory in Martingale 3 \U0001faf5':
            results['result_counts']['win3'] += 1
        elif result == '❌':
            results['result_counts']['loss'] += 1

        # Update daily stats
        if result in ['✅ WIN⁰ ✅ - Direct WIN 🏆👏', '✅ WIN¹ ✅ - Victory in Martingale 1 \U0001faf5', '✅ WIN² ✅ - Victory in Martingale 2 \U0001faf5', '✅ WIN³ ✅ - Victory in Martingale 3 \U0001faf5']:
            results['daily_stats'][date_key]['total_win'] += 1

            if result == '✅ WIN¹ ✅ - Victory in Martingale 1 \U0001faf5':
                results['daily_stats'][date_key]['win1'] += 1
            elif result == '✅ WIN² ✅ - Victory in Martingale 2 \U0001faf5':
                results['daily_stats'][date_key]['win2'] += 1
            elif result == '✅ WIN³ ✅ - Victory in Martingale 3 \U0001faf5':
                results['daily_stats'][date_key]['win3'] += 1
        elif result == '❌':
            results['daily_stats'][date_key]['loss'] += 1

    # Calculate total wins
    results['result_counts']['total_win'] = (results['result_counts']['win'] + 
                                            results['result_counts']['win1'] + 
                                            results['result_counts']['win2'] +
                                            results['result_counts']['win3'])

    # Convert defaultdict to normal dict for better usability
    results['daily_stats'] = dict(results['daily_stats'])

    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_candle = f"{output_dir}/matthew_{timestamp}.txt"
    open(filename_candle, 'w').close()
    with open(filename_candle, 'a') as file:
        file.write(f"Total WIN: {results['result_counts']['total_win']}\n")
        file.write(f"Total WIN1: {results['result_counts']['win1']}\n")
        file.write(f"Total WIN2: {results['result_counts']['win2']}\n")
        file.write(f"Total WIN3: {results['result_counts']['win3']}\n")
        file.write(f"Total Loss: {results['result_counts']['loss']}\n")
        # Write daily stats
        file.write("Daily Stats:\n")
        for date, stats in results['daily_stats'].items():
            file.write(f"{date}: TotalWin={stats['total_win']}, WIN1={stats['win1']}, WIN2={stats['win2']}, WIN3={stats['win3']}, Loss={stats['loss']}\n")

    # Return the results for further use if needed
    print(results)
    
    return results

with client:
    client.loop.run_until_complete(main())