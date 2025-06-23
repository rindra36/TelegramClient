def analyze_trading_stats(initial_amount, initial_balance, data_array):
    """
    Analyze trading statistics with different martingale scenarios
    
    Args:
        initial_amount (float): Initial trading amount
        data_array (list): List of strings containing daily trading data
    
    Returns:
        dict: Dictionary containing daily analysis results
    """
    daily_results = {}
    
    for line in data_array:
        # Split the date and data
        date, stats = line.split(': ')
        # Convert stats to dictionary
        stats_dict = dict(item.split('=') for item in stats.split(', '))
        
        # Convert values to integers
        total_win = int(stats_dict['TotalWin'])
        win1 = int(stats_dict['WIN1'])
        win2 = int(stats_dict['WIN2'])
        win3 = int(stats_dict.get('WIN3', 0))
        loss = int(stats_dict['Loss'])
        win_without_martingale = total_win - win1 - win2 - win3
        
        # Calculate total amounts for martingale strategies
        martingale1_total = initial_amount + (initial_amount * 2)  # 1 + 2 = 3
        martingale2_total = initial_amount + (initial_amount * 2) + (initial_amount * 4)  # 1 + 2 + 4 = 7
        martingale3_total = initial_amount + (initial_amount * 2) + (initial_amount * 4) + (initial_amount * 8)  # 1 + 2 + 4 + 8 = 15

        # Calculate balance at the end of the days for each scenarios
        no_martingale_balance = initial_balance + (win_without_martingale * initial_amount) - ((win1 + win2 + win3 + loss) * initial_amount)

        # For Martingale 1: Initial trades + possible doubling on loss
        martingale1_balance = initial_balance + ((win_without_martingale + win1) * initial_amount) - ((win2 + win3 + loss) * martingale1_total)

        # For Martingale 2: All trades + possible doubling twice on loss
        martingale2_balance = initial_balance + ((win_without_martingale + win1 + win2) * initial_amount) - ((win3 + loss) * martingale2_total)

        # For Martingale 3: All trades + possible doubling thrice on loss
        martingale3_balance = initial_balance + (total_win * initial_amount) - (loss * martingale3_total)
        
        # Calculate scenarios
        no_martingale_profit = no_martingale_balance - initial_balance
        
        # For Martingale 1: Initial trades + possible doubling on loss
        martingale1_profit = martingale1_balance - initial_balance
        
        # For Martingale 2: All trades + possible doubling twice on loss
        martingale2_profit = martingale2_balance - initial_balance

        # For Martingale 3: All trades + possible doubling thrice on loss
        martingale3_profit = martingale3_balance - initial_balance
        
        # Store results for this day
        daily_results[date] = {
            'Trade Statistics': {
                'Total Wins': total_win,
                'Martingale 1 Wins': win1,
                'Martingale 2 Wins': win2,
                'Martingale 3 Wins': win3,
                'Losses': loss
            },
            'Profit Scenarios': {
                'No Martingale Strategy': no_martingale_profit,
                'Martingale 1 Strategy': martingale1_profit,
                'Martingale 2 Strategy': martingale2_profit,
                'Martingale 3 Strategy': martingale3_profit,
            },
            'Balances': {
                'No Martingale Balance': no_martingale_balance,
                'Martingale 1 Balance': martingale1_balance,
                'Martingale 2 Balance': martingale2_balance,
                'Martingale 3 Balance': martingale3_balance,
            }
        }
    
    return daily_results

# Example usage
data = [
    '2025-04-30: TotalWin=18, WIN1=5, WIN2=2, Loss=2'
]

result = analyze_trading_stats(1.5, 100, data)

# Print results
for date, day_results in result.items():
    print(f"\n=== {date} ===")
    for category, stats in day_results.items():
        print(f"\n{category}:")
        for key, value in stats.items():
            if 'Strategy' in key:
                print(f"{key}: ${value:,.2f}")
            elif 'Balance' in key:
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value}")