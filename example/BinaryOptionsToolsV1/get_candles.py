# Made by Vigo Walker

from BinaryOptionsTools import pocketoption
import pandas as pd
import time as t
from datetime import datetime
import os

ssid = (r'42["auth",{"session":"vtftn12e6f5f5008moitsd6skl","isDemo":1,"uid":27658142,"platform":2}]')
demo = True
api = pocketoption(ssid, demo)

# Create output directory if it doesn't exist
output_dir = "candles_analysis"
os.makedirs(output_dir, exist_ok=True)

# Create output file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"candles_v1_analysis_{timestamp}.txt")

# Test different time frames and periods
times = [3600 * i for i in range(1, 25)]  # 1-24 hours back
time_frames = [1, 5, 15, 30]  # Different timeframes in seconds

i = 0
while True:
    i += 1
    try:
        if i == 2:
            break

        with open(output_file, 'w') as f:
            for time in times:
                for frame in time_frames:
                    output = f"\nBegin datetime : {datetime.now()}\n"
                    
                    # Get candles data
                    # candles = api.GetCandles("EURUSD_otc", frame, count=time)
                    candles = api.GetPrices("EURUSD_otc", frame, count=2)

                    output += f"End datetime : {datetime.now()}\n"
                    
                    # Convert to pandas DataFrame for analysis
                    candles_df = pd.DataFrame.from_dict(candles)
                    
                    # Prepare output text
                    output += f"Timeframe: {frame}s, Period: {time}s\n"
                    output += f"First {min(5, len(candles_df))} candles:\n{candles_df.head().to_string()}\n"
                    output += f"Last {min(5, len(candles_df))} candles:\n{candles_df.tail().to_string()}\n"
                    output += f"Number of candles: {len(candles_df)}\n"
                    output += "-" * 50 + "\n"
                    
                    # Print to console and write to file
                    print(output)
                    f.write(output)
                    f.flush()  # Ensure the output is written immediately
                    
        t.sleep(1)  # Delay to avoid overloading the API

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"An error occurred: {e}")
