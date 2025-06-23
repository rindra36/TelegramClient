from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
import pandas as pd
import asyncio
from datetime import datetime
import os

async def main(ssid: str):
    api = PocketOptionAsync(ssid)    
    await asyncio.sleep(5)  # Wait for connection
    
    # Create output directory if it doesn't exist
    output_dir = "candles_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"candles_history_analysis{timestamp}.txt")
    
    # Test different time frames and periods
    times = [3600 * i for i in range(1, 48)]  # 1-24 hours back
    time_frames = [1, 5, 15, 30, 60, 300]  # Different timeframes in seconds
    
    with open(output_file, 'w') as f:
        for time in times:
            output = f"\nBegin datetime : {datetime.now()}\n"

            # Get candles data
            candles = await api.history("EURUSD_otc", time)

            output += f"End datetime : {datetime.now()}\n"
            
            # Convert to pandas DataFrame for analysis
            candles_df = pd.DataFrame.from_dict(candles)
            
            # Prepare output text
            output += f"Period: {time}s\n"
            output += f"First {min(5, len(candles_df))} candles:\n{candles_df.head().to_string()}\n"
            output += f"Last {min(5, len(candles_df))} candles:\n{candles_df.tail().to_string()}\n"
            output += f"Number of candles: {len(candles_df)}\n"
            output += "-" * 50 + "\n"
            
            # Print to console and write to file
            print(output)
            f.write(output)
            f.flush()  # Ensure the output is written immediately

if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    asyncio.run(main(ssid))