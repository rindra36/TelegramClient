from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

import pandas as pd
import asyncio
from datetime import datetime
import os

# Main part of the code
async def main(ssid: str):
    # Create output directory if it doesn't exist
    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)

    # The api automatically detects if the 'ssid' is for real or demo account
    api = PocketOptionAsync(ssid)    
    await asyncio.sleep(5)
    
    # Candñes are returned in the format of a list of dictionaries
    time = 60
            
    candles = await api.history("EURUSD_otc", time)
    # print(f"Raw Candles: {candles}")
    candles_pd = pd.DataFrame.from_dict(candles)

    # Create filename with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/candles_EURUSD_otc_time{time}_{timestamp}.csv"

    # Save to CSV file
    candles_pd.to_csv(filename, index=False)
    print(f"Saved candles to: {filename}")

    # Still print to console for immediate feedback
    print(f"Candles: {candles_pd}")
    
if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    asyncio.run(main(ssid))
    