from BinaryOptionsToolsV2.pocketoption import PocketOption

import pandas as pd
import time
from datetime import datetime
import os

# Main part of the code
def main(ssid: str):
    # Create output directory if it doesn't exist
    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # The api automatically detects if the 'ssid' is for real or demo account
    api = PocketOption(ssid)    
    time.sleep(5)
    
    # Candñes are returned in the format of a list of dictionaries
    candles = api.get_candles("EURUSD_otc", 60, 3600)
    # print(f"Raw Candles: {candles}")
    candles_pd = pd.DataFrame.from_dict(candles)
    
    # Create filename with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/candles_EURUSD_otc_time3600_{timestamp}.csv"

    # Save to CSV file
    candles_pd.to_csv(filename, index=False)
    print(f"Saved candles to: {filename}")

    # Still print to console for immediate feedback
    print(f"Candles: {candles_pd}")
    
if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    main(ssid)
    